"""Build resumable C1 FAISS shards for the prepared study corpora."""
import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
from itertools import islice
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from tokenizers import Tokenizer

from check_medcpt_faiss import MODEL_REVISIONS, encode, load_model
from check_retrieval_corpora import build_chunks
from retrieval_fixture import TOKENIZER_SHA256


FAISS_WRITER = """
from pathlib import Path
import sys
import faiss
import numpy as np

root = Path(sys.argv[1])
faiss.omp_set_num_threads(1)
vectors = np.load(root / "vectors.npy", allow_pickle=False)
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)
path = str(root / "index.faiss")
faiss.write_index(index, path)
saved = faiss.read_index(path)
if saved.ntotal != len(vectors) or not np.array_equal(saved.reconstruct_n(0, len(vectors)), vectors):
    raise ValueError("FAISS round trip changed indexed vectors")
scores, ids = saved.search(vectors[:1], len(vectors))
reference = vectors[:1] @ vectors.T
if not np.allclose(scores, np.take_along_axis(reference, ids, axis=1), rtol=1e-5, atol=1e-4):
    raise ValueError("FAISS scores differ from direct inner products")
"""


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_shard(dataset, records, tokenizer, article_encoder, target):
    rows = {row["pmid"]: row for row in records}
    if len(rows) != len(records):
        raise ValueError("Repeated PMID in shard")
    source, chunks_by_pmid = build_chunks(dataset, rows, tokenizer)
    chunks = [chunk for column in chunks_by_pmid.values() for chunk in column]
    if not chunks:
        raise ValueError("Shard has no indexable text")
    bodies = ["".join(source[s.pmid, s.field][s.start:s.end] for s in chunk.spans if s.field != "title")
              for chunk in chunks]
    if dataset == "bioasq":
        titles = [source.get((chunk.spans[0].pmid, "title"), "") for chunk in chunks]
        vectors = encode(titles, *article_encoder, 512, pairs=bodies)
    else:
        vectors = encode(bodies, *article_encoder, 512)
    with TemporaryDirectory(dir=target) as directory:
        stage = Path(directory)
        np.save(stage / "vectors.npy", vectors, allow_pickle=False)
        subprocess.run([sys.executable, "-c", FAISS_WRITER, directory], check=True)
        with (stage / "chunks.jsonl").open("w", encoding="utf-8") as output:
            for chunk in chunks:
                output.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        (stage / "index.faiss").replace(target / "index.faiss")
        (stage / "chunks.jsonl").replace(target / "chunks.jsonl")
    return len(chunks)


def build_dataset(root, dataset, args, tokenizer, encoder, model_hash):
    prepared = root / "outputs" / dataset / "prepared"
    corpus = prepared / "corpus.jsonl"
    manifest = read(prepared / "manifest.json")
    expected_count = manifest.get("corpus_records")
    if expected_count is None:
        expected_count = sum(section["retained"] for section in manifest["counts"].values())
    if sha(corpus) != manifest["corpus_sha256"]:
        raise ValueError(f"Prepared {dataset} corpus checksum mismatch")
    target = root / "outputs" / "retrieval" / "full_indexes" / "c1" / dataset
    target.mkdir(parents=True, exist_ok=True)
    settings = {
        "dataset": dataset, "arm": "c1", "corpus_sha256": manifest["corpus_sha256"],
        "documents_total": expected_count, "documents_per_shard": args.shard_docs,
        "model_revision": MODEL_REVISIONS["article"], "model_sha256": model_hash,
        "tokenizer_sha256": TOKENIZER_SHA256["medcpt"], "encoder_cap": 512,
        "window_tokens": 128, "overlap_tokens": 0, "embedding_precision": "fp32",
        "torch_version": torch.__version__, "torch_threads": args.threads,
        "index": "IndexFlatIP", "faiss_version": version("faiss-cpu"),
        "transformers_version": version("transformers"), "tokenizers_version": version("tokenizers"),
    }
    if (target / "settings.json").exists() and read(target / "settings.json") != settings:
        raise ValueError(f"Existing {dataset} shards use different settings")
    write(target / "settings.json", settings)

    documents = chunks = new = 0
    new_seconds = 0.0
    new_documents = 0
    new_bytes = 0
    with corpus.open("rb") as source:
        for number in range((expected_count + args.shard_docs - 1) // args.shard_docs):
            lines = list(islice(source, args.shard_docs))
            expected_batch = min(args.shard_docs, expected_count - documents)
            if len(lines) != expected_batch:
                raise ValueError(f"Unexpected record count in {dataset} shard {number}")
            fingerprint = hashlib.sha256(b"".join(lines)).hexdigest()
            shard = target / f"{number:05d}"
            checkpoint = shard / "manifest.json"
            if checkpoint.exists():
                saved = read(checkpoint)
                if (saved["rows_sha256"] != fingerprint or saved["documents"] != len(lines) or
                        sha(shard / "index.faiss") != saved["index_sha256"] or
                        sha(shard / "chunks.jsonl") != saved["chunks_sha256"]):
                    raise ValueError(f"Checkpoint verification failed for {dataset}/{number}")
            else:
                if args.max_new_shards is not None and new >= args.max_new_shards:
                    break
                shard.mkdir(exist_ok=True)
                start = perf_counter()
                count = build_shard(dataset, [json.loads(line) for line in lines], tokenizer, encoder, shard)
                elapsed = perf_counter() - start
                saved = {"rows_sha256": fingerprint, "documents": len(lines), "chunks": count,
                         "index_sha256": sha(shard / "index.faiss"), "chunks_sha256": sha(shard / "chunks.jsonl"),
                         "index_bytes": (shard / "index.faiss").stat().st_size, "elapsed_seconds": elapsed}
                write(checkpoint, saved)
                new += 1
                new_seconds += elapsed
                new_documents += len(lines)
                new_bytes += saved["index_bytes"]
                print(f"{dataset} shard {number}: {len(lines)} documents, {count} chunks, {elapsed:.1f}s", flush=True)
            documents += saved["documents"]
            chunks += saved["chunks"]
        if documents == expected_count and source.read(1):
            raise ValueError("Corpus contains rows beyond its manifest count")
    progress = {
        "complete": documents == expected_count, "documents_indexed": documents, "chunks_indexed": chunks,
        "documents_total": expected_count,
    }
    if new_documents:
        progress["latest_measurement"] = {
            "new_documents": new_documents, "seconds": new_seconds,
            "linear_full_build_hours": expected_count * new_seconds / new_documents / 3600,
            "linear_full_index_bytes": round(expected_count * new_bytes / new_documents),
            "caveat": "Extrapolation from the measured shards in corpus order. Document lengths vary.",
        }
    elif (target / "progress.json").exists():
        previous = read(target / "progress.json")
        if "latest_measurement" in previous:
            progress["latest_measurement"] = previous["latest_measurement"]
    write(target / "progress.json", progress)
    print(dataset, json.dumps(progress), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--dataset", nargs="+", choices=("bioasq", "pubmedqa"), default=["bioasq", "pubmedqa"])
    parser.add_argument("--shard-docs", type=int, default=64)
    parser.add_argument("--max-new-shards", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--model-dir", type=Path, help="Local copy of the pinned article encoder")
    args = parser.parse_args()
    if args.shard_docs < 1 or args.threads < 1 or (args.max_new_shards is not None and args.max_new_shards < 0):
        parser.error("Invalid shard or thread count")
    torch.set_num_threads(args.threads)
    model = args.model_dir or args.project / "models/medcpt-article"
    token_path = args.project / "models/tokenizers/medcpt.json"
    if ((model / "revision.txt").read_text().strip() != MODEL_REVISIONS["article"]
            or sha(token_path) != TOKENIZER_SHA256["medcpt"]
            or sha(model / "tokenizer.json") != TOKENIZER_SHA256["medcpt"]):
        raise ValueError("Wrong article model revision or tokenizer")
    baseline = read(args.project / "outputs/retrieval/medcpt_faiss_fixture.json")
    model_hash = sha(model / "model.safetensors")
    if model_hash != baseline["weights_sha256"]["article"]:
        raise ValueError("Article weights differ from the verified fixture")
    encoder = load_model(model)
    tokenizer = Tokenizer.from_file(str(token_path))
    for dataset in dict.fromkeys(args.dataset):
        build_dataset(args.project, dataset, args, tokenizer, encoder, model_hash)


if __name__ == "__main__":
    main()

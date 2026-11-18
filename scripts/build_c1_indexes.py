"""Build resumable chunking-arm FAISS shards for the prepared study corpora."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
from importlib.metadata import version
from itertools import islice
import json
import shutil
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from tokenizers import Tokenizer

from check_medcpt_faiss import MODEL_REVISIONS, encode, load_model, semantic_documents
from check_retrieval_corpora import build_chunks
from retrieval_fixture import TOKENIZER_SHA256
from semantic_chunking import build_semantic_spans


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


def build_shard(dataset, records, tokenizer, article_encoder, target, arm, calibration):
    rows = {row["pmid"]: row for row in records}
    if len(rows) != len(records):
        raise ValueError("Repeated PMID in shard")
    stats = Counter()
    semantic_spans = None
    if arm == "c3":
        field = "abstract" if dataset == "bioasq" else "text"
        docs = dict(zip(rows, semantic_documents(
            [row[field] for row in rows.values()], tokenizer, article_encoder), strict=True))

        def semantic_spans(pmid, field):
            spans, counts = build_semantic_spans(pmid, field, docs[pmid], calibration)
            stats.update(counts)
            return spans

    source, chunks_by_pmid = build_chunks(dataset, rows, tokenizer, arm, stats, semantic_spans)
    chunks = [chunk for column in chunks_by_pmid.values() for chunk in column]
    if not chunks:
        raise ValueError("Shard has no indexable text")
    bodies = ["".join(source[s.pmid, s.field][s.start:s.end] for s in chunk.spans if s.field != "title")
              for chunk in chunks]
    lengths = Counter()
    if arm != "c1":
        lengths.update(len(tokens.ids) for tokens in tokenizer.encode_batch(bodies, add_special_tokens=False))
        if max(lengths) > (128 if arm == "c2" else 192):
            raise ValueError("Chunk body exceeds arm token limit")
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
    return len(chunks), {"chunker_counts": dict(stats), "body_token_histogram": dict(lengths)}


def build_dataset(root, dataset, args, tokenizer, encoder, model_hash, calibration):
    prepared = root / "outputs" / dataset / "prepared"
    corpus = prepared / "corpus.jsonl"
    manifest = read(prepared / "manifest.json")
    expected_count = manifest.get("corpus_records")
    if expected_count is None:
        expected_count = sum(section["retained"] for section in manifest["counts"].values())
    if sha(corpus) != manifest["corpus_sha256"]:
        raise ValueError(f"Prepared {dataset} corpus checksum mismatch")
    target = root / "outputs" / "retrieval" / "full_indexes" / args.arm / dataset
    target.mkdir(parents=True, exist_ok=True)
    settings = {
        "dataset": dataset, "arm": args.arm, "corpus_sha256": manifest["corpus_sha256"],
        "documents_total": expected_count, "documents_per_shard": args.shard_docs,
        "model_revision": MODEL_REVISIONS["article"], "model_sha256": model_hash,
        "tokenizer_sha256": TOKENIZER_SHA256["medcpt"], "encoder_cap": 512,
        "window_tokens": 128, "overlap_tokens": 0, "embedding_precision": "fp32",
        "torch_version": torch.__version__, "torch_threads": args.threads,
        "index": "IndexFlatIP", "faiss_version": version("faiss-cpu"),
        "transformers_version": version("transformers"), "tokenizers_version": version("tokenizers"),
    }
    if args.arm == "c2":
        settings["sentence_splitter_sha256"] = sha(Path(__file__).with_name("sentence_packing.py"))
        settings["fragmentation_fallback_sha256"] = sha(Path(__file__).with_name("c2_sentence_packing.py"))
    if args.arm == "c3":
        settings.pop("window_tokens")
        settings.update(
            sentence_splitter_sha256=sha(Path(__file__).with_name("sentence_packing.py")),
            semantic_chunker_sha256=sha(Path(__file__).with_name("semantic_chunking.py")),
            calibration=calibration, minimum_body_tokens=64, maximum_body_tokens=192,
        )
    if (target / "settings.json").exists() and read(target / "settings.json") != settings:
        raise ValueError(f"Existing {dataset} shards use different settings")
    write(target / "settings.json", settings)

    documents = chunks = new = 0
    new_seconds = 0.0
    new_documents = 0
    new_bytes = 0
    new_checkpoint_bytes = 0
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
                if shutil.disk_usage(target).free < args.min_free_mib * 1024**2:
                    raise OSError(f"Free disk space below {args.min_free_mib} MiB; existing checkpoints are intact")
                shard.mkdir(exist_ok=True)
                start = perf_counter()
                count, details = build_shard(dataset, [json.loads(line) for line in lines],
                                             tokenizer, encoder, shard, args.arm, calibration)
                elapsed = perf_counter() - start
                saved = {"rows_sha256": fingerprint, "documents": len(lines), "chunks": count,
                         "index_sha256": sha(shard / "index.faiss"), "chunks_sha256": sha(shard / "chunks.jsonl"),
                         "index_bytes": (shard / "index.faiss").stat().st_size, "elapsed_seconds": elapsed}
                if args.arm != "c1":
                    saved.update(details)
                write(checkpoint, saved)
                new += 1
                new_seconds += elapsed
                new_documents += len(lines)
                new_bytes += saved["index_bytes"]
                new_checkpoint_bytes += sum((shard / name).stat().st_size
                                            for name in ("index.faiss", "chunks.jsonl", "manifest.json"))
                print(f"{args.arm} {dataset} shard {number}: {len(lines)} documents, {count} chunks, {elapsed:.1f}s", flush=True)
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
            "linear_full_checkpoint_bytes": round(expected_count * new_checkpoint_bytes / new_documents),
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
    parser.add_argument("--arm", choices=("c1", "c2", "c3"), default="c1")
    parser.add_argument("--calibration-report", type=Path, help="Saved pilot-only C3 calibration report")
    parser.add_argument("--min-free-mib", type=int, default=0)
    args = parser.parse_args()
    if args.min_free_mib < 0:
        parser.error("Minimum free space must be non-negative")
    if args.calibration_report is not None and args.arm != "c3":
        parser.error("--calibration-report requires --arm c3")
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
    calibration = None
    if args.arm == "c3":
        path = args.calibration_report or args.project / "outputs/retrieval/medcpt_faiss_fixture_c3.json"
        report = read(path)
        calibration = report["calibration"]["selected"]
        if (report["strategy"] != "c3" or not calibration["within_target"]
                or report["weights_sha256"]["article"] != model_hash
                or not report["calibration"]["pilot_document_ids"]
                or report["tokenizer_sha256"]["medcpt"] != TOKENIZER_SHA256["medcpt"]):
            raise ValueError("C3 calibration does not match the verified model, tokenizer or pilot")
        for dataset in args.dataset:
            manifest = read(args.project / "outputs" / dataset / "prepared/manifest.json")
            if report["corpus_sha256"][dataset] != manifest["corpus_sha256"]:
                raise ValueError(f"C3 calibration corpus differs: {dataset}")
    encoder = load_model(model)
    tokenizer = Tokenizer.from_file(str(token_path))
    for dataset in dict.fromkeys(args.dataset):
        build_dataset(args.project, dataset, args, tokenizer, encoder, model_hash, calibration)


if __name__ == "__main__":
    main()

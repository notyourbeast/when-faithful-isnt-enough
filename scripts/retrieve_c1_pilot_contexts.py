"""Search completed C1 shards and pack fresh reserved-pilot contexts."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def rank_shards(queries, shards, k=50):
    """Search each shard exactly and keep the global top k, with stable ties."""
    import faiss

    faiss.omp_set_num_threads(1)
    ranked = [[] for _ in queries]
    for number, folder in enumerate(shards):
        manifest = read(folder / "manifest.json")
        index_path = folder / "index.faiss"
        if sha(index_path) != manifest["index_sha256"]:
            raise ValueError(f"Index checksum mismatch: {folder}")
        index = faiss.read_index(str(index_path))
        if (not isinstance(index, faiss.IndexFlatIP) or index.d != queries.shape[1]
                or index.ntotal != manifest["chunks"]):
            raise ValueError(f"Unexpected FAISS index: {folder}")
        scores, offsets = index.search(queries, index.ntotal)
        for i, (row, ids) in enumerate(zip(scores, offsets, strict=True)):
            candidates = [(float(score), number, int(offset)) for score, offset in zip(row, ids, strict=True)]
            if not np.isfinite(row).all() or set(ids) != set(range(index.ntotal)):
                raise ValueError(f"Invalid search results: {folder}")
            ranked[i] = sorted(ranked[i] + candidates, key=lambda item: (-item[0], item[1], item[2]))[:k]
    return ranked


def worker(query_path, output, shards):
    queries = np.load(query_path, allow_pickle=False)
    output.write_text(json.dumps(rank_shards(queries, shards)), encoding="utf-8")


def search(queries, shards):
    # FAISS and PyTorch use separate processes to avoid OpenMP runtime conflicts.
    with TemporaryDirectory() as directory:
        temp = Path(directory)
        np.save(temp / "queries.npy", queries, allow_pickle=False)
        subprocess.run([sys.executable, __file__, "--worker", str(temp / "queries.npy"),
                        str(temp / "rankings.json"), *(str(s) for s in shards)], check=True)
        return read(temp / "rankings.json")


def completed_shards(folder, corpus_hash):
    settings = read(folder / "settings.json")
    progress = read(folder / "progress.json")
    if not progress["complete"] or progress["documents_indexed"] != settings["documents_total"]:
        raise ValueError(f"Full-corpus C1 index is incomplete: {folder}")
    if settings["corpus_sha256"] != corpus_hash or settings["arm"] != "c1":
        raise ValueError(f"Index corpus or arm mismatch: {folder}")
    n = (settings["documents_total"] + settings["documents_per_shard"] - 1) // settings["documents_per_shard"]
    shards = [folder / f"{number:05d}" for number in range(n)]
    manifests = [read(shard / "manifest.json") for shard in shards]
    for number, manifest in enumerate(manifests):
        documents = min(settings["documents_per_shard"], settings["documents_total"] - number * settings["documents_per_shard"])
        if manifest["documents"] != documents:
            raise ValueError(f"Wrong document count in index shard {number}")
    if sum(m["chunks"] for m in manifests) != progress["chunks_indexed"]:
        raise ValueError(f"Index totals do not reconcile: {folder}")
    return shards, progress


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]), Path(sys.argv[3]), [Path(item) for item in sys.argv[4:]])
        return

    from collections import defaultdict

    import torch
    from tokenizers import Tokenizer

    from check_medcpt_faiss import MODEL_REVISIONS, encode, gold_for, load_model
    from check_mistral_generation import load_fresh_cases
    from check_retrieval_corpora import load_corpus
    from retrieval_fixture import CONTEXT_BUDGET, TOKENIZER_SHA256, Chunk, Span, covered_characters, pack

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--query-model-dir", type=Path)
    args = parser.parse_args()
    root = args.project

    prepared = {name: root / "outputs" / name / "prepared" for name in ("bioasq", "pubmedqa")}
    hashes = {name: read(path / "manifest.json")["corpus_sha256"] for name, path in prepared.items()}
    validated = {name: completed_shards(root / "outputs/retrieval/full_indexes/c1" / name, hashes[name])
                 for name in prepared}
    baseline = read(root / "outputs/retrieval/medcpt_faiss_fixture.json")
    model = args.query_model_dir or root / "models/medcpt-query"
    model_hash = sha(model / "model.safetensors")
    if ((model / "revision.txt").read_text().strip() != MODEL_REVISIONS["query"]
            or model_hash != baseline["weights_sha256"]["query"]):
        raise ValueError("Query encoder differs from verified retrieval fixture")
    tokenizers = {}
    for name, expected in TOKENIZER_SHA256.items():
        path = root / "models/tokenizers" / f"{name}.json"
        if sha(path) != expected:
            raise ValueError(f"Wrong {name} tokenizer")
        tokenizers[name] = Tokenizer.from_file(str(path))
    torch.set_num_threads(4)
    encoder = load_model(model)
    cases = defaultdict(dict)
    for dataset, task, qid, question, _, arm, _ in load_fresh_cases(root, tokenizers["mistral"]):
        if arm == "r1":
            cases[dataset][qid] = (task, question)

    report = {"arm": "c1", "scope": "Fresh reserved prompt comparison cases, complete corpora",
              "query_weights_sha256": model_hash, "corpus_sha256": hashes, "datasets": {}}
    for dataset, questions in cases.items():
        shards, progress = validated[dataset]
        ids = list(questions)
        rankings = search(encode([questions[qid][1] for qid in ids], *encoder, 64), shards)
        needed_by_shard = defaultdict(set)
        for ranking in rankings:
            for _, number, offset in ranking:
                needed_by_shard[number].add(offset)
        chunks = {}
        for number, needed in needed_by_shard.items():
            path = shards[number] / "chunks.jsonl"
            if sha(path) != read(shards[number] / "manifest.json")["chunks_sha256"]:
                raise ValueError(f"Chunk provenance checksum mismatch: {path}")
            with path.open(encoding="utf-8") as rows:
                for index, line in enumerate(rows):
                    if index in needed:
                        row = json.loads(line)
                        chunks[number, index] = Chunk(row["id"], tuple(Span(**span) for span in row["spans"]))
        pmids = {span.pmid for chunk in chunks.values() for span in chunk.spans}
        path = prepared[dataset] / "corpus.jsonl"
        corpus = load_corpus(path, hashes[dataset], pmids)
        source = {(pmid, field): text for pmid, row in corpus.items() for field, text in row.items() if field != "pmid"}
        gold, original_questions = gold_for(dataset, root, ids, path)
        results = []
        for qid, ranking in zip(ids, rankings, strict=True):
            task, question = questions[qid]
            if question != original_questions[qid]:
                raise ValueError(f"Question identity mismatch: {qid}")
            context, count, trace, delivered = pack(
                [chunks[number, offset] for _, number, offset in ranking], source, tokenizers["mistral"])
            if count > CONTEXT_BUDGET or any(
                context[lo:hi] != source[s.pmid, s.field][s.start:s.end] for s, lo, hi in delivered
            ):
                raise ValueError(f"Invalid context: {qid}")
            for item, (score, number, offset) in zip(trace, ranking, strict=True):
                item.update(inner_product=score, shard=number, offset=offset)
            results.append({
                "question_id": qid, "question": question, "task": task, "context": context,
                "context_tokens": count,
                "medcpt_context_tokens": len(tokenizers["medcpt"].encode(context, add_special_tokens=False).ids),
                "gold_characters": sum(s.end - s.start for s in gold[qid]),
                "covered_characters": sum(covered_characters(s, delivered) for s in gold[qid]),
                "rank_trace": trace,
                "delivered": [{"source": asdict(s), "context_span": [lo, hi]} for s, lo, hi in delivered],
            })
        report["datasets"][dataset] = {"documents": progress["documents_indexed"],
                                       "chunks": progress["chunks_indexed"], "queries": results}
    output = root / "outputs/retrieval/c1_fresh_full_contexts.json"
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(output)
    print(f"Saved full-corpus C1 contexts for {sum(len(q) for q in cases.values())} fresh pilot questions")


if __name__ == "__main__":
    main()

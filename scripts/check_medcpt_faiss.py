"""Run MedCPT and exact FAISS search on a small, fixed pilot corpus."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
from itertools import islice
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile

from importlib.metadata import version
import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import AutoModel, AutoTokenizer, __version__ as transformers_version

from check_retrieval_corpora import build_chunks, load_corpus, require
from reference_arms import oracle_input, question_only_input
from semantic_chunking import SemanticDocument, build_semantic_spans, calibrate, measure_documents
from retrieval_fixture import (
    CONTEXT_BUDGET, RANK_LIMIT, TOKENIZER_SHA256, Span, covered_characters, pack, pack_overlapping,
)

MODEL_REVISIONS = {
    "article": "d05a736da4bb84ee4057b7f7999485be6ed85465",
    "query": "d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc",
}


def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(path, local_files_only=True, use_safetensors=True)
    return tokenizer, model.cpu().float().eval()


def encode(items, tokenizer, model, max_length, pairs=None):
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(items), 16):
            batch = tokenizer(
                items[start:start + 16],
                text_pair=None if pairs is None else pairs[start:start + 16],
                return_tensors="pt", padding=True, truncation=True, max_length=max_length,
            )
            vectors.append(model(**batch).last_hidden_state[:, 0, :].numpy())
    result = np.ascontiguousarray(np.concatenate(vectors), dtype=np.float32)
    require(result.ndim == 2 and result.shape[1] == 768 and np.isfinite(result).all(),
            "Expected finite 768-dimensional MedCPT vectors")
    return result


def gold_for(dataset, root, question_ids, path):
    if dataset == "bioasq":
        audit = root / "outputs/bioasq/alignment_full/alignment.json"
        gold = {
            row["id"]: [Span(item["pmid"], item["section"], start, end)
                        for item in row["gold"] for start, end in item["spans"]]
            for row in json.loads(audit.read_bytes()) if row["id"] in question_ids
        }
        with zipfile.ZipFile(root / "data/raw/bioasq/BioASQ-training13b.zip") as archive:
            rows = json.load(archive.open("BioASQ-training13b/training13b.json"))["questions"]
        return gold, {row["id"]: row["body"] for row in rows if row["id"] in question_ids}
    queries = {}
    for suffix in ("l", "u"):
        rows = json.loads((root / f"data/raw/pubmedqa/ori_pqa{suffix}.json").read_bytes())
        queries.update({pmid: row["QUESTION"] for pmid, row in rows.items() if pmid in question_ids})
    # The prepared context text fixes the source span, including its separators.
    gold = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["pmid"] in question_ids:
                pmid = row["pmid"]
                gold[pmid] = [Span(pmid, "text", 0, len(row["text"]))]
    return gold, queries


def exact_search(matrix, queries):
    # Keep FAISS separate from PyTorch so their OpenMP libraries never meet.
    worker = """
import sys
from pathlib import Path
import faiss
import numpy as np

root = Path(sys.argv[1])
faiss.omp_set_num_threads(1)
vectors = np.load(root / "vectors.npy")
queries = np.load(root / "queries.npy")
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)
scores, indices = index.search(queries, len(vectors))
np.savez(root / "result.npz", scores=scores, indices=indices)
"""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        np.save(root / "vectors.npy", matrix)
        np.save(root / "queries.npy", queries)
        subprocess.run([sys.executable, "-c", worker, directory], check=True)
        with np.load(root / "result.npz") as result:
            return result["scores"], result["indices"]


def semantic_documents(texts, tokenizer, article_encoder):
    measured = measure_documents(texts, lambda sentences: encode(sentences, *article_encoder, 512))
    return [SemanticDocument(text, dims, tokenizer) for text, dims in zip(texts, measured, strict=True)]


def calibrate_pilot(root, fixture, tokenizer, article_encoder):
    bio = root / "outputs/bioasq/alignment_full"
    reserved = json.loads((bio / "eligibility_and_pilot.json").read_bytes())
    pilot = {qid for stratum in reserved["strata"].values() for qid in stratum["pilot_ids"]}
    raw_alignment = (bio / "alignment.json").read_bytes()
    require(hashlib.sha256(raw_alignment).hexdigest() == reserved["alignment_sha256"],
            "BioASQ calibration pool refers to another alignment")
    bio_ids = {span["pmid"] for row in json.loads(raw_alignment) if row["id"] in pilot for span in row["gold"]}
    pqa = json.loads((root / "outputs/pubmedqa/prepared/manifest.json").read_bytes())
    pqa_ids = set(pqa["pqa_l_pilot_ids"] + pqa["pqa_u_pilot_ids"])
    texts, ids_by_dataset = [], {}
    for dataset, ids, field in (("bioasq", bio_ids, "abstract"), ("pubmedqa", pqa_ids, "text")):
        path = root / f"outputs/{dataset}/prepared/corpus.jsonl"
        rows = load_corpus(path, fixture["corpus_sha256"][dataset], ids)
        ids_by_dataset[dataset] = sorted((pmid for pmid, row in rows.items() if row[field].strip()), key=int)
        texts.extend(rows[pmid][field] for pmid in ids_by_dataset[dataset])
    print(f"C3: encoding sentences in {len(texts)} pilot-linked documents", flush=True)
    docs = semantic_documents(texts, tokenizer, article_encoder)
    print(f"C3: calibrating over {sum(len(doc.distances) for doc in docs)} adjacent sentence pairs", flush=True)
    selected, trials = calibrate(docs)
    print(f"C3: percentile {selected['percentile']}, median {selected['median_tokens']} tokens", flush=True)
    return {"pilot_document_ids": ids_by_dataset, "selected": selected, "trials": trials,
            "selection": "Pooled chunk median closest to 128; ties use lower percentile. Flag results outside ±15%."}


def run_dataset(dataset, root, fixture, medcpt, mistral, article_encoder, query_encoder, strategy, setting, top5):
    path = root / f"outputs/{dataset}/prepared/corpus.jsonl"
    question_ids = sorted({row["question_id"] for row in fixture["cases"] if row["dataset"] == dataset})
    gold, queries = gold_for(dataset, root, question_ids, path)
    require(set(question_ids) == gold.keys() == queries.keys(), f"Missing pilot questions in {dataset}")
    with path.open(encoding="utf-8") as source:
        background = {json.loads(line)["pmid"] for line in islice(source, 24)}
    needed = background | {s.pmid for spans in gold.values() for s in spans}
    rows = load_corpus(path, fixture["corpus_sha256"][dataset], needed)
    chunk_stats = Counter(fragmented_sentences=0)
    semantic_spans = None
    if strategy == "c3":
        field = "abstract" if dataset == "bioasq" else "text"
        docs = dict(zip(rows, semantic_documents([row[field] for row in rows.values()], medcpt, article_encoder), strict=True))

        def semantic_spans(pmid, field):
            spans, counts = build_semantic_spans(pmid, field, docs[pmid], setting)
            chunk_stats.update(counts)
            require(all(docs[pmid].count(s.start, s.end) <= 192 for s in spans), "C3 chunk exceeds maximum")
            return spans

    source, by_pmid = build_chunks(dataset, rows, medcpt, strategy, chunk_stats, semantic_spans)
    chunks = [chunk for pmid in sorted(by_pmid, key=int) for chunk in by_pmid[pmid]]

    def body(chunk):
        spans = [s for s in chunk.spans if s.field != "title"]
        return "".join(source[s.pmid, s.field][s.start:s.end] for s in spans)

    bodies = [body(chunk) for chunk in chunks]
    titles = None
    if dataset == "bioasq":
        titles = [source.get((chunk.spans[0].pmid, "title"), "") for chunk in chunks]
        matrix = encode(titles, *article_encoder, 512, pairs=bodies)
    else:
        matrix = encode(bodies, *article_encoder, 512)
    embedding_lengths = None
    if strategy == "r0":
        encoder = article_encoder[0]
        inputs = titles if titles is not None else bodies
        pairs = bodies if titles is not None else None
        lengths = encoder(inputs, text_pair=pairs, truncation=True, max_length=512, return_length=True)
        embedding_lengths = dict(zip((chunk.id for chunk in chunks), lengths["length"], strict=True))
        require(all(0 < length <= 512 for length in embedding_lengths.values()), "Encoder token count out of bounds")
        require(len(chunks) == len(rows), "R0 must index one unit per document")
    query_vectors = encode([queries[qid] for qid in question_ids], *query_encoder, 64)
    scores, indices = exact_search(matrix, query_vectors)
    baseline = query_vectors @ matrix.T
    require(np.allclose(scores, np.take_along_axis(baseline, indices, axis=1), rtol=1e-5, atol=1e-4),
            f"{dataset}: FAISS scores differ from direct dot products")
    require(np.allclose(scores, np.sort(baseline, axis=1)[:, ::-1], rtol=1e-5, atol=1e-4),
            f"{dataset}: FAISS rank scores differ from direct sorting")
    require(all(len(set(row)) == len(chunks) for row in indices), "Missing or duplicate FAISS indices")

    results = []
    references = []

    def span_records(delivered):
        return [{"source": asdict(s), "context_span": [lo, hi]} for s, lo, hi in delivered]

    for qid, row, distances in zip(question_ids, indices, scores, strict=True):
        ranked = [chunks[i] for i in row[:5 if top5 else RANK_LIMIT]]
        if strategy == "c1_overlap":
            context, count, trace, delivered = pack_overlapping(ranked, source, mistral)
        else:
            context, count, trace, delivered = pack(
                ranked, source, mistral, budget=None if top5 else CONTEXT_BUDGET,
                overflow="whole_document" if strategy == "r0" else "skip")
        if top5:
            require(len(trace) == min(5, len(chunks)) and all(item["accepted"] for item in trace),
                    "Fixed top-5 dropped a ranked candidate")
        else:
            require(count <= CONTEXT_BUDGET, "Context exceeds budget")
        if strategy == "c1_overlap":
            spans = sorted((s.pmid, s.field, s.start, s.end) for s, _, _ in delivered)
            require(all(a[:2] != b[:2] or a[3] <= b[2] for a, b in zip(spans, spans[1:])),
                    "Unmerged overlapping evidence in delivered context")
        require(count == len(mistral.encode(context, add_special_tokens=False).ids), "Context token mismatch")
        require(all(context[start:end] == source[s.pmid, s.field][s.start:s.end] for s, start, end in delivered),
                "Delivered context contains a mismatched span")
        require(all(0 <= s.start < s.end <= len(source[s.pmid, s.field]) for s in gold[qid]), "Invalid gold span")
        covered = sum(covered_characters(s, delivered) for s in gold[qid])
        total = sum(s.end - s.start for s in gold[qid])
        require(0 <= covered <= total, "Invalid coverage")
        for item, score in zip(trace, distances):
            item["inner_product"] = float(score)
            if embedding_lengths is not None:
                item["article_encoder_tokens"] = embedding_lengths[item["chunk"]]
        results.append({
            "question_id": qid, "gold_characters": total, "covered_characters": covered,
            "context_tokens": count, "context": context, "rank_trace": trace,
            "first_ranked_gold_overlap": next((rank for rank, i in enumerate(row, 1)
                if any(covered_characters(s, [(span, 0, 0) for span in chunks[i].spans]) for s in gold[qid])), None),
            "delivered": span_records(delivered),
        })
        if strategy == "r0":
            oracle, oracle_spans = oracle_input(queries[qid], gold[qid], source, mistral)
            require(oracle["context_tokens"] <= CONTEXT_BUDGET and
                    oracle["context_tokens"] == len(mistral.encode(oracle["context"], add_special_tokens=False).ids),
                    "Invalid R1 context token count")
            require(all(oracle["context"][lo:hi] == source[s.pmid, s.field][s.start:s.end]
                        and any((s.pmid, s.field) == (g.pmid, g.field) and g.start <= s.start < s.end <= g.end
                                for g in gold[qid]) for s, lo, hi in oracle_spans), "Unexpected R1 source span")
            covered_oracle = sum(covered_characters(s, oracle_spans) for s in gold[qid])
            require(covered_oracle == sum(s.end - s.start for s, _, _ in oracle_spans), "R1 contains duplicate evidence")
            r2 = question_only_input(queries[qid])
            require(not r2["context"] and not r2["faithfulness_applicable"] and not r2["context_recall_applicable"],
                    "R2 contains context or an inapplicable score")
            references.append({"question_id": qid,
                               "r1": {**oracle, "gold_characters": total, "covered_characters": covered_oracle,
                                      "delivered": span_records(oracle_spans)},
                               "r2": r2})
    print(f"{dataset}: {len(rows)} articles, {len(chunks)} chunks, "
          f"{sum(row['covered_characters'] > 0 for row in results)}/{len(results)} contexts with evidence", flush=True)
    return {"articles": len(rows), "chunks": len(chunks), "queries": results, **chunk_stats,
            **({"reference_inputs": references} if strategy == "r0" else {}),
            "max_dot_product_error": float(np.max(np.abs(scores - np.take_along_axis(baseline, indices, axis=1))))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--strategy", choices=("c1", "c2", "c3", "r0", "c1_overlap"), default="c1")
    parser.add_argument("--top5", action="store_true", help="Deliver the first five ranked chunks regardless of context length")
    args = parser.parse_args()
    if args.top5 and args.strategy not in {"c1", "c2", "c3"}:
        parser.error("--top5 is defined only for C1, C2 and C3")
    root = args.project
    torch.set_num_threads(4)
    fixture = json.loads((root / "outputs/retrieval/corpus_fixture.json").read_bytes())
    tokenizers = {}
    for name, expected in TOKENIZER_SHA256.items():
        path = root / "models/tokenizers" / f"{name}.json"
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Wrong {name} tokenizer")
        tokenizers[name] = Tokenizer.from_file(str(path))
    models, revisions, fingerprints = {}, {}, {}
    for role in ("article", "query"):
        path = root / "models" / f"medcpt-{role}"
        revisions[role] = (path / "revision.txt").read_text().strip()
        require(revisions[role] == MODEL_REVISIONS[role], f"Wrong MedCPT {role} revision")
        with (path / "model.safetensors").open("rb") as file:
            fingerprints[role] = hashlib.file_digest(file, "sha256").hexdigest()
        models[role] = load_model(path)
    require(hashlib.sha256((root / "models/medcpt-article/tokenizer.json").read_bytes()).hexdigest()
            == TOKENIZER_SHA256["medcpt"], "MedCPT article and chunk tokenizers differ")
    calibration = None
    if args.strategy == "c3":
        if args.top5:
            baseline = json.loads((root / "outputs/retrieval/medcpt_faiss_fixture_c3.json").read_bytes())
            require(baseline["corpus_sha256"] == fixture["corpus_sha256"] and baseline["weights_sha256"] == fingerprints,
                    "Saved C3 calibration uses different inputs")
            calibration = baseline["calibration"]
        else:
            calibration = calibrate_pilot(root, fixture, tokenizers["medcpt"], models["article"])
    report = {
        "scope": f"{args.strategy.upper()} smoke test on prior pilot cases; pool deliberately includes cited sources and first 24 corpus rows",
        "strategy": args.strategy,
        "packing": "fixed top-5 without a length limit" if args.top5 else "1024-token budget",
        **({"overlap_tokens": 16, "merge_policy": "Union same-PMID, same-field overlaps at first contributing rank before budgeting"}
           if args.strategy == "c1_overlap" else {}),
        **({"reference_policy": {
            "r0": "Pretruncate documents longer than the full 1024-token budget, then skip any resulting unit that does not fit remaining capacity.",
            "r1": "Merged gold spans by numeric PMID, title before abstract, then source offset. Pack a prefix, clipping the last fitting span at a token boundary.",
            "r2": "Question only. Exclude faithfulness and context recall."
        }} if args.strategy == "r0" else {}),
        **({"calibration": calibration} if calibration else {}),
        "revisions": revisions, "weights_sha256": fingerprints,
        "tokenizer_sha256": TOKENIZER_SHA256, "corpus_sha256": fixture["corpus_sha256"],
        "versions": {"faiss": version("faiss-cpu"), "torch": torch.__version__,
                     "transformers": transformers_version, "numpy": np.__version__},
        "methods": "CPU fp32 CLS vectors, query limit 64, article limit 512, raw inner product, exact IndexFlatIP",
        "datasets": {dataset: run_dataset(dataset, root, fixture, tokenizers["medcpt"], tokenizers["mistral"],
                                          models["article"], models["query"], args.strategy,
                                          calibration["selected"] if calibration else None, args.top5)
                     for dataset in ("bioasq", "pubmedqa")},
    }
    suffix = "" if args.strategy == "c1" else f"_{args.strategy}"
    if args.top5:
        suffix = f"_{args.strategy}_top5"
    output = root / f"outputs/retrieval/medcpt_faiss_fixture{suffix}.json"
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(output)


if __name__ == "__main__":
    main()

"""Verify fixed windows, context assembly and evidence coverage on pilot records."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

from reference_arms import whole_document_spans
from c2_sentence_packing import pack_sentences

from retrieval_fixture import (
    CONTEXT_BUDGET, RANK_LIMIT, TOKENIZER_SHA256, Chunk, Span,
    covered_characters, fixed_windows, pack,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_bytes())


def load_corpus(path, expected_hash, needed):
    selected = {}
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for raw in source:
            digest.update(raw)
            row = json.loads(raw)
            if row["pmid"] in needed:
                require(row["pmid"] not in selected, f"Duplicate PMID {row['pmid']}")
                selected[row["pmid"]] = row
    require(digest.hexdigest() == expected_hash, f"Corpus checksum mismatch: {path}")
    require(selected.keys() == needed, f"Missing selected articles in {path}")
    return selected


def build_chunks(dataset, rows, tokenizer, strategy="c1", stats=None, semantic_spans=None):
    source, chunks = {}, {}
    for pmid, row in rows.items():
        title = row["title"] if dataset == "bioasq" else ""
        field = "abstract" if dataset == "bioasq" else "text"
        body = row[field]
        source[pmid, field] = body
        if strategy == "r0":
            spans = whole_document_spans(pmid, field, body)
        elif strategy == "c3":
            spans = semantic_spans(pmid, field)
        elif strategy == "c2":
            spans, fragmented = pack_sentences(tokenizer, pmid, field, body)
            if stats is not None:
                stats["fragmented_sentences"] += fragmented
        else:
            spans = fixed_windows(tokenizer, pmid, field, body, overlap=16 if strategy == "c1_overlap" else 0)
        if strategy == "c1_overlap" and spans:
            require(spans[0].start == 0 and spans[-1].end == len(body)
                    and all(a.start < b.start < a.end < b.end for a, b in zip(spans, spans[1:])),
                    f"Overlapping windows do not cover {pmid}")
        else:
            require("".join(body[s.start:s.end] for s in spans) == body or not body.strip(),
                f"Windows do not reconstruct {pmid}")
        header = ()
        if title:
            source[pmid, "title"] = title
            header = (Span(pmid, "title", 0, len(title)),)
        chunks[pmid] = [
            Chunk(f"{pmid}:{field}:{span.start}", (*header, span)) for span in spans
        ] or ([Chunk(f"{pmid}:title", header)] if header else [])
    return source, chunks


def exercise(dataset, question_id, case, ranked, source, gold, tokenizer):
    context, count, trace, delivered = pack(ranked, source, tokenizer)
    require(len(trace) == min(len(ranked), RANK_LIMIT), "Incorrect rank cutoff")
    require(count <= CONTEXT_BUDGET and count == len(tokenizer.encode(context, add_special_tokens=False).ids),
            "Context count or budget mismatch")
    require(all(context[start:end] == source[span.pmid, span.field][span.start:span.end]
                for span, start, end in delivered), "Delivered span differs from source")
    if dataset == "pubmedqa":
        require(all(span.field == "text" for span, _, _ in delivered), "Unexpected field in PubMedQA context")
    for span in gold:
        require(0 <= span.start < span.end <= len(source[span.pmid, span.field]), "Invalid gold span")
    covered = sum(covered_characters(span, delivered) for span in gold)
    total = sum(span.end - span.start for span in gold)
    # Independently count distinct source character positions to check interval arithmetic.
    delivered_chars = {(s.pmid, s.field, index) for s, _, _ in delivered for index in range(s.start, s.end)}
    direct = sum((s.pmid, s.field, index) in delivered_chars for s in gold for index in range(s.start, s.end))
    require(covered == direct, "Evidence coverage disagrees with direct character count")
    require((covered > 0 if case == "evidence_first" else covered == 0), "Unexpected evidence coverage")
    return {
        "dataset": dataset, "question_id": question_id, "case": case,
        "context": context, "context_tokens": count, "rank_trace": trace,
        "gold_characters": total, "covered_characters": covered,
        "delivered": [{"source": asdict(s), "context_span": [start, end]} for s, start, end in delivered],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    args = parser.parse_args()
    root = args.project
    tokenizers = {}
    for name, expected in TOKENIZER_SHA256.items():
        path = root / "models" / "tokenizers" / f"{name}.json"
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Wrong {name} tokenizer")
        tokenizers[name] = Tokenizer.from_file(str(path))
    medcpt, mistral = tokenizers["medcpt"], tokenizers["mistral"]
    bio_audit = root / "outputs/bioasq/alignment_full"
    bio_prepared = root / "outputs/bioasq/prepared"
    pqa_prepared = root / "outputs/pubmedqa/prepared"
    bio_manifest = read_json(bio_prepared / "manifest.json")
    pqa_manifest = read_json(pqa_prepared / "manifest.json")
    split = read_json(bio_audit / "eligibility_and_pilot.json")
    raw_alignment = (bio_audit / "alignment.json").read_bytes()
    alignment_hash = hashlib.sha256(raw_alignment).hexdigest()
    require(alignment_hash == bio_manifest["alignment_sha256"] == split["alignment_sha256"],
            "BioASQ pool and corpus reference different alignments")
    alignment = {row["id"]: row for row in json.loads(raw_alignment)}
    pilot_ids = [qid for kind in sorted(split["strata"]) for qid in split["strata"][kind]["pilot_ids"][:2]]
    # Include a pilot case with title evidence to exercise header provenance.
    title_id = next(qid for kind in sorted(split["strata"]) for qid in split["strata"][kind]["pilot_ids"]
                    if any(item["section"] == "title" for item in alignment[qid]["gold"]))
    bio_cases = {
        qid: [Span(item["pmid"], item["section"], start, end)
              for item in alignment[qid]["gold"] for start, end in item["spans"]]
        for qid in dict.fromkeys([*pilot_ids, title_id])
    }
    bio_pmids = {s.pmid for spans in bio_cases.values() for s in spans}
    bio_rows = load_corpus(bio_prepared / "corpus.jsonl", bio_manifest["corpus_sha256"], bio_pmids)
    pqa_ids = pqa_manifest["pqa_l_pilot_ids"][:2] + pqa_manifest["pqa_u_pilot_ids"][:2]
    pqa_rows = load_corpus(pqa_prepared / "corpus.jsonl", pqa_manifest["corpus_sha256"], set(pqa_ids))
    pqa_cases = {pmid: [Span(pmid, "text", 0, len(pqa_rows[pmid]["text"]))] for pmid in pqa_ids}
    results = []
    for dataset, rows, cases in (("bioasq", bio_rows, bio_cases), ("pubmedqa", pqa_rows, pqa_cases)):
        source, chunks = build_chunks(dataset, rows, medcpt)
        for case_id, gold in cases.items():
            relevant = {span.pmid for span in gold}
            candidates = [c for pmid, parts in chunks.items() if pmid in relevant for c in parts]
            evidence = [c for c in candidates if any(covered_characters(s, [(span, 0, 0) for span in c.spans])
                                                     for s in gold)]
            distractors = [c for pmid, parts in chunks.items() if pmid not in relevant for c in parts]
            require(bool(evidence and distractors), f"Fixture requires both candidate groups for {case_id}")
            results.append(exercise(dataset, case_id, "evidence_first", evidence + distractors, source, gold, mistral))
            results.append(exercise(dataset, case_id, "distractors_only", distractors, source, gold, mistral))
    require(any(item["dataset"] == "bioasq" and item["case"] == "evidence_first"
                and any(span["source"]["field"] == "title" for span in item["delivered"]) for item in results),
            "No BioASQ title span was delivered")
    report = {
        "ranking": "preset fixture order",
        "tokenizer_sha256": TOKENIZER_SHA256,
        "corpus_sha256": {"bioasq": bio_manifest["corpus_sha256"], "pubmedqa": pqa_manifest["corpus_sha256"]},
        "cases": results,
    }
    output = root / "outputs/retrieval/corpus_fixture.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Passed {len(results)} scenarios across {len(bio_cases)} BioASQ and {len(pqa_cases)} PubMedQA pilot questions")
    print(f"Maximum delivered context: {max(row['context_tokens'] for row in results)} tokens")


if __name__ == "__main__":
    main()

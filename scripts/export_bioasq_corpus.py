"""Export the cached BioASQ PubMed corpus and verify the existing gold alignment."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

from audit_bioasq_alignment import CANDIDATE_TYPES, align, merge_spans, parse_pubmed, pmid


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary_raw = (args.audit / "summary.json").read_bytes()
    alignment_raw = (args.audit / "alignment.json").read_bytes()
    summary, alignment = json.loads(summary_raw), json.loads(alignment_raw)
    with args.archive.open("rb") as source:
        archive_hash = hashlib.file_digest(source, "sha256").hexdigest()
    require(archive_hash == summary["archive_sha256"], "Archive checksum differs from alignment audit")
    with zipfile.ZipFile(args.archive) as archive:
        questions = json.load(archive.open("BioASQ-training13b/training13b.json"))["questions"]

    requested = sorted({pmid(url) for q in questions for url in q["documents"]}, key=int)
    records, books = {}, set()
    xml_hash = hashlib.sha256()
    for start in range(0, len(requested), 200):
        ids = requested[start:start + 200]
        key = hashlib.sha256(",".join(ids).encode()).hexdigest()
        raw = (args.audit / "xml" / f"{key}.xml").read_bytes()
        xml_hash.update(hashlib.sha256(raw).digest())
        articles, book_ids = parse_pubmed(raw)
        require(articles.keys() <= set(ids), f"Unexpected article ID in cache batch {key}")
        require(not records.keys() & articles.keys(), f"Duplicate article in batch {key}")
        records.update(articles)
        books.update(book_ids)
        if start % 4000 == 0 or start + 200 >= len(requested):
            print(f"Read {min(start + 200, len(requested))}/{len(requested)} requested IDs from cache", flush=True)
    excluded = (set(requested) & books) - records.keys()
    missing = set(requested) - records.keys() - excluded
    require(len(requested) == summary["requested_pmids"] and len(records) == summary["records"]
            and excluded == set(summary["excluded_book_pmids"]) and missing == set(summary["missing_pmids"]),
            "Cached article inventory differs from the completed audit")

    by_id = {row["id"]: row for row in alignment}
    candidates = [q for q in questions if q["type"] in CANDIDATE_TYPES]
    require(len(by_id) == len(alignment) == len(candidates)
            and set(by_id) == {q["id"] for q in candidates}, "Candidate question IDs differ")
    outcomes, eligible = Counter(), Counter()
    merged_span_count = 0
    for q in candidates:
        row = by_id[q["id"]]
        require(row["type"] == q["type"] and len(q["snippets"]) == len(row["snippets"]),
                f"Question metadata differs for {q['id']}")
        for annotation, saved in zip(q["snippets"], row["snippets"], strict=True):
            status, span = align(annotation, records, excluded)
            require(saved["pmid"] == pmid(annotation["document"])
                    and saved["section"] == annotation["beginSection"]
                    and saved["end_section"] == annotation["endSection"]
                    and saved["annotated_span"] == [annotation["offsetInBeginSection"], annotation["offsetInEndSection"]]
                    and (saved["status"], saved["span"]) == (status, span),
                    f"Snippet audit differs for {q['id']}")
            outcomes[status] += 1
        gold = merge_spans(row["snippets"])
        require(gold == row["gold"], f"Merged gold differs for {q['id']}")
        merged_span_count += sum(len(item["spans"]) for item in gold)
        if gold:
            eligible[q["type"]] += 1
    require(outcomes == summary["snippet_outcomes"] and eligible == summary["eligible_questions"],
            "Recomputed audit totals differ")

    args.output.mkdir(parents=True, exist_ok=True)
    corpus_hash = hashlib.sha256()
    empty_ids = []
    article_count = 0
    with TemporaryDirectory(dir=args.output) as temp:
        staging = Path(temp)
        with (staging / "corpus.jsonl").open("wb") as output:
            for identifier, fields in sorted(records.items(), key=lambda item: int(item[0])):
                if not any(text.strip() for text in fields.values()):
                    empty_ids.append(identifier)
                    continue
                raw = (json.dumps({"pmid": identifier, **fields}, ensure_ascii=False) + "\n").encode()
                output.write(raw)
                corpus_hash.update(raw)
                article_count += 1
        manifest = {
            "archive_sha256": archive_hash,
            "alignment_sha256": hashlib.sha256(alignment_raw).hexdigest(),
            "summary_sha256": hashlib.sha256(summary_raw).hexdigest(),
            "cached_xml_sha256": xml_hash.hexdigest(),
            "cached_xml_hash_method": "SHA-256 over concatenated binary batch SHA-256 digests in requested PMID order",
            "corpus_sha256": corpus_hash.hexdigest(),
            "canonical_text": summary["canonical_text"],
            "span_indexing": summary["span_indexing"],
            "corpus_records": article_count,
            "excluded_empty_pmids": empty_ids,
            "candidate_questions_verified": len(candidates),
            "snippet_outcomes": outcomes,
            "merged_gold_spans_verified": merged_span_count,
            "eligible_questions": eligible,
        }
        payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        destination = args.output / "manifest.json"
        if destination.exists():
            require(destination.read_text() == payload, "Existing corpus manifest differs")
        (staging / "manifest.json").write_text(payload, encoding="utf-8")
        (staging / "corpus.jsonl").replace(args.output / "corpus.jsonl")
        (staging / "manifest.json").replace(destination)
    print(payload)


if __name__ == "__main__":
    main()

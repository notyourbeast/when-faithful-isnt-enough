"""Cache cited PubMed records and audit BioASQ Training 13b snippet spans."""
import argparse
import hashlib
import http.client
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

from audit_bioasq_alignment_sample import normalise

SECTIONS = {"title", "abstract"}
CANDIDATE_TYPES = {"factoid", "yesno"}
PMID_URL = re.compile(r"https?://(?:www\.ncbi\.nlm\.nih\.gov/pubmed/|pubmed\.ncbi\.nlm\.nih\.gov/)([0-9]+)/?")


def pmid(url):
    match = PMID_URL.fullmatch(url)
    if not match:
        raise ValueError(f"Unrecognised BioASQ document URL: {url}")
    return match[1]


def parse_pubmed(raw):
    root = ET.fromstring(raw)
    if root.tag != "PubmedArticleSet" or root.find(".//ERROR") is not None:
        raise ValueError("Unexpected PubMed EFetch response")
    records = {}
    for citation in root.findall("./PubmedArticle/MedlineCitation"):
        article = citation.find("Article")
        if article is None:
            continue
        title = article.find("ArticleTitle")
        records[citation.findtext("PMID")] = {
            "title": "".join(title.itertext()) if title is not None else "",
            "abstract": " ".join("".join(part.itertext()) for part in article.findall("./Abstract/AbstractText")),
        }
    book_ids = {node.text for node in root.findall("./PubmedBookArticle/BookDocument/PMID") if node.text}
    return records, book_ids


def get_batch(ids, cache):
    key = hashlib.sha256(",".join(ids).encode()).hexdigest()
    path = cache / f"{key}.xml"
    if path.exists():
        return parse_pubmed(path.read_bytes())
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    body = urllib.parse.urlencode({
        "db": "pubmed", "id": ",".join(ids), "retmode": "xml",
        "tool": "faithful_rag_alignment",
    }).encode()
    for attempt in range(4):
        time.sleep(0.4 if not attempt else 2 ** attempt)
        try:
            with urllib.request.urlopen(url, data=body, timeout=35) as response:
                raw = response.read()
            parsed = parse_pubmed(raw)
            break
        except (OSError, ValueError, ET.ParseError, http.client.IncompleteRead) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code < 500 and error.code != 429:
                raise
            if attempt == 3:
                raise
    temp = path.with_suffix(".tmp")
    temp.write_bytes(raw)
    temp.replace(path)
    return parsed


def occurrences(text, needle):
    return [match.start() for match in re.finditer(f"(?={re.escape(needle)})", text)]


def normalised_offsets(source, needle):
    # Find the normalised occurrence first. Then map it through a decomposed
    # character stream carrying original offsets. Reject any mapping that
    # cannot be proven to denote the same substring.
    normal = normalise(source)
    hits = occurrences(normal, needle)
    if len(hits) != 1:
        return ("ambiguous_normalised" if hits else "no_match"), None
    pieces, positions = [], []
    for match in re.finditer(r"\S+|\s+", source):
        if match.group().isspace():
            if pieces and match.end() < len(source):
                pieces.append(" ")
                positions.append((match.start(), match.end()))
            continue
        for offset in range(match.start(), match.end()):
            chars = unicodedata.normalize("NFD", source[offset])
            pieces.extend(chars)
            positions.extend([(offset, offset + 1)] * len(chars))
    if "".join(pieces) != unicodedata.normalize("NFD", normal):
        return "normalised_mapping_unverified", None
    start = len(unicodedata.normalize("NFD", normal[:hits[0]]))
    end = start + len(unicodedata.normalize("NFD", needle))
    lo, hi = positions[start][0], positions[end - 1][1]
    if normalise(source[lo:hi]) != needle:
        return "normalised_mapping_unverified", None
    return "normalised_reanchor", [lo, hi]


def align(s, records, excluded_book_ids):
    section = s["beginSection"]
    if section not in SECTIONS or s["endSection"] != section:
        return "outside_indexed_sections", None
    identifier = pmid(s["document"])
    record = records.get(identifier)
    if record is None:
        if identifier in excluded_book_ids:
            return "book_record_excluded", None
        return "pubmed_record_unavailable", None
    source = record[section]
    if not source.strip():
        return "source_field_empty", None
    snippet = s["text"]
    if not snippet.strip():
        return "empty_snippet", None
    lo, hi = s["offsetInBeginSection"], s["offsetInEndSection"]
    if 0 <= lo < hi <= len(source) and source[lo:hi] == snippet:
        return "offset_exact", [lo, hi]
    hits = occurrences(source, snippet)
    if len(hits) > 1:
        return "ambiguous_exact", None
    if hits:
        return "exact_reanchor", [hits[0], hits[0] + len(snippet)]
    return normalised_offsets(source, normalise(snippet))


def merge_spans(rows):
    by_source = {}
    for row in rows:
        if row["span"] is not None:
            by_source.setdefault((row["pmid"], row["section"]), []).append(row["span"])
    result = []
    for (identifier, section), spans in sorted(by_source.items()):
        merged = []
        for lo, hi in sorted(spans):
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(hi, merged[-1][1])
            else:
                merged.append([lo, hi])
        result.append({"pmid": identifier, "section": section, "spans": merged})
    return result


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-limit", type=int, help="Only cache this many batches without finalising the audit")
    args = parser.parse_args()
    with args.archive.open("rb") as file:
        checksum = hashlib.file_digest(file, "sha256").hexdigest()
    with zipfile.ZipFile(args.archive) as archive:
        questions = json.load(archive.open("BioASQ-training13b/training13b.json"))["questions"]
    ids = sorted({pmid(url) for q in questions for url in q["documents"]}, key=int)
    cache = args.output / "xml"
    cache.mkdir(parents=True, exist_ok=True)
    batches = [ids[i:i + 200] for i in range(0, len(ids), 200)]
    records = {}
    book_ids = set()
    for index, batch in enumerate(batches[:args.batch_limit], 1):
        batch_records, batch_book_ids = get_batch(batch, cache)
        records.update(batch_records)
        book_ids.update(batch_book_ids)
        if index == 1 or index % 10 == 0 or index == len(batches):
            print(f"Loaded {index}/{len(batches)} batches, {len(records)} records", flush=True)
    if args.batch_limit is not None and args.batch_limit < len(batches):
        return
    excluded_book_ids = (set(ids) & book_ids) - records.keys()
    outcomes, candidate_counts, eligible = Counter(), Counter(), Counter()
    results = []
    for q in questions:
        if q["type"] not in CANDIDATE_TYPES:
            continue
        candidate_counts[q["type"]] += 1
        rows = []
        for s in q["snippets"]:
            identifier = pmid(s["document"])
            status, span = align(s, records, excluded_book_ids)
            outcomes[status] += 1
            rows.append({
                "pmid": identifier, "section": s["beginSection"], "end_section": s["endSection"],
                "annotated_span": [s["offsetInBeginSection"], s["offsetInEndSection"]],
                "status": status, "span": span,
            })
        gold = merge_spans(rows)
        if gold:
            eligible[q["type"]] += 1
        results.append({"id": q["id"], "type": q["type"], "gold": gold, "snippets": rows})
    write_json(args.output / "alignment.json", results)
    summary = {
        "archive_sha256": checksum, "canonical_text": "XML itertext, abstract blocks joined with one space",
        "span_indexing": "Zero-based half-open Unicode character offsets within title or abstract",
        "requested_pmids": len(ids), "records": len(records),
        "corpus_scope": "PubMed journal article records with title or abstract fields",
        "excluded_book_pmids": sorted(excluded_book_ids, key=int),
        "missing_pmids": sorted(set(ids) - records.keys() - excluded_book_ids, key=int),
        "candidate_questions": candidate_counts, "eligible_questions": eligible,
        "snippet_outcomes": outcomes,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "missing_pmids"}, indent=2))


if __name__ == "__main__":
    main()

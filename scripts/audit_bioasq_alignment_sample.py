"""Check BioASQ snippet offsets against a small, reproducible PubMed sample."""
import argparse
import hashlib
import json
import random
import re
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


def normalise(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def classify(annotation, source):
    if not source:
        return "source_unavailable"
    start, end, snippet = annotation
    if 0 <= start < end <= len(source) and source[start:end] == snippet:
        return "offset_exact"
    if 0 <= start <= end < len(source) and source[start:end + 1] == snippet:
        return "inclusive_end_exact_diagnostic"
    matches = [m.start() for m in re.finditer(f"(?={re.escape(snippet)})", source)]
    if matches:
        return "unique_exact_reanchor" if len(matches) == 1 else "ambiguous_exact_reanchor"
    needle = normalise(snippet)
    if needle:
        matches = list(re.finditer(f"(?={re.escape(needle)})", normalise(source)))
        if matches:
            return "unique_normalised_candidate" if len(matches) == 1 else "ambiguous_normalised_candidate"
    return "no_match"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    with zipfile.ZipFile(args.archive) as archive:
        questions = json.loads(archive.read("BioASQ-training13b/training13b.json"))["questions"]
    for question in questions:
        if question["type"] not in {"factoid", "yesno"}:
            continue
        for snippet in question["snippets"]:
            section = snippet["beginSection"]
            if section not in {"title", "abstract"} or snippet["endSection"] != section or not snippet["text"].strip():
                continue
            start, end = snippet["offsetInBeginSection"], snippet["offsetInEndSection"]
            group = (section, "same_length" if end - start == len(snippet["text"]) else "different_length")
            groups[group].append((question["id"], snippet))
    rng = random.Random(13)
    sample = []
    for group, rows in sorted(groups.items()):
        rng.shuffle(rows)
        seen = set()
        for question_id, snippet in rows:
            pmid = snippet["document"].rstrip("/").rsplit("/", 1)[-1]
            if pmid.isdecimal() and pmid not in seen:
                sample.append((group, question_id, pmid, snippet))
                seen.add(pmid)
            if len(seen) == 8:
                break
    pmids = sorted({row[2] for row in sample}, key=int)
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urllib.parse.urlencode({
        "db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "tool": "faithful_rag_alignment",
    })
    xml_path = args.output / "pubmed.xml"
    if xml_path.exists():
        data = xml_path.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
    root = ET.fromstring(data)
    if root.tag != "PubmedArticleSet":
        raise ValueError(f"Unexpected EFetch root {root.tag}")
    records = {}
    for item in root.findall("./PubmedArticle/MedlineCitation"):
        article = item.find("Article")
        pmid = item.findtext("PMID")
        if article is None or pmid is None:
            continue
        title = article.find("ArticleTitle")
        parts = ["".join(part.itertext()) for part in article.findall("./Abstract/AbstractText")]
        records[pmid] = ("".join(title.itertext()) if title is not None else "", parts)
    results = []
    for (section, group), question_id, pmid, snippet in sample:
        title, parts = records.get(pmid, ("", []))
        annotation = snippet["offsetInBeginSection"], snippet["offsetInEndSection"], snippet["text"]
        representations = {"title": title} if section == "title" else {
            "abstract_space": " ".join(parts), "abstract_newline": "\n".join(parts), "abstract_join": "".join(parts),
        }
        results.append({
            "question_id": question_id, "pmid": pmid, "section": section, "offset_length_group": group,
            "begin": annotation[0], "end": annotation[1], "snippet_length": len(annotation[2]),
            "abstract_sections": len(parts),
            "outcomes": {name: classify(annotation, source) for name, source in representations.items()},
        })
    counts = defaultdict(Counter)
    for row in results:
        for name, outcome in row["outcomes"].items():
            counts[f"{name}/{row['offset_length_group']}"][outcome] += 1
    report = {
        "sample_method": "Eight distinct PMIDs per section and offset-length group, seed 13",
        "source_url": url, "xml_sha256": hashlib.sha256(data).hexdigest(),
        "requested_pmids": pmids, "missing_pmids": sorted(set(pmids) - records.keys()),
        "normalised_matches_are_candidates_only": True,
        "counts": dict(counts), "sample": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if not xml_path.exists():
        xml_path.write_bytes(data)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"requested": len(pmids), "missing_pmids": report["missing_pmids"], "counts": counts}, indent=2))


if __name__ == "__main__":
    main()

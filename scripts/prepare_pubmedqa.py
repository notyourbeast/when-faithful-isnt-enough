"""Build a context-only PubMedQA corpus and reproducible pilot splits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


SEED = 13


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def rank(ids, namespace):
    return sorted(ids, key=lambda pmid: (sha256(f"{SEED}:{namespace}:{pmid}".encode()), pmid))


def write(path, content):
    temp = path.with_suffix(".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit = json.loads((args.raw / "source_audit.json").read_bytes())
    corpus, seen = {}, {}
    labels, unlabelled, issues, input_hashes = {}, [], [], {}
    counts = {}
    for split in ("pqa_l", "pqa_u"):
        source = audit["files"][split]
        raw = (args.raw / source["file"]).read_bytes()
        input_hashes[split] = sha256(raw)
        if input_hashes[split] != source["sha256"]:
            raise ValueError(f"{split} checksum differs from intake audit")
        records = json.loads(raw)
        counts[split] = Counter()
        for pmid in sorted(records, key=int):
            record = records[pmid]
            sections = record["CONTEXTS"]
            if not isinstance(sections, list) or any(not isinstance(item, str) for item in sections):
                raise ValueError(f"Invalid context sections for {split}/{pmid}")
            parts = [item.strip() for item in sections if item.strip()]
            if len(parts) != len(sections):
                counts[split]["records_with_blank_sections"] += 1
                issues.append({"split": split, "pmid": pmid, "action": "drop_blank_sections",
                               "sections_removed": len(sections) - len(parts)})
            text = " ".join(parts)
            reason = "empty_context" if not text else "duplicate_pmid" if pmid in corpus else "duplicate_text" if text in seen else None
            if reason:
                if split == "pqa_l":
                    raise ValueError(f"Labelled source {pmid} requires review: {reason}")
                counts[split][reason] += 1
                issue = {"split": split, "pmid": pmid, "action": reason}
                if reason == "duplicate_text":
                    issue["kept_pmid"] = seen[text]
                issues.append(issue)
                continue
            corpus[pmid] = text
            seen[text] = pmid
            counts[split]["retained"] += 1
            if split == "pqa_l":
                label = record["final_decision"]
                if label not in {"yes", "no", "maybe"}:
                    raise ValueError(f"Invalid label for {pmid}: {label}")
                labels[pmid] = label
            else:
                unlabelled.append(pmid)

    if len(labels) != 1000 or len(unlabelled) < 100:
        raise ValueError("Inputs do not support the planned pilot and main sizes")

    by_label = Counter(labels.values())
    quotas = {label: count // 10 for label, count in by_label.items()}
    order = sorted(by_label, key=lambda label: (-(by_label[label] % 10), label))
    for label in order[:100 - sum(quotas.values())]:
        quotas[label] += 1
    labelled_pilot = {pmid for label, quota in quotas.items()
                      for pmid in rank([p for p, value in labels.items() if value == label], f"pqa_l:{label}")[:quota]}
    unlabelled_pilot = rank(unlabelled, "pqa_u")[:100]
    corpus_text = "".join(json.dumps({"pmid": pmid, "text": text}, ensure_ascii=False) + "\n"
                          for pmid, text in sorted(corpus.items(), key=lambda item: int(item[0])))
    manifest = {
        "seed": SEED,
        "selection": "Rank by SHA-256 of seed:namespace:PMID, tie-break by PMID. PQA-L quotas proportional by label (largest remainder, ties by label).",
        "input_sha256": input_hashes,
        "corpus_sha256": sha256(corpus_text.encode("utf-8")),
        "corpus_policy": "Use CONTEXTS only. Trim section edges, drop blank sections, join with one space. Exclude empty PQA-U contexts and duplicate PMIDs or exact prepared texts. Prefer PQA-L over PQA-U, then lower PMID. Stop for review if policy would remove a PQA-L source.",
        "counts": counts,
        "pqa_l_pilot_by_label": quotas,
        "pqa_l_pilot_ids": sorted(labelled_pilot, key=int),
        "pqa_l_main_ids": sorted(labels.keys() - labelled_pilot, key=int),
        "pqa_u_pilot_ids": sorted(unlabelled_pilot, key=int),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != manifest_text:
        raise SystemExit("Existing manifest differs. Review before replacing it.")
    write(args.output / "corpus.jsonl", corpus_text)
    write(args.output / "cleaning_log.json", json.dumps(issues, indent=2) + "\n")
    write(manifest_path, manifest_text)
    print(json.dumps({"corpus_records": len(corpus), "counts": counts, "pqa_l_pilot_by_label": quotas}, indent=2))


if __name__ == "__main__":
    main()

"""Reserve a proportionally stratified BioASQ pilot from the alignment audit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

PILOT_SIZE = 100
SEED = 13


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def build_manifest(alignment_bytes, summary_bytes):
    rows = json.loads(alignment_bytes)
    summary = json.loads(summary_bytes)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate question IDs")
    candidates = Counter(row["type"] for row in rows)
    by_type = {kind: [] for kind in sorted(candidates)}
    for row in rows:
        if row["gold"]:
            by_type[row["type"]].append(row["id"])
    counts = {kind: len(ids) for kind, ids in by_type.items()}
    if candidates != summary["candidate_questions"] or counts != summary["eligible_questions"]:
        raise ValueError("Alignment rows do not reconcile with the audit summary")
    total = sum(counts.values())
    if total < PILOT_SIZE:
        raise ValueError("Too few eligible questions for the pilot")

    quotas = {kind: PILOT_SIZE * count // total for kind, count in counts.items()}
    order = sorted(counts, key=lambda kind: (-(PILOT_SIZE * counts[kind] % total), kind))
    for kind in order[:PILOT_SIZE - sum(quotas.values())]:
        quotas[kind] += 1

    strata = {}
    for kind, ids in by_type.items():
        ranked = sorted(ids, key=lambda qid: (digest(f"{SEED}:{kind}:{qid}".encode()), qid))
        strata[kind] = {
            "pilot_ids": sorted(ranked[:quotas[kind]]),
            "main_candidate_ids": sorted(ranked[quotas[kind]:]),
        }
    return {
        "dataset": "BioASQ Training 13b",
        "archive_sha256": summary["archive_sha256"],
        "alignment_sha256": digest(alignment_bytes),
        "summary_sha256": digest(summary_bytes),
        "seed": SEED,
        "allocation": "Proportional by eligible question type, largest remainder, ties by type name",
        "selection": "Within each type, take the lowest SHA-256 hashes of UTF-8 seed:type:question_id; break ties by question ID",
        "eligible_counts": counts,
        "pilot_counts": quotas,
        "main_candidate_counts": {kind: counts[kind] - quotas[kind] for kind in counts},
        "strata": strata,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = build_manifest(
        (args.audit_directory / "alignment.json").read_bytes(),
        (args.audit_directory / "summary.json").read_bytes(),
    )
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != payload:
            raise SystemExit("Existing manifest differs. Review before replacing the split.")
    else:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(payload)
    print(json.dumps({key: manifest[key] for key in ("eligible_counts", "pilot_counts", "main_candidate_counts")}, indent=2))


if __name__ == "__main__":
    main()

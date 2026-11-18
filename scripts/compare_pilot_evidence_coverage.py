"""Compare delivered evidence across C1–C3 for the reserved pilot pool."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
arms = ("c1", "c2", "c3")
groups = ("bioasq", "pqa_l", "pqa_u")
reports, checksums, rows = {}, {}, {}
for arm in arms:
    payload = (output / f"{arm}_full_pilot_contexts.json").read_bytes()
    checksums[arm] = hashlib.sha256(payload).hexdigest()
    reports[arm] = json.loads(payload)
    if reports[arm]["arm"] != arm:
        raise ValueError(f"Wrong arm: {arm}")
    rows[arm] = {}
    for dataset, data in reports[arm]["datasets"].items():
        for row in data["queries"]:
            key = (dataset, row["question_id"])
            if (key in rows[arm] or row["pilot_stratum"] not in groups or len(row["rank_trace"]) != 50
                    or not 0 <= row["context_tokens"] <= 1024
                    or not 0 <= row["covered_characters"] <= row["gold_characters"]
                    or row["gold_characters"] <= 0):
                raise ValueError(f"Invalid retrieval row: {arm}/{key}")
            rows[arm][key] = row
    if Counter(r["pilot_stratum"] for r in rows[arm].values()) != {group: 100 for group in groups}:
        raise ValueError(f"Unexpected pilot stratum counts for {arm}")
    if reports[arm]["corpus_sha256"] != reports["c1"]["corpus_sha256"] or rows[arm].keys() != rows["c1"].keys():
        raise ValueError(f"Unpaired pilot reports for {arm}")
    for key, row in rows[arm].items():
        if any(row[k] != rows["c1"][key][k] for k in ("pilot_stratum", "question", "task", "gold_characters")):
            raise ValueError(f"Unpaired record for {arm}/{key}")

by_group = {}
for group in groups:
    keys = [k for k, r in rows["c1"].items() if r["pilot_stratum"] == group]
    by_group[group] = {}
    for arm in arms:
        count = Counter()
        for key in keys:
            row, base = rows[arm][key], rows["c1"][key]
            n, denom = row["covered_characters"], row["gold_characters"]
            count["zero" if n == 0 else "full" if n == denom else "partial"] += 1
            count["more_than_c1" if n > base["covered_characters"] else
                  "less_than_c1" if n < base["covered_characters"] else "same_as_c1"] += 1
        by_group[group][arm] = {**count, "questions": len(keys),
            "mean_coverage": sum(rows[arm][k]["covered_characters"] / rows[arm][k]["gold_characters"]
                                 for k in keys) / len(keys)}
summary = {
    "scope": "Descriptive evidence coverage, 300 previously reserved pilot questions",
    "input_sha256": checksums,
    "by_group": by_group,
    "total": {arm: {
        **{metric: sum(by_group[g][arm].get(metric, 0) for g in groups) for metric in
           ("zero", "partial", "full", "more_than_c1", "less_than_c1", "same_as_c1", "questions")},
        "mean_coverage": sum(by_group[g][arm]["mean_coverage"] for g in groups) / len(groups),
    } for arm in arms},
}
path = output / "c1_c2_c3_pilot_coverage.json"
temp = path.with_suffix(".tmp")
temp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
temp.replace(path)
print(json.dumps(summary, indent=2))

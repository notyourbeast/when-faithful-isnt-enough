"""Run the Mistral answer-format check on reserved, labelled pilot questions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import re
import unicodedata
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import zipfile

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from reference_arms import oracle_input
from retrieval_fixture import Span

MODEL = "mistral:7b-instruct-v0.3-q4_K_M"
MODEL_DIGEST = "6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf"
TOKENIZER_REVISION = "c170c708c41dac9275d15a8fff4eca08d52bab71"
TOKENIZER_HASH = "e553af6fff7d7ad76e830608b218c5c0b0822998d5a1a96099a74cd3c1cb1a49"
OPTIONS = {"temperature": 0, "top_k": 1, "seed": 13, "num_predict": 256, "num_ctx": 4096}
ABSTAIN = "INSUFFICIENT EVIDENCE"
ANSWER_LINE = re.compile(r"^[ \t]*ANSWER:[ \t]*(.*)$", re.MULTILINE)
LABELS = {"yesno": {"yes", "no"}, "pubmedqa": {"yes", "no", "maybe"}}


def normalise_entity(text):
    lowered = "".join(char for char in text.lower() if not unicodedata.category(char).startswith("P"))
    return " ".join(word for word in lowered.split() if word not in {"a", "an", "the"})


def parse_answer(response, task):
    matches = list(ANSWER_LINE.finditer(response))
    if not matches:
        return {"status": "malformed", "answer": None, "reason": "missing_answer_line"}
    value = matches[-1][1].strip()
    if value == ABSTAIN:
        return {"status": "abstention", "answer": None, "reason": None}
    if not value or (task in LABELS and value.lower() not in LABELS[task]) or (
            task == "factoid" and (not normalise_entity(value) or value.upper() == ABSTAIN)):
        return {"status": "malformed", "answer": None, "reason": "invalid_final_label"}
    return {"status": "answer", "answer": value, "reason": None}


def score(parsed, task, reference):
    if parsed["status"] != "answer":
        return {"correct": False, "contains_synonym": False if task == "factoid" else None}
    answer = parsed["answer"]
    if task != "factoid":
        return {"correct": answer.lower() == reference.lower(), "contains_synonym": None}
    synonyms = {normalise_entity(item) for group in reference
                for item in ([group] if isinstance(group, str) else group)}
    synonyms.discard("")
    value = normalise_entity(answer)
    return {"correct": value in synonyms,
            "contains_synonym": any(f" {synonym} " in f" {value} " for synonym in synonyms)}


def user_message(question, context, task, variant="original"):
    choices = {"yesno": "yes or no", "pubmedqa": "yes, no, or maybe", "factoid": "one short entity"}[task]
    instruction = (
        "Answer the biomedical question and give a short explanation. "
        "When source text is supplied, base your answer only on that text. "
        f"End with a separate line formatted as ANSWER: <label>, using {choices}. "
        f"If there is insufficient evidence to answer, write ANSWER: {ABSTAIN}."
    )
    if variant == "label_only":
        instruction = (
            "Answer the biomedical question. When source text is supplied, base your answer only on that text. "
            "Write a short explanation on its own line. Then start a NEW line with ANSWER: followed by the label alone. "
            f"After ANSWER: write only {choices}, or exactly {ABSTAIN} if there is insufficient evidence. "
            "Keep all explanation above the ANSWER: line. Do not put a period, extra words, or Markdown "
            "around the final label. End your response immediately after that label."
        )
    if context:
        instruction += f"\n\nSource text:\n{context}"
    return f"{instruction}\n\nQuestion: {question}"


def request_json(base, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=300) as response:
        return json.load(response)


def load_cases(root):
    refs = json.loads((root / "outputs/retrieval/medcpt_faiss_fixture_r0.json").read_bytes())
    c1 = json.loads((root / "outputs/retrieval/medcpt_faiss_fixture.json").read_bytes())
    if refs["corpus_sha256"] != c1["corpus_sha256"]:
        raise ValueError("Reference and C1 fixtures use different corpora")
    split = json.loads((root / "outputs/bioasq/alignment_full/eligibility_and_pilot.json").read_bytes())
    bio_pilots = {k: set(v["pilot_ids"]) for k, v in split["strata"].items()}
    with zipfile.ZipFile(root / "data/raw/bioasq/BioASQ-training13b.zip") as archive:
        bio = {q["id"]: q for q in json.load(archive.open("BioASQ-training13b/training13b.json"))["questions"]}
    pqa_manifest = json.loads((root / "outputs/pubmedqa/prepared/manifest.json").read_bytes())
    pqa_ids = set(pqa_manifest["pqa_l_pilot_ids"])
    pqa = json.loads((root / "data/raw/pubmedqa/ori_pqal.json").read_bytes())
    for dataset, data in refs["datasets"].items():
        baseline = {row["question_id"]: row for row in c1["datasets"][dataset]["queries"]}
        selected = []
        if dataset == "bioasq":
            for task, ids in sorted(bio_pilots.items()):
                qid = next(row["question_id"] for row in data["reference_inputs"] if row["question_id"] in ids)
                selected.append((qid, task, bio[qid]["exact_answer"]))
        else:
            selected.extend((row["question_id"], "pubmedqa", pqa[row["question_id"]]["final_decision"])
                            for row in data["reference_inputs"] if row["question_id"] in pqa_ids)
        by_id = {row["question_id"]: row for row in data["reference_inputs"]}
        for qid, task, reference in selected:
            row = by_id[qid]
            question = row["r2"]["question"]
            for arm, context in (("c1", baseline[qid]["context"]), ("r1", row["r1"]["context"]), ("r2", row["r2"]["context"])):
                yield dataset, task, qid, question, reference, arm, context


def load_fresh_cases(root, tokenizer):
    """Reserve comparison inputs from pilot IDs outside the prompt-development fixture."""
    original = json.loads((root / "outputs/generation/mistral_pilot_fixture.json").read_bytes())
    used = {(row["dataset"], row["question_id"]) for row in original["results"]}
    audit = root / "outputs/bioasq/alignment_full"
    split = json.loads((audit / "eligibility_and_pilot.json").read_bytes())
    bio_ids = [next(qid for qid in stratum["pilot_ids"] if ("bioasq", qid) not in used)
               for _, stratum in sorted(split["strata"].items())]
    alignment_raw = (audit / "alignment.json").read_bytes()
    if hashlib.sha256(alignment_raw).hexdigest() != split["alignment_sha256"]:
        raise ValueError("BioASQ alignment does not match pilot manifest")
    bio_gold = {row["id"]: [Span(group["pmid"], group["section"], lo, hi)
                           for group in row["gold"] for lo, hi in group["spans"]]
                for row in json.loads(alignment_raw) if row["id"] in bio_ids}
    with zipfile.ZipFile(root / "data/raw/bioasq/BioASQ-training13b.zip") as archive:
        bio = {q["id"]: q for q in json.load(archive.open("BioASQ-training13b/training13b.json"))["questions"]
               if q["id"] in bio_ids}
    pqa_manifest = json.loads((root / "outputs/pubmedqa/prepared/manifest.json").read_bytes())
    pqa_ids = [qid for qid in pqa_manifest["pqa_l_pilot_ids"] if ("pubmedqa", qid) not in used][:2]
    pqa = json.loads((root / "data/raw/pubmedqa/ori_pqal.json").read_bytes())

    for dataset, selected in (("bioasq", bio_ids), ("pubmedqa", pqa_ids)):
        needed = {s.pmid for spans in bio_gold.values() for s in spans} if dataset == "bioasq" else set(selected)
        prepared = root / "outputs" / dataset / "prepared"
        manifest = json.loads((prepared / "manifest.json").read_bytes())
        source, found = {}, set()
        digest = hashlib.sha256()
        with (prepared / "corpus.jsonl").open("rb") as corpus:
            for line in corpus:
                digest.update(line)
                row = json.loads(line)
                pmid = row.pop("pmid")
                if pmid in needed:
                    if pmid in found:
                        raise ValueError(f"Duplicate PMID {pmid}")
                    found.add(pmid)
                    source.update({(pmid, field): text for field, text in row.items()})
        if digest.hexdigest() != manifest["corpus_sha256"] or found != needed:
            raise ValueError(f"Missing or changed prepared records for {dataset}")
        for qid in selected:
            if dataset == "bioasq":
                row = bio[qid]
                task, question, reference, gold = row["type"], row["body"], row["exact_answer"], bio_gold[qid]
            else:
                row = pqa[qid]
                task, question, reference = "pubmedqa", row["QUESTION"], row["final_decision"]
                gold = [Span(qid, "text", 0, len(source[qid, "text"]))]
            oracle, delivered = oracle_input(question, gold, source, tokenizer)
            if not delivered:
                raise ValueError(f"Missing oracle evidence for {dataset}/{qid}")
            yield dataset, task, qid, question, reference, "r1", oracle["context"]
            yield dataset, task, qid, question, reference, "r2", ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--ollama", default="http://127.0.0.1:11435")
    parser.add_argument("--prompt-variant", choices=("original", "label_only"), default="original")
    parser.add_argument("--fresh", action="store_true", help="Use four additional reserved questions, R1 and R2")
    args = parser.parse_args()
    base = args.ollama.rstrip("/")
    endpoint = urlsplit(base)
    if endpoint.scheme != "http" or endpoint.hostname != "127.0.0.1" or endpoint.username or endpoint.path:
        parser.error("Use a local Ollama HTTP endpoint at 127.0.0.1")
    root = args.project
    tokenizer_file = root / "models/tokenizers/mistral.json"
    config_file = root / "models/tokenizers/mistral_tokenizer_config.json"
    if hashlib.sha256(tokenizer_file.read_bytes()).hexdigest() != TOKENIZER_HASH:
        raise ValueError("Wrong prompt tokenizer file")
    config_raw = config_file.read_bytes()
    template = json.loads(config_raw)["chat_template"]
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file), chat_template=template, bos_token="<s>", eos_token="</s>")
    version = request_json(base, "/api/version")["version"]
    installed = request_json(base, "/api/tags")["models"]
    if not any(entry["name"] == MODEL and entry["digest"] == MODEL_DIGEST for entry in installed):
        raise ValueError("Pinned Mistral model tag and digest not found")
    results = []
    cases = list(load_fresh_cases(root, Tokenizer.from_file(str(tokenizer_file))) if args.fresh else load_cases(root))
    if len(cases) != (8 if args.fresh else 12):
        raise ValueError("Unexpected number of reserved pilot inputs")
    prefix = "mistral_pilot_fresh" if args.fresh else "mistral_pilot_fixture"
    baseline_by_key = {}
    if args.prompt_variant != "original":
        baseline = json.loads((root / f"outputs/generation/{prefix}.json").read_bytes())
        baseline_by_key = {(r["dataset"], r["question_id"], r["arm"]): r for r in baseline["results"]}
        if (baseline["ollama_model_digest"], baseline["engine_version"], baseline["options"],
                baseline["tokenizer_config_sha256"]) != (
                MODEL_DIGEST, version, OPTIONS, hashlib.sha256(config_raw).hexdigest()):
            raise ValueError("Baseline runtime, model, options or tokenizer configuration differs")
        if set(baseline_by_key) != {(dataset, qid, arm) for dataset, _, qid, _, _, arm, _ in cases}:
            raise ValueError("Baseline and comparison cases differ")
        for dataset, task, qid, question, _, arm, context in cases:
            original = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_message(question, context, task)}],
                tokenize=False, add_generation_prompt=True,
            )
            if baseline_by_key[dataset, qid, arm]["prompt"] != original:
                raise ValueError(f"Baseline question or context differs: {dataset}/{qid}/{arm}")
        print("Verified matching baseline cases, context, model and runtime settings", flush=True)
    for dataset, task, qid, question, reference, arm, context in cases:
        content = user_message(question, context, task, args.prompt_variant)
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                               add_generation_prompt=True)
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if prompt_tokens + OPTIONS["num_predict"] > OPTIONS["num_ctx"]:
            raise ValueError(f"Prompt would exceed the context window: {dataset}/{qid}/{arm}")
        raw = request_json(base, "/api/generate", {"model": MODEL, "prompt": prompt, "raw": True, "stream": False,
                                                  "options": OPTIONS, "keep_alive": "5m"})
        if not raw.get("done") or raw.get("eval_count", 0) > OPTIONS["num_predict"]:
            raise ValueError(f"Incomplete or over-limit run: {dataset}/{qid}/{arm}")
        parsed = parse_answer(raw["response"], task)
        results.append({
            "dataset": dataset, "task": task, "question_id": qid, "arm": arm, "prompt": prompt,
            "context_tokens": len(tokenizer.encode(context, add_special_tokens=False)),
            "prompt_tokens": prompt_tokens, "response": raw["response"], "finish_reason": raw.get("done_reason"),
            "generated_tokens": raw.get("eval_count"), "runtime_ns": raw.get("total_duration"),
            "parsed": parsed, "benchmark": score(parsed, task, reference),
            "faithfulness_applicable": arm != "r2",
        })
        print(f"{dataset} {arm}: {parsed['status']}, {raw.get('eval_count')} tokens", flush=True)
    suffix = "" if args.prompt_variant == "original" else f"_{args.prompt_variant}"
    output = root / f"outputs/generation/{prefix}{suffix}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": ("Four fresh reserved labelled pilot questions, R1 and R2" if args.fresh
                  else "Four reserved labelled pilot questions, C1, R1 and R2"),
        "prompt_variant": args.prompt_variant,
        "model": MODEL, "ollama_model_digest": MODEL_DIGEST, "quantisation": "Q4_K_M",
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "engine": "Ollama", "engine_version": version, "platform": platform.platform(), "options": OPTIONS,
        "parse_status_counts": Counter(row["parsed"]["status"] for row in results), "results": results,
    }
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(output)


if __name__ == "__main__":
    main()

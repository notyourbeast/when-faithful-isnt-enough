"""Exercise chunk provenance and context packing with pinned tokenizer files."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

CHUNK_SIZE = 128
CONTEXT_BUDGET = 1024
RANK_LIMIT = 50
SEPARATOR = "\n\n"
TOKENIZER_SHA256 = {
    "medcpt": "6e046044df8a2fcedb10607075dca187cae61d806c0d80a96c5b81017edc90c9",
    "mistral": "e553af6fff7d7ad76e830608b218c5c0b0822998d5a1a96099a74cd3c1cb1a49",
}


@dataclass(frozen=True)
class Span:
    pmid: str
    field: str
    start: int
    end: int


@dataclass(frozen=True)
class Chunk:
    id: str
    spans: tuple[Span, ...]


def fixed_windows(tokenizer, pmid, field, text, *, overlap=0):
    if not 0 <= overlap < CHUNK_SIZE:
        raise ValueError("Invalid fixed-window overlap")
    offsets = tokenizer.encode(text, add_special_tokens=False).offsets
    if not offsets:
        return []
    if overlap:
        spans = []
        for first in range(0, len(offsets), CHUNK_SIZE - overlap):
            stop = min(first + CHUNK_SIZE, len(offsets))
            start_char = offsets[first][0] if first else 0
            end_char = offsets[stop][0] if stop < len(offsets) else len(text)
            if (first and start_char < offsets[first - 1][1]) or end_char < offsets[stop - 1][1]:
                raise ValueError(f"Token boundary cannot be represented as a source slice: {pmid}/{field}")
            spans.append(Span(pmid, field, start_char, end_char))
            if stop == len(offsets):
                break
        return spans
    cuts = [0]
    for index in range(CHUNK_SIZE, len(offsets), CHUNK_SIZE):
        cut = offsets[index][0]
        if cut < offsets[index - 1][1] or cut <= cuts[-1]:
            raise ValueError(f"Token boundary cannot be represented as a source slice: {pmid}/{field}")
        cuts.append(cut)
    cuts.append(len(text))
    return [Span(pmid, field, start, end) for start, end in zip(cuts, cuts[1:])]


def render(spans, source):
    return "\n".join(source[(s.pmid, s.field)][s.start:s.end] for s in spans)


def truncate_spans(spans, source, tokenizer, prefix, budget):
    """Find a token-boundary prefix that fits and preserve its source offsets."""
    encoded = tokenizer.encode(prefix + render(spans, source), add_special_tokens=False)
    for _, cut in reversed(encoded.offsets[:budget]):
        remaining = cut - len(prefix)
        if remaining <= 0:
            break
        clipped = []
        for span in spans:
            if remaining <= 0:
                break
            width = min(span.end - span.start, remaining)
            clipped.append(Span(span.pmid, span.field, span.start, span.start + width))
            remaining -= width + 1  # Separator between spans.
        if clipped and len(tokenizer.encode(prefix + render(clipped, source), add_special_tokens=False).ids) <= budget:
            return tuple(clipped)
    return ()


def pack(ranked, source, tokenizer, budget=CONTEXT_BUDGET, *, overflow="skip"):
    context, trace, delivered = "", [], []
    count = 0

    def tokens(text):
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    candidates = ranked if overflow == "oracle_prefix" else ranked[:RANK_LIMIT]
    for rank, chunk in enumerate(candidates, 1):
        spans = chunk.spans
        block = render(spans, source)
        original_count = tokens(block) if overflow == "whole_document" else None
        if original_count is not None and original_count > budget:
            spans = truncate_spans(spans, source, tokenizer, "", budget)
            block = render(spans, source)
        prefix = context + SEPARATOR if context else ""
        proposed = prefix + block
        proposed_count = tokens(proposed)
        stop = overflow == "oracle_prefix" and proposed_count > budget
        if stop:
            spans = truncate_spans(spans, source, tokenizer, prefix, budget)
            proposed = prefix + render(spans, source)
            proposed_count = tokens(proposed)
        accepted = bool(spans) and (budget is None or proposed_count <= budget)
        entry = {"rank": rank, "chunk": chunk.id, "accepted": accepted,
                 "proposed_context_tokens": proposed_count}
        if overflow != "skip":
            entry["truncated"] = spans != chunk.spans
        if original_count is not None:
            entry["original_document_context_tokens"] = original_count
        trace.append(entry)
        if accepted:
            start = len(prefix)
            for span in spans:
                width = span.end - span.start
                delivered.append((span, start, start + width))
                start += width + 1
            context, count = proposed, proposed_count
        if stop:
            break
    return context, count, trace, delivered


def merge_ranked_spans(chunks):
    """Union overlaps at the highest-ranked contributor's location."""
    groups = [[] for _ in chunks]
    for rank, chunk in enumerate(chunks):
        for incoming in chunk.spans:
            target_rank, target_index = rank, len(groups[rank])
            lo, hi = incoming.start, incoming.end
            matches = []
            # Existing spans are nonoverlapping, so adding this interval can
            # touch multiple groups without changing the order of other spans.
            for group_rank, group in enumerate(groups[:rank + 1]):
                for index, span in enumerate(group):
                    if (span.pmid, span.field) == (incoming.pmid, incoming.field) and span.start < hi and lo < span.end:
                        matches.append((group_rank, index))
                        lo, hi = min(lo, span.start), max(hi, span.end)
            merged = Span(incoming.pmid, incoming.field, lo, hi)
            if matches:
                target_rank, target_index = matches[0]
                for group_rank, index in reversed(matches[1:]):
                    del groups[group_rank][index]
                groups[target_rank][target_index] = merged
            else:
                groups[rank].append(merged)
    return [Chunk(chunk.id, tuple(group)) for chunk, group in zip(chunks, groups, strict=True) if group]


def pack_overlapping(ranked, source, tokenizer):
    """Check each candidate against the budget after unioning its source spans."""
    accepted, trace = [], []
    result = "", 0, [], []
    for rank, chunk in enumerate(ranked[:RANK_LIMIT], 1):
        proposed = merge_ranked_spans([*accepted, chunk])
        packed = pack(proposed, source, tokenizer, budget=None)
        fits = packed[1] <= CONTEXT_BUDGET
        trace.append({"rank": rank, "chunk": chunk.id, "accepted": fits,
                      "proposed_context_tokens": packed[1]})
        if fits:
            result = packed
            accepted.append(chunk)
    return result[0], result[1], trace, result[3]


def covered_characters(gold, delivered):
    intervals = sorted((max(gold.start, span.start), min(gold.end, span.end))
                       for span, _, _ in delivered
                       if (gold.pmid, gold.field) == (span.pmid, span.field)
                       and span.start < gold.end and gold.start < span.end)
    total, covered_end = 0, gold.start
    for start, end in intervals:
        total += max(0, end - max(start, covered_end))
        covered_end = max(covered_end, end)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("medcpt_tokenizer", type=Path)
    parser.add_argument("mistral_tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for name, path in (("medcpt", args.medcpt_tokenizer), ("mistral", args.mistral_tokenizer)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != TOKENIZER_SHA256[name]:
            parser.error(f"{name} tokenizer checksum differs from the pinned artifact")
    medcpt = Tokenizer.from_file(str(args.medcpt_tokenizer))
    mistral = Tokenizer.from_file(str(args.mistral_tokenizer))
    text = "  " + "A café had three tables. " * 40 + "The door was blue."
    title = "Illustrative room description"
    source = {("fixture", "abstract"): text, ("fixture", "title"): title,
              ("oversize", "abstract"): " Oversized" * 2000}
    spans = fixed_windows(medcpt, "fixture", "abstract", text)
    assert len(spans) > 1 and "".join(text[s.start:s.end] for s in spans) == text
    header = Span("fixture", "title", 0, len(title))
    oversize = Span("oversize", "abstract", 0, len(source[("oversize", "abstract")]))
    ranked = [Chunk("oversize", (oversize,))] + [
        Chunk(f"fixture-{i}", (header, span)) for i, span in enumerate(spans)
    ]
    ranked.extend([ranked[0]] * (RANK_LIMIT - len(ranked)))
    ranked.append(Chunk("beyond-rank-limit", (header,)))
    context, tokens, trace, delivered = pack(ranked, source, mistral)
    assert len(trace) == RANK_LIMIT and not trace[0]["accepted"] and trace[1]["accepted"]
    assert trace[-1]["chunk"] != "beyond-rank-limit"
    assert tokens <= CONTEXT_BUDGET and tokens == len(mistral.encode(context, add_special_tokens=False).ids)
    for span, start, end in delivered:
        assert context[start:end] == source[(span.pmid, span.field)][span.start:span.end]
    cross_boundary = Span("fixture", "abstract", spans[0].end - 2, spans[0].end + 2)
    assert covered_characters(cross_boundary, delivered) == 4
    assert covered_characters(header, delivered) == len(title)
    assert covered_characters(oversize, delivered) == 0
    report = {
        "fixture": "Synthetic text and preset ranks",
        "tokenizer_sha256": TOKENIZER_SHA256,
        "chunk_size": CHUNK_SIZE, "fixture_chunks": len(spans), "context_budget": CONTEXT_BUDGET,
        "context_tokens": tokens, "rank_limit": RANK_LIMIT, "rank_trace": trace,
        "delivered": [{"pmid": s.pmid, "field": s.field, "source_span": [s.start, s.end],
                       "context_span": [start, end]} for s, start, end in delivered],
        "context": context,
        "checks": ["source reconstruction", "skip and continue", "rank cutoff", "token budget",
                   "delivered span equality", "cross-boundary coverage", "deduplicated title coverage", "skipped evidence"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Fixture passed: {len(spans)} chunks, {tokens} context tokens, {len(delivered)} delivered spans")


if __name__ == "__main__":
    main()

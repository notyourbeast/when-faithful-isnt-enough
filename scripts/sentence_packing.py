"""Deterministic sentence boundaries and C2 packing with source offsets."""
import re

from retrieval_fixture import CHUNK_SIZE, Span, fixed_windows


SENTENCE_END = re.compile(r"""[.!?]+["'”’\])]*(?:\s+|$)""")
ABBREVIATIONS = re.compile(
    r"(?i)(?:\b(?:e\.g|i\.e|et al|figs?|refs?|vs|dr|mr|mrs|prof|no|vol)\.)$"
)


def sentence_spans(text):
    """Return contiguous character ranges, retaining separators in the source."""
    start = 0
    for match in SENTENCE_END.finditer(text):
        end = match.end()
        if end == len(text):
            break
        # Restrict single-letter suppression to uppercase initials.
        prefix = text[:match.start() + 1]
        if match[0].startswith(".") and (
            ABBREVIATIONS.search(prefix)
            or re.search(r"\b[A-Z]\.$", prefix)
        ):
            continue
        yield start, end
        start = end
    if start < len(text):
        yield start, len(text)


def pack_sentences(tokenizer, pmid, field, text):
    """Greedily combine sentences up to 128 tokens, splitting oversized ones."""
    def count(start, end):
        return len(tokenizer.encode(text[start:end], add_special_tokens=False).ids)

    if not text.strip():
        return [], 0
    packed, pending = [], None
    long_sentences = 0
    for start, end in sentence_spans(text):
        units = [(start, end)]
        if count(start, end) > CHUNK_SIZE:
            long_sentences += 1
            units = [(start + span.start, start + span.end)
                     for span in fixed_windows(tokenizer, pmid, field, text[start:end])]
        for lo, hi in units:
            if count(lo, hi) > CHUNK_SIZE:
                raise ValueError(f"Sentence fragment exceeds the token limit: {pmid}/{field}/{lo}")
            if pending is not None and count(pending[0], hi) > CHUNK_SIZE:
                packed.append(Span(pmid, field, *pending))
                pending = None
            pending = (lo if pending is None else pending[0], hi)
    if pending is not None:
        packed.append(Span(pmid, field, *pending))
    return packed, long_sentences

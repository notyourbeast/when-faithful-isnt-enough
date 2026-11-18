"""Recover C2 boundaries when a standalone WordPiece fragment grows."""
from retrieval_fixture import CHUNK_SIZE, Span
from sentence_packing import pack_sentences as original_pack, sentence_spans


def pack_sentences(tokenizer, pmid, field, text):
    try:
        return original_pack(tokenizer, pmid, field, text)
    except ValueError as error:
        if not str(error).startswith("Sentence fragment exceeds the token limit:"):
            raise

    def encode(lo, hi):
        return tokenizer.encode(text[lo:hi], add_special_tokens=False)

    packed, pending, fragmented = [], None, 0
    for lo, hi in sentence_spans(text):
        units = []
        remaining = encode(lo, hi)
        if len(remaining.ids) > CHUNK_SIZE:
            fragmented += 1
        while len(remaining.ids) > CHUNK_SIZE:
            cuts = sorted({lo + start for start, _ in remaining.offsets[:CHUNK_SIZE + 1] if start > 0}, reverse=True)
            cut = next((end for end in cuts if len(encode(lo, end).ids) <= CHUNK_SIZE), None)
            if cut is None:
                raise ValueError(f"No fitting C2 fragment for {pmid}/{field}/{lo}")
            units.append((lo, cut))
            lo = cut
            remaining = encode(lo, hi)
        units.append((lo, hi))
        for start, end in units:
            if pending is not None and len(encode(pending[0], end).ids) > CHUNK_SIZE:
                packed.append(Span(pmid, field, *pending))
                pending = None
            pending = (start if pending is None else pending[0], end)
    if pending is not None:
        packed.append(Span(pmid, field, *pending))
    return packed, fragmented

"""C3 chunking and pilot-only percentile calibration."""
from collections import Counter
from functools import lru_cache
from statistics import median

import numpy as np

from retrieval_fixture import Span
from sentence_packing import sentence_spans

MIN_TOKENS = 64
MAX_TOKENS = 192
TARGET = 128
GRID = tuple(range(0, 101, 5))


def measure_documents(texts, encode):
    """Compute adjacent sentence cosine distances once per document."""
    spans = [list(sentence_spans(text)) for text in texts]
    sentences = [text[start:end] for text, sections in zip(texts, spans, strict=True)
                 if len(sections) > 1 for start, end in sections]
    if sentences:
        order = np.argsort([len(sentence) for sentence in sentences], kind="stable")
        encoded = encode([sentences[index] for index in order])
        vectors = np.empty_like(encoded)
        vectors[order] = encoded
    measured, cursor = [], 0
    for sections in spans:
        n = len(sections)
        if n < 2:
            measured.append((sections, np.empty(0)))
            continue
        v = vectors[cursor:cursor + n].astype(np.float64)
        strengths = np.linalg.norm(v, axis=1)
        if np.any(strengths == 0):
            raise ValueError("Cannot calculate cosine distance for a zero vector")
        v /= strengths[:, None]
        measured.append((sections, 1 - np.clip(np.sum(v[:-1] * v[1:], axis=1), -1, 1)))
        cursor += n
    return measured


class SemanticDocument:
    def __init__(self, text, measurements, tokenizer):
        self.text = text
        spans, self.distances = measurements
        self.ends = [end for _, end in spans]
        self.tokenizer = tokenizer
        self.encode = lru_cache(maxsize=256)(self.encode)

    def encode(self, start, end):
        return self.tokenizer.encode(self.text[start:end], add_special_tokens=False)

    def count(self, start, end):
        return len(self.encode(start, end).ids)

    def chunk(self, percentile, fallback):
        if not self.text.strip():
            return [], Counter()
        threshold = float(np.percentile(self.distances, percentile, method="linear")) if len(self.distances) >= 4 else fallback
        semantic = [end for end, distance in zip(self.ends[:-1], self.distances, strict=True)
                    if distance > threshold]
        segments = []
        start = 0
        for end in [*semantic, len(self.text)]:
            if self.count(start, end) >= MIN_TOKENS or end == len(self.text):
                segments.append((start, end))
                start = end
        if len(segments) > 1 and self.count(*segments[-1]) < MIN_TOKENS:
            segments[-2:] = [(segments[-2][0], segments[-1][1])]

        chunks, counts = [], Counter()
        for start, end in segments:
            first = len(chunks)
            while self.count(start, end) > MAX_TOKENS:
                fitting = [cut for cut in self.ends if start < cut < end and MIN_TOKENS <= self.count(start, cut) <= MAX_TOKENS]
                if fitting:
                    cut = fitting[-1]
                    counts["forced_sentence_boundaries"] += 1
                else:
                    offsets = self.encode(start, end).offsets
                    cut = start + offsets[MAX_TOKENS][0]
                    if cut <= start or self.count(start, cut) > MAX_TOKENS:
                        raise ValueError("No safe token boundary within the C3 maximum")
                    counts["forced_token_boundaries"] += 1
                chunks.append((start, cut))
                start = cut
            chunks.append((start, end))

            if len(chunks) > first + 1 and self.count(*chunks[-1]) < MIN_TOKENS:
                lo, _ = chunks[-2]
                if self.count(lo, end) <= MAX_TOKENS:
                    chunks[-2:] = [(lo, end)]
                    continue
                fitting = [cut for cut in self.ends if lo < cut < end
                           and MIN_TOKENS <= self.count(lo, cut) <= MAX_TOKENS
                           and MIN_TOKENS <= self.count(cut, end) <= MAX_TOKENS]
                if fitting:
                    cut = fitting[-1]
                else:
                    offsets = self.encode(lo, end).offsets
                    cut = next((lo + offsets[i][0] for i in range(MAX_TOKENS, MIN_TOKENS - 1, -1)
                                if MIN_TOKENS <= self.count(lo, lo + offsets[i][0]) <= MAX_TOKENS
                                and MIN_TOKENS <= self.count(lo + offsets[i][0], end) <= MAX_TOKENS), None)
                if cut is not None:
                    chunks[-2:] = [(lo, cut), (cut, end)]
                    counts["tail_rebalances"] += 1
        counts["below_minimum_chunks"] = sum(self.count(lo, hi) < MIN_TOKENS for lo, hi in chunks)
        counts["output_boundaries"] = max(0, len(chunks) - 1)
        counts["final_forced_boundaries"] = sum(
            hi not in semantic for _, hi in chunks[:-1]
        )
        if counts["below_minimum_chunks"] and self.count(0, len(self.text)) >= MIN_TOKENS:
            raise ValueError("Could not satisfy C3 minimum without losing source characters")
        return chunks, counts


def calibrate(documents):
    distances = np.concatenate([doc.distances for doc in documents])
    if not len(distances):
        raise ValueError("Calibration requires adjacent sentence pairs")
    trials = []
    for percentile in GRID:
        fallback = float(np.percentile(distances, percentile, method="linear"))
        lengths, stats = [], Counter()
        for doc in documents:
            chunks, counts = doc.chunk(percentile, fallback)
            lengths.extend(doc.count(lo, hi) for lo, hi in chunks)
            stats.update(counts)
        midpoint = median(lengths)
        trials.append({"percentile": percentile, "fallback_cosine_distance": fallback,
                       "median_tokens": midpoint, "within_target": abs(midpoint - TARGET) <= TARGET * .15,
                       "chunks": len(lengths), "stats": dict(stats)})
    return min(trials, key=lambda row: (abs(row["median_tokens"] - TARGET), row["percentile"])), trials


def build_semantic_spans(pmid, field, doc, setting):
    chunks, counts = doc.chunk(setting["percentile"], setting["fallback_cosine_distance"])
    return [Span(pmid, field, start, end) for start, end in chunks], counts

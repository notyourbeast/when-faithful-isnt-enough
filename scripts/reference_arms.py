"""Construct whole-document, oracle and question-only reference inputs."""
from retrieval_fixture import Chunk, Span, pack


def whole_document_spans(pmid, field, text):
    return [Span(pmid, field, 0, len(text))] if text else []


def oracle_input(question, gold, source, tokenizer):
    # The alignment already unions overlapping BioASQ spans. Exact duplicate
    # spans are discarded here as well to keep the ordering deterministic.
    ordered = sorted(set(gold), key=lambda s: (int(s.pmid), s.field != "title", s.start, s.end))
    chunks = [Chunk(f"oracle:{s.pmid}:{s.field}:{s.start}", (s,)) for s in ordered]
    context, count, trace, delivered = pack(chunks, source, tokenizer, overflow="oracle_prefix")
    return {"question": question, "context": context, "context_tokens": count,
            "context_recall_applicable": True, "faithfulness_applicable": True,
            "rank_trace": trace}, delivered


def question_only_input(question):
    return {"question": question, "context": "", "context_tokens": 0,
            "context_recall_applicable": False, "faithfulness_applicable": False}

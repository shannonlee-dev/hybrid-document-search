"""Downstream RAG evaluation helpers: fixed subsets, prompts and paired statistics.

Exploratory only. Retrievers, α and the frozen benchmark inputs are reused
unchanged; this module adds no retrieval logic of its own.
"""

import random
import re
from math import isfinite
from statistics import fmean

RAG_METHODS = ("tfidf", "bm25", "dense", "rrf", "score_fusion")
# 생성 context 길이를 통제하기 위한 값으로, 최종 benchmark의 Top-10과 별개다.
RAG_CONTEXT_TOP_K = 5
RAG_SEED = 20261023
PILOT_QUERIES = 50
MAX_QUERIES = 100
PAIRED_COMPARISONS = (("score_fusion", "dense"), ("rrf", "dense"))
PROMPT_TEMPLATE = (
    "다음 문서만을 근거로 질문에 답하세요.\n"
    "문서에서 답을 확인할 수 없다면 추측하지 말고 알 수 없다고 답하세요.\n\n"
    "[Context]\n{context}\n\n"
    "[Question]\n{question}\n\n"
    "[Answer]\n"
)
_ABSTENTION = re.compile(r"알\s*수\s*없|확인할\s*수\s*없")


def shuffled_query_ids(query_ids, seed=RAG_SEED):
    """Shuffle sorted IDs once, so every smaller subset is a prefix of a larger one."""
    order = sorted(query_ids)
    if len(set(order)) != len(order):
        raise ValueError("query IDs must be unique")
    random.Random(seed).shuffle(order)
    return order


def parent_id(query_id):
    """StrategyQA sub-question IDs are ``<original question ID>_<step>``."""
    return query_id.rsplit("_", 1)[0]


def context_records(hits, documents, top_k=RAG_CONTEXT_TOP_K):
    """Keep the retriever's own order and score; attach the full passage text."""
    return [
        {
            "document_id": hit.document_id,
            "rank": hit.rank,
            "score": hit.score,
            "title": documents[hit.document_id].title,
            "text": documents[hit.document_id].text,
        }
        for hit in hits[:top_k]
    ]


def context_text(record):
    """Match the indexed ``title + "\\n" + text`` form used by Sparse and Dense."""
    title = record["title"]
    return f"{title}\n{record['text']}" if title else record["text"]


def build_prompt(question, contexts):
    """One user message; the retriever name never reaches the generator."""
    context = "\n\n".join(
        f"[문서 {index}] {context_text(record)}"
        for index, record in enumerate(contexts, 1)
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


def id_context_metrics(contexts, qrels):
    """Binary qrels-based precision/recall over the generation contexts."""
    relevant = {doc for doc, label in qrels.items() if label > 0}
    if not relevant or not contexts:
        raise ValueError("contexts and positive qrels are required")
    hits = sum(record["document_id"] in relevant for record in contexts)
    return {
        "id_context_precision": hits / len(contexts),
        "id_context_recall": hits / len(relevant),
    }


def is_abstention(response):
    """Deterministic flag for answers that decline because the context lacks it."""
    return bool(_ABSTENTION.search(response or ""))


def finite_or_none(value):
    """NaN and ±infinity are undefined metrics, never zero and never serialized."""
    return value if value is not None and isfinite(value) else None


def finite_mean(values):
    """Mean over defined values plus how many were undefined (NaN/None)."""
    defined = [v for v in values if v is not None and isfinite(v)]
    return {
        "mean": fmean(defined) if defined else None,
        "n": len(defined),
        "undefined": len(values) - len(defined),
    }


def paired_bootstrap(pairs, *, resamples=10_000, seed=RAG_SEED, confidence=0.95):
    """Cluster bootstrap of mean(a - b) over (group, a, b) rows with both defined.

    Sub-questions of one original StrategyQA question are resampled together.
    """
    groups = {}
    for group, a, b in pairs:
        if a is not None and b is not None and isfinite(a) and isfinite(b):
            groups.setdefault(group, []).append(a - b)
    deltas = [d for values in groups.values() for d in values]
    if not deltas:
        return {"mean_delta": None, "ci": None, "n": 0, "groups": 0}
    keys = sorted(groups)
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        total = count = 0
        for key in rng.choices(keys, k=len(keys)):
            total += sum(groups[key])
            count += len(groups[key])
        means.append(total / count)
    means.sort()
    tail = (1 - confidence) / 2
    low = means[int(tail * (resamples - 1))]
    high = means[int((1 - tail) * (resamples - 1))]
    return {
        "mean_delta": fmean(deltas),
        "ci": [low, high],
        "n": len(deltas),
        "groups": len(keys),
        "better": sum(d > 0 for d in deltas),
        "worse": sum(d < 0 for d in deltas),
        "tied": sum(d == 0 for d in deltas),
    }

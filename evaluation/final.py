"""Evaluate final retrievers with the shared metrics and frozen team inputs."""

from dataclasses import asdict
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from time import perf_counter

from data.preparation import PINNED_REVISION
from evaluation.benchmark import (
    _code_provenance,
    _environment,
    benchmark_retriever,
)
from evaluation.constants import DEFAULT_REPEATS, DEFAULT_WARMUP, EVALUATION_TOP_K
from evaluation.data import EvaluationDataset, Query, file_sha256
from evaluation.json_io import read_json
from evaluation.metrics import evaluate_query

FINAL_METHODS = ("tfidf", "bm25", "dense", "rrf", "score_fusion")
FINAL_BM25_WEIGHT = 0.05
DATASET_REVISIONS = {
    "miracl10k": PINNED_REVISION,
    "strategyqa": "7cececfcc209fc836e08d6ab7584cba776847cfa",
}
_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = _ROOT / "docs/experiments/score-fusion/cache-release.json"


def load_final_dataset(directory, *, manifest=DEFAULT_MANIFEST):
    """Verify all four pinned inputs, then adapt the existing fusion cache loader.

    MIRACL always uses its existing dev IDs; StrategyQA uses its complete dev
    cache. No synthetic train split or resampling is introduced.
    """
    directory, manifest = Path(directory), Path(manifest)
    release = read_json(manifest)
    for name in (
        "data.json",
        "document-vectors.npy",
        "query-vectors.npy",
        "encoding.json",
    ):
        digest = release["files"].get(f"{directory.name}/{name}")
        if digest is None or file_sha256(directory / name) != digest:
            raise ValueError(f"고정 cache SHA-256 불일치: {directory.name}/{name}")
    try:
        from scripts.benchmark_score_fusion import _load_cache
    except ImportError as exc:
        raise ImportError(
            "최종 평가는 sparse/dense extras(score fusion 구현)가 필요합니다."
        ) from exc
    raw = read_json(directory / "data.json")
    split = "dev" if raw.get("splits") else None
    data, documents, vectors, queries, ids, hashes, encoding = _load_cache(
        directory, split
    )
    if data["name"] != directory.name:
        raise ValueError("cache 디렉터리 이름과 dataset 이름이 다릅니다.")
    corpus_ids = {doc.document_id for doc in documents}
    if not corpus_ids or len(corpus_ids) != len(documents):
        raise ValueError("corpus가 비었거나 document_id가 중복됩니다.")
    for doc in documents:
        if (
            not isinstance(doc.document_id, str)
            or not doc.document_id.strip()
            or not isinstance(doc.text, str)
            or not doc.text.strip()
            or (doc.title is not None and not isinstance(doc.title, str))
        ):
            raise ValueError("잘못된 공통 Document schema입니다.")
    selected_queries, qrels = [], {}
    for query_id in ids:
        text = data["queries"][query_id]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("평가 질의가 비어 있습니다.")
        judgments = data["qrels"][query_id]
        evaluate_query([], judgments)
        if set(judgments) - corpus_ids:
            raise ValueError("qrels에 corpus 밖의 document_id가 있습니다.")
        selected_queries.append(Query(query_id, text))
        qrels[query_id] = judgments
    hashes["encoding.json"] = file_sha256(directory / "encoding.json")
    provenance = {
        "dataset_id": data["name"],
        "dataset_revision": DATASET_REVISIONS.get(data["name"]),
        "manifest_sha256": file_sha256(manifest),
        "input_sha256": hashes,
        "corpus_size": len(documents),
        "query_count": len(ids),
        "query_ids": ids,
        "encoding_record": encoding,
        "cache_release": {
            key: release.get(key) for key in ("repository", "tag", "asset")
        },
    }
    dataset = EvaluationDataset(
        tuple(documents), tuple(selected_queries), qrels, "dev", provenance
    )
    return dataset, data, vectors, queries


def _build_final_retrievers(dataset, data, vectors, query_vectors, methods):
    """Reuse the score fusion retrievers from main; cached vectors never load a model."""
    from fusion.hybrid import HybridRetriever
    from indexing.faiss_index import FaissIndex
    from retrievers.bm25 import BM25Retriever
    from retrievers.dense import DenseRetriever
    from retrievers.tfidf import TfidfRetriever

    # sparse/dense extras가 없는 환경도 import 가능해야 하므로 실행 시에만 불러온다.
    score_module = import_module("fusion.score")
    cached_embedder = import_module("scripts.benchmark_score_fusion")._CachedEmbedder

    started = perf_counter()
    sparse = BM25Retriever(dataset.documents)
    sparse_seconds = perf_counter() - started
    started = perf_counter()
    dense = DenseRetriever(
        index=FaissIndex.build(vectors),
        documents=dataset.documents,
        embedder=cached_embedder(data, query_vectors),
    )
    dense_seconds = perf_counter() - started
    built, setup, configs = {}, {}, {}
    for method in methods:
        started = perf_counter()
        if method == "tfidf":
            built[method] = TfidfRetriever(dataset.documents)
            configs[method] = {"analyzer": "char_wb", "ngram_range": [2, 4]}
            setup[method] = perf_counter() - started
        elif method == "bm25":
            built[method], setup[method] = sparse, sparse_seconds
            configs[method] = {
                "analyzer": "char_wb",
                "ngram_range": [2, 4],
                "k1": 1.5,
                "b": 0.75,
                "method": "lucene",
            }
        elif method == "dense":
            built[method], setup[method] = dense, dense_seconds
            configs[method] = {"embedding": "frozen BGE-M3 query/document vectors"}
        else:
            built[method] = (
                HybridRetriever(sparse, dense)
                if method == "rrf"
                else score_module.ScoreFusionRetriever(
                    sparse, dense, bm25_weight=FINAL_BM25_WEIGHT
                )
            )
            setup[method] = sparse_seconds + dense_seconds + perf_counter() - started
            configs[method] = (
                {"rank_constant": 60, "candidate_top_k_per_component": 10}
                if method == "rrf"
                else {
                    "bm25_weight": FINAL_BM25_WEIGHT,
                    "normalization": "per-query corpus maximum",
                    "candidate_scope": "full corpus",
                    "vector_correction": False,
                }
            )
    return built, setup, configs


def run_final_benchmark(
    directory,
    *,
    manifest=DEFAULT_MANIFEST,
    methods=FINAL_METHODS,
    warmup=DEFAULT_WARMUP,
    repeats=DEFAULT_REPEATS,
    progress=None,
):
    """Evaluate the full fixed dev set with identical warm-up/repeat counts."""
    if (
        not methods
        or len(set(methods)) != len(methods)
        or any(name not in FINAL_METHODS for name in methods)
    ):
        raise ValueError("최종 평가 methods는 중복 없는 공통 검색 방식이어야 합니다.")
    for value in (warmup, repeats):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("warmup/repeats는 양의 정수여야 합니다.")
    dataset, data, vectors, query_vectors = load_final_dataset(
        directory, manifest=manifest
    )
    retrievers, setup, configs = _build_final_retrievers(
        dataset, data, vectors, query_vectors, methods
    )
    code = _code_provenance()
    for name in (
        "evaluation/final.py",
        "scripts/evaluate_final.py",
        "fusion/score.py",
        "scripts/benchmark_score_fusion.py",
        "indexing/vectors.py",
    ):
        path = _ROOT / name
        if path.is_file():
            code["sha256"][name] = file_sha256(path)
    report = {
        "schema_version": "1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": {"split": "dev", **dataset.provenance},
        "conditions": {
            "top_k": EVALUATION_TOP_K,
            "warmup_passes": warmup,
            "measurement_passes": repeats,
            "query_order": [q.query_id for q in dataset.queries],
            "method_order": list(methods),
            "latency_unit": "ms",
            "latency_scope": "cached-vector CPU retrieval and fusion; excludes model inference, initialization and HTTP",
            "p95_method": "linear interpolation at (N - 1) * 0.95",
            "quality_run": "first measured pass",
            "relevance": "binary: relevance > 0",
            "vector_correction": False,
            "bm25_weight": FINAL_BM25_WEIGHT,
            "reference_only": "Recall@100 is not included in this Top-10 run",
        },
        "environment": _environment(),
        "code": code,
        "methods": {},
    }
    for name, retriever in retrievers.items():
        if progress:
            progress(
                f"{data['name']}: {name} / {len(dataset.queries)} queries / {repeats} repeats"
            )
        result = benchmark_retriever(retriever, dataset, warmup=warmup, repeats=repeats)
        # 벤치마크 출력이 의미 있는 공통 SearchResult를 보존하는지도 확인한다.
        for row in result["queries"]:
            if row["results"] != [
                asdict(hit) for hit in retriever.search(row["text"], 10)
            ]:
                raise ValueError(f"반복 검색 결과가 다릅니다: {name}/{row['query_id']}")
        report["methods"][name] = {
            "config": configs[name],
            "setup_seconds": setup[name],
            **result,
        }
    return report

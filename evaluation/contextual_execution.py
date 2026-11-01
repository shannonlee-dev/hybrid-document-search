"""Execute local DAT requests and preserve observed routing, timing and identity."""

import hashlib
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np

from evaluation.contextual_report import ranking_record
from evaluation.data import file_sha256, load_evaluation_dataset
from evaluation.enterprise import _load_vectors
from evaluation.json_io import read_json
from fusion.contextual import ContextualRetriever, FixedPolicy

ROOT = Path(__file__).resolve().parents[1]


def source_identity():
    names = (
        "fusion/contextual.py",
        "fusion/contextual_judge.py",
        "evaluation/contextual.py",
        "evaluation/contextual_execution.py",
        "evaluation/contextual_report.py",
        "scripts/evaluate_contextual.py",
    )
    return {name: file_sha256(ROOT / name) for name in names}


def snapshot_identity(path):
    """Hash local model/tokenizer files rather than trusting a directory name."""
    path = Path(path)
    files = {
        p.relative_to(path).as_posix(): file_sha256(p)
        for p in sorted(path.rglob("*"))
        if p.is_file()
        and not {".git", ".cache"}.intersection(p.relative_to(path).parts)
    }
    if not files or not any(n.endswith((".safetensors", ".bin")) for n in files):
        raise ValueError("local model snapshot must contain model weights")
    digest = hashlib.sha256(
        "\n".join(f"{k}:{v}" for k, v in files.items()).encode()
    ).hexdigest()
    return {"files_sha256": files, "snapshot_sha256": digest}


def run_queries(dataset, query_ids, retrievers, *, run_id, repeats=3, seed=20261030):
    """Run every request uncached; metrics/JSON work stays outside search timing."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if set(retrievers) != {"dat_fixed", "dat"}:
        raise ValueError("matched dat_fixed and dat retrievers required")
    queries = {q.query_id: q.text for q in dataset.queries}
    if (
        not query_ids
        or len(set(query_ids)) != len(query_ids)
        or set(query_ids) - set(queries)
    ):
        raise ValueError("unique known query IDs required")
    doc_index = {d.document_id: i for i, d in enumerate(dataset.documents)}
    rng = np.random.default_rng(seed)
    records, samples = [], []
    for repeat in range(repeats):
        first = {}
        for count, index in enumerate(rng.permutation(len(query_ids))):
            query_id = query_ids[index]
            for method in rng.permutation(list(retrievers)):
                run = retrievers[method].search_detailed(queries[query_id], 10)
                samples.append(
                    {
                        "query_id": query_id,
                        "method": str(method),
                        "repeat": repeat,
                        **run.elapsed_ms,
                    }
                )
                if repeat == 0:
                    first[query_id, method] = ranking_record(
                        dataset,
                        query_id,
                        [doc_index[r.document_id] for r in run.results],
                        run_id=run_id,
                        method=str(method),
                        alpha=run.alpha,
                        judged=run.judged if method == "dat" else None,
                        top1=run.top1_document_ids,
                        probabilities=run.probabilities,
                        normalization="union_minmax"
                        if run.judged
                        else retrievers[method].fixed.normalization,
                    )
            if count % 100 == 0:
                print(
                    f"DAT live repeat={repeat} completed={count + 1}/{len(query_ids)}",
                    flush=True,
                )
        if repeat == 0:
            records = [first[q, method] for q in query_ids for method in retrievers]
    return records, samples


def live_evaluate(args):
    """Load supplied local models and frozen vectors; run the exact primary set."""
    import torch
    from sentence_transformers import SentenceTransformer
    from threadpoolctl import threadpool_info, threadpool_limits

    from data.loader import Document
    from evaluation.contextual_report import save_handoff
    from fusion.contextual_judge import LocalRelevanceJudge
    from retrievers.bm25 import BM25Retriever

    if args.output.exists():
        raise FileExistsError(f"run already exists: {args.output}")
    dataset = load_evaluation_dataset(args.data, "test", verify_all_splits=False)
    vectors, _, encoding = _load_vectors(args.data, args.vector_cache, dataset)
    selection = read_json(args.policy)
    if selection.get("selection_split") != "validation":
        raise ValueError("DAT policy must be selected on Validation")
    if (
        selection["input_sha256"]["dataset_manifest"]
        != dataset.provenance["manifest_sha256"]
    ):
        raise ValueError("policy belongs to another dataset")
    if "encoding" in selection["input_sha256"] and selection["input_sha256"][
        "encoding"
    ] != file_sha256(args.vector_cache / "encoding.json"):
        raise ValueError("policy encoding differs from supplied vector cache")
    if (
        encoding["model"]["model_name"] != args.encoder_name
        or encoding["model"]["revision"] != args.encoder_revision
    ):
        raise ValueError("encoder identity differs from frozen document vectors")
    fixed = FixedPolicy(selection["alpha"], selection["normalization"])
    encoder_identity = snapshot_identity(args.encoder_model)
    judge_identity = snapshot_identity(args.judge_model)
    encoder = SentenceTransformer(
        str(args.encoder_model), device=args.device, local_files_only=True
    ).eval()
    encoder.max_seq_length = encoding["max_seq_length"]
    judge = LocalRelevanceJudge(str(args.judge_model), device=args.device)
    sparse = BM25Retriever(
        [Document(d.document_id, d.text, d.title) for d in dataset.documents]
    )

    def encode(query):
        return encoder.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    synchronize = (
        torch.cuda.synchronize if args.device.startswith("cuda") else lambda: None
    )
    retrievers = {
        method: ContextualRetriever(
            dataset.documents,
            vectors,
            encode=encode,
            sparse_scores=sparse.score_all,
            judge=judge,
            fixed=fixed,
            adaptive=method == "dat",
            synchronize=synchronize,
        )
        for method in ("dat_fixed", "dat")
    }
    warmup = load_evaluation_dataset(args.data, "validation", verify_all_splits=False)
    primary = read_json(args.data / "manifest.json")["primary_slice_query_ids"]
    # 모델과 BM25가 추가로 로드한 모든 BLAS/OpenMP 라이브러리에도 제한을 적용한다.
    with threadpool_limits(2):
        effective_threadpools = threadpool_info()
        for query in warmup.queries[:10]:
            for retriever in retrievers.values():
                retriever.search(query.text, 10)
        judge.pairs = judge.truncated_pairs = 0
        records, samples = run_queries(
            dataset, primary, retrievers, run_id=args.run_id, repeats=args.repeats
        )
    # 첫 전체 회차는 품질과 추가 워밍업에 쓰고 이후 회차만 latency에 사용한다.
    protocol = {
        "scope": "local DAT-style top1 relevance-odds weighting; not full DAT reproduction",
        "quality_source": "fresh single-query inference, first complete primary pass",
        "test_status": "previously exposed official test; exploratory comparison",
        "query_scope": "official primary non_synthetic test",
        "query_count": len(primary),
        "encoder": {
            "name": args.encoder_name,
            "revision": args.encoder_revision,
            **encoder_identity,
        },
        "judge": {
            "name": args.judge_name,
            "revision": args.judge_revision,
            **judge_identity,
        },
        "encoding": encoding,
        "source_sha256": source_identity(),
        "candidate_k": 100,
        "alpha_definition": "Dense coefficient; alpha*Dense+(1-alpha)*BM25",
        "adaptive_rule": "Dense odds / (Dense odds + Sparse odds); clip probabilities to [1e-7,1-1e-7]",
        "agreement_rule": "skip judge and use frozen Fixed",
        "test_policy_tuning": False,
        "seed": 20261030,
        "selection_sha256": file_sha256(args.policy),
        "oracle": "not computed; Phase 5 analysis excluded",
    }
    environment = {
        "effective_threadpools": effective_threadpools,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "transformers": version("transformers"),
        "sentence_transformers": version("sentence-transformers"),
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "cpu_threads": 2,
        "query_encoder_dtype": "float32",
        "judge_dtype": str(judge.model.dtype),
        "dense_search": "CPU float64 full-corpus dot product with fixed document vectors",
        "query_batch": 1,
        "judge_batch": 2,
        "judge_max_tokens": judge.max_tokens,
        "judge_pairs_all_passes": judge.pairs,
        "truncated_pairs_all_passes": judge.truncated_pairs,
        "warmup_queries_per_method": min(10, len(warmup.queries)),
        "passes": args.repeats,
        "latency_passes": args.repeats - 1,
        "quality_pass_excluded_from_latency": True,
        "scope": "encoding + Dense/BM25 retrieval + optional judge tokenization/forward + fusion/top10; excludes loading, metrics, HTTP",
        "no_query_or_score_or_judge_cache": True,
        "no_external_inference_api": True,
    }
    save_handoff(
        args.output,
        args.data,
        records,
        selection=selection,
        protocol=protocol,
        samples=samples,
        environment=environment,
    )

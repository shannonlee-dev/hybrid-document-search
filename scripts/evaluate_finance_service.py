"""Run the preregistered Finance encoder service-path evaluation.

Commands (run from the repository root, in this order):
  prepare       build service corpus/index directories from frozen document vectors
  parity        path-equivalence checks 1-3 (stop if any fails)
  accuracy      BC Card Test conditions through POST /search
  latency       staged in-process latency and memory for one encoder
  http          full HTTP latency/throughput against a uvicorn server for one encoder
  cold-start    one cold service start and first query for one encoder
  external      Ko-MIRACL dev and Ko-StrategyQA conditions
Large inputs and indexes stay in --work-dir; only JSON results go to --output-dir.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from app.service import QueryCheckpoint
from evaluation import finance_service as fs
from evaluation.json_io import read_json, write_json


def _checkpoint(args):
    return QueryCheckpoint(
        base_model_path=args.base_model,
        checkpoint_path=args.checkpoint,
        identity_path=args.identity,
    )


def _bccard_paths(args):
    return args.data_dir / "corpus.jsonl", args.work_dir / "bccard-index"


def cmd_prepare(args):
    import numpy as np

    from data.loader import load_prepared_documents
    from evaluation.query_encoder_training import load_encoder_inputs

    load_encoder_inputs(
        args.data_dir, args.cache_dir, ("test",)
    )  # identity/order checks
    corpus, index_dir = _bccard_paths(args)
    documents = load_prepared_documents(corpus)
    vectors = np.load(args.cache_dir / "document-vectors.npy", allow_pickle=False)
    report = {"bccard": fs.build_index_dir(documents, vectors, index_dir)}
    for name in ("miracl10k", "strategyqa"):
        _, docs, vecs, *_ = fs.load_external(args.external_cache, name)
        fs.write_corpus(docs, args.work_dir / name / "corpus.jsonl")
        report[name] = fs.build_index_dir(docs, vecs, args.work_dir / name / "index")
    return report


def cmd_parity(args):
    corpus, index_dir = _bccard_paths(args)
    services, _ = fs.build_services(corpus, index_dir, args.device, _checkpoint(args))
    run = args.repro_run
    result = {
        "check1_checkpoint": fs.parity_checkpoint(
            _checkpoint(args),
            run / "train-seed43/selected-validation-vectors.npy",
            args.data_dir,
            args.device,
        ),
        "check2_fixed": fs.parity_fixed(
            services,
            args.data_dir,
            {
                "baseline": run / "baseline/baseline-test-vectors.npy",
                "seed43": run / "checkpoint-seed43/selected-test-vectors.npy",
            },
            Path("docs/validation/2026-11-02/finance-repro-blackwell/per-query.jsonl"),
        ),
        "check3_service_hybrid": fs.parity_service_hybrid(
            services,
            args.data_dir,
            args.cache_dir,
            Path("docs/validation/2026-10-30/qdap/main-baseline-per-query.jsonl"),
        ),
    }
    result["passed"] = (
        result["check1_checkpoint"]["bitwise_equal"]
        and result["check2_fixed"]["passed"]
        and result["check3_service_hybrid"]["passed"]
    )
    return result


def cmd_accuracy(args):
    corpus, index_dir = _bccard_paths(args)
    services, build_seconds = fs.build_services(
        corpus, index_dir, args.device, _checkpoint(args)
    )
    dataset, queries, qrels = fs.bccard_test(args.data_dir)
    rows, mismatches = fs.run_conditions(
        fs.condition_services(services), queries, qrels
    )
    fs.write_rows(args.output_dir / "bccard-per-query.jsonl", rows)
    run = args.repro_run
    drift = fs.encoding_drift(
        services,
        queries,
        {
            "baseline": run / "baseline/baseline-test-vectors.npy",
            "seed43": run / "checkpoint-seed43/selected-test-vectors.npy",
        },
        dataset,
    )
    return {
        "dataset": "BC Card Test primary",
        "queries": len(queries),
        "conditions": {
            k: {"encoder": v[0], "kind": v[1], "alpha": v[2], "role": v[3]}
            for k, v in fs.CONDITIONS.items()
        },
        "api_vs_direct_top10_mismatches": mismatches,
        "aggregates": fs.aggregate(rows),
        "comparisons": fs.compare(rows),
        "fp32_service_vs_pr41_bf16_query_vectors": drift,
        "service_build_seconds": build_seconds,
        "per_query_sha256": fs.file_sha256(args.output_dir / "bccard-per-query.jsonl"),
    }


def cmd_latency(args):
    corpus, index_dir = _bccard_paths(args)
    from app.service import build_search_service
    from evaluation.data import load_evaluation_dataset

    before = fs.gpu_snapshot()
    service = build_search_service(
        corpus,
        dense_index_path=index_dir,
        dense_device=args.device,
        dense_query_checkpoint=_checkpoint(args) if args.encoder == "fin" else None,
    )
    _, queries, _ = fs.bccard_test(args.data_dir)
    validation = load_evaluation_dataset(
        args.data_dir, "validation", verify_all_splits=False
    )
    warmup = [(q.query_id, q.text) for q in validation.queries[:50]]
    result = fs.staged_latency(service, queries, warmup, repeats=3)
    result.update(
        {
            "encoder": args.encoder,
            "gpu_before": before,
            "gpu_after": fs.gpu_snapshot(),
            "environment": fs.environment(),
        }
    )
    return result


def _server(args, port):
    corpus, index_dir = _bccard_paths(args)
    env = dict(
        os.environ,
        SEARCH_CORPUS_PATH=str(corpus),
        DENSE_INDEX_PATH=str(index_dir),
        DENSE_DEVICE=args.device,
    )
    if args.encoder == "fin":
        env.update(
            DENSE_QUERY_CHECKPOINT=str(args.checkpoint),
            DENSE_QUERY_CHECKPOINT_IDENTITY=str(args.identity),
            DENSE_QUERY_BASE_MODEL=str(args.base_model),
        )
    # The app logs one line per search; an unread pipe would fill and block the server.
    log = (args.work_dir / f"server-{args.encoder}.log").open("w")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_ready(base_url, process, timeout=900):
    import httpx

    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited with code {process.returncode}; see server log"
            )
        try:
            methods = httpx.get(f"{base_url}/search/methods", timeout=5).json()[
                "methods"
            ]
            if all(
                m["available"]
                for m in methods
                if m["method"] in ("bm25", "dense", "hybrid")
            ):
                return time.perf_counter() - started
            raise RuntimeError(f"methods unavailable: {methods}")
        except (httpx.HTTPError, OSError):
            time.sleep(0.5)
    raise TimeoutError("server did not become ready")


def cmd_http(args):
    from evaluation.data import load_evaluation_dataset

    port = 18000 + (1 if args.encoder == "fin" else 0)
    base_url = f"http://127.0.0.1:{port}"
    process = _server(args, port)
    try:
        ready = _wait_ready(base_url, process)
        _, queries, _ = fs.bccard_test(args.data_dir)
        validation = load_evaluation_dataset(
            args.data_dir, "validation", verify_all_splits=False
        )
        warmup = [(q.query_id, q.text) for q in validation.queries[:50]]
        before = fs.gpu_snapshot()
        results = {
            m: fs.http_latency(base_url, queries, warmup, m)
            for m in ("hybrid", "dense")
        }
        return {
            "encoder": args.encoder,
            "server_ready_seconds": ready,
            "results": results,
            "server_process_memory": fs._proc_memory(process.pid),
            "gpu_before": before,
            "gpu_after": fs.gpu_snapshot(),
            "server_pid": process.pid,
            "environment": fs.environment(),
        }
    finally:
        process.terminate()
        process.wait(timeout=60)


def cmd_cold_start(args):
    import torch

    from app.schemas import RetrievalMethod
    from app.service import build_search_service

    corpus, index_dir = _bccard_paths(args)
    started = time.perf_counter()
    service = build_search_service(
        corpus,
        dense_index_path=index_dir,
        dense_device=args.device,
        dense_query_checkpoint=_checkpoint(args) if args.encoder == "fin" else None,
    )
    ready = time.perf_counter() - started
    _, queries, _ = fs.bccard_test(args.data_dir)
    t = time.perf_counter()
    service.search(queries[0][1], RetrievalMethod.HYBRID, fs.TOP_K)
    first = (time.perf_counter() - t) * 1e3
    return {
        "encoder": args.encoder,
        "service_ready_seconds": ready,
        "first_hybrid_query_ms": first,
        "torch_cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "process_memory": fs._proc_memory(),
    }


def cmd_external(args):
    import numpy as np

    from app.schemas import RetrievalMethod
    from fusion.score import DEFAULT_BM25_WEIGHT, ScoreFusionRetriever
    from retrievers.dense import DenseRetriever
    from scripts.benchmark_score_fusion import _CachedEmbedder

    previous = {"miracl10k": "miracl-dev.json", "strategyqa": "strategyqa.json"}
    report = {}
    for name in ("miracl10k", "strategyqa"):
        data, _, _, query_vectors, queries, qrels = fs.load_external(
            args.external_cache, name
        )
        services, _ = fs.build_services(
            args.work_dir / name / "corpus.jsonl",
            args.work_dir / name / "index",
            args.device,
            _checkpoint(args),
        )
        # Path check: cached BGE query vectors through the service retrievers.
        service = services["bge"]
        cached = DenseRetriever(
            index=service.retriever(RetrievalMethod.DENSE).index,
            documents=service.retriever(RetrievalMethod.DENSE).documents,
            embedder=_CachedEmbedder(data, query_vectors),
        )
        cached_fusion = ScoreFusionRetriever(
            service.retriever(RetrievalMethod.BM25),
            cached,
            bm25_weight=DEFAULT_BM25_WEIGHT,
        )
        path_check = {}
        old = read_json(Path("docs/experiments/score-fusion") / previous[name])[
            "quality"
        ]
        for label, retriever in (("dense", cached), ("score_0.05", cached_fusion)):
            rows = [
                fs.query_metrics(
                    [r.document_id for r in retriever.search(t, fs.TOP_K)], qrels[q]
                )
                for q, t in queries
            ]
            mine = {
                m: sum(r[m] for r in rows) / len(rows)
                for m in ("recall@5", "recall@10", "mrr@10", "ndcg@10")
            }
            path_check[label] = {
                "recomputed": mine,
                "previous": {m: old[label][m] for m in mine},
                "max_abs_difference": max(abs(mine[m] - old[label][m]) for m in mine),
            }
        # Live FP32 BGE query vectors vs cached BGE query vectors.
        index_of = {q: i for i, q in enumerate(data["query_ids"])}
        live = service.retriever(RetrievalMethod.DENSE).embedder
        cos = np.asarray(
            [
                float(
                    live.encode_query(t)[0].astype(np.float64)
                    @ query_vectors[index_of[q]].astype(np.float64)
                )
                for q, t in queries
            ]
        )
        mapped = fs.condition_services(services)
        conditions = ["A-bge", "A-fin", "B-bge", "B-fin", "D-bge", "D-fin"]
        rows, mismatches = fs.run_conditions(mapped, queries, qrels, conditions)
        fs.write_rows(args.output_dir / f"{name}-per-query.jsonl", rows)
        report[name] = {
            "queries": len(queries),
            "path_check_cached_bge": path_check,
            "live_vs_cached_bge_query_cosine": {
                "min": float(cos.min()),
                "mean": float(cos.mean()),
            },
            "api_vs_direct_top10_mismatches": mismatches,
            "aggregates": fs.aggregate(rows),
            "comparisons": fs.compare(
                rows, {k: v for k, v in fs.PAIRS.items() if k in ("A", "B", "D")}
            ),
            "per_query_sha256": fs.file_sha256(
                args.output_dir / f"{name}-per-query.jsonl"
            ),
        }
    return report


COMMANDS = {
    "prepare": cmd_prepare,
    "parity": cmd_parity,
    "accuracy": cmd_accuracy,
    "latency": cmd_latency,
    "http": cmd_http,
    "cold-start": cmd_cold_start,
    "external": cmd_external,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, required=True, help="original BC Card vector cache"
    )
    parser.add_argument("--external-cache", type=Path, required=True)
    parser.add_argument(
        "--base-model", type=Path, required=True, help="local BGE-M3 snapshot"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--identity",
        type=Path,
        default=Path("config/query_checkpoints/finance-seed43-fbbd5ed0.json"),
        help="checkpoint identity JSON (fail-closed base model check)",
    )
    parser.add_argument(
        "--repro-run",
        type=Path,
        required=True,
        help="PR #41 reproduction run directory",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=("bge", "fin"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)
    if args.command in ("latency", "http", "cold-start") and not args.encoder:
        parser.error("--encoder is required for latency, http and cold-start")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    name = (
        args.command
        + (f"-{args.encoder}" if args.encoder else "")
        + (f"-{args.tag}" if args.tag else "")
    )
    out = args.output_dir / f"{name}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    result = COMMANDS[args.command](args)
    write_json(out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

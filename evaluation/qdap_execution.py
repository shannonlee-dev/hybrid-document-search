"""Frozen QDAP evaluation, fresh live latency, and independent metric replay."""

import gc
import platform
import time
from pathlib import Path

import numpy as np
import torch

from evaluation.json_io import read_json, write_json
from evaluation.latency import latency_summary
from evaluation.qdap import (
    PoolBatch,
    predict_probabilities,
    probability_alphas,
    rank_pools,
    ranking_metrics,
)
from evaluation.qdap_benchmark import (
    METHODS,
    batch_utilities,
    judgments_for,
    legacy_ranking,
    primary,
    rows,
    sha,
    split_scores,
)
from fusion.adaptive import Router, query_features
from fusion.qdap import QDAPLarge, QDAPSmall


def frozen_selection(args):
    selection = read_json(args.output / "selection.json")
    if selection["selection_split"] != "validation":
        raise ValueError("Validation selection required")
    if sha(args.output / "protocol.json") != selection["protocol_sha256"]:
        raise ValueError("protocol changed after selection")

    for source, digest in selection["evaluation_input_sha256"].items():
        path = Path(source)
        # Only the three obsolete DAT cache files are no longer evaluation inputs.
        if path.parts[-5:-1] == (
            "artifacts",
            "paper-guided",
            "dat-local",
            "bccard",
        ) and path.name in ("dense.npy", "sparse.npy", "queries.json"):
            continue
        if sha(path) != digest:
            raise ValueError(f"evaluation input changed after selection: {source}")
    for config in selection["selected"].values():
        if sha(config["checkpoint"]) != config["checkpoint_sha256"]:
            raise ValueError("selected checkpoint changed")
    return selection


def load_large(selection, device):
    from transformers import AutoModel, AutoTokenizer

    encoder = AutoModel.from_pretrained(
        selection["model"],
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    predictor = QDAPLarge(encoder).to(device)
    predictor.load_state_dict(
        torch.load(
            selection["selected"]["L"]["checkpoint"],
            map_location=device,
            weights_only=True,
        )
    )
    predictor.eval()
    tokenizer = AutoTokenizer.from_pretrained(selection["model"], local_files_only=True)
    return predictor, tokenizer


def tokens(tokenizer, texts, device, length):
    batch = tokenizer(
        texts, padding=True, truncation=True, max_length=length, return_tensors="pt"
    )
    return {k: v.to(device) for k, v in batch.items()}


def test_inputs(args):
    data = args.inputs / "data/processed/bccard"
    docs = rows(data / "corpus.jsonl")
    qs = [q for q in rows(data / "queries_test.jsonl") if primary(q)]
    if len(qs) != 721:
        raise ValueError("official non-synthetic Test must have 721 queries")
    return data, docs, qs


def evaluate(args):
    """Read Test labels only after the immutable Validation selection exists."""
    selection = frozen_selection(args)
    out = args.output
    if (out / "test-summary.json").exists():
        raise ValueError("Test already evaluated; use verify to replay")
    frozen_hash = sha(out / "selection.json")
    data, docs, qs = test_inputs(args)
    ids = np.array([d["document_id"] for d in docs])
    labels = judgments_for(data, "test", qs, docs)
    dense, sparse = split_scores(args.inputs, "test", qs)
    pools = PoolBatch.from_scores(dense, sparse, ids)
    curves = batch_utilities(pools, labels)
    all_test = rows(data / "queries_test.jsonl")
    selected = [i for i, q in enumerate(all_test) if primary(q)]
    vectors = np.load(
        args.inputs / "artifacts/bccard-adaptive-vectors-resumed/query-vectors-test.npy"
    )[selected]
    psmall = predict_probabilities(
        selection["selected"]["S"]["checkpoint"], vectors, device=args.device
    )
    large, tokenizer = load_large(selection, args.device)
    with torch.inference_mode():
        plarge = np.concatenate(
            [
                large(
                    **tokens(
                        tokenizer,
                        [q["text"] for q in qs[start : start + 16]],
                        args.device,
                        128,
                    )
                )
                .cpu()
                .numpy()
                for start in range(0, len(qs), 16)
            ]
        )
    del large
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    router_path = args.inputs / "docs/experiments/adaptive/router.json"
    router = Router.load(router_path)
    ranking = {
        "BM25": np.array([np.lexsort((ids, -s))[:10] for s in sparse]),
        "Dense": np.array([np.lexsort((ids, -d))[:10] for d in dense]),
        "Existing Fixed": np.array(
            [legacy_ranking(d, s, ids, 0.05) for d, s in zip(dense, sparse)]
        ),
        "Paper Fixed": rank_pools(
            pools, np.full(len(qs), selection["paper_fixed_alpha"])
        ),
    }
    routed, modes = [], []
    for q, d, s in zip(qs, dense, sparse, strict=True):
        mode = int(router.select([query_features(q["text"], s, d, ids)])[0])
        modes.append(mode)
        routed.append(legacy_ranking(d, s, ids, router.state["weights"][mode]))
    ranking["Existing Router"] = np.array(routed)
    alphas = {}
    for variant, probability in [("S", psmall), ("L", plarge)]:
        alphas[variant] = probability_alphas(
            probability, selection["selected"][variant]["readout"]
        )
        ranking[f"QDAP-{variant}"] = rank_pools(pools, alphas[variant])
    oracle = curves.argmax(1) / 100
    ranking["Oracle alpha"] = rank_pools(pools, oracle)
    result, per_query = {}, {}
    for name in METHODS:
        result[name], per_query[name] = ranking_metrics(ranking[name], labels)
    np.savez(out / "test-rankings.npz", **ranking)
    np.savez(out / "test-per-query.npz", **per_query)
    np.savez(
        out / "test-pools.npz",
        pools=pools.indices,
        pairs=pools.pairs,
        valid=pools.valid,
        quality=curves,
    )
    np.savez(
        out / "test-predictions.npz",
        S=psmall,
        L=plarge,
        S_alpha=alphas["S"],
        L_alpha=alphas["L"],
        oracle_alpha=oracle,
    )
    write_json(out / "test-queries.json", qs)
    if sha(out / "selection.json") != frozen_hash:
        raise ValueError("selection changed during Test")
    write_json(
        out / "test-summary.json",
        {
            "queries": len(qs),
            "selection_sha256": frozen_hash,
            "rows": result,
            "router_modes": {str(k): modes.count(k) for k in range(3)},
            "paper_fixed_alpha": selection["paper_fixed_alpha"],
            "existing_fixed_bm25_weight": 0.05,
            "oracle": "post-hoc nDCG argmax among 101 alphas; not deployable; not used for selection",
            "test_qrels_sha256": sha(data / "qrels_test.jsonl"),
            "router_sha256": sha(router_path),
            "quality_precision": "original frozen FP32 query/document caches, float64 dot products/normalization; S FP32 head, L BF16 encoder and FP32 head",
        },
    )
    print("TEST RESULTS", result, flush=True)


def latency(args):
    """Measure the same live queries per method; no cached scores or predictions."""
    from sentence_transformers import SentenceTransformer

    from data.loader import Document
    from retrievers.bm25 import BM25Retriever

    selection = frozen_selection(args)
    if not torch.cuda.is_available():
        raise ValueError("GPU required for matched latency")
    _, docs, qs = test_inputs(args)
    ids = np.array([d["document_id"] for d in docs])
    vectors = np.load(
        args.inputs / "artifacts/bccard-adaptive-vectors-resumed/document-vectors.npy"
    ).astype(float)
    bm25 = BM25Retriever([Document(**doc) for doc in docs])
    dense_model = SentenceTransformer(
        selection["model"], device=args.device, local_files_only=True
    ).eval()
    head = QDAPSmall(1024).eval()
    head.load_state_dict(
        torch.load(
            selection["selected"]["S"]["checkpoint"],
            map_location="cpu",
            weights_only=True,
        )
    )
    large, tokenizer = load_large(selection, args.device)
    stream = torch.cuda.Stream()
    router = Router.load(args.inputs / "docs/experiments/adaptive/router.json")
    cached = np.load(args.output / "test-rankings.npz")

    def search(text, method):
        torch.cuda.synchronize()
        start = time.perf_counter()
        lp = None
        if method == "QDAP-L":
            with torch.cuda.stream(stream), torch.inference_mode():
                lp = large(**tokens(tokenizer, [text], args.device, 128))
        d = s = qvector = None
        if method != "BM25":
            encoded = tokens(dense_model.tokenizer, [text], args.device, 512)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                qvector = (
                    torch.nn.functional.normalize(
                        dense_model(encoded)["sentence_embedding"].float(), dim=-1
                    )
                    .cpu()
                    .numpy()
                )
            d = vectors @ qvector[0].astype(float)
        if method != "Dense":
            s = bm25.score_all(text).astype(float)
        if method in ("BM25", "Dense"):
            order = np.lexsort((ids, -(s if method == "BM25" else d)))[:10]
        elif method == "Existing Fixed":
            order = legacy_ranking(d, s, ids, 0.05)
        elif method == "Existing Router":
            mode = int(router.select([query_features(text, s, d, ids)])[0])
            order = legacy_ranking(d, s, ids, router.state["weights"][mode])
        else:
            pools = PoolBatch.from_scores(d[None], s[None], ids)
            if method == "Paper Fixed":
                alpha = np.array([selection["paper_fixed_alpha"]])
            else:
                if method == "QDAP-S":
                    with torch.inference_mode():
                        probability = head(torch.from_numpy(qvector)).numpy()
                else:
                    stream.synchronize()
                    probability = lp.cpu().numpy()
                alpha = probability_alphas(
                    probability, selection["selected"][method[-1]]["readout"]
                )
            order = rank_pools(pools, alpha)[0]
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000, order

    selected = np.random.default_rng(370).choice(len(qs), 30, replace=False)
    methods = list(METHODS[:-1])
    samples = []
    for repeat, method_order in enumerate((methods, methods[::-1])):
        for method in method_order:
            for i in selected[:10]:
                search(qs[i]["text"], method)
            for i in selected:
                elapsed, ranking = search(qs[i]["text"], method)
                samples.append(
                    {
                        "method": method,
                        "round": repeat,
                        "query_id": qs[i]["query_id"],
                        "total_ms": elapsed,
                        "ranking": ranking.tolist(),
                        "cached_ranking_equal": bool(
                            np.array_equal(ranking, cached[method][i])
                        ),
                    }
                )
            print("Measured", repeat, method, flush=True)
    aggregates = {
        method: latency_summary(
            [r["total_ms"] for r in samples if r["method"] == method]
        )
        for method in methods
    }
    write_json(args.output / "latency-samples.json", samples)
    write_json(
        args.output / "latency.json",
        {
            "methods": aggregates,
            "hardware": torch.cuda.get_device_name(0),
            "platform": platform.platform(),
            "scope": "warm batch1: live tokenizer/GPU BF16 encoder, CPU4 full-corpus Dense/BM25, fusion/router/top10; L overlaps independent prediction encoder using a separate CUDA stream; excludes loading/HTTP/concurrent requests",
            "sampling": "same30 Test queries x2 per method, reversed method order,10 warmup per block",
            "precision_note": "latency uses fresh BF16 query encoding; quality table uses original FP32 caches; raw ranking parity flags are retained",
            "selection_sha256": sha(args.output / "selection.json"),
            "oracle_latency": None,
        },
    )
    print("LATENCY", aggregates, flush=True)


def verify(args):
    """Recompute every final Test metric with the repository's independent API."""
    from evaluation.metrics import (
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
        reciprocal_rank_at_k,
    )
    from retrievers.base import SearchResult

    selection = frozen_selection(args)
    data, docs, qs = test_inputs(args)
    labels = judgments_for(data, "test", qs, docs)
    summary = read_json(args.output / "test-summary.json")
    if sha(args.output / "selection.json") != summary["selection_sha256"]:
        raise ValueError("selection changed since Test")
    rankings = np.load(args.output / "test-rankings.npz")
    values = {}
    for method in METHODS:
        checks = []
        for ranking, judgments in zip(rankings[method], labels, strict=True):
            hits = [
                SearchResult(docs[int(index)]["document_id"], rank, 0.0)
                for rank, index in enumerate(ranking, 1)
            ]
            rel = {
                docs[index]["document_id"]: grade for index, grade in judgments.items()
            }
            checks.append(
                [
                    ndcg_at_k(hits, rel, 10),
                    precision_at_k(hits, rel, 1),
                    recall_at_k(hits, rel, 10),
                    reciprocal_rank_at_k(hits, rel, 10),
                ]
            )
        measured = dict(
            zip(
                ("ndcg@10", "p@1", "recall@10", "mrr@10"),
                np.mean(checks, axis=0).tolist(),
                strict=True,
            )
        )
        for metric, value in measured.items():
            np.testing.assert_allclose(
                value, summary["rows"][method][metric], atol=1e-12
            )
        values[method] = measured
    pools = PoolBatch.load(args.output / "test-pools.npz")
    predictions = np.load(args.output / "test-predictions.npz")
    for variant in ("S", "L"):
        alpha = probability_alphas(
            predictions[variant], selection["selected"][variant]["readout"]
        )
        np.testing.assert_array_equal(
            rank_pools(pools, alpha), rankings[f"QDAP-{variant}"]
        )
    timing = read_json(args.output / "latency.json")
    samples = read_json(args.output / "latency-samples.json")
    for method, stats in timing["methods"].items():
        selected = [s for s in samples if s["method"] == method]
        assert len(selected) == 60 and len({s["query_id"] for s in selected}) == 30
        actual = latency_summary([s["total_ms"] for s in selected])
        for metric in ("mean_ms", "p95_ms"):
            np.testing.assert_allclose(actual[metric], stats[metric], atol=1e-12)
    protocol = read_json(args.output / "protocol.json")
    for path, digest in protocol["input_sha256"].items():
        if sha(path) != digest:
            raise ValueError("retrieval input changed during training")
    result = {
        "passed": True,
        "independent_query_rankings": len(qs) * len(METHODS),
        "latency_samples": len(samples),
        "metrics": values,
        "selection_and_checkpoint_hashes": "verified",
        "retrieval_inputs_unchanged": True,
    }
    write_json(args.output / "verification.json", result)
    print("VERIFIED", result, flush=True)

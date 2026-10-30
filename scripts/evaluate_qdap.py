"""Train and replay an isolated QDAP-S experiment on explicit frozen inputs."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from evaluation.json_io import read_json, write_json
from evaluation.qdap import (
    DEFAULT_SEEDS,
    FIXED_ALPHA,
    FitConfig,
    PoolBatch,
    fit_qdap,
    graded_utilities,
    predict_probabilities,
    probability_alphas,
    rank_pools,
    ranking_metrics,
    select_candidate,
    validate_partitions,
)


def main(argv=None):
    """Run fit, utilities, validation selection or frozen test evaluation."""
    parser = _parser()
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    if args.stage == "fit":
        _fit(args)
        return
    pools = PoolBatch.load(args.pools)
    queries, judgments = _judgments(args)
    if len(queries) != len(pools.indices):
        parser.error("query manifest and pool rows must align")
    if args.stage == "utilities":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, graded_utilities(pools, judgments))
        return
    vectors = _vectors(args, len(queries))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = _hashes(
        [args.pools, args.queries, args.qrels, args.corpus]
        + ([args.vectors] if args.vectors else [])
    )
    if args.stage == "select":
        manifest = read_json(args.candidates)
        records = []
        for candidate in manifest["candidates"]:
            checkpoint = (args.candidates.parent / candidate["checkpoint"]).resolve()
            digest = _sha(checkpoint)
            if digest != candidate["checkpoint_sha256"]:
                raise ValueError("candidate checkpoint hash changed")
            probability = predict_probabilities(checkpoint, vectors, device=args.device)
            for readout in ("mean", "argmax"):
                alpha = probability_alphas(probability, readout)
                metrics, _ = ranking_metrics(rank_pools(pools, alpha), judgments)
                records.append(
                    {
                        **candidate,
                        "checkpoint": str(checkpoint),
                        "readout": readout,
                        "name": f"{candidate['name']}:{readout}",
                        "metrics": metrics,
                    }
                )
        selected = select_candidate(records)
        write_json(
            args.output_dir / "selection.json",
            {
                "selection_split": "validation",
                "criterion": "maximum nDCG@10; stable candidate/readout order on ties",
                "selected": selected,
                "validation_candidates": records,
                "input_sha256": inputs,
                "device": args.device,
                "versions": _versions(),
            },
        )
    else:
        frozen = read_json(args.selection)
        if frozen.get("selection_split") != "validation":
            raise ValueError("selection must be frozen on validation")
        if inputs[str(args.pools)] == frozen["input_sha256"].get(str(args.pools)):
            raise ValueError("test evaluation cannot reuse the validation pool")
        candidate = frozen["selected"]
        checkpoint = Path(candidate["checkpoint"])
        if _sha(checkpoint) != candidate["checkpoint_sha256"]:
            raise ValueError("selected checkpoint hash changed")
        probability = predict_probabilities(checkpoint, vectors, device=args.device)
        alpha = probability_alphas(probability, candidate["readout"])
        rankings = rank_pools(pools, alpha)
        metrics, _ = ranking_metrics(rankings, judgments)
        baseline, _ = ranking_metrics(
            rank_pools(pools, np.full(len(queries), FIXED_ALPHA)), judgments
        )
        parity = None
        if args.expected_rankings:
            expected = np.load(args.expected_rankings, allow_pickle=False)
            parity = bool(np.array_equal(rankings, expected))
            if not parity:
                raise ValueError("rankings differ from archived expected rankings")
        np.save(args.output_dir / "rankings.npy", rankings)
        np.save(args.output_dir / "alphas.npy", alpha)
        write_json(
            args.output_dir / "evaluation.json",
            {
                "evaluation_split": "test",
                "queries": len(queries),
                "selected": candidate,
                "metrics": metrics,
                "fixed_alpha": FIXED_ALPHA,
                "fixed_metrics": baseline,
                "delta_ndcg": metrics["ndcg@10"] - baseline["ndcg@10"],
                "archived_ranking_parity": parity,
                "alpha": {
                    "mean": float(alpha.mean()),
                    "min": float(alpha.min()),
                    "max": float(alpha.max()),
                    "unique": len(np.unique(alpha)),
                },
                "input_sha256": inputs,
                "selection_sha256": _sha(args.selection),
                "expected_rankings_sha256": _sha(args.expected_rankings)
                if args.expected_rankings
                else None,
                "device": args.device,
                "versions": _versions(),
            },
        )


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    commands = parser.add_subparsers(dest="stage", required=True)
    fit = commands.add_parser("fit")
    fit.add_argument("--vectors", type=Path, required=True)
    fit.add_argument("--utilities", type=Path, required=True)
    fit.add_argument("--development-queries", type=Path, required=True)
    fit.add_argument("--evaluation-queries", type=Path, nargs="+", required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    fit.add_argument(
        "--ce-modes",
        choices=("literal", "reference"),
        nargs="+",
        default=["literal", "reference"],
    )
    for option, default in (
        ("fit-size", 6000),
        ("calibration-size", 1000),
        ("epochs", 5),
        ("batch-size", 64),
        ("accumulation", 2),
    ):
        fit.add_argument(f"--{option}", type=int, default=default)
    for stage in ("utilities", "select", "evaluate"):
        command = commands.add_parser(stage)
        for option in ("pools", "queries", "corpus", "qrels"):
            command.add_argument(f"--{option}", type=Path, required=True)
        if stage == "utilities":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument(
                "--vectors", type=Path, help="Otherwise use vectors stored in pool NPZ"
            )
            command.add_argument("--output-dir", type=Path, required=True)
        if stage == "select":
            command.add_argument("--candidates", type=Path, required=True)
        if stage == "evaluate":
            command.add_argument("--selection", type=Path, required=True)
            command.add_argument("--expected-rankings", type=Path)
    return parser


def _fit(args):
    config = FitConfig(
        fit_size=args.fit_size,
        calibration_size=args.calibration_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation=args.accumulation,
    )
    development = _rows(args.development_queries)
    if len(development) != config.fit_size + config.calibration_size:
        raise ValueError("development manifest must match exact fit/calibration sizes")
    evaluation = [row for path in args.evaluation_queries for row in _rows(path)]
    validate_partitions(
        development[: config.fit_size], development[config.fit_size :], evaluation
    )
    vectors = np.load(args.vectors, allow_pickle=False)
    utilities = np.load(args.utilities, allow_pickle=False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    candidates = []
    for mode in args.ce_modes:
        for seed in args.seeds:
            result = fit_qdap(
                vectors,
                utilities,
                seed=seed,
                ce_mode=mode,
                config=config,
                device=args.device,
            )
            name = f"{mode}-{seed}"
            checkpoint = args.output_dir / f"{name}.pt"
            torch.save(result.state_dict, checkpoint)
            candidates.append(
                {
                    "name": name,
                    "ce_mode": mode,
                    "seed": seed,
                    "checkpoint": checkpoint.name,
                    "checkpoint_sha256": _sha(checkpoint),
                    "chosen_epoch": result.chosen_epoch,
                    "history": result.history,
                }
            )
    write_json(
        args.output_dir / "candidates.json",
        {
            "config": asdict(config),
            "candidates": candidates,
            "input_sha256": _hashes(
                [
                    args.vectors,
                    args.utilities,
                    args.development_queries,
                    *args.evaluation_queries,
                ]
            ),
            "device": args.device,
            "versions": _versions(),
        },
    )


def _judgments(args):
    documents = _rows(args.corpus)
    indices = {row["document_id"]: i for i, row in enumerate(documents)}
    if len(indices) != len(documents):
        raise ValueError("corpus document IDs must be unique")
    queries = _rows(args.queries)
    query_ids = [row["query_id"] for row in queries]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query IDs must be unique")
    by_query = {query_id: {} for query_id in query_ids}
    for row in _rows(args.qrels):
        if row["query_id"] in by_query:
            index = indices[row["document_id"]]
            if index in by_query[row["query_id"]]:
                raise ValueError("duplicate query/document qrel")
            by_query[row["query_id"]][index] = row["relevance"]
    if any(not judgments for judgments in by_query.values()):
        raise ValueError("every pool query needs explicit qrels")
    return queries, [by_query[query_id] for query_id in query_ids]


def _vectors(args, count):
    if args.vectors:
        vectors = np.load(args.vectors, allow_pickle=False)
    else:
        with np.load(args.pools, allow_pickle=False) as data:
            vectors = data["vectors"]
    if len(vectors) != count:
        raise ValueError("query vectors must align with ordered query manifest")
    return vectors


def _rows(path):
    content = path.read_text(encoding="utf-8")
    return (
        json.loads(content)
        if content.lstrip().startswith("[")
        else [json.loads(line) for line in content.splitlines() if line.strip()]
    )


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hashes(paths):
    return {str(path): _sha(path) for path in paths}


def _versions():
    return {"torch": torch.__version__, "numpy": np.__version__}


if __name__ == "__main__":
    main()

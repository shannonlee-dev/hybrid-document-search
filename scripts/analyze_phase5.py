"""Phase 5 (#36): replay the frozen #24 router, paired statistics, cross-model joins.

Runs are saved enterprise reports (`scripts.evaluate_enterprise evaluate`).
Methods are addressed as `NAME:METHOD`, where `--run NAME=PATH` registers a
report. Every comparison first checks run identity and refuses protocol
mismatches. Nothing here retrieves with a new setting except `replay-adaptive`,
which only re-executes the committed frozen router.
"""

import csv
import json
import sys
from pathlib import Path

from scripts._cli import CliArgumentParser, _positive_int

DEFAULT_ROUTER = Path("docs/experiments/adaptive/router.json")
DEFAULT_DATA = Path("data/processed/bccard")
# cross-model 표에서 쓰는 열 이름. 없는 열이 있으면 해당 분석만 건너뛴다.
OVERLAP_PAIRS = (
    ("bm25", "bge_dense"),
    ("bm25", "kanana_dense"),
    ("bge_dense", "kanana_dense"),
)


def replay_adaptive(data_dir, cache_dir, router_path, *, expected_router_sha256=None):
    """Re-execute the frozen #24 router on test and keep per-query modes.

    Quality only: one pass, no latency claim. The mode oracle uses the same
    tie rule as `evaluation.adaptive.evaluate_frozen`.
    """
    from dataclasses import asdict

    from data.bccard import load_query_metadata
    from evaluation.adaptive import _retriever
    from evaluation.data import file_sha256
    from evaluation.enterprise import (
        query_metrics,
        run_enterprise_baselines,
        slice_definitions,
        summarize,
    )
    from evaluation.metrics import evaluate_query, ndcg_at_k
    from fusion.adaptive import MODES, Router, training_labels

    router_sha256 = file_sha256(Path(router_path))
    if expected_router_sha256 and router_sha256 != expected_router_sha256:
        raise ValueError("router.json이 PR #30 frozen router와 다릅니다.")
    router = Router.load(Path(router_path))
    dataset, adaptive, _ = _retriever(data_dir, cache_dir, "test", router)
    report = run_enterprise_baselines(
        data_dir, cache_dir, "test", methods=("score_fusion",), warmup=1, repeats=1
    )
    slices = slice_definitions(load_query_metadata(data_dir, "test"))
    rows = {"adaptive": [], "adaptive_oracle": []}
    for query in dataset.queries:
        judged = dataset.qrels[query.query_id]
        hits, details = adaptive.search_details(query.text, 10)
        sparse, dense = adaptive.score_arrays(query.text)
        by_mode = [
            adaptive.rank_scores(sparse, dense, weight, 10)
            for weight in router.state["weights"]
        ]
        best = int(training_labels([[ndcg_at_k(h, judged, 10) for h in by_mode]])[0])
        for name, mode, results in (
            ("adaptive", details["mode"], hits),
            ("adaptive_oracle", MODES[best], by_mode[best]),
        ):
            metrics = evaluate_query(results, judged)
            row = {
                "query_id": query.query_id,
                "text": query.text,
                "mode": mode,
                "metrics": metrics,
                "results": [asdict(hit) for hit in results],
            }
            row["metrics"] |= query_metrics(row["results"], judged)
            row["metrics"]["mrr@10"] = row["metrics"]["rr@10"]
            rows[name].append(row)
    for name, method_rows in rows.items():
        per_query = {row["query_id"]: row["metrics"] for row in method_rows}
        report["methods"][name] = {
            "config": {
                "router_sha256": router_sha256,
                "weights": router.state["weights"],
                "selection": "frozen router"
                if name == "adaptive"
                else "best frozen mode by binary nDCG@10; post-hoc upper bound",
            },
            "latency": None,
            "queries": method_rows,
            "slices": {
                slice_name: summarize(per_query, ids)
                for slice_name, ids in slices.items()
                if ids
            },
        }
    report["conditions"]["replay"] = (
        "PR #30 frozen router re-executed on this vector cache; quality only"
    )
    return report


def _parse_runs(values):
    runs = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name or name in runs:
            raise ValueError(f"--run은 고유한 NAME=PATH 형식이어야 합니다: {value}")
        runs[name] = Path(path)
    return runs


def _resolve(ref, runs, reports):
    from evaluation.json_io import read_json

    name, sep, method = ref.partition(":")
    if not sep or name not in runs:
        raise ValueError(
            f"NAME:METHOD 형식이어야 하고 NAME은 --run에 있어야 합니다: {ref}"
        )
    if name not in reports:
        reports[name] = read_json(runs[name])
    return name, method


def _identity(ref, runs, reports, manifest):
    from evaluation.provenance import run_identity

    name, method = _resolve(ref, runs, reports)
    return run_identity(
        reports[name], method, manifest=manifest, report_path=runs[name]
    )


def compare(args):
    from evaluation.json_io import read_json
    from evaluation.paired import paired_comparison, per_query_metrics
    from evaluation.provenance import compare_identities

    runs, reports = _parse_runs(args.run), {}
    manifest = read_json(args.data_dir / "manifest.json")
    primary = manifest["primary_slice_query_ids"]
    identities, comparisons = {}, []
    for pair in args.pair:
        left, sep, right = pair.partition(",")
        if not sep:
            raise ValueError(f"--pair는 A,B 형식이어야 합니다: {pair}")
        for ref in (left, right):
            if ref not in identities:
                identities[ref] = _identity(ref, runs, reports, manifest)
        check = compare_identities(identities[left], identities[right])
        a = per_query_metrics(reports[left.split(":")[0]], left.split(":")[1])
        b = per_query_metrics(reports[right.split(":")[0]], right.split(":")[1])
        comparisons.append(
            {
                "a": left,
                "b": right,
                "identity": check,
                **paired_comparison(
                    a, b, primary, resamples=args.resamples, seed=args.seed
                ),
            }
        )
    return {
        "slice": "non_synthetic (primary)",
        "primary_slice_query_ids_sha256": next(iter(identities.values()))["protocol"][
            "primary_slice_query_ids_sha256"
        ],
        "runs": identities,
        "comparisons": comparisons,
    }


def cross_model(args):
    from evaluation import cross_model as cm
    from evaluation.data import load_evaluation_dataset
    from evaluation.json_io import read_json
    from evaluation.provenance import compare_identities

    runs, reports = _parse_runs(args.run), {}
    manifest = read_json(args.data_dir / "manifest.json")
    primary = manifest["primary_slice_query_ids"]
    columns, identities = {}, {}
    for value in args.column:
        label, sep, ref = value.partition("=")
        if not sep or label in columns:
            raise ValueError(
                f"--column은 고유한 LABEL=NAME:METHOD 형식이어야 합니다: {value}"
            )
        identities[label] = _identity(ref, runs, reports, manifest)
        name, method = ref.split(":")
        columns[label] = {
            row["query_id"]: row for row in reports[name]["methods"][method]["queries"]
        }
    first = next(iter(identities.values()))
    checks = {
        label: compare_identities(first, identity)
        for label, identity in identities.items()
    }
    dataset = load_evaluation_dataset(args.data_dir, "test")
    texts = {query.query_id: query.text for query in dataset.queries}
    pairs = [p for p in OVERLAP_PAIRS if set(p) <= set(columns)]
    rows = cm.query_table(columns, dataset.qrels, texts, primary, pairs=pairs)
    dense_runs = [name for name in ("bge_dense", "kanana_dense") if name in columns]
    summary = {
        "slice": "non_synthetic (primary)",
        "query_count": len(rows),
        "columns": {
            label: value.split("=", 1)[1]
            for label, value in zip(columns, args.column, strict=True)
        },
        "identity": checks,
        "top10_overlap_mean": {
            f"{a}|{b}": sum(r[f"top10_overlap:{a}|{b}"] for r in rows) / len(rows)
            for a, b in pairs
        },
        "top1_differs": {
            f"{a}|{b}": sum(r[f"top1_differs:{a}|{b}"] for r in rows) for a, b in pairs
        },
        "sparse_complementarity": cm.sparse_complementarity(rows, "bm25", dense_runs),
        "fusion_effect": {
            encoder: cm.fusion_effect(
                rows, "bm25", f"{encoder}_dense", f"{encoder}_fixed"
            )
            for encoder in ("bge", "kanana")
            if {f"{encoder}_dense", f"{encoder}_fixed"} <= set(columns)
        },
    }
    if len(dense_runs) == 2:
        summary["encoder_agreement"] = cm.encoder_agreement(rows, *dense_runs)
    oracles = {
        encoder: cm.method_oracle(
            rows, ("bm25", f"{encoder}_dense", f"{encoder}_fixed"), f"{encoder}_fixed"
        )
        for encoder in ("bge", "kanana")
        if {f"{encoder}_dense", f"{encoder}_fixed"} <= set(columns)
    }
    if len(oracles) == 2:
        both = set(oracles["bge"]["headroom_queries"]) & set(
            oracles["kanana"]["headroom_queries"]
        )
        summary["method_oracle_shared_headroom_queries"] = len(both)
    summary["method_oracle"] = {
        encoder: {**value, "headroom_queries": len(value["headroom_queries"])}
        for encoder, value in oracles.items()
    }
    if {"adaptive", "adaptive_oracle", "bge_fixed"} <= set(columns):
        summary["adaptive_recovery"] = cm.adaptive_recovery(
            rows,
            fixed="bge_fixed",
            adaptive="adaptive",
            oracle="adaptive_oracle",
            disagreement="top1_differs:bm25|bge_dense",
        )
    return summary, rows


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            # 원문 질의의 줄바꿈은 CSV 한 줄로 접는다. 질의는 query_id로 식별한다.
            writer.writerow(
                row
                | {
                    "text": " ".join(row["text"].split()),
                    "relevant": " ".join(row["relevant"]),
                }
            )


def main(argv=None):
    from evaluation.paired import DEFAULT_RESAMPLES, DEFAULT_SEED

    parser = CliArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "command", choices=("replay-adaptive", "compare", "cross-model")
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("artifacts/bccard-bge-m3")
    )
    parser.add_argument("--router", type=Path, default=DEFAULT_ROUTER)
    parser.add_argument("--run", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--pair", action="append", default=[], metavar="A,B")
    parser.add_argument(
        "--column", action="append", default=[], metavar="LABEL=NAME:METHOD"
    )
    parser.add_argument("--resamples", type=_positive_int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threads", type=_positive_int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, help="cross-model per-query CSV")
    args = parser.parse_args(argv)
    from evaluation.json_io import write_json

    try:
        if args.command == "replay-adaptive":
            import faiss

            from evaluation.json_io import read_json
            from evaluation.runtime_config import configure_sparse_runtime

            configure_sparse_runtime(args.threads)
            faiss.omp_set_num_threads(args.threads)
            frozen = read_json(args.router.parent / "summary.json")["router_sha256"]
            report = replay_adaptive(
                args.data_dir,
                args.cache_dir,
                args.router,
                expected_router_sha256=frozen,
            )
            write_json(args.output, report)
            printed = {
                name: report["methods"][name]["slices"]["non_synthetic"]
                for name in report["methods"]
            }
        elif args.command == "compare":
            if not args.pair:
                parser.error("compare에는 --pair가 하나 이상 필요합니다.")
            result = compare(args)
            write_json(args.output, result)
            printed = {
                f"{c['a']} -> {c['b']}": {
                    m: [v["difference"], v["ci95"]] for m, v in c["metrics"].items()
                }
                for c in result["comparisons"]
            }
        else:
            summary, rows = cross_model(args)
            write_json(args.output, summary)
            if args.rows_output:
                _write_rows(args.rows_output, rows)
            printed = {key: summary[key] for key in ("top1_differs", "method_oracle")}
    except ImportError as exc:
        parser.error(f"sparse/dense extras가 필요합니다: {exc}")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(f"Phase 5 분석 실패: {exc}")
    print(json.dumps(printed, ensure_ascii=False, indent=2), file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

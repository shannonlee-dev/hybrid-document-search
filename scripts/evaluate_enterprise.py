"""Encode BC Card vectors once, then evaluate fixed Phase 3 baselines per split."""

import json
import sys
from pathlib import Path

from evaluation.constants import DEFAULT_REPEATS, DEFAULT_THREADS, DEFAULT_WARMUP
from scripts._cli import CliArgumentParser, _positive_int

PREFERENCE_SLICES = ("all", "non_synthetic", "synthetic")


def main(argv=None):
    parser = CliArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("encode", "evaluate", "analyze"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/bccard"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("artifacts/bccard-bge-m3")
    )
    parser.add_argument(
        "--encoder",
        choices=("bge", "kanana"),
        default="bge",
        help="encode only; kanana is the Phase 5 external model at its pinned revision",
    )
    parser.add_argument("--split", choices=("validation", "test"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--warmup", type=_positive_int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=_positive_int, default=DEFAULT_REPEATS)
    parser.add_argument("--threads", type=_positive_int, default=DEFAULT_THREADS)
    args = parser.parse_args(argv)
    if args.command != "encode" and (args.split is None or args.output_dir is None):
        parser.error(f"{args.command}에는 --split과 --output-dir이 필요합니다.")
    try:
        if args.command == "encode":
            from evaluation.enterprise import encode_bccard
            from evaluation.runtime_config import configure_runtime

            runtime = configure_runtime(args.threads, cuda=True)
            record = encode_bccard(args.data_dir, args.cache_dir, encoder=args.encoder)
            print(
                json.dumps(
                    {
                        "runtime_gpu": runtime.get("gpu"),
                        "dtype": record["dtype"],
                        "max_seq_length": record["max_seq_length"],
                        "embedding_dimension": record["embedding_dimension"],
                        "gpu": record.get("gpu"),
                        "document_encoding_seconds": record[
                            "document_encoding_seconds"
                        ],
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "analyze":
            _analyze(args)
        else:
            _evaluate(args)
    except ImportError as exc:
        parser.error(f"sparse/dense extras가 필요합니다: {exc}")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(f"기업 데이터 평가 실패: {exc}")
    return 0


def _evaluate(args):
    from evaluation.benchmark import check_output_directory
    from evaluation.json_io import write_json
    from evaluation.runtime_config import configure_sparse_runtime

    check_output_directory(args.output_dir)
    import faiss

    from data.bccard import load_query_metadata
    from evaluation.enterprise import (
        preference_analysis,
        run_enterprise_baselines,
        slice_definitions,
        write_summary,
    )

    runtime = configure_sparse_runtime(args.threads)
    faiss.omp_set_num_threads(args.threads)
    report = run_enterprise_baselines(
        args.data_dir,
        args.cache_dir,
        args.split,
        warmup=args.warmup,
        repeats=args.repeats,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    report["environment"]["effective_runtime"] = {
        "requested_threads": args.threads,
        "faiss_omp_max_threads": faiss.omp_get_max_threads(),
        **runtime,
    }
    write_json(args.output_dir / "report.json", report)
    write_summary(report, args.output_dir / "summary.json")
    printed = {
        name: {
            "latency_ms": [data["latency"]["mean_ms"], data["latency"]["p95_ms"]],
            **{key: data["slices"][key] for key in ("all", "non_synthetic")},
        }
        for name, data in report["methods"].items()
    }
    if args.split != "test":
        slices = slice_definitions(load_query_metadata(args.data_dir, args.split))
        selected = {
            name: ids
            for name, ids in slices.items()
            if name in PREFERENCE_SLICES or name.startswith("provenance=")
        }
        analysis = {
            metric: preference_analysis(report, selected, metric)
            for metric in ("ndcg@10", "graded_ndcg@10")
        }
        write_json(args.output_dir / "preference.json", analysis)
        printed["preference_ndcg@10"] = analysis["ndcg@10"]["slices"]["all"]
    print(json.dumps(printed, ensure_ascii=False, indent=2))


def _analyze(args):
    """Write the validation-only handoff for Issue #24 from a saved report."""
    from data.bccard import load_query_metadata
    from evaluation.data import load_evaluation_dataset
    from evaluation.enterprise import validation_handoff
    from evaluation.json_io import read_json, write_json

    if args.split == "test":
        raise ValueError("handoff 분석은 validation에서만 계산합니다.")
    report = read_json(args.output_dir / "report.json")
    dataset = load_evaluation_dataset(args.data_dir, args.split)
    handoff = validation_handoff(
        report,
        load_query_metadata(args.data_dir, args.split),
        dataset.qrels,
        dataset.documents,
    )
    write_json(args.output_dir / "handoff.json", handoff)
    print(json.dumps({k: handoff[k] for k in ("wins", "oracle")}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())

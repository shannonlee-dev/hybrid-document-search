"""Run dataset-separated final evaluation from the pinned team vector cache."""

import json
import sys
from pathlib import Path

from evaluation.benchmark import check_output_directory, write_report
from evaluation.constants import DEFAULT_REPEATS, DEFAULT_THREADS, DEFAULT_WARMUP
from evaluation.final import DEFAULT_MANIFEST, FINAL_METHODS, run_final_benchmark
from scripts._cli import CliArgumentParser, _positive_int


def main(argv=None):
    parser = CliArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("artifacts/score-fusion-cache")
    )
    parser.add_argument("--cache-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("miracl10k", "strategyqa"),
        default=["miracl10k", "strategyqa"],
    )
    parser.add_argument(
        "--methods", nargs="+", choices=FINAL_METHODS, default=list(FINAL_METHODS)
    )
    parser.add_argument("--warmup", type=_positive_int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=_positive_int, default=DEFAULT_REPEATS)
    parser.add_argument("--threads", type=_positive_int, default=DEFAULT_THREADS)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("datasets는 중복될 수 없습니다.")
    try:
        check_output_directory(args.output_dir)
        import faiss
        from threadpoolctl import threadpool_info, threadpool_limits

        faiss.omp_set_num_threads(args.threads)
        reports = {}
        with threadpool_limits(limits=args.threads):
            for name in args.datasets:
                report = run_final_benchmark(
                    args.cache_dir / name,
                    manifest=args.cache_manifest,
                    methods=tuple(args.methods),
                    warmup=args.warmup,
                    repeats=args.repeats,
                    progress=lambda message: print(
                        message, file=sys.stderr, flush=True
                    ),
                )
                report["environment"]["effective_runtime"] = {
                    "requested_threads": args.threads,
                    "faiss_omp_max_threads": faiss.omp_get_max_threads(),
                    "threadpools": threadpool_info(),
                }
                reports[name] = report
        for name, report in reports.items():
            write_report(report, args.output_dir / name)
    except ImportError as exc:
        parser.error(f"score fusion 구현과 sparse/dense extras가 필요합니다: {exc}")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.error(f"최종 평가 실패: {exc}")
    print(
        json.dumps(
            {
                name: {
                    method: result["metrics"]
                    for method, result in report["methods"].items()
                }
                for name, report in reports.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

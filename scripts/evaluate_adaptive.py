"""Prepare vectors, fit/freeze the #24 router, then evaluate the frozen test once."""

import json
from pathlib import Path

from scripts._cli import CliArgumentParser, _positive_int


def main(argv=None):
    parser = CliArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("encode", "fit", "evaluate"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/bccard"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("artifacts/bccard-adaptive-vectors")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/bccard-adaptive")
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        help="fit-only JSON weight candidates, registered before train/validation selection",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        help="reusable train/validation table checkpoints for fit",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=_positive_int, default=2)
    parser.add_argument("--warmup", type=_positive_int, default=1)
    parser.add_argument("--repeats", type=_positive_int, default=5)
    args = parser.parse_args(argv)
    if (
        args.candidates is not None or args.table_dir is not None
    ) and args.command != "fit":
        parser.error(
            "--candidates / --table-dir are only accepted for fit; evaluation uses frozen weights"
        )
    try:
        from evaluation.runtime_config import (
            configure_runtime,
            configure_sparse_runtime,
        )

        if args.command == "encode":
            from evaluation.enterprise import encode_bccard

            configure_runtime(args.threads, cuda=args.device.startswith("cuda"))
            encode_bccard(
                args.data_dir,
                args.cache_dir,
                splits=("train", "validation", "test"),
                device=args.device,
            )
        else:
            import faiss

            from evaluation.adaptive import evaluate_frozen, fit_router

            configure_sparse_runtime(args.threads)
            faiss.omp_set_num_threads(args.threads)
            if args.command == "fit":
                report = fit_router(
                    args.data_dir,
                    args.cache_dir,
                    args.output_dir,
                    progress=print,
                    table_dir=args.table_dir,
                    **({"candidates_path": args.candidates} if args.candidates else {}),
                )
                print(json.dumps(report["selected"], ensure_ascii=False, indent=2))
            else:
                report = evaluate_frozen(
                    args.data_dir,
                    args.cache_dir,
                    args.output_dir,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    progress=print,
                )
                print(
                    json.dumps(
                        {
                            name: data["slices"]["non_synthetic"]
                            for name, data in report["methods"].items()
                        },
                        indent=2,
                    )
                )
    except (ImportError, OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Select on validation, replay frozen test, or evaluate a frozen query encoder."""

import json
from pathlib import Path

from scripts._cli import CliArgumentParser, _positive_int


def _encoders(values):
    result = {}
    for value in values:
        name, separator, directory = value.partition("=")
        if (
            not separator
            or not name
            or not directory
            or name == "baseline"
            or name in result
        ):
            raise ValueError(
                "--encoder requires unique NAME=DIRECTORY, excluding baseline"
            )
        result[name] = Path(directory)
    return result


def main(argv=None):
    parser = CliArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "baseline",
            "select",
            "evaluate",
            "checkpoint",
            "verify",
            "export",
        ),
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--score-dir",
        type=Path,
        help="raw Dense/Sparse scores and corpus/query identities",
    )
    parser.add_argument(
        "--encoder", action="append", default=[], metavar="NAME=DIRECTORY"
    )
    parser.add_argument("--model", type=Path, help="local SentenceTransformer snapshot")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument(
        "--baseline-vectors-dir",
        type=Path,
        help="verified BF16 baseline vectors and metadata",
    )
    parser.add_argument(
        "--test-source",
        type=Path,
        help="checkpoint evaluation output with BF16 baseline Test vectors",
    )
    parser.add_argument(
        "--shared-alpha",
        type=float,
        help="predeclared common Dense alpha for both encoders",
    )
    parser.add_argument("--run-dir", type=Path, help="source run for compact export")
    parser.add_argument("--canonical-summary", type=Path)
    parser.add_argument("--layers", type=_positive_int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=_positive_int, default=4)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            from evaluation.query_encoder_report import verify_results

            result = verify_results(args.data_dir, args.output_dir)
        elif args.command == "export":
            from evaluation.query_encoder_report import compact_export

            if args.run_dir is None:
                raise ValueError("export requires --run-dir")
            result = compact_export(
                args.data_dir,
                args.run_dir,
                args.output_dir,
                canonical_summary=args.canonical_summary,
            )
        else:
            if args.cache_dir is None:
                raise ValueError(
                    "prepare/select/evaluate/checkpoint require --cache-dir"
                )
            result = _execute(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ImportError, OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
    return 0


def _execute(args):
    from evaluation import query_encoder
    from evaluation.runtime_config import configure_runtime, configure_sparse_runtime

    if args.command == "baseline":
        from evaluation.query_encoder_training import prepare_baseline_vectors

        if args.model is None:
            raise ValueError("baseline encoding requires --model")
        configure_runtime(args.threads, cuda=args.device.startswith("cuda"))
        return prepare_baseline_vectors(
            args.data_dir,
            args.cache_dir,
            args.model,
            args.output_dir,
            test_source=args.test_source,
            device=args.device,
        )
    if args.command in ("prepare", "select", "evaluate"):
        configure_sparse_runtime(args.threads)
        if args.command == "prepare":
            return query_encoder.prepare_scores(
                args.data_dir, args.cache_dir, args.output_dir
            )
        else:
            if args.score_dir is None:
                raise ValueError("select/evaluate require --score-dir")
            action = (
                query_encoder.select_cached_encoders
                if args.command == "select"
                else query_encoder.evaluate_cached_encoders
            )
            return action(
                args.data_dir,
                args.cache_dir,
                args.score_dir,
                _encoders(args.encoder),
                args.output_dir,
                baseline_vectors_dir=args.baseline_vectors_dir,
                **(
                    {"model_path": args.model, "shared_alpha": args.shared_alpha}
                    if args.command == "select"
                    else {}
                ),
            )
    if args.model is None or args.checkpoint_dir is None:
        raise ValueError("checkpoint requires --model and --checkpoint-dir")
    configure_runtime(args.threads, cuda=args.device.startswith("cuda"))
    from evaluation.query_encoder_training import evaluate_checkpoint

    return evaluate_checkpoint(
        args.data_dir,
        args.cache_dir,
        args.model,
        args.checkpoint_dir,
        args.output_dir,
        layers=args.layers,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())

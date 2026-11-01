"""Train query-only final transformer blocks against fixed corpus vectors."""

import json
from pathlib import Path

from scripts._cli import CliArgumentParser, _positive_int


def main(argv=None):
    parser = CliArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, required=True, help="local SentenceTransformer snapshot"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=_positive_int, default=256)
    parser.add_argument("--layers", type=_positive_int, default=4)
    parser.add_argument("--batch-size", type=_positive_int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=_positive_int, default=2)
    args = parser.parse_args(argv)
    try:
        from evaluation.query_encoder_training import train_query_encoder
        from evaluation.runtime_config import configure_runtime

        configure_runtime(args.threads, cuda=args.device.startswith("cuda"))
        result = train_query_encoder(
            args.data_dir,
            args.cache_dir,
            args.model,
            args.output_dir,
            seed=args.seed,
            steps=args.steps,
            layers=args.layers,
            batch_size=args.batch_size,
            device=args.device,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ImportError, OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

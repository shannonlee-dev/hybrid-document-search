"""Ko-miracl 공통 subset 준비 명령입니다."""

import argparse
import json
import re
from pathlib import Path

from data.preparation import PINNED_REVISION, download_sources, prepare_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ko-miracl 공통 subset을 준비합니다.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/ko-miracl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument("--corpus-size", type=int, default=10000)
    parser.add_argument("--train-queries", type=int, default=100)
    parser.add_argument("--dev-queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--download", action="store_true", help="고정 revision의 원본을 다운로드합니다."
    )
    args = parser.parse_args(argv)
    for name in ("corpus_size", "train_queries", "dev_queries"):
        if getattr(args, name) <= 0:
            parser.error(f"{name}는 양의 정수여야 합니다.")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision은 고정된 40자리 commit SHA여야 합니다.")
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        parser.error("output_dir은 비어 있는 디렉터리여야 합니다.")
    try:
        raw_dir = (
            download_sources(args.raw_dir, args.revision)
            if args.download
            else args.raw_dir
        )
        manifest = prepare_dataset(
            raw_dir,
            args.output_dir,
            corpus_size=args.corpus_size,
            train_queries=args.train_queries,
            dev_queries=args.dev_queries,
            seed=args.seed,
            revision=args.revision,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest["output_counts"], ensure_ascii=False, indent=2))
    print(f"준비 완료: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

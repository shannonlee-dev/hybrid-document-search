"""BC Card 기업 특화 검색 데이터 준비 명령입니다."""

import argparse
import json
from pathlib import Path

from data.bccard import (
    PAIR_ID,
    PAIR_REVISION,
    TRIPLET_ID,
    TRIPLET_REVISION,
    download_sources,
    prepare_bccard,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BC Card Pair/Triplet을 공통 평가 형식으로 준비합니다."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/bccard"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/bccard")
    )
    parser.add_argument(
        "--download", action="store_true", help="고정 revision의 원본을 다운로드합니다."
    )
    args = parser.parse_args(argv)
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        parser.error("output_dir은 비어 있는 디렉터리여야 합니다.")
    try:
        if args.download:
            directories = download_sources(args.raw_dir)
        else:
            directories = {
                "pair": args.raw_dir / PAIR_ID.split("/")[1] / PAIR_REVISION,
                "triplet": args.raw_dir / TRIPLET_ID.split("/")[1] / TRIPLET_REVISION,
            }
        manifest = prepare_bccard(
            directories["pair"], directories["triplet"], args.output_dir
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest["output_counts"], ensure_ascii=False, indent=2))
    print(f"준비 완료: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

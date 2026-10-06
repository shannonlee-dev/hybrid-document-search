"""준비된 corpus에서 TF-IDF 검색을 실행하는 CLI입니다."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from data.loader import load_prepared_documents


def positive_int(value: str) -> int:
    """잘못된 Top-K를 인덱싱 전에 거부합니다."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("top_k는 양의 정수여야 합니다.") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("top_k는 양의 정수여야 합니다.")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="준비된 corpus를 TF-IDF로 검색합니다.")
    parser.add_argument(
        "--corpus", type=Path, default=Path("data/processed/corpus.jsonl")
    )
    parser.add_argument("--query", required=True, help="검색할 질의")
    parser.add_argument(
        "--top-k", type=positive_int, default=5, help="최대 결과 수 (기본 5)"
    )
    args = parser.parse_args(argv)

    try:
        documents = load_prepared_documents(args.corpus)
        # --help와 입력 검증은 Sparse extra 없이도 사용할 수 있습니다.
        from retrievers.tfidf import TfidfRetriever

        results = TfidfRetriever(documents).search(args.query, args.top_k)
    except ModuleNotFoundError as exc:
        if exc.name == "sklearn" or (exc.name and exc.name.startswith("sklearn.")):
            parser.error(
                "Sparse 의존성이 필요합니다. uv sync --extra sparse로 설치하고 "
                "uv run --extra sparse python -m scripts.search로 실행하세요."
            )
        raise
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            [asdict(result) for result in results],
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

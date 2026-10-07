"""선택한 검색 방식의 의존성만 로드하고 공통 SearchResult JSON을 출력한다."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from scripts._cli import (
    DEFAULT_CORPUS_PATH,
    CliArgumentParser,
    _dependency_hint,
    _positive_int,
    _rebuild_hint,
)

if TYPE_CHECKING:
    from retrievers.base import SearchResult

_DEFAULT_TOP_K = 5


def main(argv: list[str] | None = None) -> int:
    args, unknown = _build_parser().parse_known_args(argv)
    if unknown:
        args.parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    _print_search_results(args.run(args))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="python -m scripts.search",
        description="TF-IDF 또는 Dense 방식으로 검색하고 공통 SearchResult JSON을 출력합니다.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(
        dest="mode", required=True, title="검색 방식", metavar="{tfidf,dense}"
    )
    tfidf = commands.add_parser(
        "tfidf",
        help="준비된 corpus를 TF-IDF로 검색",
        description="준비된 JSONL corpus를 읽어 TF-IDF로 검색합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog=(
            "예시: 준비된 corpus 검색\n"
            f"  python -m scripts.search tfidf --corpus {DEFAULT_CORPUS_PATH} "
            "--query '제주' --top-k 3"
        ),
    )
    tfidf.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=f"준비된 JSONL corpus (기본: {DEFAULT_CORPUS_PATH})",
    )
    dense = commands.add_parser(
        "dense",
        help="저장된 Dense 인덱스 검색",
        description="빌드해 둔 Dense 인덱스에서 검색합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog=(
            "예시: 저장된 인덱스 검색\n"
            "  python -m scripts.search dense --index indexes/dense-sample "
            "--query '고양이는 어떤 소리로 우나요?' --top-k 3 --device cpu\n\n"
            "인덱스가 없으면 먼저 빌드해 주세요.\n"
            "  python -m scripts.build_index --corpus tests/dense/fixtures/dense_corpus.jsonl "
            "--index indexes/dense-sample --device cpu\n\n"
            "검색에는 빌드할 때 사용한 모델의 로컬 경로 또는 캐시가 필요합니다."
        ),
    )
    dense.add_argument(
        "--index", type=Path, required=True, help="빌드해 둔 인덱스 디렉터리"
    )
    dense.add_argument("--device", help="cpu, cuda 또는 cuda:0 (생략하면 자동 선택)")
    for child, run in ((tfidf, _run_tfidf), (dense, _run_dense)):
        child.add_argument("--query", required=True, help="검색할 문장 또는 키워드")
        child.add_argument(
            "--top-k",
            type=_positive_int,
            default=_DEFAULT_TOP_K,
            help=f"반환할 최대 문서 수 (기본: {_DEFAULT_TOP_K})",
        )
        child.set_defaults(parser=child, run=run)
    return parser


def _run_tfidf(args: argparse.Namespace) -> list["SearchResult"]:
    try:
        from data.loader import load_prepared_documents

        documents = load_prepared_documents(args.corpus)
        from retrievers.tfidf import TfidfRetriever

        return TfidfRetriever(documents).search(args.query, args.top_k)
    except ModuleNotFoundError as exc:
        if exc.name == "sklearn" or (exc.name and exc.name.startswith("sklearn.")):
            args.parser.error(
                "Sparse 의존성이 필요합니다. uv sync --extra sparse로 설치하고 "
                "다음 명령으로 실행하세요:\n"
                "  uv run --extra sparse python -m scripts.search tfidf --query '제주'"
            )
        raise
    except PermissionError as exc:
        args.parser.error(
            f"corpus를 읽을 권한이 없습니다: {args.corpus}\n"
            f"파일·디렉터리의 접근 권한을 확인해 주세요.\n상세: {exc}"
        )
    except (OSError, ValueError) as exc:
        args.parser.error(
            f"corpus를 읽거나 TF-IDF 검색을 실행할 수 없습니다: {args.corpus}\n"
            f"--corpus 경로와 JSONL 형식을 확인해 주세요.\n상세: {exc}"
        )


def _run_dense(args: argparse.Namespace) -> list["SearchResult"]:
    # 인덱스 복원 실패와 질의 임베딩 실패는 복구 방법이 달라 단계 경계를 유지한다.
    loading = True
    try:
        missing = [
            name
            for name in ("metadata.json", "index.faiss")
            if not (args.index / name).is_file()
        ]
        if missing:
            args.parser.error(
                f"검색에 필요한 인덱스 파일이 없습니다: {args.index}\n"
                f"빠진 파일: {', '.join(missing)}\n"
                f"--index가 빌드 결과 디렉터리인지 확인해 주세요.\n{_rebuild_hint(args.index)}"
            )
        if args.device is not None and not args.device.strip():
            args.parser.error(
                "--device에 cpu, cuda 또는 cuda:0을 입력하거나 옵션을 생략해 주세요."
            )
        from retrievers.dense import DenseRetriever

        retriever = DenseRetriever.load(args.index, device=args.device)
        loading = False
        return retriever.search(args.query, args.top_k)
    except ImportError as exc:
        args.parser.error(_dependency_hint(exc))
    except PermissionError as exc:
        args.parser.error(
            f"파일을 읽을 권한이 없습니다. 인덱스와 모델 경로의 접근 권한을 확인해 주세요.\n상세: {exc}"
        )
    except (ValueError, OSError, RuntimeError) as exc:
        if loading:
            args.parser.error(
                f"인덱스를 읽을 수 없습니다: {args.index}\n"
                "파일이 손상되었거나 현재 버전과 맞지 않을 수 있습니다. "
                "올바른 --index 경로인지 확인하고, 필요하면 다시 빌드해 주세요.\n"
                f"{_rebuild_hint(args.index)}\n상세: {exc}"
            )
        args.parser.error(
            "검색을 실행할 수 없습니다. 검색어와 빌드할 때 사용한 모델 경로·캐시를 확인해 주세요.\n"
            "모델이 캐시에 없으면 다운로드 연결이 필요합니다. GPU 실행에 문제가 있으면 "
            "--device cpu로 다시 실행해 주세요.\n"
            f"상세: {exc}"
        )


def _print_search_results(results: list["SearchResult"]) -> None:
    print(
        json.dumps(
            [asdict(result) for result in results],
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

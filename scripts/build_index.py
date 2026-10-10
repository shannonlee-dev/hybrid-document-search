"""Build a Dense index from prepared documents and report stage-specific failures."""

import argparse
import json
import shlex
import sys
from pathlib import Path
from time import perf_counter

from scripts._cli import CliArgumentParser, _dependency_hint, _positive_int


def _build_index(argv: list[str] | None = None) -> None:
    """Embed and save a corpus, reporting recovery guidance for failed stages."""
    parser = CliArgumentParser(
        prog="python -m scripts.build_index",
        description="준비된 corpus의 문서를 임베딩하고 Dense 인덱스를 저장합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시: 작은 샘플로 빌드\n"
            "  python -m scripts.build_index --corpus tests/dense/fixtures/dense_corpus.jsonl "
            "--index indexes/dense-sample --device cpu\n\n"
            "빌드한 인덱스 검색\n"
            "  python -m scripts.search dense --index indexes/dense-sample "
            "--query '고양이는 어떤 소리로 우나요?' --top-k 3"
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="입력 JSONL 파일 경로 (각 문서에 document_id와 text 필요)",
    )
    parser.add_argument(
        "--index", type=Path, required=True, help="인덱스를 저장할 디렉터리"
    )
    parser.add_argument("--device", help="cpu, cuda 또는 cuda:0 (생략하면 자동 선택)")
    parser.add_argument(
        "--model", help="모델 이름 또는 로컬 경로 (기본: dense_models.toml 설정)"
    )
    parser.add_argument(
        "--revision",
        help="모델의 고정 40자리 commit SHA (등록된 후보는 TOML의 SHA 사용)",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        help="한 번에 임베딩할 문서 수 (기본: 32)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="벡터 정규화를 끄고 원시 내적 점수 사용 (기본: cosine similarity)",
    )
    parser.add_argument(
        "--query-prefix", help="질의 앞에 붙일 문자열 (생략하면 모델별 기본값)"
    )
    parser.add_argument(
        "--passage-prefix", help="문서 앞에 붙일 문자열 (생략하면 모델별 기본값)"
    )
    parser.add_argument(
        "--timings", action="store_true", help="빌드 단계별 소요 시간(초)을 JSON에 포함"
    )
    args = parser.parse_args(argv)
    started = perf_counter()
    stage = "corpus 읽기"
    hint = (
        "--corpus에 읽을 수 있는 UTF-8 JSONL 파일을 지정해 주세요.\n"
        "각 문서에는 고유한 document_id와 비어 있지 않은 text가 필요하며, title은 선택입니다."
    )
    try:
        if not args.corpus.is_file():
            parser.error(
                f"corpus 경로가 파일이 아니거나 존재하지 않습니다: {args.corpus}\n{hint}"
            )
        if args.model is not None and not args.model.strip():
            parser.error("--model에 모델 이름 또는 로컬 경로를 입력해 주세요.")
        from data.loader import load_prepared_documents
        from retrievers.dense import DenseConfig, DenseRetriever

        documents = load_prepared_documents(args.corpus)
        if not documents:
            parser.error(f"corpus에 문서가 없습니다: {args.corpus}\n{hint}")
        stage = "임베딩 설정 확인"
        hint = "--model과 --device에 비어 있지 않은 값을 지정해 주세요."
        defaults = DenseConfig()
        config = DenseConfig(
            model_name=args.model if args.model is not None else defaults.model_name,
            revision=args.revision,
            device=args.device,
            batch_size=args.batch_size
            if args.batch_size is not None
            else defaults.batch_size,
            normalize_embeddings=not args.no_normalize,
            query_prefix=args.query_prefix,
            passage_prefix=args.passage_prefix,
        )
        stage = "문서 임베딩 및 인덱스 생성"
        hint = (
            "--model의 이름 또는 로컬 경로와 모델 다운로드 연결·캐시를 확인해 주세요.\n"
            "GPU 실행에 문제가 있으면 --device cpu로 실행하고, 메모리가 부족하면 "
            "--batch-size를 낮춰 주세요."
        )
        retriever = DenseRetriever.build(documents, config)
        stage = "인덱스 저장"
        hint = "--index 경로에 쓸 수 있는지, 디스크 공간이 충분한지 확인해 주세요."
        retriever.save(args.index)
        summary = {"index": str(args.index), "documents": len(documents)}
        if args.timings:
            summary["timings"] = {
                **retriever.build_timings,
                "total_preparation_seconds": perf_counter() - started,
            }
        print(json.dumps(summary))
        search_command = shlex.join(
            [
                "python",
                "-m",
                "scripts.search",
                "dense",
                "--index",
                str(args.index),
                "--query",
                "검색어",
            ]
        )
        print(
            f"인덱스 저장 완료: {args.index} (문서 {len(documents)}개)\n"
            f"검색어를 바꿔 검색해 보세요:\n  {search_command}",
            file=sys.stderr,
        )
    except ImportError as exc:
        parser.error(_dependency_hint(exc))
    except PermissionError as exc:
        parser.error(
            f"{stage} 중 접근 권한이 부족합니다. 파일·디렉터리 권한을 확인해 주세요.\n상세: {exc}"
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(f"{stage}에 실패했습니다.\n{hint}\n상세: {exc}")


def main(argv: list[str] | None = None) -> int:
    """Run the build CLI and report user cancellation without a traceback."""
    try:
        _build_index(argv)
    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

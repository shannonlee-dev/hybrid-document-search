"""Evaluate retrieval quality and latency on shared prepared data."""

import argparse
import json
import sys
from pathlib import Path

from evaluation.benchmark import (
    METHODS,
    check_output_directory,
    run_benchmark,
    write_report,
)
from evaluation.constants import CUDA_DEVICE, DEFAULT_REPEATS, DEFAULT_WARMUP
from scripts._cli import CliArgumentParser, _positive_int


def _build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="python -m scripts.evaluate",
        description="공통 Top-10 검색 품질 및 warm-up 이후 latency 평균/P95를 평가합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog=(
            "Sparse 예시:\n"
            "  python -m scripts.evaluate --methods tfidf bm25 --output-dir artifacts/benchmarks/sparse-dev\n\n"
            "네 가지 방식 예시(공통 corpus로 만든 Dense 인덱스 필요):\n"
            "  python -m scripts.evaluate --methods tfidf bm25 dense hybrid --index indexes/dense-ko-miracl --output-dir artifacts/benchmarks/all-dev"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="manifest와 준비된 JSONL 디렉터리",
    )
    parser.add_argument(
        "--split", choices=("train", "dev"), default="dev", help="평가 split (기본 dev)"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=["tfidf", "bm25"],
        help="평가할 방식 (기본 tfidf bm25)",
    )
    parser.add_argument("--index", type=Path, help="공통 corpus로 준비한 Dense 인덱스")
    parser.add_argument("--threads", type=_positive_int, help="실효 library threads")
    parser.add_argument("--device", help="Dense 실행 장치 (예: cpu)")
    parser.add_argument(
        "--warmup",
        type=_positive_int,
        default=DEFAULT_WARMUP,
        help="측정 전 전체 query warm-up pass 수 (기본 1)",
    )
    parser.add_argument(
        "--repeats",
        type=_positive_int,
        default=DEFAULT_REPEATS,
        help="전체 query 반복 측정 pass 수 (기본 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="새 결과 디렉터리; 기존 결과는 덮어쓰지 않음",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate retrieval quality and latency and save reports to a new directory."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    uses_dense = "dense" in args.methods or "hybrid" in args.methods
    strict_runtime = (
        uses_dense and args.threads is not None and args.device == CUDA_DEVICE
    )
    try:
        runtime = None
        if args.threads is not None:
            if uses_dense:
                # 관련 수치 라이브러리까지 로드한 뒤 실효 스레드 수를 제한한다.
                import sentence_transformers  # noqa: F401

                from evaluation.runtime_config import configure_runtime

                runtime = configure_runtime(
                    args.threads, cuda=args.device == CUDA_DEVICE
                )
            else:
                from evaluation.runtime_config import configure_sparse_runtime

                runtime = configure_sparse_runtime(args.threads)
        check_output_directory(args.output_dir)
        report = run_benchmark(
            args.data_dir,
            split=args.split,
            methods=tuple(args.methods),
            dense_index=args.index,
            device=args.device,
            warmup=args.warmup,
            repeats=args.repeats,
            strict_runtime=strict_runtime,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        if runtime is not None:
            report["environment"]["effective_runtime"] = runtime
        write_report(report, args.output_dir)
    except ImportError as exc:
        parser.error(
            "선택한 방식의 검색 의존성을 설치해 주세요.\n"
            "  uv sync --locked --extra sparse --extra dense\n"
            "Sparse만 평가할 때는 --extra sparse만 필요합니다.\n"
            f"상세: {exc}"
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(
            f"평가를 완료할 수 없습니다. 준비 데이터, manifest와 인덱스/출력 경로를 확인해 주세요.\n상세: {exc}"
        )
    print(
        json.dumps(
            {
                method: {"metrics": result["metrics"], "latency": result["latency"]}
                for method, result in report["methods"].items()
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

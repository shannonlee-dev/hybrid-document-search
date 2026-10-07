"""CLI 입력 검증과 한국어 오류 안내를 공유한다."""

import argparse
import shlex
import sys
from pathlib import Path

DEFAULT_CORPUS_PATH = Path("data/processed/corpus.jsonl")
_POSITIVE_INT_ERROR = "1 이상의 정수를 입력해 주세요."
_ARGUMENT_ERROR_TRANSLATIONS = (
    ("the following arguments are required: ", "필수 옵션을 입력해 주세요: "),
    ("unrecognized arguments: ", "지원하지 않는 인수입니다: "),
)


class CliArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        """argparse의 입력 오류를 번역하고 실패한 명령의 도움말로 안내한다."""
        if message.startswith("argument ") and message.endswith(
            ": expected one argument"
        ):
            option = message.removeprefix("argument ").removesuffix(
                ": expected one argument"
            )
            message = f"{option} 옵션 뒤에 값을 입력해 주세요."
        elif message.startswith("argument ") and ": invalid choice: " in message:
            message = (
                message.removeprefix("argument ")
                .replace(": invalid choice: ", ": 지원하지 않는 값입니다: ", 1)
                .replace("(choose from ", "(선택 가능한 값: ", 1)
            )
        for prefix, replacement in _ARGUMENT_ERROR_TRANSLATIONS:
            if message.startswith(prefix):
                message = replacement + message.removeprefix(prefix)
                break
        self.print_usage(sys.stderr)
        self.exit(
            2,
            f"\n오류: {message}\n\n옵션 설명과 실행 예시를 확인하려면:\n"
            f"  {self.prog} --help\n",
        )


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(_POSITIVE_INT_ERROR) from exc
    if number <= 0:
        raise argparse.ArgumentTypeError(_POSITIVE_INT_ERROR)
    return number


def _dependency_hint(exc: ImportError) -> str:
    if exc.name in {"data.loader", "data.preprocess"}:
        return (
            "공통 데이터 코드가 준비되지 않아 실행할 수 없습니다.\n"
            "Dataset/Sparse의 공통 데이터 코드가 병합된 버전에서 실행해 주세요.\n"
            f"상세: {exc}"
        )
    return (
        "Dense 실행에 필요한 라이브러리를 불러올 수 없습니다.\n"
        "저장소 루트에서 의존성을 설치한 뒤 uv run --locked --extra dense로 실행해 주세요.\n"
        "  uv sync --locked --extra dense\n"
        f"상세: {exc}"
    )


def _rebuild_hint(index: Path) -> str:
    command = shlex.join(
        [
            "python",
            "-m",
            "scripts.build_index",
            "--corpus",
            str(DEFAULT_CORPUS_PATH),
            "--index",
            str(index),
        ]
    )
    return (
        "먼저 인덱스를 빌드해 주세요. --corpus는 실제 준비된 JSONL 파일로 바꿔 주세요.\n"
        f"  {command}"
    )

"""모델과 데이터를 다운로드하지 않고 CLI 오류 안내와 파일 생성 여부를 검증한다."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


def test_subcommand_help_without_dependencies():
    result = subprocess.run(
        [sys.executable, "-S", "-m", "scripts.search", "dense", "--help"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert result.stderr == ""
    assert "--index" in result.stdout and "--device" in result.stdout
    assert "python -m scripts.build_index" in result.stdout
    assert "python -m scripts.search dense" in result.stdout


@pytest.mark.parametrize("option", ["--corpus", "--model"])
def test_unknown_option_uses_selected_help(option):
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "scripts.search",
            "dense",
            "--query",
            "제주",
            option,
            "unused",
            "--index",
            "missing-index",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "usage: python -m scripts.search dense" in result.stderr
    assert "python -m scripts.search dense --help" in result.stderr
    assert f"지원하지 않는 인수입니다: {option}" in result.stderr
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


def test_empty_query_invalid_top_k_is_rejected_before_loading(capsys):
    from scripts.search import main

    with pytest.raises(SystemExit) as exc:
        main(["dense", "--query", "", "--top-k", "0", "--index", "missing-index"])
    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "1 이상의 정수" in captured.err
    assert "python -m scripts.search dense --help" in captured.err
    assert captured.out == ""


def _run_cli(module, args, cwd):
    root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=cwd,
        env={
            **os.environ,
            "PYTHONPATH": str(root),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize(
    ("module", "prefix"), [("scripts.build_index", []), ("scripts.search", ["dense"])]
)
def test_help_works_without_loading_dense_dependencies(module, prefix, tmp_path):
    result = _run_cli(module, [*prefix, "--help"], tmp_path)
    assert result.returncode == 0
    assert "예시" in result.stdout
    assert "--index" in result.stdout
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("module", "args", "detail"),
    [
        ("scripts.build_index", [], "--corpus"),
        ("scripts.search", ["dense"], "--query"),
        ("scripts.search", ["dense", "--query"], "값을 입력"),
        (
            "scripts.build_index",
            ["--corpus", "missing.jsonl", "--index", "index", "--batch-size", "0"],
            "1 이상의 정수",
        ),
        (
            "scripts.search",
            ["dense", "--index", "index", "--query", "고양이", "--top-k", "-1"],
            "1 이상의 정수",
        ),
        (
            "scripts.search",
            ["dense", "--index", "index", "--query", "고양이", "--top-k", "abc"],
            "1 이상의 정수",
        ),
        (
            "scripts.search",
            ["dense", "--index", "index", "--query", "고양이", "--model", "unused"],
            "--model",
        ),
    ],
)
def test_invalid_inputs_show_help_and_do_not_create_artifacts(
    module, args, detail, tmp_path
):
    result = _run_cli(module, args, tmp_path)
    assert result.returncode == 2
    assert detail in result.stderr
    command = f"python -m {module}" + (" dense" if module == "scripts.search" else "")
    assert f"{command} --help" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("files", "missing"),
    [
        ([], "metadata.json, index.faiss"),
        (["metadata.json"], "index.faiss"),
        (["index.faiss"], "metadata.json"),
    ],
)
def test_missing_index_shows_a_shell_safe_build_command(files, missing, tmp_path):
    index = tmp_path / "index with spaces $(echo unsafe)"
    index.mkdir()
    for name in files:
        (index / name).write_text("placeholder", encoding="utf-8")
    result = _run_cli(
        "scripts.search",
        ["dense", "--index", str(index), "--query", "고양이"],
        tmp_path,
    )
    assert result.returncode == 2
    assert f"빠진 파일: {missing}" in result.stderr
    command = next(
        line.strip()
        for line in result.stderr.splitlines()
        if line.strip().startswith("python -m scripts.build_index ")
    )
    assert shlex.split(command) == [
        "python",
        "-m",
        "scripts.build_index",
        "--corpus",
        "data/processed/corpus.jsonl",
        "--index",
        str(index),
    ]
    assert "Traceback" not in result.stderr
    assert result.stdout == ""
    assert sorted(path.name for path in index.iterdir()) == sorted(files)


@pytest.mark.parametrize("directory", [False, True])
def test_invalid_corpus_path_shows_how_to_select_a_prepared_file(directory, tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    if directory:
        corpus.mkdir()
    result = _run_cli(
        "scripts.build_index",
        ["--corpus", str(corpus), "--index", str(tmp_path / "index")],
        tmp_path,
    )
    assert result.returncode == 2
    assert "corpus" in result.stderr
    assert "JSONL 파일" in result.stderr
    assert "--corpus" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "index").exists()

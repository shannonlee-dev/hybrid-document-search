"""오프라인 fixture로 준비 corpus 검색 CLI를 검증합니다."""

import builtins
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.search import main

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ko_miracl_prepared_corpus.jsonl"


@pytest.fixture
def sparse_extra():
    pytest.importorskip("sklearn")


def test_root_help_lists_search_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert "tfidf" in captured.out and "dense" in captured.out
    assert captured.err == ""


@pytest.mark.parametrize("command", [[], ["tfidf"]])
def test_subcommand_help_without_dependencies(command):
    result = subprocess.run(
        [sys.executable, "-S", "-m", "scripts.search", *command, "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert result.stderr == ""
    if command == ["tfidf"]:
        assert "--corpus" in result.stdout
        assert "python -m scripts.search tfidf" in result.stdout


@pytest.mark.parametrize(
    ("mode", "option"),
    [
        ("tfidf", "--index"),
        ("tfidf", "--device"),
    ],
)
def test_unknown_option_uses_selected_help(mode, option):
    args = [mode, "--query", "제주", option, "unused"]
    result = subprocess.run(
        [sys.executable, "-S", "-m", "scripts.search", *args],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert f"usage: python -m scripts.search {mode}" in result.stderr
    assert f"python -m scripts.search {mode} --help" in result.stderr
    assert f"지원하지 않는 인수입니다: {option}" in result.stderr
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("args", [[], ["unknown"]])
def test_missing_or_unknown_search_command_uses_root_help(args, capsys):
    with pytest.raises(SystemExit) as exc:
        main(args)
    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "오류:" in captured.err
    assert "tfidf" in captured.err and "dense" in captured.err
    assert "python -m scripts.search --help" in captured.err
    assert captured.out == ""


def test_empty_query_invalid_top_k_is_rejected_before_loading(capsys):
    args = ["tfidf", "--query", "", "--top-k", "0", "--corpus", "missing.jsonl"]
    with pytest.raises(SystemExit) as exc:
        main(args)
    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "1 이상의 정수" in captured.err
    assert "python -m scripts.search tfidf --help" in captured.err
    assert captured.out == ""


def test_backend_dependencies_are_independent(monkeypatch, capsys):
    pytest.importorskip("sklearn")
    args = ["tfidf", "--corpus", str(FIXTURE_PATH), "--query", "제주"]
    forbidden = ("retrievers.dense", "faiss", "sentence_transformers")
    original_import = builtins.__import__

    def _isolated_import(name, *args, **kwargs):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden):
            raise AssertionError(f"Unexpected backend import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _isolated_import)
    import scripts.search as search

    importlib.reload(search)
    assert search.main(args) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)
    assert captured.err == ""


def test_internal_non_string_query_is_not_hidden(capsys):
    from scripts.search import _build_parser

    pytest.importorskip("sklearn")
    argv = ["tfidf", "--corpus", str(FIXTURE_PATH), "--query", "제주"]
    args = _build_parser().parse_args(argv)
    args.query = None
    with pytest.raises(TypeError, match="query"):
        args.run(args)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_sparse_permission_failure_includes_corpus_and_help(monkeypatch, capsys):
    from data import loader

    def _denied(path):
        raise PermissionError("corpus access denied")

    monkeypatch.setattr(loader, "load_prepared_documents", _denied)
    with pytest.raises(SystemExit) as exc:
        main(["tfidf", "--corpus", str(FIXTURE_PATH), "--query", "제주"])
    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "권한" in captured.err
    assert str(FIXTURE_PATH) in captured.err
    assert "corpus access denied" in captured.err
    assert "python -m scripts.search tfidf --help" in captured.err
    assert captured.out == ""


def test_help_without_loading_corpus(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["tfidf", "--help"])
    assert exc.value.code == 0
    assert "--corpus" in capsys.readouterr().out


@pytest.mark.parametrize("top_k", ["0", "-1", "abc", "1.5"])
def test_invalid_top_k_is_rejected_before_loading(top_k, capsys):
    with pytest.raises(SystemExit) as exc:
        main(
            ["tfidf", "--query", "제주", "--corpus", "missing.jsonl", "--top-k", top_k]
        )
    assert exc.value.code == 2
    assert "1 이상의 정수" in capsys.readouterr().err


def test_missing_query_is_rejected():
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "tfidf",
            ]
        )
    assert exc.value.code == 2


def test_missing_corpus_reports_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["tfidf", "--query", "제주", "--corpus", str(tmp_path / "missing.jsonl")])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "missing.jsonl" in captured.err
    assert "python -m scripts.search tfidf --help" in captured.err
    assert not captured.out


def test_default_corpus_path(tmp_path, monkeypatch, sparse_extra, capsys):
    directory = tmp_path / "data" / "processed"
    directory.mkdir(parents=True)
    (directory / "corpus.jsonl").write_bytes(FIXTURE_PATH.read_bytes())
    monkeypatch.chdir(tmp_path)
    assert main(["tfidf", "--query", "제주", "--top-k", "1"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["document_id"] == "fixture-002#0"


def test_cli_serializes_ranked_search_results(sparse_extra, capsys):
    assert (
        main(
            [
                "tfidf",
                "--corpus",
                str(FIXTURE_PATH),
                "--query",
                "대한민국",
                "--top-k",
                "2",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    results = json.loads(captured.out)
    assert not captured.err
    assert len(results) == 2
    assert [result["rank"] for result in results] == [1, 2]
    assert results[0]["score"] >= results[1]["score"] > 0
    assert all(
        set(result) == {"document_id", "rank", "score", "title", "snippet"}
        for result in results
    )
    assert all(result["document_id"].startswith("fixture-") for result in results)
    assert "대한민국" in captured.out


@pytest.mark.parametrize("query", ["", "   ", "\n\t", "zzzzzz"])
def test_cli_no_matches_returns_json_list(query, sparse_extra, capsys):
    assert main(["tfidf", "--corpus", str(FIXTURE_PATH), "--query", query]) == 0
    assert json.loads(capsys.readouterr().out) == []


@pytest.mark.parametrize(
    "records, message",
    [
        ('{"_id":"raw#0","text":"본문"}\n', "document_id"),
        (
            '{"document_id":"doc#0","text":"본문"}\n{"document_id":"doc#0","text":"중복"}\n',
            "중복",
        ),
        ("", "문서"),
    ],
)
def test_cli_invalid_corpus(tmp_path, sparse_extra, capsys, records, message):
    path = tmp_path / "corpus.jsonl"
    path.write_text(records, encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["tfidf", "--corpus", str(path), "--query", "서울"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert message in captured.err
    assert str(path) in captured.err
    assert "python -m scripts.search tfidf --help" in captured.err


def test_missing_sparse_extra_reports_installation_command(monkeypatch, capsys):
    original_import = builtins.__import__

    def _without_sparse(name, *args, **kwargs):
        if name == "retrievers.tfidf":
            raise ModuleNotFoundError("No module named sklearn", name="sklearn")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _without_sparse)
    with pytest.raises(SystemExit) as exc:
        main(["tfidf", "--corpus", str(FIXTURE_PATH), "--query", "제주"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "uv sync --extra sparse" in captured.err
    assert "python -m scripts.search tfidf" in captured.err
    assert "python -m scripts.search tfidf --help" in captured.err
    assert captured.out == ""


def test_module_entry_point(sparse_extra):
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "scripts.search",
            "tfidf",
            "--corpus",
            str(FIXTURE_PATH),
            "--query",
            "제주",
            "--top-k",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    results = json.loads(completed.stdout)
    assert results[0]["document_id"] == "fixture-002#0"
    assert results[0]["title"] is None
    assert results[0]["snippet"] == "제주도는 대한민국의 섬입니다."
    assert not completed.stderr

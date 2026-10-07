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


@pytest.fixture(params=["tfidf", "bm25"])
def sparse_mode(request):
    pytest.importorskip("sklearn")
    if request.param == "bm25":
        pytest.importorskip("bm25s")
    return request.param


def test_root_help_lists_search_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert all(mode in captured.out for mode in ("tfidf", "bm25", "dense"))
    assert captured.err == ""


@pytest.mark.parametrize("command", [[], ["tfidf"], ["bm25"]])
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
    if command:
        assert "--corpus" in result.stdout
        assert f"python -m scripts.search {command[0]}" in result.stdout


@pytest.mark.parametrize(
    ("mode", "option"),
    [
        ("tfidf", "--index"),
        ("tfidf", "--device"),
        ("bm25", "--index"),
        ("bm25", "--device"),
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
    assert all(mode in captured.err for mode in ("tfidf", "bm25", "dense"))
    assert "python -m scripts.search --help" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("mode", ["tfidf", "bm25"])
def test_empty_query_invalid_top_k_is_rejected_before_loading(mode, capsys):
    args = [mode, "--query", "", "--top-k", "0", "--corpus", "missing.jsonl"]
    with pytest.raises(SystemExit) as exc:
        main(args)
    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "1 이상의 정수" in captured.err
    assert f"python -m scripts.search {mode} --help" in captured.err
    assert captured.out == ""


def test_backend_dependencies_are_independent(monkeypatch, sparse_mode, capsys):
    args = [sparse_mode, "--corpus", str(FIXTURE_PATH), "--query", "제주"]
    other_sparse = "retrievers.bm25" if sparse_mode == "tfidf" else "retrievers.tfidf"
    forbidden = (other_sparse, "retrievers.dense", "faiss", "sentence_transformers")
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


def test_internal_non_string_query_is_not_hidden(sparse_mode, capsys):
    from scripts.search import _build_parser

    argv = [sparse_mode, "--corpus", str(FIXTURE_PATH), "--query", "제주"]
    args = _build_parser().parse_args(argv)
    args.query = None
    with pytest.raises(TypeError, match="query"):
        args.run(args)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("mode", ["tfidf", "bm25"])
def test_sparse_permission_failure_includes_corpus_and_help(mode, monkeypatch, capsys):
    from data import loader

    def _denied(path):
        raise PermissionError("corpus access denied")

    monkeypatch.setattr(loader, "load_prepared_documents", _denied)
    with pytest.raises(SystemExit) as exc:
        main([mode, "--corpus", str(FIXTURE_PATH), "--query", "제주"])
    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "권한" in captured.err
    assert str(FIXTURE_PATH) in captured.err
    assert "corpus access denied" in captured.err
    assert f"python -m scripts.search {mode} --help" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("mode", ["tfidf", "bm25"])
def test_help_without_loading_corpus(mode, capsys):
    with pytest.raises(SystemExit) as exc:
        main([mode, "--help"])
    assert exc.value.code == 0
    assert "--corpus" in capsys.readouterr().out


@pytest.mark.parametrize("top_k", ["0", "-1", "abc", "1.5"])
@pytest.mark.parametrize("mode", ["tfidf", "bm25"])
def test_invalid_top_k_is_rejected_before_loading(mode, top_k, capsys):
    with pytest.raises(SystemExit) as exc:
        main([mode, "--query", "제주", "--corpus", "missing.jsonl", "--top-k", top_k])
    assert exc.value.code == 2
    assert "1 이상의 정수" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["tfidf", "bm25"])
def test_missing_query_is_rejected(mode):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                mode,
            ]
        )
    assert exc.value.code == 2


@pytest.mark.parametrize("mode", ["tfidf", "bm25"])
def test_missing_corpus_reports_error(mode, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main([mode, "--query", "제주", "--corpus", str(tmp_path / "missing.jsonl")])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "missing.jsonl" in captured.err
    assert f"python -m scripts.search {mode} --help" in captured.err
    assert not captured.out


def test_default_corpus_path(tmp_path, monkeypatch, sparse_mode, capsys):
    directory = tmp_path / "data" / "processed"
    directory.mkdir(parents=True)
    (directory / "corpus.jsonl").write_bytes(FIXTURE_PATH.read_bytes())
    monkeypatch.chdir(tmp_path)
    assert main([sparse_mode, "--query", "제주", "--top-k", "1"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["document_id"] == "fixture-002#0"


def test_cli_serializes_ranked_search_results(sparse_mode, capsys):
    assert (
        main(
            [
                sparse_mode,
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
def test_cli_no_matches_returns_json_list(query, sparse_mode, capsys):
    assert main([sparse_mode, "--corpus", str(FIXTURE_PATH), "--query", query]) == 0
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
def test_cli_invalid_corpus(tmp_path, sparse_mode, capsys, records, message):
    path = tmp_path / "corpus.jsonl"
    path.write_text(records, encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main([sparse_mode, "--corpus", str(path), "--query", "서울"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert message in captured.err
    assert str(path) in captured.err
    assert f"python -m scripts.search {sparse_mode} --help" in captured.err


@pytest.mark.parametrize(
    ("mode", "dependency"),
    [("tfidf", "sklearn"), ("bm25", "bm25s"), ("bm25", "sklearn.feature_extraction")],
)
def test_missing_sparse_extra_reports_installation_command(
    mode, dependency, monkeypatch, capsys
):
    original_import = builtins.__import__

    def _without_sparse(name, *args, **kwargs):
        if name == f"retrievers.{mode}":
            raise ModuleNotFoundError(f"No module named {dependency}", name=dependency)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _without_sparse)
    with pytest.raises(SystemExit) as exc:
        main([mode, "--corpus", str(FIXTURE_PATH), "--query", "제주"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "uv sync --extra sparse" in captured.err
    assert f"python -m scripts.search {mode}" in captured.err
    assert f"python -m scripts.search {mode} --help" in captured.err
    assert captured.out == ""


def test_module_entry_point(sparse_mode):
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "scripts.search",
            sparse_mode,
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

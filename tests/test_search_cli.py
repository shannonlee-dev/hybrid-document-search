"""오프라인 fixture로 준비 corpus 검색 CLI를 검증합니다."""

import builtins
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


def test_help_without_loading_corpus(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--corpus" in capsys.readouterr().out


@pytest.mark.parametrize("top_k", ["0", "-1", "abc", "1.5"])
def test_invalid_top_k_is_rejected_before_loading(top_k, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--query", "제주", "--corpus", "missing.jsonl", "--top-k", top_k])
    assert exc.value.code == 2
    assert "양의 정수" in capsys.readouterr().err


def test_missing_query_is_rejected():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_missing_corpus_reports_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--query", "제주", "--corpus", str(tmp_path / "missing.jsonl")])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "missing.jsonl" in captured.err
    assert not captured.out


def test_default_corpus_path(tmp_path, monkeypatch, sparse_extra, capsys):
    directory = tmp_path / "data" / "processed"
    directory.mkdir(parents=True)
    (directory / "corpus.jsonl").write_bytes(FIXTURE_PATH.read_bytes())
    monkeypatch.chdir(tmp_path)
    assert main(["--query", "제주", "--top-k", "1"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["document_id"] == "fixture-002#0"


def test_cli_serializes_ranked_search_results(sparse_extra, capsys):
    assert (
        main(["--corpus", str(FIXTURE_PATH), "--query", "대한민국", "--top-k", "2"])
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


@pytest.mark.parametrize("query", ["   ", "zzzzzz"])
def test_cli_no_matches_returns_json_list(query, sparse_extra, capsys):
    assert main(["--corpus", str(FIXTURE_PATH), "--query", query]) == 0
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
        main(["--corpus", str(path), "--query", "서울"])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_missing_sparse_extra_reports_installation_command(monkeypatch, capsys):
    original_import = builtins.__import__

    def without_sparse(name, *args, **kwargs):
        if name == "retrievers.tfidf":
            raise ModuleNotFoundError("No module named sklearn", name="sklearn")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_sparse)
    with pytest.raises(SystemExit) as exc:
        main(["--corpus", str(FIXTURE_PATH), "--query", "제주"])
    assert exc.value.code == 2
    assert "uv sync --extra sparse" in capsys.readouterr().err


def test_module_entry_point(sparse_extra):
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "scripts.search",
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

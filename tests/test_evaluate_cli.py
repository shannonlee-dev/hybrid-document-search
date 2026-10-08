"""평가 CLI의 실제 Sparse 실행, 출력 파일과 입력 오류를 검증합니다."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.evaluate import main


def test_help_without_search_dependencies():
    result = subprocess.run(
        [sys.executable, "-B", "-S", "-m", "scripts.evaluate", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 0
    assert "--methods" in result.stdout and "--index" in result.stdout
    assert "--output-dir" in result.stdout
    assert result.stderr == ""
    assert "\\n" not in result.stdout


def test_sparse_end_to_end_json_outputs(evaluation_directory, tmp_path, capsys):
    pytest.importorskip("sklearn")
    pytest.importorskip("bm25s")
    directory = tmp_path / "benchmark"
    assert (
        main(
            [
                "--data-dir",
                str(evaluation_directory),
                "--output-dir",
                str(directory),
                "--repeats",
                "2",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert set(summary) == {"tfidf", "bm25"}
    assert "검색기 준비" in captured.err and "평가:" in captured.err
    result = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    assert result["dataset"]["split"] == "dev"
    assert result["conditions"]["top_k"] == 10
    for method in summary:
        assert summary[method]["metrics"] == {
            "recall@5": 1,
            "recall@10": 1,
            "mrr@10": 1,
            "ndcg@10": 1,
        }
        assert summary[method]["latency"]["sample_count"] == 4
    assert set(path.name for path in directory.iterdir()) == {
        "results.json",
        "summary.csv",
        "queries.csv",
        "latencies.csv",
        "comparison.md",
    }


@pytest.mark.parametrize(
    "args, detail",
    [
        (["--repeats", "0"], "1 이상의 정수"),
        (["--warmup", "0"], "1 이상의 정수"),
        (["--split", "test"], "지원하지 않는"),
        (["--methods", "unknown"], "지원하지 않는"),
        (["--methods", "dense"], "--index"),
        (["--methods", "tfidf", "tfidf"], "중복"),
        (["--device", " "], "device"),
    ],
)
def test_input_errors_do_not_create_outputs(tmp_path, capsys, args, detail):
    directory = tmp_path / "result"
    with pytest.raises(SystemExit) as exc:
        main(["--output-dir", str(directory), *args])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert detail in captured.err
    assert "python -m scripts.evaluate --help" in captured.err
    assert not captured.out and not directory.exists()


def test_existing_output_is_not_overwritten(tmp_path, capsys):
    directory = tmp_path / "existing"
    directory.mkdir()
    original = directory / "results.json"
    original.write_text("keep")
    with pytest.raises(SystemExit) as exc:
        main(["--output-dir", str(directory)])
    assert exc.value.code == 2
    assert original.read_text() == "keep"
    assert "output-dir" in capsys.readouterr().err


def test_missing_data_reports_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--data-dir",
                str(tmp_path / "missing"),
                "--output-dir",
                str(tmp_path / "result"),
            ]
        )
    assert exc.value.code == 2
    assert "manifest" in capsys.readouterr().err

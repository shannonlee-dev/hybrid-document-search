"""Require an explicit local model before starting the QDAP benchmark."""

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")


def test_benchmark_requires_explicit_model(monkeypatch, capsys, tmp_path):
    from scripts.run_qdap_benchmark import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_qdap_benchmark",
            "train",
            "--inputs",
            str(tmp_path),
            "--output",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "--model" in capsys.readouterr().err

"""Require explicit model and run identities before starting QDAP work."""

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
            "--variant",
            "L",
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


@pytest.mark.parametrize(
    "stage, extra, missing",
    [
        ("export", [], "--run-id"),
        ("audit", [], "--main-run-id"),
        ("all", ["--run-id", "new-qdap-run"], "--main-run-id"),
        ("export", ["--run-id", "   "], "--run-id"),
    ],
)
def test_export_requires_stage_specific_run_id(stage, extra, missing, capsys, tmp_path):
    from scripts.export_qdap_results import main

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--stage",
                stage,
                "--inputs",
                str(tmp_path),
                "--run",
                str(tmp_path),
                "--output",
                str(tmp_path),
                *extra,
            ]
        )
    assert error.value.code == 2
    assert missing in capsys.readouterr().err


def test_export_dispatches_distinct_explicit_run_ids(monkeypatch, tmp_path):
    from scripts import export_qdap_results

    exported = []
    monkeypatch.setattr(
        export_qdap_results,
        "export_frozen",
        lambda args: exported.append(("qdap", args.run_id)),
    )
    monkeypatch.setattr(
        export_qdap_results,
        "audit_main",
        lambda args: exported.append(("main", args.main_run_id)),
    )
    export_qdap_results.main(
        [
            "--stage",
            "all",
            "--inputs",
            str(tmp_path),
            "--run",
            str(tmp_path),
            "--output",
            str(tmp_path),
            "--run-id",
            "qdap-new-experiment",
            "--main-run-id",
            "main-new-audit",
        ]
    )
    assert exported == [("qdap", "qdap-new-experiment"), ("main", "main-new-audit")]

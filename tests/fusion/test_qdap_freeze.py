"""Freeze real Hugging Face checkpoint formats before reading Test labels."""

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")


@pytest.mark.parametrize("weight_name", ["pytorch_model.bin", "model.safetensors"])
def test_freeze_hashes_actual_snapshot_format(tmp_path, weight_name):
    from evaluation.json_io import read_json, write_json
    from evaluation.qdap_benchmark import sha
    from scripts.run_qdap_benchmark import freeze

    out, inputs, model = [tmp_path / name for name in ("out", "inputs", "model")]
    out.mkdir()
    model.mkdir()
    (model / weight_name).write_bytes(b"original model weights")
    (model / "config.json").write_text("{}")
    write_json(out / "protocol.json", {})
    np.savez(out / "validation.npz", quality=np.zeros((2, 101)))
    for variant, seeds in (("S", (42, 43, 44)), ("L", (42,))):
        for mode in ("literal", "reference"):
            for seed in seeds:
                directory = out / f"{variant}-{mode}-{seed}"
                directory.mkdir()
                (directory / "selected.pt").write_bytes(b"selected weights")
                write_json(
                    directory / "checkpoint.json",
                    {"sha256": sha(directory / "selected.pt")},
                )
                write_json(
                    directory / "training.json",
                    {
                        "history": [{"epoch": i} for i in range(1, 6)],
                        "selected": {
                            "epoch": 1,
                            "readout": "mean",
                            "metrics": {"ndcg@10": 0.5},
                        },
                    },
                )
    for name in [
        "data/processed/bccard/manifest.json",
        "artifacts/bccard-adaptive-vectors-resumed/query-vectors-test.npy",
        "artifacts/bccard-adaptive-vectors-resumed/document-vectors.npy",
        "artifacts/bccard-adaptive-vectors-resumed/encoding.json",
        "docs/experiments/adaptive/router.json",
    ]:
        path = inputs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unchanged input")
    freeze(SimpleNamespace(output=out, inputs=inputs, model=model))
    record = read_json(out / "selection.json")
    assert record["selection_split"] == "validation"
    assert record["evaluation_input_sha256"][str(model / weight_name)] == sha(
        model / weight_name
    )

"""평가 pipeline 테스트에서 사용하는 공통 준비 데이터 fixture."""

import json
from pathlib import Path

import pytest

from evaluation.data import file_sha256

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def evaluation_directory(tmp_path):
    directory = tmp_path / "prepared"
    directory.mkdir()
    (directory / "corpus.jsonl").write_bytes(
        (FIXTURES / "ko_miracl_prepared_corpus.jsonl").read_bytes()
    )
    for name in ("queries_train", "queries_dev", "qrels_train", "qrels_dev"):
        (directory / f"{name}.jsonl").write_bytes(
            (FIXTURES / f"evaluation_{name}.jsonl").read_bytes()
        )
    refresh_manifest(directory)
    return directory


def refresh_manifest(directory):
    counts, hashes = {}, {}
    for path in directory.glob("*.jsonl"):
        counts[path.name] = sum(
            bool(line.strip()) for line in path.read_text().splitlines()
        )
        hashes[path.name] = file_sha256(path)
    manifest = {
        "schema_version": "1",
        "dataset_id": "synthetic-fixture",
        "dataset_revision": "0" * 40,
        "settings": {"seed": 42},
        "output_counts": counts,
        "output_sha256": hashes,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

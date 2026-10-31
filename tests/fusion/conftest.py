"""Shared prepared inputs for QDAP score and report tests."""

import json

import pytest


@pytest.fixture
def score_inputs(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    pytest.importorskip("bm25s")

    from evaluation.qdap_benchmark import sha

    data = tmp_path / "data/processed/bccard"
    cache = tmp_path / "artifacts/bccard-adaptive-vectors-resumed"
    data.mkdir(parents=True)
    cache.mkdir(parents=True)
    docs = [
        {"document_id": "a", "text": "카드 결제"},
        {"document_id": "b", "text": "분실 신고"},
    ]
    (data / "corpus.jsonl").write_text("".join(json.dumps(r) + "\n" for r in docs))
    np.save(cache / "document-vectors.npy", np.eye(2, dtype=np.float32))
    encoding = {
        "model": {"model_name": "BAAI/bge-m3", "revision": "1" * 40},
        "dtype": "torch.float32",
        "max_seq_length": 8192,
        "splits": {},
        "vector_sha256": {"document-vectors.npy": sha(cache / "document-vectors.npy")},
    }
    for split in ("train", "validation", "test"):
        queries = [
            dict(
                query_id=f"{split}-{i}",
                text=d["text"],
                parent_doc_id=f"{split}-{i}",
                provenance="base",
                page_type=None,
                site=None,
            )
            for i, d in enumerate(docs)
        ]
        (data / f"queries_{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in queries)
        )
        (data / f"qrels_{split}.jsonl").write_text(
            "".join(
                json.dumps(
                    dict(
                        query_id=q["query_id"],
                        document_id=d["document_id"],
                        relevance=3,
                    )
                )
                + "\n"
                for q, d in zip(queries, docs)
            )
        )
        name = f"query-vectors-{split}.npy"
        np.save(cache / name, np.eye(2, dtype=np.float32))
        encoding["vector_sha256"][name] = sha(cache / name)
        encoding["splits"][split] = {"query_ids": [q["query_id"] for q in queries]}
    manifest = {
        "schema_version": "1",
        "dataset_id": "score-fixture",
        "dataset_revision": "0" * 40,
        "splits": ["train", "validation", "test"],
        "settings": {},
        "primary_slice_query_ids": ["test-0", "test-1"],
        "output_sha256": {p.name: sha(p) for p in data.glob("*.jsonl")},
        "output_counts": {
            p.name: len(p.read_text().splitlines()) for p in data.glob("*.jsonl")
        },
    }
    (data / "manifest.json").write_text(json.dumps(manifest))
    encoding["dataset_manifest_sha256"] = sha(data / "manifest.json")
    (cache / "encoding.json").write_text(json.dumps(encoding))
    return tmp_path

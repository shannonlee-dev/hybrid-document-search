"""작은 parquet fixture로 BC Card 변환, 등급 qrels, split 격리와 manifest를 검증합니다."""

import json

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from data.bccard import (  # noqa: E402
    PAIR_FILES,
    TRIPLET_FILES,
    load_query_metadata,
    parent_and_index,
    prepare_bccard,
    provenance,
)
from evaluation.data import file_sha256, load_evaluation_dataset  # noqa: E402

CARD = "crawl_bc-personal_card_"
FAQ = "crawl_bc-personal_faq_1000"


def _pair():
    documents = {
        f"{CARD}2000": "카드 상품 개요",
        f"{CARD}2001": "연회비 안내",
        f"{CARD}2002": "포인트 적립",
        f"{CARD}2005": "해지 안내",
        FAQ: "Q: 분실 신고는? A: 고객센터",
        "base_bcai_000001_000": "비씨카드 소개",
        "base_bcai_000002_000": "결제일 변경 방법",
        "base_bcai_000003_000": "가맹점 수수료",
        "base_bcai_000004_000": "같은 본문",
        "base_bcai_000005_000": "같은 본문",
    }
    rows = [
        {
            "chunk_id": document_id,
            "source_chunk_id": document_id,
            "positive": text,
            # 실제 원본처럼 FAQ 직접 질의도 source/page 정보는 출처 판단에 쓰지 않는다.
            "page_type": "faq" if document_id == FAQ else None,
            "title": None,
        }
        for document_id, text in documents.items()
    ]
    rows.append(
        {
            "chunk_id": f"synth_{CARD}2001_q1",
            "source_chunk_id": f"{CARD}2001",
            "positive": "연회비 안내",
            "page_type": "service",
            "title": "카드",
        }
    )
    return rows


def _query(query_id, text, parent, relevant=None):
    row = {"anchor": text, "anchor_chunk_id": query_id, "anchor_parent_doc_id": parent}
    if relevant is not None:
        row["relevant_chunk_ids"] = list(relevant)
        row["relevance_scores"] = list(relevant.values())
    return row


def _splits():
    return {
        "train": [
            _query(
                "base_bcai_000001_000",
                "비씨카드는 어떤 회사인가요?",
                "base_bcai_000001",
            ),
            _query(f"synth_{CARD}2001_q1", "연회비는 얼마인가요?", f"{CARD}2000"),
        ],
        "validation": [
            _query(
                "base_bcai_000002_000",
                "결제일은 어떻게 바꾸나요?",
                "base_bcai_000002",
                {"base_bcai_000002_000": 3},
            ),
            _query(FAQ, "분실 신고는?", FAQ, {FAQ: 3}),
        ],
        "test": [
            _query(
                "base_bcai_000003_000",
                "가맹점 수수료는?",
                "base_bcai_000003",
                {"base_bcai_000003_000": 3},
            ),
            _query(
                "base_bcai_000004_000",
                "비씨카드는 어떤 회사인가요?",
                "base_bcai_000004",
                {"base_bcai_000004_000": 3},
            ),
        ],
    }


def _write(directory, pair=None, splits=None):
    pair_dir, triplet_dir = directory / "pair", directory / "triplet"
    for path in (pair_dir / PAIR_FILES[0], *(triplet_dir / n for n in TRIPLET_FILES)):
        path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(pair or _pair()), pair_dir / PAIR_FILES[0])
    for name, rows in zip(TRIPLET_FILES, (splits or _splits()).values(), strict=True):
        pq.write_table(pa.Table.from_pylist(rows), triplet_dir / name)
    return pair_dir, triplet_dir


@pytest.fixture
def prepared(tmp_path):
    output = tmp_path / "prepared"
    manifest = prepare_bccard(*_write(tmp_path / "raw"), output)
    return output, manifest


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_corpus_keeps_every_source_chunk_with_stable_ids(prepared):
    output, manifest = prepared
    corpus = _jsonl(output / "corpus.jsonl")
    ids = [row["document_id"] for row in corpus]
    assert ids == sorted(ids) and len(ids) == 10
    assert all(row["title"] is None for row in corpus)
    # 같은 본문의 문서는 별도 ID로 유지하고 manifest에 기록한다.
    assert manifest["identical_text_groups"] == [
        ["base_bcai_000004_000", "base_bcai_000005_000"]
    ]


def test_train_qrels_follow_documented_grading_rule(prepared):
    output, _ = prepared
    qrels = [
        row
        for row in _jsonl(output / "qrels_train.jsonl")
        if row["query_id"] == f"synth_{CARD}2001_q1"
    ]
    assert {row["document_id"]: row["relevance"] for row in qrels} == {
        f"{CARD}2000": 2,
        f"{CARD}2001": 3,
        f"{CARD}2002": 2,
        f"{CARD}2005": 1,
    }


def test_provenance_uses_id_prefix_not_source_column(prepared):
    output, manifest = prepared
    metadata = load_query_metadata(output, "validation")
    assert metadata[FAQ]["provenance"] == "faq_direct"
    assert metadata[FAQ]["site"] == "bc-personal"
    assert provenance(f"synth_{CARD}2001_q1") == "synthetic_crawl"
    assert provenance("synth_base_bcai_000013_000_q1") == "synthetic_base"
    with pytest.raises(ValueError, match="접두사"):
        provenance("other_1")
    assert manifest["primary_slice_query_ids"] == [
        "base_bcai_000003_000",
        "base_bcai_000004_000",
    ]


def test_eval_queries_flag_text_seen_in_train(prepared):
    output, manifest = prepared
    flags = {
        query_id: row["text_in_train"]
        for query_id, row in load_query_metadata(output, "test").items()
    }
    assert flags == {"base_bcai_000003_000": False, "base_bcai_000004_000": True}
    assert manifest["leakage_checks"]["train&test"]["identical_query_texts"] == 1


def test_prepared_splits_load_with_shared_evaluation_loader(prepared):
    output, manifest = prepared
    for split, count in (("train", 2), ("validation", 2), ("test", 2)):
        dataset = load_evaluation_dataset(output, split)
        assert len(dataset.queries) == count and len(dataset.documents) == 10
    assert manifest["output_sha256"] == {
        name: file_sha256(output / name) for name in manifest["output_sha256"]
    }


def test_preparation_is_deterministic(tmp_path, prepared):
    output, _ = prepared
    again = tmp_path / "again"
    prepare_bccard(tmp_path / "raw/pair", tmp_path / "raw/triplet", again)
    for name in ("manifest.json", "corpus.jsonl", "qrels_test.jsonl"):
        assert (again / name).read_bytes() == (output / name).read_bytes()


def test_parent_keeps_zero_padding():
    assert parent_and_index("crawl_bc-merchant_event_main_000") == (
        "crawl_bc-merchant_event_main_000",
        0,
    )
    assert parent_and_index(f"{CARD}29008") == (f"{CARD}29000", 8)
    assert parent_and_index("base_bcai_011324_002") == ("base_bcai_011324", 2)


def _break(mutate):
    splits = _splits()
    mutate(splits)
    return splits


@pytest.mark.parametrize(
    ("splits", "message"),
    [
        (
            _break(lambda s: s["test"].append(dict(s["train"][0]))),
            "query_id",
        ),
        (
            _break(
                lambda s: s["test"][0].update(
                    relevant_chunk_ids=["base_bcai_000003_000", "missing"],
                    relevance_scores=[3, 1],
                )
            ),
            "corpus에 없는",
        ),
        (
            _break(lambda s: s["test"][0].update(relevance_scores=[2])),
            "등급 규칙",
        ),
        (
            _break(lambda s: s["test"][0].update(relevance_scores=[3, 1])),
            "relevance 배열",
        ),
        (
            _break(lambda s: s["test"][0].update(anchor_parent_doc_id="base_bcai_9")),
            "anchor_parent_doc_id",
        ),
        (
            _break(
                lambda s: s["validation"].append(
                    _query(
                        f"{CARD}2002",
                        "포인트",
                        f"{CARD}2000",
                        {f"{CARD}2000": 1, f"{CARD}2001": 2, f"{CARD}2002": 3}
                        | {f"{CARD}2005": 1},
                    )
                )
            ),
            "parent_doc",
        ),
    ],
)
def test_invalid_sources_are_rejected(tmp_path, splits, message):
    with pytest.raises(ValueError, match=message):
        prepare_bccard(*_write(tmp_path / "raw", splits=splits), tmp_path / "out")


def test_conflicting_corpus_text_is_rejected(tmp_path):
    pair = _pair()
    pair[-1]["positive"] = "다른 본문"
    with pytest.raises(ValueError, match="문서 본문"):
        prepare_bccard(*_write(tmp_path / "raw", pair=pair), tmp_path / "out")


def test_non_empty_output_is_rejected(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep").write_text("x")
    with pytest.raises(ValueError, match="비어 있는"):
        prepare_bccard(*_write(tmp_path / "raw"), output)
    assert (output / "keep").read_text() == "x"

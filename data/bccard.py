"""BC Card 금융 검색 데이터를 공통 corpus/query/qrels 형식과 고정 평가 protocol로 준비합니다.

Pair의 `source_chunk_id`가 corpus, Triplet의 공식 train/validation/test가 질의입니다.
validation/test qrels는 공식 graded label을 그대로 쓰고, label이 없는 train에는
dataset card에 문서화된 chunk-distance 규칙을 적용합니다. 규칙이 공식 label을
재현하지 못하면 준비를 중단합니다.
"""

import json
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

from data.preparation import _check_revision, _write_jsonl, file_sha256

PAIR_ID = "BCCard/BCAI-Finance-Kor-Embedding-Pair"
TRIPLET_ID = "BCCard/BCAI-Finance-Kor-Embedding-Triplet"
PAIR_REVISION = "e022cb013f2907e0716ebe40d13f30ed93ffa9b0"
TRIPLET_REVISION = "f63d59969dba9916bd34c86c82112331890b11da"
LICENSE = "cc-by-4.0"
SPLITS = ("train", "validation", "test")
PAIR_FILES = ("data/train-00000-of-00001.parquet",)
TRIPLET_FILES = tuple(f"data/{split}-00000-of-00001.parquet" for split in SPLITS)

# Pair의 `source` 열은 FAQ 직접 질의(crawl_*)도 synth로 표시하므로 ID 접두사로 구분한다.
PROVENANCE_PREFIXES = (
    ("synth_base_", "synthetic_base"),
    ("synth_crawl_", "synthetic_crawl"),
    ("base_", "base"),
    ("crawl_", "faq_direct"),
)
NON_SYNTHETIC = ("base", "faq_direct")
PRIMARY_SLICE = {"split": "test", "provenance": list(NON_SYNTHETIC)}
EXTERNAL_UPLOAD_STATUS = "EXTERNAL_UPLOAD_REQUIRES_MANUAL_TERMS_CHECK"

_CRAWL_ID = re.compile(r"(crawl_.+_)(\d+)")
_BASE_ID = re.compile(r"(base_bcai_\d+)_(\d+)")
_SITE = re.compile(r"crawl_(bc-[a-z]+)_")


def provenance(query_id: str) -> str:
    """Triplet anchor_chunk_id 접두사에 기록된 질의 출처를 반환합니다."""
    for prefix, name in PROVENANCE_PREFIXES:
        if query_id.startswith(prefix):
            return name
    raise ValueError(f"알 수 없는 질의 ID 접두사: {query_id}")


def parent_and_index(document_id: str) -> tuple[str, int]:
    """card 규칙대로 parent_doc와 chunk 순번을 구합니다.

    crawl은 끝 번호를 1000 단위로 내림하고, base는 마지막 chunk 번호를 제거합니다.
    """
    if match := _CRAWL_ID.fullmatch(document_id):
        digits = match.group(2)
        number = int(digits)
        return f"{match.group(1)}{number // 1000 * 1000:0{len(digits)}d}", number % 1000
    if match := _BASE_ID.fullmatch(document_id):
        return match.group(1), int(match.group(2))
    raise ValueError(f"parent_doc를 구할 수 없는 document_id: {document_id}")


def graded_qrels(source_id: str, siblings: dict[str, list[tuple[str, int]]]) -> dict:
    """같은 parent_doc 안에서 자기 chunk 3, 인접 chunk 2, 나머지 1을 부여합니다."""
    parent, index = parent_and_index(source_id)
    return {
        document_id: 3 if other == index else 2 if abs(other - index) <= 1 else 1
        for document_id, other in siblings[parent]
    }


def download_sources(raw_root: str | Path) -> dict[str, Path]:
    """고정 revision의 parquet과 dataset card를 별도 디렉터리에 받고 기존 파일은 재사용합니다."""
    directories = {}
    for name, dataset, revision, files in (
        ("pair", PAIR_ID, PAIR_REVISION, PAIR_FILES),
        ("triplet", TRIPLET_ID, TRIPLET_REVISION, TRIPLET_FILES),
    ):
        _check_revision(revision)
        directory = Path(raw_root) / dataset.split("/")[1] / revision
        for filename in (*files, "README.md"):
            destination = directory / filename
            if destination.is_file():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{filename}"
            print(f"다운로드: {dataset}/{filename}", flush=True)
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, delete=False
            ) as target:
                temporary = Path(target.name)
                try:
                    with urlopen(url, timeout=60) as response:
                        shutil.copyfileobj(response, target)
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            temporary.replace(destination)
        directories[name] = directory
    return directories


def _read_parquet(path: Path, columns: list[str]) -> list[dict]:
    import pyarrow.parquet as pq

    if not path.is_file():
        raise ValueError(f"원본 파일이 없습니다: {path}")
    return pq.read_table(path, columns=columns).to_pylist()


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}는 비어 있지 않은 문자열이어야 합니다.")
    return value


def prepare_bccard(pair_dir: str | Path, triplet_dir: str | Path, output_dir) -> dict:
    """원본을 검증해 corpus, split별 queries/qrels와 고정 protocol manifest를 만듭니다."""
    pair_dir, triplet_dir, output = Path(pair_dir), Path(triplet_dir), Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output_dir은 비어 있는 디렉터리여야 합니다.")
    sources = {f"pair/{name}": pair_dir / name for name in PAIR_FILES}
    sources |= {f"triplet/{name}": triplet_dir / name for name in TRIPLET_FILES}

    pair_rows = _read_parquet(
        sources[f"pair/{PAIR_FILES[0]}"],
        ["chunk_id", "source_chunk_id", "positive", "page_type", "title"],
    )
    source_of = {}
    texts = {}
    metadata = {}
    for row in pair_rows:
        chunk_id = _text(row["chunk_id"], "chunk_id")
        document_id = _text(row["source_chunk_id"], "source_chunk_id")
        text = _text(row["positive"], "positive")
        if chunk_id in source_of:
            raise ValueError(f"중복 chunk_id: {chunk_id}")
        source_of[chunk_id] = document_id
        if texts.setdefault(document_id, text) != text:
            raise ValueError(
                f"source_chunk_id의 문서 본문이 여러 개입니다: {document_id}"
            )
        metadata.setdefault(document_id, row["page_type"])

    siblings = defaultdict(list)
    for document_id in sorted(texts):
        parent, index = parent_and_index(document_id)
        siblings[parent].append((document_id, index))
    by_text = defaultdict(list)
    for document_id in sorted(texts):
        by_text[texts[document_id]].append(document_id)
    duplicate_groups = sorted(ids for ids in by_text.values() if len(ids) > 1)

    columns = ["anchor", "anchor_chunk_id", "anchor_parent_doc_id"]
    splits = {}
    for split, name in zip(SPLITS, TRIPLET_FILES, strict=True):
        extra = [] if split == "train" else ["relevant_chunk_ids", "relevance_scores"]
        splits[split] = _read_parquet(sources[f"triplet/{name}"], columns + extra)

    payloads = {}
    seen_queries = {}
    parents = {}
    positives = {}
    query_texts = {}
    stats = {}
    for split, rows in splits.items():
        queries, qrels = [], []
        parents[split], positives[split], query_texts[split] = set(), set(), set()
        label_counts, provenance_counts = Counter(), Counter()
        for row in rows:
            query_id = _text(row["anchor_chunk_id"], "anchor_chunk_id")
            text = _text(row["anchor"], "anchor")
            if query_id in seen_queries:
                raise ValueError(
                    f"query_id가 {seen_queries[query_id]}와 {split}에 중복됩니다: {query_id}"
                )
            seen_queries[query_id] = split
            if query_id not in source_of:
                raise ValueError(f"Pair에 없는 anchor_chunk_id: {query_id}")
            positive = source_of[query_id]
            parent = parent_and_index(positive)[0]
            if parent != row["anchor_parent_doc_id"]:
                raise ValueError(
                    f"anchor_parent_doc_id가 card 규칙과 다릅니다: {query_id}"
                )
            expected = graded_qrels(positive, siblings)
            if split == "train":
                judgments = expected
            else:
                ids, scores = row["relevant_chunk_ids"], row["relevance_scores"]
                if len(ids) != len(scores) or len(set(ids)) != len(ids):
                    raise ValueError(f"relevance 배열이 올바르지 않습니다: {query_id}")
                judgments = dict(zip(ids, scores, strict=True))
                if set(judgments) - texts.keys():
                    raise ValueError(f"corpus에 없는 relevant_chunk_id: {query_id}")
                if judgments != expected:
                    raise ValueError(
                        f"공식 qrels가 문서화된 등급 규칙과 다릅니다: {query_id}"
                    )
            if judgments.get(positive) != 3:
                raise ValueError(f"질의의 원본 chunk가 rel=3이 아닙니다: {query_id}")
            origin = provenance(query_id)
            site = _SITE.match(positive)
            queries.append(
                {
                    "query_id": query_id,
                    "text": text,
                    "provenance": origin,
                    "page_type": metadata[positive],
                    "site": site.group(1) if site else None,
                    "parent_doc_id": parent,
                }
            )
            qrels.extend(
                {"query_id": query_id, "document_id": doc_id, "relevance": score}
                for doc_id, score in sorted(judgments.items())
            )
            label_counts.update(judgments.values())
            provenance_counts[origin] += 1
            parents[split].add(parent)
            positives[split].add(positive)
            query_texts[split].add(text)
        payloads[f"queries_{split}.jsonl"] = queries
        payloads[f"qrels_{split}.jsonl"] = qrels
        stats[split] = {
            "provenance": dict(sorted(provenance_counts.items())),
            "relevance_labels": {str(k): v for k, v in sorted(label_counts.items())},
            "qrels_source": (
                "reconstructed with the dataset-card chunk-distance rule"
                if split == "train"
                else "official relevant_chunk_ids / relevance_scores"
            ),
        }

    overlaps = {}
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        key = f"{left}&{right}"
        if parents[left] & parents[right] or positives[left] & positives[right]:
            raise ValueError(f"split 간 parent_doc 또는 원본 chunk가 겹칩니다: {key}")
        overlaps[key] = {
            "parent_doc_ids": 0,
            "positive_source_chunks": 0,
            "identical_query_texts": len(query_texts[left] & query_texts[right]),
        }
    for split in ("validation", "test"):
        relevant = {row["document_id"] for row in payloads[f"qrels_{split}.jsonl"]}
        if relevant & positives["train"]:
            raise ValueError(f"{split} 정답 chunk가 train 원본 chunk와 겹칩니다.")

    # train과 같은 문장의 평가 질의는 민감도 분석용으로만 표시하고 제외하지 않는다.
    for split in ("validation", "test"):
        for query in payloads[f"queries_{split}.jsonl"]:
            query["text_in_train"] = query["text"] in query_texts["train"]

    documents = [
        {"document_id": document_id, "text": texts[document_id], "title": None}
        for document_id in sorted(texts)
    ]
    payloads = {"corpus.jsonl": documents, **payloads}
    test_queries = payloads["queries_test.jsonl"]
    primary_ids = [
        query["query_id"]
        for query in test_queries
        if query["provenance"] in PRIMARY_SLICE["provenance"]
    ]
    manifest = {
        "schema_version": "1",
        "dataset_id": TRIPLET_ID,
        "dataset_revision": TRIPLET_REVISION,
        "datasets": {
            "corpus": {"id": PAIR_ID, "revision": PAIR_REVISION, "license": LICENSE},
            "queries": {
                "id": TRIPLET_ID,
                "revision": TRIPLET_REVISION,
                "license": LICENSE,
            },
        },
        "usage_terms": {
            "license": LICENSE,
            "attribution": "BC Card AX Team, BCAI Finance Kor Embedding Pair/Triplet (2026)",
            "external_upload": EXTERNAL_UPLOAD_STATUS,
        },
        "splits": list(SPLITS),
        "settings": {"seed": None, "split_source": "official Triplet splits"},
        "protocol": {
            "router_train": "train",
            "model_selection": "validation",
            "final_test": "test",
            "primary_slice": {**PRIMARY_SLICE, "query_count": len(primary_ids)},
            "secondary_slices": [
                {"split": "test", "provenance": "all"},
                {"split": "test", "provenance": ["synthetic_base", "synthetic_crawl"]},
                {"split": "test", "text_in_train": False},
            ],
            "corpus": "shared Pair source_chunk_id corpus for every split",
            "document_text": "Pair positive only; title is not indexed",
            "relevance": "graded 1-3; binary metrics use relevance > 0",
            "test_isolation": "no training, tuning or method selection on test",
        },
        "primary_slice_query_ids": primary_ids,
        "split_stats": stats,
        "leakage_checks": overlaps,
        "corpus_checks": {
            "documents": len(documents),
            "pair_rows": len(pair_rows),
            "parent_docs": len(siblings),
            "identical_text_groups": len(duplicate_groups),
            "documents_in_identical_text_groups": sum(map(len, duplicate_groups)),
        },
        "identical_text_groups": duplicate_groups,
        "output_counts": {name: len(rows) for name, rows in payloads.items()},
        "source_sha256": {name: file_sha256(path) for name, path in sources.items()},
        "script_sha256": file_sha256(Path(__file__)),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        stage = Path(staging)
        for name, rows in payloads.items():
            _write_jsonl(stage / name, rows)
        manifest["output_sha256"] = {
            name: file_sha256(stage / name) for name in payloads
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        output.mkdir(exist_ok=True)
        for path in stage.iterdir():
            path.replace(output / path.name)
    return manifest


def load_query_metadata(directory: str | Path, split: str) -> dict[str, dict]:
    """평가 slice 계산용으로 준비된 질의의 출처·분류 필드를 읽습니다."""
    path = Path(directory) / f"queries_{split}.jsonl"
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    return {row["query_id"]: row for row in rows}

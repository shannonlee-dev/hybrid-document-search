"""가짜 transport·provider로 상용 검색 결과 정규화, 원본 ID mapping, 재시도와 재개를 검증합니다."""

import json

import pytest

from commercial.base import ProviderError, ProviderHit, ProviderResponse
from commercial.naver import NaverRagClient, parse_conversation
from data.loader import Document
from evaluation.data import Query
from evaluation.external import (
    RECORDS_FILE,
    NormalizationError,
    RunAborted,
    index_documents,
    normalize_hits,
    read_jsonl,
    require_terms_review,
    run_external_queries,
)

ID_MAP = {"doc-a": "doc-a", "doc-b": "doc-b", "doc-c": "doc-c"}
METADATA = {
    "provider": "fake",
    "provider_settings": {"service_id": "svc", "base_url": "https://fake.invalid"},
    "split": "test",
    "top_k": 3,
    "dataset_manifest_sha256": "m",
    "query_set_sha256": "q",
    "id_map_sha256": "i",
}


def _conversation(chunks):
    return {
        "code": 0,
        "data": {
            "results": [
                {"blockId": "QUERY_REWRITE", "value": "rewritten"},
                {"blockId": "RETRIEVAL", "value": chunks},
                {
                    "blockId": "GENERATOR",
                    "value": "answer",
                    "metadata": {"modelName": "HCX-007", "totalTokens": 12},
                },
            ]
        },
    }


class _Transport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((method, url, headers, json.loads(body) if body else None))
        status, payload = self.responses.pop(0)
        return status, {}, json.dumps(payload).encode()


def test_naver_response_maps_retrieval_block_and_usage():
    response = parse_conversation(
        _conversation(
            [
                {"docId": "doc-b", "score": 0.9, "rank": 1, "chunk_no": 0, "text": "x"},
                {"docId": "doc-a", "rank": 2, "chunkNo": 3},
            ]
        )
    )
    assert response.hits == (
        ProviderHit("doc-b", 1, 0.9, None, "0"),
        ProviderHit("doc-a", 2, None, None, "3"),
    )
    assert response.usage == {"modelName": "HCX-007", "totalTokens": 12}


def test_naver_search_request_sets_k_verbose_and_bearer_without_leaking_key():
    transport = _Transport((200, _conversation([])))
    client = NaverRagClient("secret-key", "svc", transport=transport)
    assert client.search("질의", 10).hits == ()
    method, url, headers, body = transport.calls[0]
    assert (method, url) == (
        "POST",
        "https://kr-pub-gateway.rag.naverncp.com/api/v1/svc/svc/conversation",
    )
    assert headers["Authorization"] == "Bearer secret-key"
    assert body["verbose"] is True
    assert body["blocks"]["RETRIEVAL"]["parameters"]["k"] == 10
    assert "secret-key" not in json.dumps(client.describe())


def test_naver_add_document_keeps_source_chunk_id():
    transport = _Transport(
        (200, {"status": {"code": 200}, "result": {"inserted": ["d1"]}})
    )
    NaverRagClient("k", "svc", transport=transport).add_document("d1", "본문")
    _, url, _, body = transport.calls[0]
    assert url.endswith("/doc/d1")
    assert body == {"body": "본문", "extra": {"source_chunk_id": "d1"}}
    transport = _Transport(
        (200, {"status": {"code": 200}, "result": {"inserted": ["x"]}})
    )
    with pytest.raises(ProviderError) as error:
        NaverRagClient("k", "svc", transport=transport).add_document("d1", "본문")
    assert error.value.abort


@pytest.mark.parametrize(
    ("status", "payload", "kind", "transient", "abort"),
    [
        (401, {"code": 40100, "message": "unauthorized"}, "auth", False, True),
        (404, {"code": 40400}, "not_found", False, True),
        (429, {"code": 50003}, "rate_limited", True, False),
        (200, {"code": 50003}, "rate_limited", True, False),
        (402, {"code": 50009}, "quota_exceeded", False, True),
        (503, {}, "server_error", True, False),
        (400, {"code": 40000}, "request_rejected", False, False),
    ],
)
def test_naver_errors_are_classified(status, payload, kind, transient, abort):
    client = NaverRagClient("k", "svc", transport=_Transport((status, payload)))
    with pytest.raises(ProviderError) as error:
        client.search("q", 10)
    assert (error.value.kind, error.value.transient, error.value.abort) == (
        kind,
        transient,
        abort,
    )


@pytest.mark.parametrize(
    "payload",
    [{"data": {}}, {"data": {"results": [{"blockId": "GENERATOR"}]}}, ["x"]],
)
def test_naver_malformed_response_is_rejected(payload):
    client = NaverRagClient("k", "svc", transport=_Transport((200, payload)))
    with pytest.raises(ProviderError) as error:
        client.search("q", 10)
    assert error.value.kind == "malformed_response"


def test_client_requires_credentials_from_environment():
    with pytest.raises(ValueError, match="NAVER_RAG_API_KEY"):
        NaverRagClient.from_env({})


def test_rechunked_duplicates_keep_best_rank_and_are_not_padded():
    hits = [
        ProviderHit("doc-b", 2, 0.8, chunk="1"),
        ProviderHit("doc-a", 1, 0.9, chunk="0"),
        ProviderHit("doc-a", 3, 0.7, chunk="1"),
        ProviderHit("doc-b", 4, None, chunk="2"),
    ]
    rows, diagnostics = normalize_hits(hits, ID_MAP, 3)
    assert [row["original_document_id"] for row in rows] == ["doc-a", "doc-b"]
    assert [row["rank"] for row in rows] == [1, 2]
    assert rows[1]["provider_chunk"] == "1"
    assert diagnostics["duplicate_hits"] == 2
    assert diagnostics["short_of_top_k"] and diagnostics["unique_count"] == 2


def test_missing_ranks_and_scores_use_provider_order():
    rows, diagnostics = normalize_hits(
        [ProviderHit("doc-c"), ProviderHit("doc-a", rank=1)], ID_MAP, 2
    )
    assert [row["original_document_id"] for row in rows] == ["doc-c", "doc-a"]
    assert rows[0]["score"] is None and diagnostics["rank_source"] == "order"


@pytest.mark.parametrize(
    ("hit", "kind"),
    [
        (ProviderHit(None), "missing_result_id"),
        (ProviderHit(" "), "missing_result_id"),
        (ProviderHit("stranger"), "unmapped_id"),
        (ProviderHit("doc-a", original_id="doc-b"), "conflicting_original_id"),
    ],
)
def test_bad_ids_fail_instead_of_shifting_ranks(hit, kind):
    with pytest.raises(NormalizationError) as error:
        normalize_hits([ProviderHit("doc-a"), hit], ID_MAP, 10)
    assert error.value.kind == kind


def test_metadata_mapping_with_non_identity_provider_ids():
    rows, _ = normalize_hits(
        [ProviderHit("p-1", original_id="doc-a")], {"p-1": "doc-a"}, 10
    )
    assert rows[0]["original_document_id"] == "doc-a"
    assert rows[0]["provider_document_id"] == "p-1"


class _Provider:
    name = "fake"

    def __init__(self, script):
        self.script = script
        self.calls = []

    def describe(self):
        return {}

    def search(self, query, top_k):
        self.calls.append(query)
        outcome = self.script[query].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return ProviderResponse(
            tuple(ProviderHit(i) for i in outcome), raw={"q": query}
        )


def _queries(*ids):
    return [Query(i, i) for i in ids]


def test_transient_errors_retry_with_bounded_backoff(tmp_path):
    waits = []
    provider = _Provider(
        {
            "q1": [
                ProviderError(
                    "slow", kind="rate_limited", transient=True, retry_after=5
                ),
                ProviderError("timeout", kind="timeout", transient=True),
                ["doc-a"],
            ],
            "q2": [ProviderError("down", kind="server_error", transient=True)] * 4,
            "q3": [ProviderError("bad", kind="request_rejected")],
        }
    )
    records = run_external_queries(
        provider,
        _queries("q1", "q2", "q3"),
        tmp_path,
        metadata=METADATA,
        id_map=ID_MAP,
        sleeper=waits.append,
    )
    assert waits == [5, 4.0, 2.0, 4.0, 8.0]
    assert records["q1"]["status"] == "ok" and records["q1"]["attempts"] == 3
    assert records["q1"]["rate_limit_wait_seconds"] == 5
    assert records["q2"]["status"] == "failed" and records["q2"]["attempts"] == 4
    assert records["q3"]["error"]["kind"] == "request_rejected"
    assert records["q3"]["attempts"] == 1


def test_auth_failure_aborts_and_resume_skips_successes(tmp_path):
    provider = _Provider(
        {
            "q1": [["doc-a", "doc-b"]],
            "q2": [ProviderError("denied", kind="auth", abort=True), ["doc-c"]],
            "q3": [["stranger"], ["doc-b"]],
        }
    )
    with pytest.raises(RunAborted, match="auth"):
        run_external_queries(
            provider,
            _queries("q1", "q2", "q3"),
            tmp_path,
            metadata=METADATA,
            id_map=ID_MAP,
        )
    assert [r["query_id"] for r in read_jsonl(tmp_path / RECORDS_FILE)] == ["q1"]
    # 중단 중 잘린 마지막 줄은 무시하고 이어서 실행한다.
    with (tmp_path / RECORDS_FILE).open("a") as target:
        target.write('{"query_id": "q2", "sta')
    records = run_external_queries(
        provider, _queries("q1", "q2", "q3"), tmp_path, metadata=METADATA, id_map=ID_MAP
    )
    assert provider.calls == ["q1", "q2", "q2", "q3"]
    assert records["q2"]["results"][0]["original_document_id"] == "doc-c"
    assert records["q3"]["status"] == "failed"
    assert records["q3"]["error"]["kind"] == "unmapped_id"
    records = run_external_queries(
        provider, _queries("q1", "q2", "q3"), tmp_path, metadata=METADATA, id_map=ID_MAP
    )
    assert records["q3"]["status"] == "ok"


def test_resume_refuses_changed_run_conditions(tmp_path):
    provider = _Provider({"q1": [["doc-a"]]})
    run_external_queries(
        provider, _queries("q1"), tmp_path, metadata=METADATA, id_map=ID_MAP
    )
    with pytest.raises(ValueError, match="top_k"):
        run_external_queries(
            provider,
            _queries("q1"),
            tmp_path,
            metadata=METADATA | {"top_k": 5},
            id_map=ID_MAP,
        )


@pytest.mark.parametrize(
    "changed_settings",
    [
        {"service_id": "other-svc", "base_url": "https://fake.invalid"},
        {"service_id": "svc", "base_url": "https://changed.invalid"},
        {"service_id": "svc", "base_url": "https://fake.invalid", "k": 50},
    ],
)
def test_search_resume_refuses_changed_provider_settings(tmp_path, changed_settings):
    provider = _Provider({"q1": [["doc-a"]], "q2": [["doc-a"]]})
    run_external_queries(
        provider, _queries("q1"), tmp_path, metadata=METADATA, id_map=ID_MAP
    )
    with pytest.raises(ValueError, match="provider_settings"):
        run_external_queries(
            provider,
            _queries("q1", "q2"),
            tmp_path,
            metadata=METADATA | {"provider_settings": changed_settings},
            id_map=ID_MAP,
        )
    # 거부는 provider.search() 호출 전에 일어난다: q1만 이미 호출됐어야 한다.
    assert provider.calls == ["q1"]


class _IndexClient:
    name = "fake"

    def __init__(self, settings=None):
        self._settings = settings or {
            "service_id": "svc",
            "base_url": "https://fake.invalid",
        }
        self.calls = []

    def describe(self):
        return self._settings

    def add_document(self, document_id, text):
        self.calls.append(document_id)
        return {}


INDEX_IDENTITY = {
    "provider": "fake",
    "provider_settings": {"service_id": "svc", "base_url": "https://fake.invalid"},
    "dataset_manifest_sha256": "m",
    "corpus_sha256": "c",
}


def _documents(*ids):
    return [Document(document_id=i, text=f"text-{i}", title="") for i in ids]


def test_index_resume_with_same_identity_skips_completed_documents(tmp_path):
    client = _IndexClient()
    checkpoint = tmp_path / "indexed.jsonl"
    index_documents(
        client, _documents("d1", "d2", "d3"), checkpoint, run_identity=INDEX_IDENTITY
    )
    assert client.calls == ["d1", "d2", "d3"]
    client.calls.clear()
    # 같은 identity + 같은 document set으로 재개하면 이미 색인된 문서는 건너뛴다.
    checkpoint.write_text(
        "\n".join(checkpoint.read_text(encoding="utf-8").splitlines()[:2]) + "\n",
        encoding="utf-8",
    )
    index_documents(
        client, _documents("d1", "d2", "d3"), checkpoint, run_identity=INDEX_IDENTITY
    )
    assert client.calls == ["d3"]


def test_index_resume_refuses_changed_service_setting(tmp_path):
    client = _IndexClient()
    checkpoint = tmp_path / "indexed.jsonl"
    index_documents(client, _documents("d1"), checkpoint, run_identity=INDEX_IDENTITY)
    other = _IndexClient(
        {"service_id": "other-svc", "base_url": "https://fake.invalid"}
    )
    with pytest.raises(ValueError, match="provider_settings"):
        index_documents(
            other,
            _documents("d1", "d2"),
            checkpoint,
            run_identity=INDEX_IDENTITY | {"provider_settings": other.describe()},
        )
    # 거부는 add_document() 호출 전에 일어난다.
    assert other.calls == []


def test_index_resume_refuses_changed_document_set(tmp_path):
    client = _IndexClient()
    checkpoint = tmp_path / "indexed.jsonl"
    index_documents(
        client, _documents("d1", "d2"), checkpoint, run_identity=INDEX_IDENTITY
    )
    with pytest.raises(ValueError, match="document_set_sha256"):
        index_documents(
            client, _documents("d1", "d3"), checkpoint, run_identity=INDEX_IDENTITY
        )
    assert client.calls == ["d1", "d2"]


def test_terms_gate_requires_explicit_approval(tmp_path):
    with pytest.raises(ValueError, match="EXTERNAL_UPLOAD_REQUIRES_MANUAL_TERMS_CHECK"):
        require_terms_review(None, "naver_cloud_rag")
    record = {
        "provider": "naver_cloud_rag",
        "decision": "blocked",
        "reviewer": "someone",
        "reviewed_at": "2026-10-29",
        "evidence": ["https://example.invalid/terms"],
    }
    path = tmp_path / "review.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="승인"):
        require_terms_review(path, "naver_cloud_rag")
    path.write_text(json.dumps(record | {"decision": "approved"}))
    assert require_terms_review(path, "naver_cloud_rag")["decision"] == "approved"


def test_cli_refuses_live_commands_without_terms_review(tmp_path, capsys):
    from scripts.benchmark_commercial import main

    with pytest.raises(SystemExit):
        main(["search", "--output-dir", str(tmp_path / "out")])
    assert "EXTERNAL_UPLOAD_REQUIRES_MANUAL_TERMS_CHECK" in capsys.readouterr().err


def test_evaluation_uses_shared_qrels_and_counts_failures(tmp_path):
    pytest.importorskip("pyarrow")
    from data.bccard import prepare_bccard
    from evaluation.data import load_evaluation_dataset
    from evaluation.external import (
        evaluate_external_run,
        identity_id_map,
        run_metadata,
    )
    from evaluation.metrics import evaluate_query
    from retrievers.base import SearchResult
    from tests.test_bccard import _write

    data = tmp_path / "prepared"
    prepare_bccard(*_write(tmp_path / "raw"), data)
    dataset = load_evaluation_dataset(data, "test")
    first, second = dataset.queries
    positive = max(dataset.qrels[first.query_id], key=dataset.qrels[first.query_id].get)
    other = next(d.document_id for d in dataset.documents if d.document_id != positive)
    provider = _Provider(
        {
            first.text: [[other, positive, other]],
            second.text: [ProviderError("bad", kind="request_rejected")],
        }
    )
    id_map = identity_id_map(dataset.documents)
    run_dir = tmp_path / "run"
    run_external_queries(
        provider,
        dataset.queries,
        run_dir,
        metadata=run_metadata(provider, dataset, id_map, 10),
        id_map=id_map,
    )
    report = evaluate_external_run(run_dir, data)
    expected = evaluate_query(
        [SearchResult(other, 1, 0.0), SearchResult(positive, 2, 0.0)],
        dataset.qrels[first.query_id],
    )
    summary = report["slices"]["all"]
    assert summary["query_count"] == 2 and summary["failed"] == 1
    assert summary["ndcg@10"] == pytest.approx(expected["ndcg@10"] / 2)
    assert summary["p@1"] == 0.0
    assert report["queries"]["duplicate_hits_removed"] == 1
    assert report["queries"]["short_of_top_k"] == 1
    assert report["latency"]["request"]["sample_count"] == 1

# 검색 서비스와 FastAPI 연동

관련 Issue: Phase 2 (Retrieval API integration)
담당: Role 3 (Hybrid integration + FastAPI)

## 구조

```text
HTTP POST /search
      |
      v
   FastAPI (app/api.py)
      |
      v
SearchService (app/service.py)
      |
      +-- tfidf  -> TfidfRetriever      (구현됨, sparse extra)
      +-- bm25   -> BM25Retriever       (미병합, unavailable)
      +-- dense  -> DenseRetriever      (미병합, unavailable)
      +-- hybrid -> HybridRetriever     (BM25 + Dense + RRF, unavailable)
      |
      v
SearchResult -> SearchHit -> SearchResponse
```

서비스는 검색 알고리즘이나 RRF를 구현하지 않습니다. 요청을 선택된 retriever로 보내고 결과를 그대로 반환합니다.

## 초기화 순서

1. `import app.api`는 corpus, index, 모델을 읽지 않습니다. 테스트와 문서 생성 도구가 가볍게 import할 수 있습니다.
2. 서버가 시작되면 lifespan에서 `build_search_service(corpus_path)`를 한 번 호출합니다.
3. corpus를 한 번 읽고, 사용할 수 있는 retriever만 메모리에서 만듭니다.
4. 요청마다 인덱스를 다시 만들지 않습니다. Dense 임베딩도 요청 시점에 실행하지 않습니다.

테스트에서는 `create_app(service=...)`로 미리 만든 서비스를 주입하여 이 단계를 건너뜁니다.

## 설정

| 환경 변수 | 기본값 | 의미 |
| --- | --- | --- |
| `SEARCH_CORPUS_PATH` | `data/processed/corpus.jsonl` | 준비된 공통 corpus JSONL 경로 |

Dense 인덱스 경로와 장치 설정은 Role 2 구현이 병합된 뒤 추가합니다. 지금은 설정 항목을 만들지 않습니다.

## 실행

저장소 루트에서 실행합니다. 상대 경로 기본값이 현재 디렉터리를 기준으로 하기 때문입니다.

```bash
uv sync --locked --extra sparse
uv run --extra sparse uvicorn app.api:app
```

## 사용 가능한 검색 방식

| method | 상태 (이번 PR 기준) | 조건 |
| --- | --- | --- |
| `tfidf` | 사용 가능 | `corpus.jsonl`이 있고 sparse extra가 설치되어 있어야 함 |
| `bm25` | 미병합 (503) | BM25 retriever가 병합되면 `build_search_service`에 등록 |
| `dense` | 미병합 (503) | Dense retriever가 병합되면 `build_search_service`에 등록 |
| `hybrid` | 미병합 (503) | BM25와 Dense가 모두 있어야 함. TF-IDF 대체 경로 없음 |

Hybrid 기준선은 BM25 + Dense + RRF입니다. BM25가 없을 때 TF-IDF + Dense로 조용히 바꾸지 않습니다.

## 필요한 artifact

- `data/processed/corpus.jsonl` (TF-IDF에 필요)
- Dense 인덱스 (Dense 병합 후 필요)

재현 명령:

```bash
uv run --extra sparse python -m scripts.prepare_dataset --download
```

생성된 파일과 인덱스는 커밋하지 않습니다.

## 엔드포인트

### `GET /health`

프로세스가 살아 있으면 항상 `200`을 반환합니다. 검색 준비 상태는 확인하지 않습니다.

```json
{"status": "ok"}
```

### `GET /search/methods`

각 방식의 사용 가능 여부와 사용할 수 없는 이유를 반환합니다.

```json
{
  "methods": [
    {"method": "tfidf", "available": true, "reason": null},
    {"method": "bm25", "available": false, "reason": "BM25 retriever is not merged yet."}
  ]
}
```

### `POST /search`

요청:

```json
{
  "query": "대한민국의 수도는 어디인가요?",
  "method": "tfidf",
  "top_k": 5
}
```

- `query`: 필수. 공백만 있으면 안 됩니다.
- `method`: `tfidf`, `bm25`, `dense`, `hybrid` 중 하나. 기본값 `hybrid`.
- `top_k`: 양의 정수. 기본값 10. 문자열이나 bool은 허용하지 않습니다.

응답 (`200`):

```json
{
  "query": "대한민국의 수도는 어디인가요?",
  "method": "tfidf",
  "results": [
    {
      "document_id": "fixture-001#0",
      "rank": 1,
      "score": 0.41,
      "title": "서울",
      "snippet": "서울은 대한민국의 수도입니다."
    }
  ]
}
```

`score`는 방식마다 의미가 다릅니다. TF-IDF는 코사인 유사도, Hybrid는 RRF 순위 점수입니다. 서로 비교하지 마세요.

## 오류 응답

| 상태 | 조건 | 본문 |
| --- | --- | --- |
| `422` | 빈 query, 잘못된 method, `top_k` 0 이하 또는 정수가 아님 | FastAPI 검증 오류 |
| `503` | 요청한 방식이 설정되지 않았거나 서비스가 초기화되지 않음 | `{"detail": "<method> search is unavailable: <이유>"}` |
| `500` | 예상하지 못한 내부 오류 | 일반 메시지. 스택과 파일 경로는 응답에 없고 서버 로그에만 남음 |

503은 빈 결과로 바꾸지 않습니다. 사용할 수 없는 방식이 검색에 성공한 것처럼 보이지 않도록 합니다.

## 시작 실패 정책

| 상황 | 동작 |
| --- | --- |
| corpus 파일 없음 | 프로세스는 계속 실행됩니다. 모든 방식이 503이며, 이유에 준비 명령을 포함합니다. |
| corpus 형식 오류 | 위와 같습니다. 세부 내용은 서버 로그에만 남깁니다. |
| sklearn 없음 | TF-IDF만 503입니다. 설치 명령을 이유에 포함합니다. |
| BM25 또는 Dense 미병합 | 해당 방식과 Hybrid만 503입니다. |

선택 기능이 없다고 프로세스가 종료되지는 않습니다. 대신 상태 엔드포인트와 503 이유로 드러납니다.

## 지연 시간 관측

`POST /search`는 검색 호출 시간을 `time.perf_counter()`로 재고 INFO 로그에 `method`, `top_k`, `hits`, `latency_ms`를 남깁니다. 질의 본문은 로그에 남기지 않습니다.

이 값은 검색 실행 시간만 포함합니다. 인덱스 빌드, 모델 로딩, 첫 요청 준비 시간은 포함하지 않습니다. 최종 비교 수치는 Role A의 평가 러너가 기준입니다.

## 테스트

기본 테스트는 오프라인이며 모델이나 데이터셋을 다운로드하지 않습니다.

```bash
uv run --locked pytest
```

- `tests/test_service.py`: fake retriever로 라우팅, top_k, 순위, 메타데이터, RRF 연결, 503 조건을 검증합니다.
- `tests/test_search_api.py`: 주입한 fake 서비스로 FastAPI 계약, 422/503, 한국어 질의를 검증합니다.
- `tests/test_integration.py`: 실제 TF-IDF와 `tests/fixtures/ko_miracl_prepared_corpus.jsonl`로 lifespan 전체 경로를 검증합니다. sklearn이 없으면 건너뜁니다.

## 실제 데이터 스모크 테스트

전체 데이터셋이 준비된 환경에서 다음 순서로 확인합니다. 이 저장소에는 준비된 데이터가 없으므로 결과를 보고하지 않았습니다.

```bash
uv run --extra sparse python -m scripts.prepare_dataset --download
uv run --extra sparse uvicorn app.api:app
curl -X POST localhost:8000/search -H "Content-Type: application/json" \
  -d '{"query": "대한민국의 수도는 어디인가요?", "method": "tfidf", "top_k": 5}'
```

## 남은 제한

- BM25 미병합: `bm25`와 `hybrid`는 사용할 수 없습니다.
- Dense 미병합: `dense`와 `hybrid`는 사용할 수 없습니다. 연결 지점은 `build_search_service`입니다.
- 실제 데이터 스모크 테스트는 아직 실행하지 않았습니다.
- 인증, 요청 제한, 캐시는 없습니다. 이번 단계의 범위가 아닙니다.

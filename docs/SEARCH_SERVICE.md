# 검색 서비스와 FastAPI 연동

## 운영 흐름

1. **환경 준비**: [실행](#실행)의 명령으로 Sparse·Dense 의존성을 설치합니다.
2. **데이터 준비**: [데이터 준비 절차](DATA_PREPARATION.md)에 따라 공통 corpus를 만듭니다.
3. **인덱스 생성**: Dense·Hybrid를 쓰려면 [필요한 artifact](#필요한-artifact)의 명령으로 인덱스를 만듭니다.
4. **API 실행**: corpus·인덱스 경로를 [설정](#설정)하고 서버를 시작합니다.
5. **준비 상태 확인**: `GET /search/methods`에서 사용할 방식의 `available` 값을 확인합니다.
6. **UI 실행**: 별도 터미널에서 `uv run --no-sync streamlit run frontend/app.py`를 실행합니다.

UI의 API 주소는 `SEARCH_API_URL`로 설정하며 기본값은 `http://127.0.0.1:8000`입니다.
검색이 불가능하면 [시작 실패 정책](#시작-실패-정책)에 따라 원인을 해결한 뒤 서버를 재시작합니다.
`GET /health`는 프로세스 생존만 확인합니다. 평가 재현은 [최종 평가](FINAL_EVALUATION.md)를 참고하세요.

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
      +-- tfidf  -> TfidfRetriever      (sparse extra 필요)
      +-- bm25   -> BM25Retriever       (sparse extra 필요)
      +-- dense  -> DenseRetriever      (dense extra + 저장된 FAISS 인덱스 필요)
      +-- hybrid -> ScoreFusionRetriever (BM25 + Dense score fusion, 둘 다 있어야 함)
      |
      v
SearchResult -> SearchHit -> SearchResponse
```

`tfidf`/`bm25`는 corpus와 sparse extra, `dense`/`hybrid`는 추가로 dense extra와 저장된 FAISS 인덱스가 필요합니다.

`hybrid`의 내부 구현은 `fusion.score.ScoreFusionRetriever`
(Dense score에 query별 corpus 최대로 정규화한 BM25 score를 `DEFAULT_BM25_WEIGHT=0.05`만큼
더하는 방식)입니다. 기존 RRF(`fusion.hybrid.HybridRetriever`)는 evaluation baseline으로
남아 있습니다. 결합 규칙은 [SCORE_FUSION.md](SCORE_FUSION.md)를 참조하세요.

서비스는 검색 알고리즘 자체를 구현하지 않습니다. 요청을 선택된 retriever로 보내고 결과를 그대로 반환합니다.

## 초기화 순서

1. `import app.api`는 corpus, index, 모델을 읽지 않습니다. 테스트와 문서 생성 도구가 가볍게 import할 수 있습니다.
2. 서버가 시작되면 lifespan에서 `build_search_service(corpus_path)`를 한 번 호출합니다.
3. corpus를 한 번 읽고 retriever를 만듭니다. Dense는 저장 인덱스를 복원하고 warm-up 검색으로
   모델 로딩을 확인합니다. BM25와 Dense가 준비되면 `ScoreFusionRetriever`도 한 번 초기화합니다.
4. 요청마다 인덱스를 다시 만들지 않습니다. document embedding도 요청 시점에 실행하지 않고
   저장된 index/vector를 재사용합니다. query embedding은 Dense/Hybrid 요청마다 수행합니다.

테스트에서는 `create_app(service=...)`로 미리 만든 서비스를 주입하여 이 단계를 건너뜁니다.

## 설정

| 환경 변수 | 기본값 | 의미 |
| --- | --- | --- |
| `SEARCH_CORPUS_PATH` | `data/processed/corpus.jsonl` | 준비된 공통 corpus JSONL 경로 |
| `DENSE_INDEX_PATH` | 없음 | `scripts.build_index`로 저장한 Dense 인덱스 디렉터리. 설정하지 않으면 Dense는 `not_configured`(503) |
| `DENSE_DEVICE` | 자동 선택 | Dense 모델을 올릴 장치 (`cpu`, `cuda`, `cuda:0` 등). 선택 사항 |

## 실행

저장소 루트에서 실행합니다. 상대 경로 기본값이 현재 디렉터리를 기준으로 하기 때문입니다.

```bash
uv sync --locked --extra sparse --extra dense
export DENSE_INDEX_PATH=indexes/dense  # Dense/Hybrid를 쓰려면 설정
uv run --extra sparse --extra dense uvicorn app.api:app
```

Dense/Hybrid 없이 TF-IDF와 BM25만 쓰려면 `--extra sparse`만 설치하고 `DENSE_INDEX_PATH`를 설정하지 않으면 됩니다.

## 사용 가능한 검색 방식

| method | 상태 | 조건 |
| --- | --- | --- |
| `tfidf` | 사용 가능 | `corpus.jsonl`이 있고 sparse extra 설치 |
| `bm25` | 사용 가능 | `corpus.jsonl`이 있고 sparse extra 설치 (`bm25s`) |
| `dense` | `DENSE_INDEX_PATH` 설정 시 사용 가능 | dense extra 설치 + 저장된 FAISS 인덱스가 현재 corpus와 일치해야 함 |
| `hybrid` | `bm25`와 `dense`가 모두 사용 가능할 때 사용 가능 | BM25 + Dense score fusion. TF-IDF 대체 경로 없음 |

Hybrid는 BM25 + Dense score fusion입니다. BM25나 Dense가 없을 때 TF-IDF로 조용히 바꾸지 않습니다. 각 방식의 정확한 상태는 `state` 값(아래 `GET /search/methods`)으로 구분합니다: `available`, `not_implemented`, `not_configured`, `corpus_unavailable`, `dependency_missing`, `index_missing`, `load_failed`, `upstream_unavailable`.

## 필요한 artifact

- `data/processed/corpus.jsonl` (TF-IDF, BM25, Dense, Hybrid 공통)
- Dense를 쓰려면 저장된 FAISS 인덱스 디렉터리 (`metadata.json` + `index.faiss`)

재현 명령:

```bash
uv run --extra sparse python -m scripts.prepare_dataset --download
uv run --extra dense python -m scripts.build_index \
  --corpus data/processed/corpus.jsonl --index indexes/dense
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
    {"method": "tfidf", "available": true, "state": "available", "reason": null},
    {
      "method": "dense",
      "available": false,
      "state": "not_configured",
      "reason": "set DENSE_INDEX_PATH to a saved index directory."
    }
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

- `query`: 필수. 공백만 있으면 안 되며 최대 1,000자입니다.
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

`score`는 방식마다 의미가 다릅니다. TF-IDF는 코사인 유사도, Hybrid는 Dense + 정규화된 BM25의 fused score입니다. 서로 비교하지 마세요.

## 오류 응답

| 상태 | 조건 | 본문 |
| --- | --- | --- |
| `422` | 빈 query, 1,000자 초과 query, 잘못된 method, `top_k` 0 이하 또는 정수가 아님 | FastAPI 검증 오류 |
| `503` | 요청한 방식이 설정되지 않았거나 서비스가 초기화되지 않음 | `{"detail": "<method> search is unavailable: <이유>"}` |
| `500` | 예상하지 못한 내부 오류 | 일반 메시지. 스택과 파일 경로는 응답에 없고 서버 로그에만 남음 |

503은 빈 결과로 바꾸지 않습니다. 사용할 수 없는 방식이 검색에 성공한 것처럼 보이지 않도록 합니다.

## 시작 실패 정책

| 상황 | 동작 |
| --- | --- |
| corpus 파일 없음 | 프로세스는 계속 실행됩니다. 모든 방식이 503이며, 이유에 준비 명령을 포함합니다. |
| corpus 형식 오류 | 모든 방식이 `corpus_unavailable`로 503입니다. 세부 내용은 서버 로그에 남깁니다. |
| sklearn 없음 | TF-IDF·BM25는 `dependency_missing`, Hybrid는 `upstream_unavailable`로 503입니다. |
| `DENSE_INDEX_PATH` 미설정 | Dense는 `not_configured`, Hybrid는 `upstream_unavailable`로 503입니다. |
| Dense 인덱스 경로 없음/손상/의존성 없음 | Dense는 `index_missing`/`load_failed`/`dependency_missing`, Hybrid는 `upstream_unavailable`로 503입니다. |
| Dense 인덱스가 현재 corpus와 다름 | Dense는 `load_failed`, Hybrid는 `upstream_unavailable`로 503입니다. 인덱스를 임의로 다시 빌드하지 않습니다. |
| Dense는 정상이나 score fusion의 정규화 검증 실패 | Hybrid만 `load_failed`로 503입니다. TF-IDF·BM25·Dense와 health는 유지하며 검증 오류의 세부 내용은 서버 로그에 남깁니다. |

선택 기능이 없다고 프로세스가 종료되지는 않습니다. 대신 상태 엔드포인트와 503 이유로 드러납니다.

## 지연 시간 관측

`POST /search`는 검색 호출 시간을 `time.perf_counter()`로 재고 INFO 로그에 `method`, `top_k`, `hits`, `latency_ms`를 남깁니다. 질의 본문은 로그에 남기지 않습니다.

이 값은 검색 실행 시간만 포함합니다. 인덱스 빌드와 시작 시 모델 로딩은 제외합니다.
검색 방식별 비교는 [최종 평가](FINAL_EVALUATION.md)를 참조하세요.

## 테스트

기본 테스트는 오프라인이며 모델이나 데이터셋을 다운로드하지 않습니다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked --extra sparse --extra dense \
  pytest tests/test_service.py tests/test_service_build.py tests/test_search_api.py tests/test_integration.py
```

- `tests/test_service.py`, `tests/test_service_build.py`: fake retriever와 초기화 상태(test double)로 라우팅, top_k, 순위, 메타데이터, score fusion 연결, 각 `MethodState`를 검증합니다.
- `tests/test_search_api.py`: 주입한 fake 서비스로 FastAPI 계약, 422/503, 한국어 질의를 검증합니다.
- `tests/test_integration.py`: 실제 TF-IDF/BM25와 `tests/fixtures/ko_miracl_prepared_corpus.jsonl`로 lifespan 전체 경로를 검증합니다. 실제 Dense와 Hybrid(BM25+Dense score fusion)도 로컬 결정적 임베딩 함수로 FAISS 인덱스를 만들어 HTTP까지 검증합니다 (모델 다운로드 없음). sklearn이나 faiss가 없으면 해당 테스트는 건너뜁니다.

## 실제 데이터 스모크 테스트

실제 데이터와 모델로 확인하는 절차입니다.
MIRACL 1만 문서·저장 벡터와 실제 BGE-M3 질의 임베딩으로 확인한 API/UI 실행 결과는
[2026-10-20 검증 기록](validation/2026-10-20/REPORT.md#데모)에 있습니다.
해당 실행은 검증된 문서 벡터를 재사용했으며, 아래 문서 임베딩 생성 과정은 재실행하지 않았습니다.

```bash
uv run --extra sparse --extra dense python -m scripts.prepare_dataset --download
uv run --extra dense python -m scripts.build_index \
  --corpus data/processed/corpus.jsonl --index indexes/dense
DENSE_INDEX_PATH=indexes/dense uv run --extra sparse --extra dense uvicorn app.api:app
curl -X POST localhost:8000/search -H "Content-Type: application/json" \
  -d '{"query": "대한민국의 수도는 어디인가요?", "method": "hybrid", "top_k": 5}'
```

## 남은 제한

- UI 검증은 실제 API에 연결한 AppTest 기준이며 브라우저 수동 검증은 포함하지 않습니다. 준비된 데이터와 인덱스는 Git에 포함하지 않습니다.
- 인증, 요청 제한, 캐시는 없습니다. 이번 단계의 범위가 아닙니다.

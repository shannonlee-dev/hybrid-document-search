# 상용 RAG retrieval 비교

4차 #26. #25 BC Card corpus · test 질의 · qrels로 상용 한국어 RAG/search의 Top-K를 평가하고 로컬 방식과 비교합니다.
retrieval 비교만. 답변 생성 품질은 다루지 않습니다.

## 상태

`COMMERCIAL_RETRIEVAL_RESULT: NONE` (2026-10-29 기준)

| 대상 | 상태 |
| --- | --- |
| NAVER Cloud RAG | `READY / BLOCKED, NOT RUN`. adapter · 정규화 · 평가 구현 완료. 약관 미확인 · account/credential 없음 · 비용 unknown |
| Upstage File Search | `EXCLUDED`. chunk 단위 원본 ID mapping 불가, 문서 수 제한. 실행 실패가 아니라 현재 protocol 밖 |
| commercial result | none. 업로드 0건, provider 호출 0건, test 질의 0/1,000 |
| 로컬 행 | PR #28 결과 그대로 |
| Adaptive Hybrid | PR #30 병합 전까지 pending |

NAVER 경로는 순수 retriever와 같지 않은 **commercial retrieval pipeline**입니다(query rewrite, 필수 답변 생성, provider chunking, 비공개 embedding · reranking).
같은 qrels로 평가하지만 drop-in retriever 비교로 보지 않습니다. 상용 API latency는 로컬 캐시 vector CPU latency와 따로 보고합니다.

## Provider 가능성 검토

공식 문서만 근거로 사용. 문서에 없는 항목은 unknown. 확인일 2026-10-29.

| 항목 | NAVER Cloud RAG | Upstage File Search |
| --- | --- | --- |
| 현재 제품 / API | RAG (VPC), RAG API v1 `https://kr-pub-gateway.rag.naverncp.com/api/v1/svc/{serviceId}` | File Search (Beta), `https://api.upstage.ai/v2/vector_stores` |
| 접근 조건 | VPC 환경, 콘솔에서 이용 신청 필요 | API key. Beta, 사양이 예고 없이 바뀔 수 있음 |
| 프로그램 색인 | 문서 추가 · ID 지정 추가(`POST /doc/{docId}`, `body` 텍스트) · 파일 추가 | Files API 업로드 후 vector store에 file 추가. raw text 업로드 문서 없음 |
| 한국어 | 지원(한국어 · 영어 · 일본어) | 명시 없음(hwp/hwpx 형식만 지원 목록에 있음) |
| chunking | 색인 단계에 CHUNK 있음. 검색 요청의 RETRIEVAL `chunkSize` 256–4096(기본 512), `overlapSize` 0–chunkSize/2(기본 128). 끄는 옵션 없음 | `chunking_strategy`는 `auto`만. 크기 · overlap 비공개 |
| 원본 ID 보존 | `docId` 직접 지정(영문 · 숫자 · `-` · `_`). `extra.{field}` 사용자 metadata | file `attributes`(16키). 검색 결과에 attributes · chunk ID 노출 문서 없음 |
| 결과 노출 | RETRIEVAL block: `docId`, `score`(0–1), `rank`, `text`, `chunk_no`, offset | `score`, `filename`, `content` |
| retrieval 전용 호출 | **없음**. `POST /conversation`이 QUERY_REWRITE → RETRIEVAL → GENERATOR 전체 실행, `verbose: true`일 때 RETRIEVAL 결과 노출 | `POST /vector_stores/{id}/search` (retrieval 전용) |
| 최대 Top-K | `k` 기본 5, 최대값 문서 없음 | `max_num_results` 1–20 |
| 개수 · 크기 제한 | 서비스 999개, Object Storage 파일 100MB. 문서 수 · 본문 길이 제한 문서 없음 | vector store 50개/user, **file 500개/store**, batch 50 |
| 요청 제한 | 수치 없음. 50003 Too many requests, 50009 quota exceeded | 문서 없음 |
| 가격 | RAG 자체 요금 공개 문서 없음(unknown). 답변 생성에 CLOVA Studio 연동 시 추가 요금 가능 | 가격표상 storage · processing · search Free(beta) |
| 삭제 | `DELETE /doc/{docId}`, 색인 상태 `GET /doc/{docId}/status`, `GET /doc-count` | files delete 있음. vector store file 삭제 문서 없음 |
| 보관 · 학습 사용 | 가이드는 RAG 서비스 이용약관(`ncloud.com/policy/terms/rag`)을 참조. 약관 본문이 JS 렌더링이라 확인 불가 → unknown | beta 공지: 민감 · 기밀 데이터 업로드 자제, 정식 출시 시 약관 미동의 사용자 데이터 삭제 가능. 학습 사용 unknown |

근거: [RAG 개요](https://guide.ncloud-docs.com/docs/rag-overview), [RAG 사양](https://guide.ncloud-docs.com/docs/rag-spec),
[RAG 사용 준비](https://guide.ncloud-docs.com/docs/rag-procedure), [RAG API 개요](https://api.ncloud-docs.com/docs/rag-overview),
[대화 생성](https://api.ncloud-docs.com/docs/rag-create-chat), [문서 ID 지정 추가](https://api.ncloud-docs.com/docs/rag-add-doc-by-id),
[색인 상태](https://api.ncloud-docs.com/docs/rag-get-status), [문서 삭제](https://api.ncloud-docs.com/docs/rag-delete-doc),
[Upstage File Search](https://console.upstage.ai/docs/capabilities/search/file-search), [Upstage Files](https://console.upstage.ai/docs/agents/files),
[Upstage 가격](https://www.upstage.ai/pricing/api).
NAVER 가이드 페이지 일부는 `stale` 표시가 있습니다.

### NAVER: 기술적으로 가능, 실행 차단

- ID 왕복은 가능. `docId` = `source_chunk_id`, BC Card ID 36,281개 모두 허용 문자(최장 59자)만 씀.
- 차단 사유
  1. **약관 미확인.** RAG 서비스 이용약관 본문을 확인하지 못했습니다(보관 · 학습 사용 · GENERATOR가 OpenAI/Anthropic일 때의 제3자 전송 포함).
  2. **credential 없음.** `NAVER_RAG_API_KEY`, `NAVER_RAG_SERVICE_ID` 없음. 계정 · VPC · 이용 신청은 사람 결정.
  3. **비용 unknown.** retrieval 전용 호출이 없어 질의마다 답변 생성이 같이 실행됩니다. RAG · 생성 단가가 공개 문서에 없음.
- retrieval 비교 자체는 가능: RETRIEVAL block에 docId · rank · score가 있어 qrels 평가에 충분합니다.
  다만 QUERY_REWRITE가 질의를 바꿀 수 있고 API로 끌 수 없습니다. 조건 차이로 기록합니다.

### Upstage: 제외

- 500 files/store × 50 stores = 최대 25,000 file < BC Card 36,281 chunk. 1 chunk = 1 file 불가.
- 여러 chunk를 한 file로 묶으면 `auto` re-chunking 뒤 검색 결과가 file 정보만 주므로 chunk ID는 본문 fuzzy matching으로만 복원 가능. qrels가 chunk 단위라 비교 불가.
- beta 공지가 민감 데이터 업로드를 말리고 학습 사용 조건은 unknown.
- 이 조건에서는 adapter를 만들지 않았습니다. file 단위 attributes가 검색 결과에 노출되고 store당 file 제한이 풀리면 같은 정규화 계층에 adapter만 추가하면 됩니다.

## 업로드 gate

`EXTERNAL_UPLOAD_REQUIRES_MANUAL_TERMS_CHECK` 유지.

- dataset 쪽: 두 BC Card dataset 모두 CC-BY-4.0, card에 외부 처리 금지 문구 없음([#25 기록](BC_CARD_DATASET.md#external-service-note-26)).
  CC-BY-4.0은 provider의 보관 · 학습 조건을 대신 판단해 주지 않습니다.
- provider 쪽: 위 표의 보관 · 학습 사용이 unknown → 업로드하지 않음.
- 코드: 외부로 데이터를 보내는 `index` · `smoke` · `search`는 `--terms-review` 기록이 `decision: approved`일 때만 실행. 기록은 run 메타데이터에 같이 저장.

검토 기록 형식(로컬, Git 제외 경로 권장):

```json
{
  "provider": "naver_cloud_rag",
  "decision": "approved",
  "reviewer": "<이름>",
  "reviewed_at": "<YYYY-MM-DD>",
  "evidence": ["<확인한 약관 URL과 조항>"],
  "attribution": "BC Card AX Team, BCAI Finance Kor Embedding Pair / Triplet (2026), Hugging Face"
}
```

## 비교 조건 (live 실행 시)

결과를 보기 전에 문서만 보고 고정했습니다. test qrels로 provider 설정을 고르지 않습니다.

| 항목 | 설정 |
| --- | --- |
| corpus | #25 `corpus.jsonl` 36,281개, `positive` 본문만, title 미색인, 텍스트 변환 없음 |
| 색인 단위 | 1 `source_chunk_id` = 1 NAVER 문서. `docId` = `source_chunk_id`, `extra.source_chunk_id`에 같은 값 |
| chunking | RETRIEVAL `chunkSize` 4096(문서상 최대), `overlapSize` 0. BC Card chunk 최장 2,307자라 대부분 재분할 없을 것으로 예상하지만 provider 단위(token/문자)가 문서에 없어 보장 못 함 |
| 질의 | 공식 test 1,000개 전체를 원문 그대로 순서대로. primary = 비합성 721, secondary = 전체 1,000 |
| Top-K | `k = 10` 요청. 부족분 padding 없음 |
| GENERATOR | 서비스 기본값. 끌 수 없음. 생성 결과는 평가에 쓰지 않음 |
| 동시성 | 1 (순차) |

### ID mapping · 정규화

- provider hit → `docId` → ID map(`docId` → `source_chunk_id`). metadata `extra.source_chunk_id`가 오면 map과 일치해야 함.
- 모든 hit에 `rank`가 있으면 rank 순, 아니면 응답 순서.
- re-chunking으로 같은 원본이 여러 번 나오면 **가장 높은 순위 하나만** 남김. 중복 수는 기록. 여러 sub-chunk를 여러 정답으로 세지 않음.
- 중복 제거 후 10개 미만이면 그대로 둠(`short_of_top_k`).
- ID 없음 · map에 없는 ID · metadata 불일치 → 해당 질의 실패. hit를 빼고 순위를 당기지 않음. 본문 fuzzy matching 없음.
- score 없음 허용(`null`). 지표는 rank만 사용.

### 실패 처리

- timeout · 네트워크 · 5xx · 429/50003: 최대 4회, 2 · 4 · 8초 backoff(최대 60초), `Retry-After` 우선.
- 401/403 · 404(서비스 없음) · 50009 quota: 즉시 run 중단, 재시도 안 함.
- 그 외 4xx · 응답 형식 오류: 해당 질의 실패로 기록.
- 질의마다 JSONL checkpoint. 재실행 시 성공 질의는 건너뛰고 실패 질의만 다시 시도. 실행 조건(provider · split · top_k · manifest · query set · ID map hash)이 바뀌면 재개 거부.
- 평가에서 실패 · 누락 질의는 빈 결과(0점)로 분모에 남기고 slice별 `failed` · `missing` 수를 같이 보고.

### 지표

#25 qrels와 공통 지표 함수(`evaluation.metrics.evaluate_query`, `evaluation.enterprise.query_metrics` · `summarize` · `slice_definitions`)를 그대로 사용.
P@1 · Recall@10 · MRR@10 · nDCG@10(이진), graded nDCG@10.

## 결과

held-out test. 로컬 행은 [#25 결과](BC_CARD_DATASET.md#baseline-results) 그대로.

### Primary: 비합성 test (721)

| 방식 | P@1 | Recall@10 | MRR@10 | nDCG@10 | 실패 / 누락 |
| --- | ---: | ---: | ---: | ---: | --- |
| BM25 | 0.4189 | 0.7406 | 0.5232 | 0.5755 | – |
| Dense (BGE-M3) | 0.6560 | 0.8988 | 0.7460 | 0.7835 | – |
| Fixed Score Fusion | 0.6852 | 0.8946 | 0.7615 | 0.7940 | – |
| Adaptive Hybrid (#24) | 대기 | | | | PR #30 미병합 |
| NAVER Cloud RAG | 미실행 | | | | 약관 · credential 차단 |
| Upstage File Search | 제외 | | | | chunk ID mapping 불가 |

### Secondary: 전체 test (1,000)

| 방식 | P@1 | Recall@10 | MRR@10 | nDCG@10 | 실패 / 누락 |
| --- | ---: | ---: | ---: | ---: | --- |
| BM25 | 0.4030 | 0.6044 | 0.5013 | 0.4887 | – |
| Dense (BGE-M3) | 0.5800 | 0.7264 | 0.6699 | 0.6459 | – |
| Fixed Score Fusion | 0.6100 | 0.7280 | 0.6881 | 0.6583 | – |
| Adaptive Hybrid (#24) | 대기 | | | | PR #30 미병합 |
| NAVER Cloud RAG | 미실행 | | | | 약관 · credential 차단 |
| Upstage File Search | 제외 | | | | chunk ID mapping 불가 |

### Latency · 비용 범위

같은 열에 두지 않습니다.

| 대상 | 측정 범위 |
| --- | --- |
| 로컬 3방식 | 캐시 vector CPU 검색 · 결합, 2 threads, query encoding 제외(#25) |
| 상용 | 질의당 HTTPS 요청 end-to-end(네트워크 · query rewrite · retrieval · 답변 생성 포함), 최종 시도 기준 mean / P95, 동시성 1. 재시도 수 · rate-limit 대기 별도 |

비용은 계산하지 않고 provider 콘솔 청구액을 기록할 예정. 응답의 GENERATOR token 수는 `usage_totals`로 합산합니다.
색인 시간은 업로드 시간과 `doc-count`가 corpus 수에 도달할 때까지의 대기 시간으로 나눠 기록합니다.

## 조건 차이 · 한계

- **답변 생성 강제:** NAVER는 retrieval만 호출할 수 없어 질의마다 LLM이 돌고 latency · 비용에 포함됩니다.
- **query rewrite:** provider가 질의를 바꿀 수 있고 끌 수 없음. 로컬 방식은 원문 질의.
- **re-chunking:** 색인 CHUNK 단계는 끌 수 없음. 같은 원본의 sub-chunk 중복은 best rank로 합치지만, 중복이 Top-10 자리를 차지하면 고유 문서 10개 미만이 될 수 있습니다(`short_of_top_k`).
- **비공개 모델:** embedding · hybrid 방식 · reranker 비공개. 문서 예시의 rank 1 score가 0.0417이라 cosine이 아닌 다른 결합 점수일 수 있으나 확인 불가.
- **score:** 0–1 범위만 공개, 로컬 점수와 비교 불가. 평가는 rank만 사용.
- **Top-K 의미:** provider Top-K는 chunk 단위. 중복 제거 후 원본 문서 단위로 평가.
- **답변 평가 없음:** BC Card에는 reference answer가 없어 answer correctness를 주장하지 않습니다.

## 재현

offline 평가는 credential 없이 동작합니다. live 단계는 gate 통과 후에만.

```bash
export NAVER_RAG_API_KEY=...        # 커밋 금지
export NAVER_RAG_SERVICE_ID=...
# 선택: NAVER_RAG_BASE_URL (기본 https://kr-pub-gateway.rag.naverncp.com)

# 1. ID 왕복 smoke: validation 문서 3개 색인 → 본문으로 검색 → 같은 ID 확인(품질 아님)
uv run --locked python -m scripts.benchmark_commercial smoke \
  --terms-review <review.json> --documents 3 --output-dir artifacts/commercial/naver-smoke
# 2. 전체 corpus 색인(재개 가능)
uv run --locked python -m scripts.benchmark_commercial index \
  --terms-review <review.json> --output-dir artifacts/commercial/naver-index
# 3. 작은 질의 subset → mapping 확인 후 같은 output-dir로 전체 1,000 재개
uv run --locked python -m scripts.benchmark_commercial search --split test --limit 20 \
  --terms-review <review.json> --output-dir artifacts/commercial/naver-test
uv run --locked python -m scripts.benchmark_commercial search --split test \
  --terms-review <review.json> --output-dir artifacts/commercial/naver-test
# 4. offline 평가(로컬 baseline은 docs/experiments/bccard/test-summary.json에서 읽음)
uv run --locked python -m scripts.benchmark_commercial evaluate \
  --run-dir artifacts/commercial/naver-test --output-dir artifacts/commercial/naver-test/eval
```

- `artifacts/`는 Git 제외. `raw-responses.jsonl`은 corpus 본문을 포함하므로 커밋하지 않습니다.
- 커밋 대상: `summary.json`(slice 집계만, 질의별 결과 없음), `run.json`(provider 설정 · hash · 검토 기록).
- 3단계 `--limit 20` subset은 mapping · 실패율 확인용입니다. subset 지표로 설정을 바꾸지 않습니다.

### Offline 확인

- `tests/test_commercial.py`: NAVER 응답 파싱 · 오류 분류, re-chunk 중복 → best rank, score 없음, Top-K 부족, ID 없음 · 미등록 · 불일치,
  재시도 · 중단 · checkpoint 재개(잘린 마지막 줄 포함), 약관 gate, 공통 지표 · 실패 질의 분모 유지. credential 불필요.
- 실제 prepared data로 파이프라인 확인: #25 Dense test run을 가짜 provider로 재생(원본마다 sub-chunk 중복 1개 추가, rank 없음)하고
  `evaluate_external_run`으로 평가하면 721 · 1,000 모두 Dense 지표와 차이 0, 중복 10,000개 제거.

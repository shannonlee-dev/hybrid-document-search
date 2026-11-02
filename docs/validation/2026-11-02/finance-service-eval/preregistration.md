# Finance encoder 서비스 경로 검증 사전 등록

작성: 2026-11-02, 구현과 측정 전. 기준 main: `4ce79a1`(PR #41 병합).
이 문서를 커밋한 뒤 실험을 시작함. 이후 조건을 바꾸면 변경 내역과 이유를 결과 문서에 따로 남김.

## 질문

1. Finance Seed43이 실제 Hybrid 검색 경로에서도 원본 BGE-M3 대비 개선을 유지하는가.
2. 그 개선에 드는 latency와 메모리 비용은 어느 정도인가.
3. 금융 밖 데이터셋에서도 성능을 유지하는가.

## 고정 입력

- Encoder: 원본 BAAI/bge-m3 `5617a9f6`(로컬 snapshot, 파일 해시 10개는 PR #41 protocol과 동일).
- Finance Seed43: PR #41 재현 run의 선택 checkpoint, step 4096, SHA-256 `fbbd5ed0c4d82c42745e21afb54c9a43000d38d39a975595b61b52f314e42971`. 마지막 4 block만 바뀜. PR #40 checkpoint(`d36f5c66…`)는 로컬에 없어서 쓰지 않음.
- 문서 vector: 두 encoder 모두 원본 BGE-M3 FP32 문서 vector를 그대로 씀. Query 쪽만 바뀜.
  - BC Card: 원본 cache `document-vectors.npy` `4d5f1c8d…`(release `finance-encoder-inputs-v1`).
  - Ko-MIRACL 10k, Ko-StrategyQA: `benchmark-inputs-v1` cache(`prepare_score_fusion_cache.py --verify-only` 통과한 파일).
- BM25: `char_wb` 2–4 gram, k1 1.5, b .75, Lucene. 서비스 `BM25Retriever` 그대로.
- 학습, checkpoint 선택, α 선택은 다시 하지 않음. Test나 외부 데이터 결과를 보고 어떤 설정도 바꾸지 않음.

## 서비스 경로와 adapter

서비스는 `app.service.build_search_service` → `SearchService` → retriever, API는 `POST /search`.
- Dense 인덱스: 위 문서 vector로 FAISS `IndexFlatIP`를 만들어 `DenseRetriever.save` 형식으로 저장하고, 서비스가 `DenseRetriever.load`로 읽게 함.
- Query encoding: 두 encoder 모두 서비스 `DenseEmbedder` 경로 그대로 FP32, `normalize_embeddings=True`, prompt 없음, `max_seq_length` 8192, 요청당 query 1개.
- Finance encoder는 opt-in adapter로만 붙임. 원본 snapshot을 로컬에서 읽고 checkpoint SHA를 확인한 뒤 마지막 4 block에 load함. 설정이 없으면 서비스 동작은 지금과 같음.

## 조건

BC Card Test primary 721개 query, `top_k=10`.

| ID | Encoder | 검색 | 역할 |
|---|---|---|---|
| A-bge | 원본 BGE | 서비스 Hybrid: 전체 corpus `Dense + 0.05×BM25/max(BM25)`, 동점은 document_id 순 | co-primary A |
| A-fin | Finance Seed43 | 같음 | co-primary A |
| B-bge | 원본 BGE | α=.84 Fixed: Dense/BM25 Top100 union, union min-max, `α·Dense+(1−α)·BM25`, 동점은 corpus 순 | co-primary B |
| B-fin | Finance Seed43 | 같음, α=.84 | co-primary B |
| S-fin89 | Finance Seed43 | α=.89 Fixed(PR #41 Validation 선택값) | secondary |
| D-bge | 원본 BGE | Dense-only(서비스 FAISS 검색) | secondary |
| D-fin | Finance Seed43 | Dense-only | secondary |

α=.84 Fixed는 서비스에 없는 결합식이라 `fusion/` adapter retriever로 구현함. API 경로로 부를 때는 `SearchService`의 `hybrid` 자리에 이 retriever를 주입함. API schema와 기본 동작은 바꾸지 않음.

## 지표와 판정

지표는 P@1, Recall@5, Recall@10, MRR@10, nDCG@10(binary, relevance > 0, `evaluation.metrics`).
Paired bootstrap은 Test query 단위로 돌림. Seed 20261102, 20,000 resamples, percentile 95% CI.

Q1 판정은 co-primary A와 B 각각에서 Finance − BGE nDCG@10으로 함.
- 95% CI 하한 > 0이면 유지됨.
- CI가 0을 포함하면 개선이 확인되지 않음.
- CI 상한 < 0이면 퇴보.
- 두 co-primary를 따로 보고하고 하나로 합치지 않음. 다중 비교 보정은 하지 않으며, 그 점을 결과에 적음.

나머지 지표와 secondary 조건은 기술 통계와 CI만 보고함. 퇴보한 query도 분석하되 그것으로 아무것도 바꾸지 않음.

## 경로 일치 검증 (실패하면 해당 결과를 쓰지 않고 중단)

1. Adapter checkpoint load: 같은 checkpoint를 BF16 autocast·512 tokens·batch 16으로 Validation 인코딩한 결과가 PR #41 `selected-validation-vectors.npy`와 bitwise 같아야 함.
2. α Fixed adapter: PR #41 BF16 Test vector를 넣으면 PR #41 per-query top10과 네 지표가 721개 query 모두에서 같아야 함(baseline·seed43, α .84/.83/.89 중 PR #41에 있는 조합).
3. 서비스 Hybrid 경로: cached FP32 BGE query vector를 넣으면 main canonical `score_fusion` per-query 결과(`docs/experiments/bccard/test-summary.json` 기준)와 같아야 함.
4. API 경로: `POST /search`로 받은 top10이 같은 retriever 직접 호출 결과와 모든 query에서 같아야 함.

FP32 서비스 인코딩과 PR #41 BF16 vector의 차이(cosine, top10 일치, 지표 차이)는 기술적으로 기록만 함.

## 비용 측정

정확도 실험과 분리해서 돌림. GPU0 하나만 쓰고, 측정 전후로 `nvidia-smi`에서 다른 프로세스와 GPU1 사용률을 기록함. 다른 작업이 돌고 있으면 그 사실과 함께 보고함.

- 구간: query encoding(`encode_query`), Dense scoring(서비스 Hybrid 내부 NumPy 전체 내적, Dense-only는 FAISS 검색), BM25 `score_all`, fusion/ranking, retriever 전체, HTTP 요청 전체(uvicorn localhost, httpx).
- 구간 측정 코드는 서비스와 같은 함수를 단계별로 호출하고, 그 결과 순위가 서비스 결과와 같은지 확인함.
- Warm-up은 Validation query 50개. 측정은 Test 721개 × 3회, 고정 순서, 동시 요청 1. 처리량은 HTTP 동시 요청 1과 4에서 측정함.
- Cold start는 새 프로세스에서 `build_search_service` 완료까지 시간과 첫 query latency를 encoder별 3회 측정함.
- 메모리는 `torch.cuda.max_memory_allocated`, `nvidia-smi` 프로세스 사용량, `/proc` VmRSS/VmHWM을 기록함.
- 평균, P50, P95, P99와 분포를 보고함. 판정 기준은 두지 않고 기술 통계로만 보고함.

## 외부 일반화

Ko-MIRACL dev 50개, Ko-StrategyQA 592개. 기존 score fusion 평가와 같은 query 집합과 문서 vector를 씀.
- Primary: Dense-only와 서비스 Hybrid(.05) 각각에서 Finance − BGE.
- Secondary: α=.84 Fixed. 이 α는 BC Card Validation에서 고른 값이라 외부 데이터에 맞춘 값이 아님.
- 판정 규칙은 Q1과 같은 CI 규칙을 데이터셋별로 적용함. 데이터셋을 합쳐 하나의 결론을 내지 않음.
- Ko-MIRACL은 50개라 검정력이 낮음. CI가 0을 포함하면 결론을 내리지 않음.
- 경로 확인: cached BGE query vector로 기존 `score_0.05`·`dense` 지표가 재현되는지 확인하고, live FP32 BGE query vector와 cached vector의 cosine을 기록함.

## 하지 않는 것

재학습, α·checkpoint 재선택, Test 기반 튜닝, 응답 생성 단계 추가, 서비스 기본 동작 변경, checkpoint·vector·대형 데이터 커밋.

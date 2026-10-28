# 4차 개발 분담과 Git 규칙

4차는 실제 기업 특화 한국어 데이터에서 query별로 Sparse/Dense 비중을 바꾸는 경량 Adaptive Hybrid를 검증하고, 같은 데이터로 상용 한국어 RAG/search 서비스와 비교한다.

## 왜 4차인가

3차 최종 평가(Ko-MIRACL, nDCG@10) 결과다. 수치는 [최신 평가 결과](BENCHMARK_RESULTS.md)를 따른다.

| 방식 | nDCG@10 |
| --- | ---: |
| TF-IDF | 0.5253 |
| BM25 | 0.5939 |
| Dense (BGE-M3) | 0.8549 |
| RRF Hybrid | 0.7844 |
| Fixed Score Fusion | 0.8649 |

의미 중심 benchmark에서는 Dense가 이미 강했고, RRF처럼 단순 결합하면 Dense보다 낮아졌다. 고정 weight Score Fusion만 소폭 개선했다.
[RAGAS 탐색 실험(PR #23)](https://github.com/shannonlee-dev/hybrid-document-search/pull/23)에서는 이 작은 retrieval 개선이 측정 가능한 답변 품질 개선으로 이어지지 않았다.

그래서 4차는 고정 결합 공식을 더 늘리지 않고, 경량 router가 query 특성에 따라 다른 Sparse/Dense 비중을 고를 수 있는지와 그 동작이 기업 데이터·상용 서비스 비교에서도 유지되는지를 본다.

- Query-Adaptive Hybrid Search(DOI [`10.3390/make8040091`](https://doi.org/10.3390/make8040091))의 query별 Sparse/Dense weighting 아이디어만 참고한다.
  full QDAP 재현, 101-bin alpha predictor, antagonist training, Dense encoder 재학습은 하지 않는다.
- 개선을 가정하지 않는다. 미개선·악화 결과도 그대로 보고한다.

## 반가희 — 기업 특화 데이터셋과 최종 평가

- Issue: [#25](https://github.com/shannonlee-dev/hybrid-document-search/issues/25) · 브랜치: `feat/a-enterprise-evaluation`
- 3차 선행 작업(완료): #19 / PR #20 — 공통 평가 파이프라인, 최종 결과 보고

| 작업 | 내용 |
| --- | --- |
| 데이터셋 통합 | `BCCard/BCAI-Finance-Kor-Embedding-Pair` / `-Triplet`의 revision, schema, split, corpus 구성, qrels, license를 실제 metadata로 확인하고 기존 JSONL 계약으로 변환 |
| Split 관리 | train / validation / held-out test 분리, ID 중복·누출 검증, manifest 기록. 실제 구조와 다르면 임의 split을 만들지 않음 |
| Held-out 평가 | method 결과 확인 전 primary slice 고정. direct / synthetic 구분은 metadata가 뒷받침할 때만 |
| 공통 지표 | Recall@10, MRR@10, nDCG@10, P@1, latency mean / P95 (#19 파이프라인 재사용) |
| 최종 비교 | BM25 · Dense · Fixed Score Fusion · Adaptive Hybrid · 상용 서비스 비교표와 실패 사례 분석 |

router 구현, 상용 API 연동, credential, 외부 provider 설정은 담당하지 않는다.

## 이상헌 — 경량 Adaptive Hybrid router

- Issue: [#24](https://github.com/shannonlee-dev/hybrid-document-search/issues/24) · 브랜치: `feat/b-adaptive-hybrid-router`
- 선행 작업: #17 / PR #18 — Fixed Score Fusion 구현과 분석

| 작업 | 내용 |
| --- | --- |
| 세 가지 mode | `keyword-heavy` / `balanced` / `semantic-heavy` (잠정). 기존 Score Fusion 재사용, weight는 train/validation으로 정한 뒤 고정 |
| Query feature | query 길이, 숫자 비율, 한글/영문 비율, BM25 top score·gap, Dense top similarity·gap, Sparse/Dense Top-K overlap |
| 경량 classifier | Logistic Regression, shallow tree / boosting. 매우 작은 MLP는 근거가 있을 때만 |
| 선택 규칙 | train으로 label 생성·학습, validation으로 선택, test 기반 튜닝 금지 |
| Oracle 상한 | 세 mode 중 query별 최고 mode. 평가용 상한으로만 사용 |
| 비용 분석 | 총 latency와 feature 추출·router 추론 overhead를 따로 기록 |

## 백동재 — 상용 RAG 외부 비교

- Issue: [#26](https://github.com/shannonlee-dev/hybrid-document-search/issues/26) · 브랜치: `feat/c-commercial-rag-benchmark`
- 3차 선행 작업(완료): #21 / PR #22 — 최종 서비스 통합과 E2E
- 탐색 참고: PR #23 — RAGAS downstream 실험(negative/qualifying result, 병합 요구사항 아님)

| 작업 | 내용 |
| --- | --- |
| 사전 확인 | 데이터 업로드 가능 여부(license·약관), 접근성, 제한, 비용, 한국어 지원, 인덱싱 동작, Top-K 노출 여부 |
| 대상 서비스 | NAVER Cloud RAG (1차), Upstage File Search (가능하면 2차) |
| 결과 정규화 | 같은 corpus 인덱싱, 원본 ID mapping 보존, 외부 Top-K를 프로젝트 평가 형식으로 변환 |
| 조건 차이 기록 | re-chunking, 비공개 embedding, 숨겨진 reranking, score 비노출, Top-K 의미 차이 |
| 답변 비교 | retrieval 비교 뒤 선택 사항. 정확도는 신뢰할 수 있는 reference answer가 있을 때만, Groundedness / Context Relevancy는 보조 지표 |

새 RAG 서비스를 만들지 않고, 상용 서비스가 더 낫다고 가정하지 않는다. credential은 커밋하지 않는다.

## 작업 순서

1. #25에서 데이터셋을 확인하고 split을 고정한다. #24, #26의 입력이 된다.
2. #24 router를 train/validation으로 확정·고정한 뒤 test로 평가한다.
3. #26은 사전 확인을 거쳐 쓸 수 있는 서비스만 인덱싱하고 결과를 정규화한다.
4. #25에서 같은 query / qrels로 최종 비교표를 만든다.

## Git 규칙

공통 규칙은 [CONTRIBUTING.md](../CONTRIBUTING.md)를 따른다. 4차에서 추가로 지킬 것:

- 브랜치는 최신 `main`에서 새로 만든다. 3차 브랜치는 재사용하지 않는다.
- PR은 `Implements #번호` / `Closes #번호`로 Issue를 연결한다.
- 공통 인터페이스·스키마·의존성을 바꾸거나 다른 담당 영역에 영향을 줄 때는 먼저 공유한다.
- 머지 전 `ruff check`, `ruff format --check`, `pytest`를 통과하고 담당 외 팀원 1명 이상의 리뷰를 받는다.
- 원시 산출물과 credential은 커밋하지 않는다. 데이터·결과는 manifest와 요약만 남긴다.

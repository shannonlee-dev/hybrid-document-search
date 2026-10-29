# 4차 개발 분담 + Git 컨벤션

> 기존 검색 방식 비교에서 한 단계 더 나아가, 실제 기업 특화 한국어 데이터에서 query별로 Sparse/Dense 검색 비중을 조절하는 경량 Adaptive Hybrid를 검증하고, 동일 데이터를 실제 상용 한국어 RAG/search 서비스에서도 평가하여 외부 타당성을 확인한다.

## 왜 4차인가

3차 최종 평가(Ko-MIRACL, nDCG@10)의 결과는 다음과 같다. 수치는 [최신 평가 결과](BENCHMARK_RESULTS.md)를 기준으로 한다.

| 방식 | nDCG@10 |
| --- | ---: |
| TF-IDF | 0.5253 |
| BM25 | 0.5939 |
| Dense (BGE-M3) | 0.8549 |
| RRF Hybrid | 0.7844 |
| Fixed Score Fusion | 0.8649 |

- 의미 중심 benchmark에서는 Dense가 이미 강했다.
- RRF는 Dense 단독보다 낮았다. Sparse + Dense를 단순 결합하면 오히려 나빠질 수 있다.
- 신중하게 정한 고정 weight의 Score Fusion만 Dense를 소폭 개선했다.
- [RAGAS 탐색 실험(PR #23)](https://github.com/shannonlee-dev/hybrid-document-search/pull/23)에서는 이 작은 retrieval 개선이 측정 가능한 답변 품질 개선으로 이어지지 않았다.

따라서 4차는 고정 결합 공식을 하나 더 추가하지 않고, **query별 routing**과 **외부 환경으로의 전이**를 검증한다.

- 질문: 경량 router가 한국어 query의 특성에 따라 다른 Sparse/Dense 비중을 선택할 수 있는가, 그리고 그 동작이 실제 기업 특화 데이터와 상용 서비스 비교에서도 유효한가?
- 참고 연구: Query-Adaptive Hybrid Search (DOI [`10.3390/make8040091`](https://doi.org/10.3390/make8040091)). query별 Sparse/Dense weighting 아이디어만 차용한다.
- 범위 밖: full QDAP 재현, 101-bin alpha predictor, antagonist training, Dense encoder 재학습.
- 개선을 가정하지 않는다. 미개선·악화 결과도 그대로 보고한다.

## 반가희 - Enterprise Dataset / Final Evaluation

- Issue: [#25](https://github.com/shannonlee-dev/hybrid-document-search/issues/25) · 브랜치: `feat/a-enterprise-evaluation`
- 3차 선행 작업(완료): #19 / PR #20 — 공통 평가 파이프라인, 최종 결과 보고

| 작업 | 내용 |
| --- | --- |
| 데이터셋 통합 | `BCCard/BCAI-Finance-Kor-Embedding-Pair` / `-Triplet`의 revision, schema, split, corpus 구성, qrels, license를 실제 metadata로 확인 후 기존 JSONL 계약으로 변환 |
| Split 관리 | train / validation / held-out test 분리, ID 중복·누출 검증, manifest 기록. 실제 구조와 다르면 임의 split을 만들지 않음 |
| Held-out 평가 | method 결과 확인 전 primary slice 고정. direct / synthetic 구분은 metadata가 뒷받침할 때만 |
| 공통 지표 | Recall@10, MRR@10, nDCG@10, P@1, latency mean / P95 (#19 파이프라인 재사용) |
| 최종 비교 | BM25 · Dense · Fixed Score Fusion · Adaptive Hybrid · 상용 서비스 결과 비교표와 실패 사례 분석 |

담당 외: router 구현, 상용 API 연동, credential, 외부 provider 설정.

## 이상헌 - Lightweight Adaptive Hybrid / Query Router

- Issue: [#24](https://github.com/shannonlee-dev/hybrid-document-search/issues/24) · 브랜치: `feat/b-adaptive-hybrid-router`
- 선행 작업: #17 / PR #18 — Fixed Score Fusion 구현 및 분석
- 구현·재현 계약: [Adaptive Hybrid](ADAPTIVE_HYBRID.md) · [실행 결과](ADAPTIVE_RESULTS.md)

| 작업 | 내용 |
| --- | --- |
| 세 가지 mode | `keyword-heavy` / `balanced` / `semantic-heavy` (잠정). 기존 Score Fusion 재사용, weight는 train/validation으로 선택 후 고정 |
| Query feature | query 길이, 숫자 비율, 한글/영문 비율, BM25 top score·gap, Dense top similarity·gap, Sparse/Dense Top-K overlap |
| 경량 classifier | Logistic Regression, shallow tree / boosting, 근거가 있을 때만 매우 작은 MLP |
| 선택 규칙 | train query별 fusion 결과로 label 생성·학습, validation으로 세 mode weight·router 구성 확정, test 전에 동결하고 test 기반 재튜닝 금지 |
| Ablation·결과 보존 | Dense 단독 · Fixed Score Fusion · Adaptive Hybrid · Oracle mode selection 비교. 개선되지 않아도 결과를 그대로 기록 |
| Oracle 상한 | 세 mode 중 query별 최고 mode 선택. 평가용 상한으로만 사용 |
| 비용 분석 | 총 latency와 feature 추출·router 추론 overhead 분리 기록 |

## 백동재 - Commercial RAG Integration / External Validation

- Issue: [#26](https://github.com/shannonlee-dev/hybrid-document-search/issues/26) · 브랜치: `feat/c-commercial-rag-benchmark`
- 3차 선행 작업(완료): #21 / PR #22 — 최종 서비스 통합 및 E2E
- 탐색 참고: PR #23 — RAGAS downstream 실험(negative/qualifying result, 병합 요구사항 아님)

| 작업 | 내용 |
| --- | --- |
| 사전 가능성 검토 | 데이터 업로드 가능 여부(license·약관), 접근성, 제한, 비용, 한국어 지원, 인덱싱 동작, Top-K 노출 여부 |
| 대상 서비스 | NAVER Cloud RAG (1차), Upstage File Search (가능한 경우 2차) |
| 결과 정규화 | 동일 corpus 인덱싱, 원본 ID mapping 보존, 외부 Top-K를 프로젝트 평가 형식으로 변환 |
| 조건 차이 기록 | re-chunking, 비공개 embedding, 숨겨진 reranking, score 비노출, Top-K 의미 차이 |
| E2E 비교 | retrieval 비교 이후 선택 진행. 정확도는 신뢰할 수 있는 reference answer가 있을 때만, Groundedness / Context Relevancy는 보조 지표 |

새 RAG 서비스를 만들지 않는다. 상용 서비스의 우수성을 가정하지 않는다. credential은 커밋하지 않는다.

## 작업 순서와 의존성

1. #25 데이터셋 확인 · split 고정 → #24, #26의 입력
2. #24 router를 train/validation으로 확정 · 고정 → test 평가
3. #26 가능성 검토 → 사용 가능한 서비스만 인덱싱 · 결과 정규화
4. #25에서 같은 query / qrels로 최종 비교표 작성

## Git 컨벤션

상세 규칙은 [CONTRIBUTING.md](../CONTRIBUTING.md)를 따른다.

- 작업 흐름: Issue → 작업 브랜치 → 구현·검증 → PR → 리뷰 → `main`에 머지
- 브랜치는 최신 `main`에서 분기하고 `<type>/<작업명>`을 사용한다. 3차 브랜치는 재사용하지 않는다.
- 커밋과 PR 제목은 Conventional Commits(`<type>: <변경 내용>`)를 따른다.

| Type | 용도 |
| --- | --- |
| `feat` | 기능 추가 |
| `fix` | 버그 수정 |
| `test` | 테스트 |
| `docs` | 문서 |
| `refactor` | 동작 변화 없는 구조 개선 |
| `perf` | 성능 개선 |
| `chore` | 설정, 의존성, CI |

- Issue 제목에는 `[Feature]` / `[Bug]` 접두사를 붙인다.
- PR은 `Implements #번호` / `Closes #번호`로 Issue를 연결하고, PR 템플릿의 Summary · Testing · AI assistance · Checklist를 작성한다.
- 공통 인터페이스·스키마·의존성을 바꾸거나 다른 담당 영역에 영향을 줄 때는 먼저 공유한다.
- 머지 전 `ruff check`, `ruff format --check`, `pytest`를 통과하고 담당 외 팀원 1명 이상의 리뷰를 받는다.
- 원시 산출물과 credential은 커밋하지 않는다. 데이터·결과는 manifest와 요약만 남긴다.

# 외부 baseline 재검토 프로토콜 (계산 전 고정)

작성 2026-11-04. 이 파일은 아래 수치를 계산하기 전에 커밋한다. 이후 바꾸지 않으며, 바꿀 일이 생기면 별도 addendum으로 남긴다.

## 이미 알려진 것 (사전 등록의 한계)

아래 비교의 **평균값은 이미 공개 문서에 있다**(PHASE5_EXTERNAL_VALIDATION.md, BCCARD_FINANCE_SERVICE_EVAL.md).
Kanana Dense .8029 / Kanana + Fixed .8087, Finance Dense .8087 / Finance 서비스 Hybrid .8116(nDCG@10, primary 721).
따라서 이 프로토콜은 결과를 모르는 상태의 확증 설계가 아니다. 새로 계산하는 것은 paired CI, Recall@5, 개선 집중도,
민감도 분석, 근접 중복 검사이며 이 값들은 아직 보지 않았다.

## 모델 적합성 판정

- `kakaocorp/kanana-2-3b-base`, `kanana-2-3b-instruct`는 HF `pipeline_tag=text-generation`인 생성형 모델이라 retriever로 쓰지 않는다.
- 비교 대상은 검색용 `kakaocorp/kanana-2-3b-embedding`(`sentence-similarity`), revision `2c85dbe61af8ff033d2c9d46a686c665153069aa`(2026-11-04 HF HEAD와 같음).
  공식 card 조건(task prompt 없음, mean pooling + normalize, cosine, bf16, 최대 4,096 tokens, full 2,560-dim)은 PR #37 설정과 같다.
- PR #37에서 이미 인코딩·평가했으므로 다시 실행하지 않고 커밋된 per-query 결과를 쓴다.
- `nlpai-lab/KURE-v1`은 `BAAI/bge-m3` fine-tune이라 독립 외부 encoder로 보지 않는다. 새로 평가하지 않는다.

## 입력 (모두 커밋된 파일)

| run | 파일 | 조건 |
|---|---|---|
| BGE / Kanana Dense·Fixed, BM25 | `docs/experiments/phase5/primary-queries.csv` | `*_rank`(top10 안 relevant 순위), `*_ndcg@10` |
| BGE / Finance 서비스 경로 | `docs/validation/2026-11-02/finance-service-eval/bccard-per-query.jsonl` | `A-*`, `D-*`, `B-*` |
| 중복 라벨 충돌 질의 | `docs/validation/2026-11-02/finance-strategyqa-analysis/tie-sensitivity-per-query.jsonl` | `binary_label_conflict_query` |

Primary 721개는 정답 chunk가 정확히 하나라 relevant 순위 r(1–10 또는 없음)에서 지표가 정해진다:
P@1 = [r=1], Recall@k = [r≤k], MRR@10 = 1/r, nDCG@10 = 1/log2(r+1). 커밋된 `*_ndcg@10`과 이 값이 721개 모두 같아야 진행한다.

**연결 조건(gate).** Phase 5 CSV의 `bge_fixed`와 PR #42 `A-bge`, `bge_dense`와 `D-bge`가 721개 질의 모두에서 순위·nDCG가 같아야 한다.
같으면 두 파이프라인이 같은 corpus·query·qrels·BM25·결합식(`Dense + .05×BM25/max(BM25)`)·전체 corpus 후보·동점 규칙을 쓴 것으로 보고
Kanana(Phase 5 경로)와 Finance(서비스 경로)를 비교한다. 다르면 비교를 중단하고 불일치만 보고한다.

## 비교

Primary(새 비교, 2개):

- P1 Dense-only: Kanana Dense → Finance Dense(`D-fin`)
- P2 고정 Fusion: Kanana + Fixed .05 → Finance 서비스 Hybrid .05(`A-fin`)

Reference(구현 확인용, 기존 결과 재현): BGE Dense → Kanana Dense, BGE Fixed → Kanana Fixed, BGE → Finance(A, D).

Finance α Fixed(`B-fin`)는 Kanana에 같은 결합식 결과가 없어 비교하지 않는다.

## 통계

- `evaluation/paired.py::paired_comparison`, 차이는 B − A, 지표 nDCG@10·Recall@5·Recall@10·MRR@10·P@1.
- 20,000 resamples, seed `20261102`, percentile 95% CI. P1·P2에는 Bonferroni 97.5% CI도 함께 적는다.
  Reference 행은 원래 문서의 seed로도 계산해 공개값과 같은지 확인한다.
- 판정: P1·P2 각각 nDCG@10 95% CI가 0을 포함하면 "차이 확인 안 됨". CI 하한 > 0이면 Finance 우위, 상한 < 0이면 Kanana 우위.

## 집중도

모든 primary·reference 비교에서 nDCG@10 차이가 0이 아닌 질의 수, 개선 질의 상위 10개가 개선 합에서 차지하는 비율,
차이가 큰 상위 10개(|Δ| 기준)를 뺀 평균 차이를 보고한다. 해석 기준은 두지 않고 기술만 한다.

## 민감도

1. 중복 vector 라벨 충돌 질의 10개(PR #43)를 뺀 711개에서 P1·P2 재계산.
2. 근접 중복 질의를 뺀 집합에서 P1·P2와 Finance − BGE(A, D) 재계산(아래 정의).

## 근접 중복 검사 (데이터 누수)

- 데이터: `scripts.prepare_bccard --download`(고정 revision)로 준비하고 커밋된 manifest SHA와 같아야 한다.
- 텍스트 정규화: casefold + 공백 축약(Finance 학습의 exact 중복 규칙과 같음).
- 유사도: 문자 3-gram(`char_wb`) TF-IDF cosine. fit은 train 질의 전체.
- 대상: primary test 721 질의 ↔ train 43,394 질의(질의 측), primary test 정답 chunk ↔ train positive chunk(문서 측).
- 임계값: **cosine ≥ .90을 근접 중복**으로 정의한다. .80, .95 구간 수도 기술로 보고한다.
- 민감도 2의 제외 집합은 질의 측 ≥ .90 질의다. 원문은 커밋하지 않고 질의 ID·최대 유사도만 남긴다.

## 하지 않는 것

새 인코딩, 학습, α·weight 탐색, Test 기반 설정 선택, GPU 실행. Latency는 기존 기록을 조건과 함께 다시 표로 정리만 한다.

# Lightweight Adaptive Hybrid (#24)

## 1. 목표

질문 특성에 따라 `keyword-heavy`, `balanced`, `semantic-heavy` 중 적절한 검색 모드를 선택하는 경량 Adaptive Hybrid Router를 구현하고 평가한다.

**가중치 20조합 × Router 3종 = 60개 후보**를 비교하며, 기존 Fixed Score Fusion(α=0.05)을 기준선으로 사용한다.

## 2. 가중치 및 Router 구성

검색 점수는 다음과 같이 계산한다.

`score = dense_cosine + α × normalized_bm25`

- **α 후보:** `[0, 0.05, 0.1, 0.2, 0.4, 0.8]`
- **가중치 조합:** 6개 중 3개를 선택한 20조합
- **모드 관계:** α_keyword > α_balanced > α_semantic
- **Router:** Logistic Regression, Small MLP, Shallow Tree
- **Feature 10개:** 질문 길이, 숫자 여부·비율, 한글·영문 비율, BM25·Dense 상위 점수와 점수 차이, Top-10 overlap

BM25·Dense 검색은 질문당 한 번만 수행하고, α별 결과를 캐싱하여 60개 후보에서 재사용한다.

## 3. 학습 및 평가 절차

| 단계 | 데이터 | 작업 |
|---|---|---|
| Train | 43,394개 | α별 nDCG@10으로 최적 모드 라벨 생성 및 Router 학습 |
| Validation | 1,000개 (비합성 745개) | 60개 후보 중 최종 가중치·Router 선택 |
| Freeze | - | 모델·가중치·해시 저장 및 설정 동결 |
| Test | 1,000개 (비합성 721개) | 동결된 설정으로 최종 평가 1회 |

- Validation의 비합성 질문 평균 nDCG@10을 선택 기준으로 사용한다.
- Test 결과를 이용한 재튜닝은 금지한다.
- 성능이 개선되지 않아도 결과를 그대로 기록한다.

## 4. 비교 및 평가 지표

**비교 방식**
- Dense BGE-M3
- Fixed Score Fusion (α=0.05)
- Adaptive Hybrid
- Oracle Mode Selection (평가용 상한)
- BM25 단독 (보조 기준선)

**평가 지표**
- Recall@10, MRR@10, nDCG@10, P@1
- Latency 평균 / P95
- Router 추론 비용 및 모드 선택 분포
- Fixed 대비 개선·동률·악화 사례

Primary 평가는 비합성 Test 721개, Secondary 평가는 전체 Test 1,000개를 사용한다.

Latency는 query encoding을 제외한 캐시 벡터 기반 CPU 검색 시간이며, warmup 1회 후 5회 측정한다. 통계적 유의성은 별도로 검정하지 않는다.

## 5. 실행 및 결과 관리

- **학습:** `scripts.evaluate_adaptive fit`
- **평가:** `scripts.evaluate_adaptive evaluate`
- **설정:** `config/adaptive_candidates.json`
- **최종 결과:** `docs/ADAPTIVE_RESULTS.md`
- **실험 산출물:** `artifacts/bccard-adaptive/`

학습·선택 과정과 최종 모델을 저장하고, 입력 및 설정 해시로 재현성을 검증한다.
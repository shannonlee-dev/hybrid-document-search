# Finance encoder의 Ko-StrategyQA 퇴보 탐색 분석

이 분석은 exploratory임. Ko-StrategyQA 결과를 이미 본 뒤 원인을 찾으려는 분석이라 여기 나온 연관은 가설 생성용이고 인과 증명이 아님. 사전 등록은 [`preregistration.md`](validation/2026-11-02/finance-strategyqa-analysis/preregistration.md)이고 분석 코드 작성 전에 커밋했음(`a1f9ee1`). 이 브랜치는 PR #42(`feat/finance-encoder-service-eval`)를 base로 하는 stacked 브랜치임.

요약하면 퇴보는 Dense representation에서 생겨 Hybrid로 이어졌음. 사전 등록한 drift 가설과 "금융 쪽으로 끌려감" 가설은 지지되지 않았음. 가장 일관된 차이는 margin이었음. Finance는 두 데이터셋에서 모두 query-문서 점수를 전반적으로 낮추는데, BC Card에서는 non-relevant 점수를 더 크게 낮추고 StrategyQA에서는 relevant와 non-relevant를 거의 똑같이 낮췄음. 그 결과 StrategyQA에서는 상위 근접 순위가 섞이며 1위가 다른 문서로 바뀌는 손실이 생겼음. 손실은 lexical overlap이 낮고 정답 passage가 여러 개인 query에 몰렸음.

## 입력과 경로 확인

PR #42 evidence의 per-query 결과(D, A, B 조건)를 그대로 쓰고, query vector는 서비스와 같은 FP32 경로로 다시 계산했음. Finance encoder는 identity를 확인한 `AdaptedQueryEmbedder`임. 문서 vector는 원본 BGE-M3로 고정했음.

Features 계산의 Dense top10은 StrategyQA에서 커밋된 D-bge·D-fin과 592개 모두 같았음. BC Card는 BGE 39개, Finance 40개 query에서 top10 순서가 달랐음(`features-parity.json`). 커밋된 행은 FAISS 경로이고 여기서는 NumPy 내적이라 생긴 차이로 보임. BGE 39개는 qdap 감사가 기록한 FAISS–NumPy 차이 수와 같음. 순위 지표 차이는 커밋된 행에서 가져왔고, 첫 relevant 순위와 margin 같은 특성만 NumPy 경로 값임.

## 순위 변화

| | StrategyQA (592) | BC Card (721) |
|---|---|---|
| Dense W/T/L (nDCG@10) | 55/438/99 | 110/545/66 |
| 서비스 Hybrid W/T/L | 53/442/97 | 91/569/61 |
| Fixed α=.84 W/T/L | 55/449/88 | 89/583/49 |
| Dense ΔnDCG@10 / ΔMRR@10 / ΔRecall@10 | −.0148 / −.0155 / −.0104 | +.0252 / +.0271 / +.0194 |

![rank movement](validation/2026-11-02/finance-strategyqa-analysis/rank-movement.svg)

Dense에서 첫 relevant 문서의 전체 corpus 순위를 보면 StrategyQA는 top10 진입 2개, 이탈 6개, top10 안 상승 24개, 하락 39개, 그대로 463개였음. BC Card는 진입 33개, 이탈 19개, 상승 77개, 하락 47개였음. StrategyQA 손실은 대부분 top10 안에서 한두 칸 밀린 경우임. 사례 검토에서도 1위가 2~8위로 내려간 경우가 많았음.

Dense 손실이 Hybrid로 이어지는지(H4)를 보면 query별 ΔnDCG의 Dense–서비스 Hybrid Pearson은 .647, Dense–Fixed α는 .619임(BC Card .828, .766). Dense 손실 99개 중 서비스 Hybrid에서도 손실인 query가 74개, Fixed α에서는 66개임. 손실은 fusion이 아니라 Dense 쪽에서 생겼고, BM25가 일부만 완화한 것으로 보임.

## Embedding 변화

| 항목 (평균) | StrategyQA | BC Card |
|---|---:|---:|
| Query drift 1 − cos(q_bge, q_fin) | .0706 | .1281 |
| 전체 문서 점수의 BGE–Finance 상관 | .954 | .901 |
| Top10 / Top100 이웃 overlap | .783 / .781 | .653 / .645 |
| 최고 relevant 점수 변화 | −.0744 | −.0761 |
| 최고 non-relevant 점수 변화 | −.0717 | −.0939 |
| Margin 변화 (relevant − non-relevant) | −.0027 | +.0177 |
| BC Card corpus 중심과의 cos 변화 | −.1616 | −.2021 |
| 자기 corpus 중심과의 cos 변화 | −.1637 | −.2021 |

![drift vs delta](validation/2026-11-02/finance-strategyqa-analysis/drift-vs-delta.svg)

H1(drift가 크면 손실이 큼)은 지지되지 않았음. Finance가 query를 바꾼 정도는 StrategyQA가 BC Card보다 오히려 작음. StrategyQA 안에서 drift와 Dense ΔnDCG의 Spearman은 −.098(p .017)로 약함. 5분위 그림에서는 drift가 가장 큰 구간의 손실이 가장 컸지만(평균 −.043), 나머지 구간 사이에서는 일관된 경향이 없었음.

H3(금융 쪽으로 끌림)도 지지되지 않았음. Finance는 두 데이터셋 모두에서 query를 BC Card corpus 중심에서 **멀어지게** 했고, 자기 corpus 중심에서도 비슷하게 멀어졌음. 특정 domain 방향보다는 모든 문서와의 유사도를 함께 낮추는 변화에 가까움. 이 이동량과 ΔnDCG의 상관은 .047(p .26)임.

H2(margin)는 두 데이터셋 차이를 가장 잘 보여줬음. Finance는 relevant와 non-relevant 점수를 둘 다 약 .07~.09 낮췄음. BC Card에서는 non-relevant를 더 크게 낮춰 margin이 늘었고, StrategyQA에서는 거의 같이 낮춰 margin이 평균적으로 그대로였음. Margin 변화와 Dense ΔnDCG의 Spearman은 StrategyQA .313, BC Card .417임. 다만 margin은 1위 문서가 relevant인지와 정의상 엮여 있어서 이 상관은 독립 증거가 아님. "점수 축이 전체적으로 이동했는데 StrategyQA에서는 판별력이 늘지 않았다"는 기술로만 읽어야 함.

정규화 전 norm은 측정하지 못했음. BGE-M3 SentenceTransformer pipeline에 `Normalize` 모듈이 있어서 `normalize_embeddings=False`로도 항상 1이 나왔음.

## Query 특성별 손실

Dense ΔnDCG@10 평균과 bootstrap 95% CI(seed 20261102, 5,000회). 구간 CI는 기술용이고 다중 비교 보정 없이 판정에 쓰지 않음.

| 특성 | StrategyQA | BC Card |
|---|---|---|
| Lexical overlap 하위 / 중간 / 상위 3분위 | −.027 [−.046, −.010] / −.013 / −.004 [−.013, +.006] | +.044 [+.016, +.073] / +.029 / +.002 [−.021, +.027] |
| 정답 1개 / 2개 이상 | −.005 [−.019, +.008] (211) / −.020 [−.031, −.011] (381) | +.025 (721, 전부 1개) / 없음 |
| 길이 하위 / 중간 / 상위 3분위 | −.006 / −.021 / −.019 | +.006 / +.029 / +.043 |
| 금융 용어 있음 / 없음 | −.072 (4개) / −.014 (588) | +.023 (508) / +.031 (213) |

두 데이터셋 모두 lexical overlap이 낮은 query에서 Finance 효과가 가장 컸음. 방향만 반대임. BM25나 표면 일치에 기대기 어려운 query일수록 query representation 변화가 결과를 크게 바꾼다는 해석과 맞음. StrategyQA 손실은 정답 passage가 여러 개인 query에 몰렸음. 이건 다단계 근거가 필요한 query의 proxy로 정한 기준임. 금융 용어 사전은 StrategyQA에서 4개만 걸려 정보가 없음. BC Card에서는 금융 용어가 없는 query에서도 개선이 있었음.

## 사례 검토와 사후 측정

StrategyQA Dense 손실 상위 15개와 개선 상위 5개를 직접 읽었음(`manual-review.json`, ID와 표시만 커밋). 손실 15개는 모두 금융과 무관한 일반 지식이었고 11개는 엔티티 사실 질문(과일 특징, 인물, 게임 장비, 동물 등)이었음. 4개는 짧거나 모호했고 2개는 "1번"처럼 앞 단계를 가리키는 분해 질문이었음. 개선 5개도 같은 유형 구성이었음. 그래서 이 유형 표시만으로는 손실과 개선을 가를 수 없었음.

Low overlap 표시는 처음에 눈으로 붙였다가 자동 규칙(하위 3분위)으로 다시 계산했고 한 건이 바뀌었음. Ko-StrategyQA query는 원래 질문을 분해한 단계 질문이라 이전 단계 답 없이는 모호한 경우가 섞여 있음. 데이터에 reasoning 유형 metadata는 없음.

사례에서 1위가 한 칸 밀리는 경우가 많아서, 새 1위가 정답과 같은 출처(같은 Wikipedia 제목)인지를 사후에 셌음(`posthoc-same-source.json`, 사전 등록 밖). BGE 1위가 relevant였다가 Finance 1위가 non-relevant로 바뀐 22개 중 같은 출처는 2개뿐임. 손실은 대부분 다른 문서의 passage가 올라와서 생긴 것이고, 라벨 단위(같은 문서의 다른 문단) 문제로는 설명되지 않았음. BC Card는 같은 parent의 non-relevant가 1위에 온 경우가 없었음.

## BC Card와 StrategyQA 차이에 대해 말할 수 있는 것

절대 성능은 relevance 정의와 query 분포가 달라 직접 비교하지 않았고, 같은 분석을 양쪽에 적용한 상대적 패턴만 비교했음.

- 관찰된 것: Finance는 두 데이터셋 모두에서 query-문서 유사도를 전반적으로 낮춤. BC Card에서는 그 이동이 relevant 쪽에 유리하게 작동해 margin이 늘었고, StrategyQA에서는 판별력이 늘지 않은 채 상위 근접 순위만 섞였음. 효과는 두 데이터셋 모두 lexical overlap이 낮은 query에서 가장 컸음.
- 관찰되지 않은 것: StrategyQA query가 더 많이 변했다거나 금융 방향으로 끌려갔다는 증거는 없었음. 라벨 단위 문제로도 설명되지 않았음.
- 추론(검증 안 됨): 학습은 BC Card positive와 hard negative 사이의 판별에 맞춰 점수 공간을 재배치했음. 그 재배치가 금융 밖 엔티티·일반 지식 query에서는 판별력을 주지 못하고, 근접 경쟁 passage의 순서를 무작위에 가깝게 흔든 것일 수 있음.

## 아직 검증되지 않은 가설

- 손실이 학습 데이터 분포 밖 query에서 생기는지: BC Card train query와의 최근접 거리(또는 밀도)와 StrategyQA 손실의 관계. 이번 분석에서는 train query vector를 보지 않았음.
- 점수 축 전체 이동이 특정 방향(공통 성분) 제거인지: 차이 vector(q_fin − q_bge)의 query 간 일관성과 주성분 분석.
- Tether weight나 학습 step이 일반 지식 query의 판별력 손실과 관련 있는지: 다른 checkpoint(step 0~2048)의 StrategyQA 결과 비교. 기존 checkpoint 평가만으로 가능하고 재학습은 필요 없음.
- Seed 의존성: Seed42에서도 같은 패턴이 나오는지.
- Ko-MIRACL 같은 다른 일반 도메인 데이터셋에서 margin 패턴이 재현되는지. MIRACL dev는 50개라 검정력이 부족함.

이 가설들은 별도 사전 등록 후 검증해야 함. 개선 실험(재학습, α 변경, threshold 조정)은 하지 않았고, 원인 검증이 먼저임.

## 재현

```bash
# features: GPU와 로컬 데이터 필요, per-query-summary.jsonl과 features-parity.json 생성
uv run --locked --extra sparse --extra dense python -m scripts.analyze_strategyqa_regression features \
  --data-dir $FINANCE_DATA --cache-dir $FINANCE_CACHE --external-cache artifacts/score-fusion-cache \
  --base-model $FINANCE_MODEL --checkpoint $REPRO_RUN/train-seed43/selected-last-block.pt
# cases: 수동 검토용 원문을 stdout에만 출력 (파일 저장 없음)
# posthoc: 사후 same-source 집계 (로컬 qrels 필요)
uv run --locked --extra sparse --extra dense python -m scripts.analyze_strategyqa_regression posthoc \
  --data-dir $FINANCE_DATA --cache-dir $FINANCE_CACHE --external-cache artifacts/score-fusion-cache
# analyze / check: 커밋된 파일만 사용
uv run --locked --extra sparse --extra dense python -m scripts.analyze_strategyqa_regression analyze
uv run --locked --extra sparse --extra dense python -m scripts.analyze_strategyqa_regression check
uv run --locked --extra sparse --extra dense pytest -q tests/test_strategyqa_regression_evidence.py
```

`check`는 `posthoc-same-source.json`과 `features-parity.json`을 해시로만 확인함(재계산에 로컬 데이터가 필요함). Query 원문, vector, 모델은 커밋하지 않았음.

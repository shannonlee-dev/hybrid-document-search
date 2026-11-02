# Ko-StrategyQA 퇴보 원인 분석 사전 등록 (exploratory)

작성: 2026-11-02, 분석 코드 작성과 per-query 특성 조회 전. 기준 브랜치: PR #42 `feat/finance-encoder-service-eval` `6dbcc4b`.

이 분석은 exploratory임. Ko-StrategyQA Test 결과(세 조건 모두 nDCG@10 −.012~−.015, CI가 0 아래)를 이미 알고 있음. 여기서 나오는 연관은 가설 생성용이고 독립 검증이나 인과 증명이 아님. 결과를 보고 평가 기준을 새로 만들어 유의성을 주장하지 않음. 재학습, α 변경, threshold 튜닝, 모델 수정은 하지 않음.

## 입력

- PR #42 evidence의 per-query 결과(`strategyqa-per-query.jsonl`, `bccard-per-query.jsonl`). 순위 지표는 여기서 다시 계산하지 않고 그대로 씀.
- Query vector: 서비스 경로와 같은 FP32 `DenseEmbedder`(원본 BGE)와 identity를 확인한 `AdaptedQueryEmbedder`(Finance Seed43 `fbbd5ed0…`). 문서 vector는 각 데이터셋의 원본 BGE-M3 cache 그대로 고정.
- Dense score는 FP32 문서 vector와 query vector의 내적, 순위는 stable argsort. 이 순위가 커밋된 D-bge·D-fin top10과 같은지 먼저 확인하고 다르면 차이를 기록함.
- 데이터셋: Ko-StrategyQA 592개, BC Card Test primary 721개. 같은 분석을 둘 다에 적용함.

## 가설

- H1 Representation drift: Finance query vector가 BGE에서 멀어진 정도(1 − cos)가 StrategyQA에서 BC Card보다 크고, drift가 클수록 Dense nDCG@10 손실이 큼.
- H2 Margin: StrategyQA에서 Finance가 relevant 문서 최고 점수와 non-relevant 최고 점수의 차이(margin)를 줄임. Margin 변화가 nDCG 변화와 연관됨.
- H3 금융 방향 이동: Finance가 query를 금융 문서 쪽으로 끌어당김. BC Card corpus 중심(BGE 문서 vector 평균)과의 cosine 증가량이 StrategyQA 손실과 연관됨.
- H4 Fusion은 원인이 아님: Dense-only 손실이 Hybrid 두 조건에도 비슷하게 이어짐. Query별 ΔnDCG의 Dense와 Hybrid 사이 상관이 높음.
- H5 Query 유형: 손실이 금융 용어 없는 일반 지식 query, 정답이 여러 paragraph인 query(다단계 근거 proxy), 짧은 query, 정답 문서와 lexical overlap이 낮은 query에 몰림.

## 측정 항목 (query 단위, 세 조건 구분)

- 순위: ΔnDCG@10, ΔMRR@10, ΔRecall@10, W/T/L(D, A, B 각각, 커밋된 행 기준). 첫 relevant 문서의 전체 corpus Dense 순위(BGE, Finance)와 이동량, top10 진입·이탈.
- Embedding: cos(q_bge, q_fin), 정규화 전 norm(두 encoder), relevant 최고 점수 변화, non-relevant 최고 점수 변화, margin 변화, top10·top100 Dense 이웃 overlap, BC Card corpus 중심과의 cosine 변화, 자기 corpus 중심과의 cosine 변화.
- Query 특성(고정 규칙, 결과 보기 전 정의):
  - 길이: 문자 수, BGE tokenizer 토큰 수(특수 토큰 제외).
  - 정답 수: relevance > 0인 문서 수. 1개와 2개 이상으로 나눔.
  - 금융 용어: 아래 고정 사전의 단어가 query에 하나라도 있으면 true. 카드, 은행, 대출, 이자, 금리, 결제, 보험, 주식, 투자, 세금, 연회비, 포인트, 계좌, 수수료, 적금, 예금, 환율, 신용, 할부, 한도, 금융, 돈, 가격, 비용. 오분류 가능성이 큼(예: 일반 문맥의 "돈", "가격").
  - Lexical overlap: query와 relevant 문서(제목+본문) 사이 문자 2-gram Jaccard의 최댓값.
- 연관: Spearman 상관(H1, H2, H3 각각 drift·margin 변화·금융 중심 이동과 Dense ΔnDCG). H4는 query별 ΔnDCG의 D–A, D–B Pearson·Spearman.
- 구간 비교: 길이 3분위, 정답 수(1 / 2+), 금융 용어 유무, overlap 3분위별 평균 ΔnDCG(Dense)와 bootstrap 95% CI(seed 20261102, 5,000 resamples). 구간별 CI는 기술용이며 다중 비교 보정 없이 판정에 쓰지 않음.

## 사례 검토

StrategyQA Dense ΔnDCG@10이 가장 크게 하락한 query 상위 15개와 가장 크게 개선된 query 상위 5개를 직접 읽고 다음 중 해당하는 유형을 표시함(복수 가능): 다단계 추론, 사실·인과 추론, 짧거나 모호함, 금융과 무관한 일반 지식, 정답과 lexical overlap 낮음, 기타. 자동 특성과 수동 표시가 다르면 둘 다 기록함. 사례는 설명용이며 이것으로 유의성을 주장하지 않음. 커밋에는 query ID와 유형 표시만 넣고 원문은 넣지 않음.

## 산출물

Per-query 요약(ID와 수치만), analysis JSON, 순위 이동·drift 그래프, 수동 검토 표, MANIFEST, 재현 스크립트, 최소 테스트, 후속 검증이 필요한 가설 목록.

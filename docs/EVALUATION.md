# 검색 평가 지표

관련 Issue: #12 · 담당: @bangahee · 브랜치: `feat/bm25-evaluation`

## 공통 평가 기준

- 검색 깊이: Top-10 (`top_k=10`).
- 공통 결과표: Recall@5 / Recall@10 / MRR@10 / nDCG@10.
- Latency: 검색기 준비 후 warm-up과 반복 측정으로 평균 / P95를 기록합니다.
  모델 및 인덱스 최초 준비 시간은 별도로 기록합니다.
- Dense 모델 선정은 서비스 Top-10에 맞춰 Recall@10 / MRR@10 / nDCG@10을 우선합니다.
- Recall@100은 **참고용** 지표입니다. 공통 결과표의 필수 항목 또는 주된 모델 선정 기준으로 사용하지 않습니다.
  참고 지표를 계산할 때에만 별도로 100개 결과를 검색하며, Top-10 latency와 혼합하지 않습니다.
- 공통 corpus / query / qrels / split / manifest를 유지합니다.
  모델 및 파라미터 선택은 train에서 진행하고 고정된 dev로 최종 비교합니다.
- 10k subset 결과는 전체 MIRACL 공식 benchmark 성능이 아닙니다.

## 계산 규칙

지표 계산에는 검색기의 원점수를 사용하지 않습니다. 공통 `SearchResult` 목록 순서와
`document_id`를 qrels에 연결합니다. score로 다시 정렬하지 않으므로 기존 동점 순서도 보존합니다.

현재 Ko-MIRACL 라벨은 0 / 1입니다. 모든 지표에서 `relevance > 0`을 관련 문서로
취급하고 이진 gain 1을 사용합니다. 0 이하의 라벨은 gain 0입니다.
미판정 문서는 해당 평가에서 gain 0으로 처리합니다. 이 규칙은 미판정 문서가
실제로 비관련임을 확정한다는 의미가 아닙니다.

| 지표 | 한 질의의 계산 |
| --- | --- |
| Recall@K | 상위 K에서 찾은 관련 문서 수 / 해당 질의의 전체 관련 qrels 문서 수 |
| RR@K | 상위 K의 첫 관련 문서가 rank r이면 1/r, 없으면 0 |
| nDCG@K | 상위 K의 DCG / 전체 관련 qrels로 구성한 이상적인 상위 K의 DCG |

```text
DCG@K = sum(gain_i / log2(i + 1)), i = 1 ... K
gain_i = 1 if relevance > 0 else 0
IDCG@K = sum(1 / log2(i + 1)), i = 1 ... min(K, 전체 관련 문서 수)
```

질의별 RR@10의 평균이 **MRR@10**입니다. Recall과 nDCG도 질의마다 동일한 가중치로
평균하여 보고합니다(macro average). 검색 결과가 없는 질의도 평균에 포함합니다.
관련 문서가 없는 질의의 지표는 모두 0으로 정의하고 평균에 포함합니다.
평가 대상 질의 집합이 비어 있으면 오류를 발생시킵니다.

검색 결과가 K개보다 적으면 있는 결과만 사용하며, Recall 분모와 IDCG를 줄이지 않습니다.
K보다 긴 목록이 들어오면 상위 K만 점수에 반영하지만 전체 목록의 유효성은 검증합니다.

## 입력 및 검증

- `results`: 공통 `SearchResult` iterable. 원본 목록을 변경하지 않습니다.
- 한 질의의 `qrels`: `{document_id: relevance}` mapping.
- 전체 `run`: `{query_id: SearchResult 목록}` mapping.
- 전체 `qrels`: `{query_id: {document_id: relevance}}` mapping.
- 모든 ID는 비어 있지 않은 문자열, relevance는 bool을 제외한 유한한 숫자여야 합니다.
- 검색 결과의 ID는 고유하고, rank는 입력 목록 순서대로 1부터 연속해야 합니다.
- K는 bool을 제외한 양의 정수입니다.
- 전체 run에 평가 대상 밖의 query ID가 있으면 오류로 중단합니다.
  평가 대상이지만 run에 없는 query는 빈 결과로 처리합니다.

파이프라인에서는 **선택한 모든 query ID**를 전체 qrels mapping에 포함해야 합니다.
판정이 없는 질의라면 빈 mapping을 넣습니다. qrels 행이 있는 질의만 평균에 넣으면 안 됩니다.
파일 스키마와 corpus 참조 검증은 기존 데이터 규칙을 따르며 이후 평가 loader에서 연결합니다.

## Python 사용 예시

평가 코어는 표준 라이브러리와 공통 `SearchResult`만 사용하므로 검색 모델이나
`ranx` 설치 없이 fixture 테스트를 실행할 수 있습니다.

```python
from evaluation.metrics import evaluate_query, evaluate_run, recall_at_k
from retrievers.base import SearchResult

results = [
    SearchResult("unjudged#0", rank=1, score=9),
    SearchResult("relevant#0", rank=2, score=8),
]
qrels = {"relevant#0": 1, "not-retrieved#0": 1}

print(evaluate_query(results, qrels))  # Recall@5/10=0.5, RR@10=0.5
print(evaluate_run({"q1": results}, {"q1": qrels}))  # 집계 결과의 키는 mrr@10
# 참고용 Recall@100: 100개까지 검색한 별도 결과 목록에 적용합니다.
# recall_at_k(reference_results, qrels, k=100)
```

`recall_at_k`, `reciprocal_rank_at_k`, `ndcg_at_k`는 개별 지표를 계산합니다.
`evaluate_query`는 Recall@5 / Recall@10 / RR@10 / nDCG@10을,
`evaluate_run`은 공통 집계 지표 Recall@5 / Recall@10 / MRR@10 / nDCG@10을 반환합니다.
참고용 Recall@100은 기본 집계에 포함되지 않습니다.

## 테스트 및 다음 단계

```bash
uv run pytest tests/test_metrics.py
uv run ruff check .
uv run ruff format --check .
```

`tests/fixtures/evaluation_cases.json`은 테스트 전용 합성 fixture입니다.
런타임 query / qrels JSONL 스키마를 변경하지 않습니다.
순위 2 / 4의 관련 문서, 순위 10 / 11 경계, 미판정과 0 라벨, 누락 결과를
손계산 값과 비교합니다. 중복 ID와 잘못된 rank 및 숫자 입력도 검증합니다.

이 단계는 지표 코어와 unit test 구현입니다. 실제 Ko-MIRACL query/qrels loader,
검색 방식별 실행, latency 측정, JSON/CSV 저장과 최종 비교표는 다음 단계에서 구현합니다.
실제 데이터 benchmark와 `ranx` 대조 검증은 아직 수행하지 않았습니다.

## 로컬 검증 결과

2026-10-07, Python 3.12.15 환경에서 확인했습니다.

- Metric unit test: 56 passed.
- 전체 테스트: 353 passed, 53 skipped. Dense 의존성 관련 테스트는 미설치로 건너뛰었습니다.
- 전체 Ruff lint / format 검사와 `git diff --check` 통과.
- `python -B -S`로 외부 site-packages 없이 합성 fixture 평가 성공.
  Macro 결과는 Recall@5=0.1333333333, Recall@10=0.3333333333,
  MRR@10=0.12, nDCG@10=0.1574508168입니다.

위 수치는 합성 fixture의 정확성 검증 값이며 실제 Ko-MIRACL 검색 품질 결과가 아닙니다.

## 참고 자료

- [ranx 공식 지표 정의](https://amenra.github.io/ranx/metrics/)
- [공통 query/qrels 스키마](SCHEMAS.md)
- [공통 평가 기준에 대한 Issue #12 답변](https://github.com/shannonlee-dev/hybrid-document-search/issues/12#issuecomment-6035283641)

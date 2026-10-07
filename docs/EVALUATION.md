# 검색 평가 지표 및 Benchmark

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
평가 loader가 query/qrels 스키마, split별 참조, 공통 corpus ID와 manifest를 검증합니다.

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

## Metric 코어 테스트

```bash
uv run pytest tests/test_metrics.py
uv run ruff check .
uv run ruff format --check .
```

`tests/fixtures/evaluation_cases.json`은 테스트 전용 합성 fixture입니다.
런타임 query / qrels JSONL 스키마를 변경하지 않습니다.
순위 2 / 4의 관련 문서, 순위 10 / 11 경계, 미판정과 0 라벨, 누락 결과를
손계산 값과 비교합니다. 중복 ID와 잘못된 rank 및 숫자 입력도 검증합니다.

지표의 손계산 검증은 위 fixture 테스트로 수행합니다. `ranx`와의 대조 검증은 수행하지 않았습니다.
실제 데이터 평가와 결과 저장은 아래 benchmark pipeline을 사용합니다.

## Metric 코어 검증 결과 (Step 3)

2026-10-07, Python 3.12.15 환경에서 확인했습니다.

- Metric unit test: 56 passed.
- 전체 테스트: 353 passed, 53 skipped. Dense 의존성 관련 테스트는 미설치로 건너뛰었습니다.
- 전체 Ruff lint / format 검사와 `git diff --check` 통과.
- `python -B -S`로 외부 site-packages 없이 합성 fixture 평가 성공.
  Macro 결과는 Recall@5=0.1333333333, Recall@10=0.3333333333,
  MRR@10=0.12, nDCG@10=0.1574508168입니다.

위 수치는 합성 fixture의 정확성 검증 값이며 실제 Ko-MIRACL 검색 품질 결과가 아닙니다.


## 준비 데이터 검증

`evaluation/data.py`는 `data/processed/`의 `manifest.json`과 준비된 5개 JSONL을 읽습니다.
데이터를 다운로드하거나 다시 선별하지 않습니다.

- corpus와 train/dev query/qrels 모두의 SHA-256 및 manifest 레코드 수를 확인합니다.
- manifest에 선택 ID 목록이 있으면 파일의 ID 및 순서와 비교합니다.
- 중복 query ID, train/dev query ID 중복, split 밖 query 참조 및 corpus 밖 문서 참조를 거부합니다.
- 같은 query/document의 동일한 중복 판정은 한 번만 계산하며, 다른 relevance가 중복되면 중단합니다.
- 판정 행이 없는 query도 빈 qrels mapping으로 포함합니다.
- 원본 passage ID, query text, 문서 제목 및 본문을 보존합니다.

## Benchmark 실행

프로젝트 루트에서 준비 데이터가 있는 상태로 실행합니다. 표준 라이브러리 기반 평가이므로
Sparse 비교에는 `sparse` extra만 필요하며 `evaluation` extra의 `ranx`는 필수가 아닙니다.

```bash
# train: 설정 선택과 개발 확인에 사용
uv run --locked --extra sparse python -m scripts.evaluate \
  --split train --methods tfidf bm25 --warmup 1 --repeats 5 \
  --output-dir artifacts/benchmarks/sparse-train

# dev: 설정을 고정한 뒤 최종 비교
uv run --locked --extra sparse python -m scripts.evaluate \
  --split dev --methods tfidf bm25 --warmup 1 --repeats 5 \
  --output-dir artifacts/benchmarks/sparse-dev
```

기본값은 `--data-dir data/processed`, `--split dev`, `--methods tfidf bm25`,
전체 query warm-up 1회와 측정 5회입니다. 반복 횟수는 CLI 기본값이며 팀의 고정 합의값은 아닙니다.
네 방식 최종 비교에서는 동일한 횟수를 명시합니다. 검색 깊이는 공통 합의에 따라 항상 10입니다.
기존 결과가 있는 output-dir은 거부하므로 다시 실행할 때에는 새 디렉터리를 지정합니다.

Dense 담당자가 **동일한 공통 corpus**로 준비한 FAISS 인덱스가 있으면 네 방식을 함께 실행합니다.
인덱스의 문서 ID / 순서 / 제목 / 본문이 평가 corpus와 다르면 중단합니다.
실행 장치와 환경은 팀에서 동일하게 맞추며 아래 예시는 CPU 기준입니다.

```bash
uv run --locked --extra sparse --extra dense python -m scripts.evaluate \
  --split dev --methods tfidf bm25 dense hybrid \
  --index indexes/dense-ko-miracl --device cpu --warmup 1 --repeats 5 \
  --output-dir artifacts/benchmarks/all-dev
```

Dense 인덱스 생성은 기존 `scripts.build_index`를 사용하며 이 명령은 저장된 인덱스를 재사용합니다.
모델이 캐시에 없으면 준비 중 모델 다운로드가 발생할 수 있습니다.
Hybrid는 기존 `HybridRetriever`에 BM25와 Dense를 주입하고 `rank_constant=60`을 사용합니다.
각 구성 검색기의 후보 수는 10, 최종 반환 수도 최대 10입니다. 공통 계약이나 RRF 구현은 변경하지 않습니다.

## 시간 측정 및 품질 집계

1. corpus 검증은 검색기 setup 및 query latency 측정 밖에서 수행합니다.
2. TF-IDF/BM25 인덱싱 또는 Dense 인덱스 로드와 첫 실제 검색을 `setup_seconds`에 기록합니다.
   첫 검색은 Dense의 지연 모델 로드를 준비 시간에 포함하기 위한 readiness 확인입니다.
3. 전체 query를 고정 순서로 warm-up합니다. 이 pass의 시간은 `warmup_seconds`로 별도 기록합니다.
4. 동일 query 순서로 지정 횟수만큼 반복합니다. 각 요청의 `retriever.search(query, 10)` 호출만
   `perf_counter`로 측정하며 query 전처리/embedding과 결과 생성은 호출 시간에 포함합니다.
   결과 검증, metric 계산 및 파일 저장은 query latency에 포함하지 않습니다.
5. 검색 품질은 첫 측정 pass의 결과로 계산합니다. 질의별 지표를 전체 query에 대해 macro 평균합니다.
6. Latency 평균/P95는 모든 query × 반복의 요청 샘플을 합쳐 계산합니다.
   P95는 정렬한 N개 샘플에서 `(N - 1) * 0.95` 위치를 선형 보간합니다.

Hybrid는 준비된 BM25/Dense 객체를 재사용합니다. Hybrid의 `setup_seconds`는 두 구성 검색기의
준비 시간과 Hybrid 객체 구성 시간을 합한 값이며 전체 실행의 추가 준비 시간으로 해석하지 않습니다.
Dense의 문서 embedding/인덱스 최초 생성 시간은 이 pipeline에서 측정하지 않습니다.
Dense 담당자의 빌드 benchmark에서 별도로 기록해야 합니다.
API 응답 시간과 동시 요청 처리량은 이 검색기 query-time 측정 범위에 포함되지 않습니다.

## 결과 파일

| 파일 | 내용 |
| --- | --- |
| `results.json` | 전체 지표, 질의별 첫 측정 검색 결과, 원시 latency, 실행 조건과 출처 |
| `summary.csv` | 방식별 공통 지표, latency 평균/P95, setup/warm-up 시간과 샘플 수 |
| `queries.csv` | 질의별 Recall@5/10, RR@10, nDCG@10 및 latency 요약 |
| `latencies.csv` | 방식 / query ID / 반복 번호 / 요청 latency(ms) |
| `comparison.md` | 공통 지표와 latency 비교표 |

JSON은 dataset revision, manifest/input 체크섬, 선택 query ID, 검색기 설정,
Dense 인덱스 체크섬과 실제 장치, Python/패키지/OS/CPU 환경 및 thread 환경 변수를 기록합니다.
관련 코드 파일 체크섬과 Git HEAD / 작업 트리 변경 여부도 보존하므로 미커밋 실행을 구분할 수 있습니다.
모든 결과는 UTF-8로 저장하며 `artifacts/`는 Git에서 제외됩니다.
검증에 실패한 방식은 결과표에서 조용히 제외하지 않고 실행을 중단합니다.
Recall@100은 이 공통 pipeline의 출력에 포함하지 않습니다. 별도 참고 측정의 결과와 latency를
Top-10 공통 비교표에 합치지 않습니다.

## Pipeline 검증 결과 (Step 4)

2026-10-07, Python 3.12.15 환경에서 확인했습니다.

- 신규 loader / benchmark / CLI 테스트: 62 passed.
- 전체 테스트: 415 passed, 53 skipped. Dense 의존성 미설치에 따른 관련 테스트 skip.
- 전체 Ruff lint / format 및 `git diff --check` 통과.
- 실제 Sparse fixture CLI로 TF-IDF/BM25 평가 및 5개 결과 파일 저장 확인.
- 네 방식의 공통 데이터 재사용과 실제 RRF 연결은 stub 검색기로 검증했습니다.
  실제 모델/FAISS를 사용한 Dense/Hybrid benchmark는 별도 인덱스와 의존성이 필요합니다.
- 고정된 준비 Ko-MIRACL train/dev Sparse 실행 결과는 [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)에 기록합니다.

## 참고 자료

- [ranx 공식 지표 정의](https://amenra.github.io/ranx/metrics/)
- [공통 query/qrels 스키마](SCHEMAS.md)
- [공통 평가 기준에 대한 Issue #12 답변](https://github.com/shannonlee-dev/hybrid-document-search/issues/12#issuecomment-6035283641)

# 최종 검색 평가 및 결과 보고

고정 입력으로 TF-IDF·BM25·Dense·RRF·score fusion의 품질과 검색 시간을 비교한다.
서비스 실행과 검증 범위는 [SEARCH_SERVICE.md](SEARCH_SERVICE.md)를 참조한다.

## 구현

`scripts/benchmark_score_fusion.py`의 cache loader와 cached embedder,
공통 `evaluation.metrics`·`benchmark_retriever`·`write_report`를 재사용한다.

## 비교 조건

| 항목 | 최종 공통 기준 |
|---|---|
| 방법 | TF-IDF, BM25, Dense, 기존 RRF, 최종 score fusion |
| Top-K / 지표 | 10 / Recall@5, Recall@10, MRR@10, binary nDCG@10 |
| 관련성 | relevance > 0이면 1, 그 외 및 미판정은 0 |
| 집계 | 질의별 동일 가중치 산술 평균 |
| RRF | 각 검색기의 Top-10, rank_constant=60 |
| Score fusion | 전체 corpus, Dense + 0.05 × BM25 / query별 corpus 최대 BM25 |
| 벡터 보정 | 적용하지 않음 |
| Latency | 전체 dev 질의 warm-up 1 pass, 전체 질의 측정 5 passes, mean / 선형 보간 P95 |
| 초기화 | 모델·인덱스 준비 및 warm-up은 latency에서 제외하고 별도 기록 |
| Recall@100 | 순수 참고용이며 공통 표에서 제외 |

최종 가중치는 선정용 MIRACL Train100에서 정한 값을 고정한다.
Dev 결과를 보고 가중치나 보정 조건을 변경하지 않는다.
미판정 문서를 실제 비관련이라고 단정하지 않으며 두 데이터셋의 평균을 합치지 않는다.

## 데이터셋과 입력 보존

| 데이터셋 | Corpus | 최종 평가 질의 | revision |
|---|---:|---:|---|
| Ko-MIRACL | 기존 10,000 passages | 기존 dev 50 | 5c7690518e481375551916f24241048cf7b017d0 |
| Ko-StrategyQA | 9,251 documents | 기존 dev 592 | 7cececfcc209fc836e08d6ab7584cba776847cfa |

MIRACL은 기존 seed 42 / Train100·Dev50 subset을 유지한다. 캐시를 새로 샘플링하거나
StrategyQA에 가짜 train split을 만들지 않는다. corpus의 원문·ID·행 순서와 qrels를 보존한다.
MIRACL 10k는 관련 판정 문서를 포함한 subset으로 공식 전체 MIRACL benchmark가 아니다.
StrategyQA revision은 원본 파일 내용의 일치로 고정한 버전이며 최초 다운로드 당시의
revision을 소급 확인한 것으로 해석하지 않는다.

캐시의 data.json, document-vectors.npy, query-vectors.npy, encoding.json 모두를
`docs/experiments/score-fusion/cache-release.json`의 SHA-256과 대조한다. 모델 revision·전처리·벡터 해시,
입력 manifest 해시와 실행 소스 해시를 results.json에 보존한다.
Train/dev 중복, corpus 밖 qrel, 잘못된 관련성 값은 오류로 처리한다.

## 재현 명령

저장소 루트에서 실행한다. Dense extra 설치는 모델 가중치 다운로드와 다르다.

```bash
uv sync --locked --extra sparse --extra dense
uv run --no-sync python -m scripts.prepare_score_fusion_cache
uv run --no-sync python -m scripts.prepare_score_fusion_cache --verify-only
uv run --no-sync python -m scripts.evaluate_final \
  --cache-dir artifacts/score-fusion-cache \
  --threads 2 --warmup 1 --repeats 5 \
  --output-dir artifacts/final-evaluation/run-01
```

기존 출력은 덮어쓰지 않으므로 재실행에는 새 경로를 사용한다.
두 데이터셋은 같은 명령·동일 환경에서 순차 실행하며, 측정 중 다른 benchmark를 함께 돌리지 않는다.
--datasets miracl10k 또는 --datasets strategyqa로 하나만 실행할 수 있다.
--methods dense rrf score_fusion으로 최종 세 방식만 지정할 수도 있다.

데이터셋마다 results.json, summary.csv, queries.csv, latencies.csv, comparison.md를 생성한다.
같은 표의 모든 방법은 같은 dev 질의 집합·반복 횟수·스레드 설정을 사용한다.
두 데이터셋 계산이 모두 성공한 후 보고서 저장을 시작한다.
원시 결과·캐시·모델·인덱스는 artifacts/ 아래 로컬 보관하고 Git에 올리지 않는다.
검토용 요약 수치·실험 조건·재현 절차만 docs에 남긴다.

## Latency 측정 범위

1. evaluate_final: 캐시된 query 벡터 조회 + CPU 검색·정렬·결합 + SearchResult 생성.
   모델 추론·초기화·HTTP는 제외한다. Dense는 FAISS, score fusion은 NumPy 전체 내적이므로
   검색 경로 차이도 포함된다. Sparse에는 원래 query embedding 단계가 없다.
2. scripts.evaluate: 실제 저장된 Dense 인덱스와 모델을 사용한 retriever.search 시간.
   query embedding을 포함하며 HTTP는 제외한다.
3. API/UI E2E: 서비스 초기화·HTTP·직렬화 등 별도 측정 경계를 명시하는 통합 검증이다.

첫 번째 표의 CPU 시간을 두 번째 또는 세 번째의 시간으로 표기하지 않는다.
Mac 재현 시간과 기존 WSL2/CUDA 시간은 같은 표에서 방법별로 섞지 않는다.
결과는 단일 세션·순차 검색이며 동시 요청 처리량이나 통계적 유의성을 뜻하지 않는다.

실제 인덱스를 가진 환경의 공통 평가 연결:

```bash
uv run --no-sync python -m scripts.evaluate \
  --data-dir data/processed --split dev \
  --methods tfidf bm25 dense rrf score_fusion \
  --index indexes/bge-m3 --device cpu --threads 2 \
  --warmup 1 --repeats 5 --output-dir artifacts/final-evaluation/live-01
```

위 명령의 corpus와 인덱스는 ID·순서·본문·제목이 정확히 같아야 한다.
`scripts.evaluate`의 `hybrid`는 RRF의 별칭이다. API의 `hybrid`는 score fusion이며,
평가에서는 `rrf` / `score_fusion`으로 구분한다.

## 검증과 보고

```bash
uv run --no-sync pytest tests/test_final_evaluation.py tests/test_benchmark.py tests/test_evaluate_cli.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
git diff --check
```

최종 수치·실행 환경·기존 결과와의 대조 및 결과 해석은
[최신 평가 결과](BENCHMARK_RESULTS.md)에 정리한다.

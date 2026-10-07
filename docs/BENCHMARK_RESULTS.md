# Sparse Benchmark 결과 (2026-10-07)

관련 Issue: #12 · 브랜치: `feat/bm25-evaluation`

## 실행 조건

- 공통 Ko-MIRACL 준비 corpus 10,000 passages / train 100 queries / dev 50 queries / seed 42.
- Dataset revision: `5c7690518e481375551916f24241048cf7b017d0`.
- Manifest SHA-256: `1d199e8f6d2e1f446110625b03c5c26e1dff5fed280e3a43672294b9f551520a`.
- 검색 깊이 10, 공통 지표 Recall@5 / Recall@10 / MRR@10 / nDCG@10.
- 전체 query warm-up 1회 후 측정 5회. train은 방식별 500, dev는 방식별 250 요청 샘플.
- 파라미터 조정 없이 기존 TF-IDF와 BM25 기본값을 고정하여 train 실행 후 dev 실행.
- TF-IDF: char_wb 2–4 n-gram / L2. BM25: 같은 분석기 / Lucene / k1=1.5 / b=0.75.
- 동일 로컬 환경에서 방식별 순차 실행. query-time latency는 search 호출만 포함.
- Python 3.12.15 / macOS-26.6.2-arm64-arm-64bit / CPU 논리 코어 10.
- 패키지: scikit-learn 1.9.1, bm25s 0.3.12, numpy 2.5.3, scipy 1.18.1.
- Dense 의존성과 실제 공통 FAISS 인덱스가 없어 이번 실행은 TF-IDF/BM25만 비교.
- Recall@100은 참고용이며 이 공통 결과표에는 포함하지 않음.

## Dev 결과

| Method | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Mean ms | P95 ms |
| --- | --- | --- | --- | --- | --- | --- |
| tfidf | 0.576667 | 0.722667 | 0.495238 | 0.525324 | 5.542394 | 6.291021 |
| bm25 | 0.587667 | 0.750667 | 0.597913 | 0.593881 | 0.919895 | 1.062710 |
| Dense | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |
| Hybrid (BM25 + Dense / RRF) | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |

현재 dev subset에서는 BM25가 TF-IDF보다 네 지표 모두 높았으며, 측정된 평균/P95 latency도 낮았습니다.
이 결과는 현재 문자 n-gram baseline과 10k 평가 subset에 한정됩니다. 전체 MIRACL 공식 benchmark 성능이나
모든 문서 도메인에서의 우열을 의미하지 않습니다. Latency는 실행 환경과 시스템 부하에 따라 달라집니다.

## Train 확인 결과

| Method | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Mean ms | P95 ms |
| --- | --- | --- | --- | --- | --- | --- |
| tfidf | 0.555944 | 0.689778 | 0.547524 | 0.539715 | 5.502145 | 6.376935 |
| bm25 | 0.565167 | 0.698722 | 0.566567 | 0.559138 | 0.931118 | 1.055085 |

Train 값은 개발 확인용입니다. Dense 모델 및 파라미터 선택은 train을 사용하며,
dev 결과를 보고 반복 튜닝하지 않습니다. Dense 모델 선정에서는 Recall@10 / MRR@10 / nDCG@10을 우선하고
Recall@100은 순수 참고용으로 별도 기록합니다.

## 준비 시간

| Split | Method | Setup seconds | Warm-up seconds |
| --- | --- | --- | --- |
| train | tfidf | 2.289527 | 0.551795 |
| train | bm25 | 2.210030 | 0.093616 |
| dev | tfidf | 2.238267 | 0.295036 |
| dev | bm25 | 2.043683 | 0.047082 |

Setup은 메모리 인덱싱과 첫 readiness 검색입니다. 위 비교표의 latency에 포함하지 않았습니다.
Dense의 문서 embedding/인덱스 최초 생성 시간은 Dense 담당자가 별도로 측정해야 합니다.

## 저장 위치 및 재현

실제 생성 파일은 Git에서 제외되는 다음 경로에 저장했습니다.

- `artifacts/benchmarks/sparse-train-20261007/`
- `artifacts/benchmarks/sparse-dev-20261007/`

각 경로에 `results.json`, `summary.csv`, `queries.csv`, `latencies.csv`, `comparison.md`가 있습니다.
JSON에는 전체 query와 검색 결과, 원시 timing, 환경, 검색기 설정 및 입력/코드 체크섬을 기록했습니다.
실행 당시 Git HEAD는 `9d90df530e04a697de3151e23bae12147ffcc654`였으며 Step 4 구현이 미커밋 상태여서
`working_tree_dirty=true`입니다. 실제 실행 코드 버전은 JSON의 파일별 SHA-256으로 확인합니다.

새 output-dir을 지정하여 재현합니다.

```bash
uv run --locked --extra sparse python -m scripts.evaluate \
  --split dev --methods tfidf bm25 --warmup 1 --repeats 5 \
  --output-dir artifacts/benchmarks/sparse-dev-rerun
```

## 남은 공통 검증

- Dense 담당자의 고정 기본 모델 및 동일 corpus FAISS 인덱스 준비.
- 네 방식을 같은 환경에서 한 번에 실행하고 공통 최종 비교표 갱신.
- 실제 Dense/Hybrid 결과 기반 통합 검증과 팀원 리뷰.

Benchmark 실행 옵션과 지표 계산 정의는 [EVALUATION.md](EVALUATION.md)를 참조합니다.

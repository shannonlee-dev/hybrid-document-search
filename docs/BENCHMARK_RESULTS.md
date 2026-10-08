# 검색 방식 Dev 비교

2026-10-08 프로젝트 기본 모델을 BGE-M3로 변경하고 네 검색 방식을 다시 측정했다.
기존 Ko-MIRACL corpus와 검증된 BGE-M3 인덱스를 재사용했다.
3모델 Train/Dev 비교와 인덱스 구축·복원 기록은 최초 전체 실험의 측정값을 유지한다.

| 모델 / 방식 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Mean ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF | 0.576667 | 0.722667 | 0.495238 | 0.525324 | 17.720120 | 28.155618 |
| BM25 | 0.587667 | 0.750667 | 0.597913 | 0.593881 | 1.704552 | 2.129968 |
| Dense (BGE-M3) | 0.813667 | 0.958333 | 0.861024 | 0.854857 | 21.594187 | 30.962093 |
| Hybrid (BM25 + BGE-M3) | 0.765333 | 0.927667 | 0.790571 | 0.784433 | 23.784875 | 33.535504 |

Ko-MIRACL 10,000 passages, Train 100 / Dev 50 queries, seed 42. Python 3.12.3 / WSL2 / NVIDIA RTX 4060, embedding `cuda:0`, CPU FAISS. FP32, L2 정규화, batch size 1, Top-K 10, 전체 질의 warm-up 1회, 측정 5회. PyTorch intra/inter-op·FAISS·OMP/MKL/OpenBLAS 2 threads, tokenizer 병렬화 비활성화, TF32 비활성화.

방식별 50개 질의와 latency 샘플 250개를 사용했다. Hybrid는 BM25 + BGE-M3,
RRF=60, 구성 검색기별 후보 Top-10이다. 준비 시간과 warm-up은 query latency에서 제외한다.
Dense 3모델 Train/Dev 비교는 [DENSE_RETRIEVAL.md](DENSE_RETRIEVAL.md)에 모았다.

## 결과 해석

Dev nDCG@10 최고 방식은 Dense (BGE-M3)이다.
Dense 단독과 Hybrid의 차이를 실제 측정 그대로 보고하며 Hybrid 개선을 전제하지 않는다.
기본 모델은 `config/dense_models.toml`의 `default_model = "bge"`로 지정하고
Train/Dev 점수로 자동 선택하지 않는다. 검색 알고리즘과 평가 조건은 유지했다.

## 원문·qrels 대조 사례

Dense와 Hybrid의 질의별 nDCG 차이가 가장 큰 양쪽 사례를 사후에 골랐다.
대표성을 통계적으로 보장하는 표본이 아니며 개별 검색 동작을 설명하기 위한 사례다.

### 1391: 발해는 언제 건국되나요?

- Dense (BGE-M3): 관련 문서 순위 1; 1위 `122372#0`, 제목 ‘발해’.
- Hybrid (BM25 + BGE-M3): 관련 문서 순위 6; 1위 `122372#12`, 제목 ‘발해’.

### 606: 조선에서 가장 어린 왕은 누구인가?

- Dense (BGE-M3): 관련 문서 순위 4; 1위 `11027#0`, 제목 ‘조선 현종’.
- Hybrid (BM25 + BGE-M3): 관련 문서 순위 1; 1위 `867281#3`, 제목 ‘빈 (지위)’.

각 결과의 문서 ID를 corpus 및 qrels와 대조했다. 전체 Top-10, 판정값, 제목·원문은
[cases.json](../results/retrieval/dev/cases.json)에 보존했다.

## 출처와 재현

- [평가 결과 JSON](../results/retrieval/dev/results.json)
- [요약 CSV](../results/retrieval/dev/summary.csv)
- [질의별 결과](../results/retrieval/dev/queries.csv)
- [원시 latency 샘플](../results/retrieval/dev/latencies.csv)
- [실행 환경·명령](../results/retrieval/dev/execution.json)
- [전체 실험 해시 및 검증](../results/experiment.json)

기존 데이터·인덱스·3모델 측정 코드 commit: `5a6a60024d602b23312eafdb3b3c371d282ddcc9`.
이번 네 방식 재평가의 코드 SHA와 변경 상태는 결과 JSON의 `code` 및
`experiment.json`의 `retrieval_code`에 별도로 기록했다. 원본 재평가는
`artifacts/ko-miracl-bge-default/retrieval/`에 보존했다.
계산 규칙과 재개 방법은 [EVALUATION.md](EVALUATION.md)를 참조한다.

고정된 기존 데이터와 Dev를 다시 사용하는 재현성·결과 정리 실험이다. 새로운 독립 테스트나 전체 MIRACL 공식 벤치마크 성능을 뜻하지 않는다. 단일 GPU 환경의 순차 실행이며 동시 요청 처리량·API 응답 시간은 측정하지 않았다. latency는 실행 당시 시스템 부하에 영향을 받는다.

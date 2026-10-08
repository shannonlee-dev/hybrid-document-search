# 검색 평가 및 전체 실험

## 지표와 측정 규칙

공통 Top-K는 10이고 Recall@5/10, MRR@10, 이진 nDCG@10을 보고한다.
검색기의 원점수로 다시 정렬하지 않고 `SearchResult` 순서와 원본 document ID를 qrels에 연결한다.
`relevance > 0`은 gain 1, 나머지 및 미판정 문서는 gain 0이다.
미판정이 실제 비관련이라는 뜻은 아니다.

| 지표 | 계산 |
| --- | --- |
| Recall@K | 상위 K에서 찾은 관련 문서 수 / 질의의 전체 관련 qrels 문서 수 |
| RR@10 | 첫 관련 문서 순위 r의 1/r; 없으면 0 |
| nDCG@10 | DCG@10 / 전체 관련 문서로 구성한 이상적 DCG@10 |

`DCG@K = sum(gain_i / log2(i + 1))`. RR@10을 질의별 동일 가중치로 평균한 값이 MRR@10이다.
Recall과 nDCG도 macro average하며 빈 결과와 관련 문서가 없는 질의는 0으로 포함한다.
관련 문서가 없는 경우 0, 평가 질의 집합이 비어 있으면 오류다.
Top-K보다 결과가 적어도 분모와 IDCG를 줄이지 않는다. Recall@100은 이번 결과에 포함하지 않는다.

계산은 `evaluation/metrics.py`와 `evaluation/latency.py`에 구현돼 있다.
검색기 준비·첫 readiness 검색은 `setup_seconds`, 전체 질의 warm-up은 `warmup_seconds`로 분리한다.
그 뒤 `retriever.search(query, 10)`만 측정한다. 질의 처리·embedding·결과 생성은 포함하고
지표 계산·검증·저장은 제외한다. 품질은 첫 측정 pass의 결과다.
Mean/P95는 전체 질의 × 5회 샘플이며 P95는 정렬 샘플의 `(N - 1) * 0.95`를 선형 보간한다.
Hybrid 준비 시간은 BM25/Dense 준비 시간과 결합 비용의 합이다.

## 단일 명령과 자동 재개

```bash
uv run --locked --extra sparse --extra dense python -m scripts.run_experiment
uv run --locked --extra sparse --extra dense python -m scripts.run_experiment --fresh
```

통합 실행기는 Python 3.12 이상·Linux·작동하는 CUDA가 필요하며, WSL2나 특정 GPU 모델을 강제하지 않는다.
CUDA를 감지한 뒤 실제 할당·연산을 확인하고, 사용할 수 없으면 CPU로 대체하지 않고 종료한다.
Python·OS·GPU·CUDA·패키지 버전은 실행 환경과 단계 입력에 기록한다.
작업 공간 lock과 결과 승격은 Linux API를 사용한다.
아래 환경은 저장된 벤치마크의 측정 조건이다.

Ko-MIRACL 10,000 passages, Train 100 / Dev 50 queries, seed 42. Python 3.12.3 / WSL2 / NVIDIA RTX 4060, embedding `cuda:0`, CPU FAISS. FP32, L2 정규화, batch size 1, Top-K 10, 전체 질의 warm-up 1회, 측정 5회. PyTorch intra/inter-op·FAISS·OMP/MKL/OpenBLAS 2 threads, tokenizer 병렬화 비활성화, TF32 비활성화.

측정 코드는 실행 전에 커밋해야 한다. 모델 revision은 [Dense 문서](DENSE_RETRIEVAL.md)에 고정돼 있다.
후보 모델과 고정 revision은 `config/dense_models.toml`의 `[models.<alias>]`에서 읽는다.
기본 모델은 같은 파일의 `default_model = "bge"`로 지정하며 `config_default` 정책으로 기록한다.
`retrievers/model_config.py`는 TOML을 읽고 검증하며, 실험 실행기와 Dense 검색기가 같은 설정을 사용한다.
Train/Dev 순위는 비교용으로만 기록한다. TOML 변경도 단계 입력 해시와 실행 전 변경 검사에 포함한다.

실행 순서는 환경·revision 사전 점검 → 데이터 준비 → 모델별 다운로드·build·별도 subprocess 복원 →
Train 3모델 → Dev 3모델 → 기본 모델(BGE-M3) 기반 Dev 4방식 → 검증·staging·최종 승격이다.
다운로드 실패는 최대 3회 시도하고, 다른 단계 실패는 실패 상태와 오류를 기록한 뒤 종료한다.

`artifacts/ko-miracl-full/checkpoint.json`에는 단계 상태·설정·코드 해시·의존 단계의 출력 경로 및 SHA·
실행 generation·완료 시각·오류를 저장한다. 데이터와 모델 revision, manifest/index SHA는
단계 설정 및 의존 출력 해시를 통해 연결한다. 파일 존재만으로 완료를 판단하지 않는다.
입출력 해시·설정이 바뀌면 해당 단계와 의존 단계를 다시 실행한다.
정상 Dense 인덱스는 재개 과정에서 재임베딩하지 않는다. 완료 후 기본 명령은 checkpoint와 결과를 검사한다.

`--fresh`는 소유 marker가 있는 실험 작업 공간만 삭제한다. 심볼릭 링크와 경로 이탈을 거부하고
동시 실행은 lock으로 막는다. 전역 모델 캐시와 다른 팀원의 인덱스는 보존한다.
원본 데이터는 다시 다운로드하며 모델 가중치는 정확한 revision의 캐시를 재사용할 수 있다.
기존 `results/`는 Git에 보존돼 있어야 한다. 모든 검증이 끝난 staging만 Linux 원자적 디렉터리 교환으로
승격하므로 실패·중간 결과는 최종 결과에 섞이지 않는다.

## 최종 산출물

```text
results/
  experiment.json
  dense/
    runtime.json
    train/ e5.json bge.json kure.json comparison.csv
    dev/   e5.json bge.json kure.json comparison.csv
  retrieval/dev/
    results.json summary.csv
```

패키징 시 위 파일만 생성해 `results/`에 저장하고 Git에 커밋한다.

Train/Dev 모델별·검색 방식별 JSON에는 질의 원문, Top-10 문서 ID·순위·점수, 지표·latency 샘플,
측정 조건·환경·코드 출처를 저장한다. 원시 JSON에서 검색 결과의 `title`·`snippet`을 제거하고
`qrels`와 변환 설명인 `export`를 추가한다. 지표와 Mean/P95는 이 JSON으로 재계산할 수 있다.
질의별 검색 결과와 판정값은 모델별·검색 방식별 JSON에서 확인한다.

`experiment.json`의 `files_sha256`은 자기 자신을 제외한 결과 파일 해시,
`raw_results`는 원시 JSON의 로컬 경로와 해시를 기록한다.

원시 출력은 `artifacts/`에 보관한다. JSON과 내용이 겹치는 `queries.csv`·`latencies.csv`·`comparison.md`,
로그·checkpoint·중간 결과·원본 corpus·모델·FAISS 바이너리는 Git에서 제외한다.

## 검증

`PACKAGE_FILES`의 출력 경로를 패키징·검증이 공유하며 완료 상태·필수 파일·체크섬을 검사한다.
산출물 변경 시 공통 경로·생성 처리·문서를 맞추고 `files_sha256`을 갱신한다.

패키징 전에 corpus/query/qrels와 원본·manifest 해시, 10000/100/50개 수량, 모델별 revision,
FP32·정규화·장치·실효 스레드, 인덱스 복원을 검증한다. 원시 결과로 모든 질의별/집계 품질과
latency를 재계산하고 Train 500개·Dev 250개 샘플을 확인한다. 세 모델·네 방식 중 하나라도 누락되면 실패한다.

```bash
uv run --locked --extra sparse --extra dense ruff check .
uv run --locked --extra sparse --extra dense ruff format --check .
uv run --locked --extra sparse --extra dense pytest
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked --extra sparse --extra dense pytest tests/dense
git diff --check
```

Fixture 테스트는 checkpoint 재개·의존 단계 무효화·fresh 삭제 범위 보호,
지표·latency 재계산, 환경 검사와 Dense 저장·복원을 확인한다.

개별 평가 CLI는 계속 사용할 수 있다. 기본 데이터 경로는 `data/processed/`이며
새 실험의 데이터에는 `--data-dir artifacts/ko-miracl-full/data/prepared`를 명시한다.
Sparse만 평가할 때 `--threads 2`는 Dense 의존성 없이 수치 라이브러리의 스레드를 제한한다.
Dense/Hybrid 평가의 `--threads 2 --device cuda:0`은 엄격한 revision·FP32·CUDA 확인을 활성화한다.

결과는 고정된 10k subset의 측정값이다. 독립적인 전체 실험 반복은 수행하지 않았으며,
질의별 5회 측정으로 신뢰구간을 제공하지 않는다. 전체 MIRACL 성능·동시 요청 처리량·API 응답 시간은
측정하지 않았다. GPU·OS·시스템 부하가 다른 latency와 구축 시간은 같은 조건의 수치로 비교할 수 없다.

[데이터](DATASET.md) · [Dense 비교](DENSE_RETRIEVAL.md) · [네 방식 결과](BENCHMARK_RESULTS.md)

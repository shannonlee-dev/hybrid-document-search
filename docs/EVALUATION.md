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

기존 `evaluation/metrics.py`와 `evaluation/latency.py`의 계산을 유지했다.
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

프로젝트의 Python 최소 버전은 3.12이며, 통합 실행기는 CUDA 사용 가능 여부를 검사한다.
Python 3.12로 세부 버전을 고정하거나 WSL2·특정 GPU 모델을 요구하지 않는다.
CUDA가 없으면 CPU로 대체하지 않고 종료한다. 실제 Python·OS·GPU·CUDA·패키지 버전은
실행 환경과 단계 입력에 기록한다. 작업 공간 lock과 결과 승격은 Linux API를 사용한다.
아래 환경은 저장된 벤치마크의 측정 조건이다.

Ko-MIRACL 10,000 passages, Train 100 / Dev 50 queries, seed 42. Python 3.12.3 / WSL2 / NVIDIA RTX 4060, embedding `cuda:0`, CPU FAISS. FP32, L2 정규화, batch size 1, Top-K 10, 전체 질의 warm-up 1회, 측정 5회. PyTorch intra/inter-op·FAISS·OMP/MKL/OpenBLAS 2 threads, tokenizer 병렬화 비활성화, TF32 비활성화.

측정 코드는 실행 전에 커밋해야 한다. 모델 revision은 [Dense 문서](DENSE_RETRIEVAL.md)에 고정돼 있다.
후보 모델과 고정 revision은 `config/dense_models.toml`의 `[models.<alias>]`에서 읽는다.
기본 모델은 같은 파일의 `default_model = "bge"`로 지정하며 `config_default` 정책으로 기록한다.
`retrievers/model_config.py`는 TOML을 읽고 검증하며, 실험 실행기와 Dense 검색기가 같은 설정을 사용한다.
Train/Dev 순위는 비교용으로만 기록한다. TOML 변경도 단계 입력 해시와 실행 전 변경 검사에 포함한다.

실행 순서는 환경·revision 사전 점검 → 신규 데이터 → 모델별 다운로드·build·별도 subprocess 복원 →
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
    train/ e5.json bge.json kure.json comparison.csv selection.json
    dev/   e5.json bge.json kure.json comparison.csv
  retrieval/dev/
    results.json summary.csv queries.csv latencies.csv comparison.md cases.json execution.json
```

Dense split별 JSON은 기존 평가 CLI 출력의 변경 없는 복사본이다. 비교 CSV와 문서는 요약본이다.
`experiment.json`은 자기 자신을 제외한 결과 파일 SHA, 코드·lock·데이터·모델 출처와 실행 명령을 기록한다.
로그·checkpoint·원본 corpus·모델·FAISS 바이너리는 로컬 작업 공간에만 두며 Git에 넣지 않는다.

## 검증

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

Fixture 테스트는 최초 실행·실패 checkpoint·재개·완료 후 검증·fresh·해시/revision 불일치·
실패 결과 차단·삭제 범위 보호와 실제 단계 그래프의 인덱스 재사용을 포함한다.
별도 기본 의존성 환경은 354 passed, 선택 의존성 미설치로 158 skipped다.
Ruff lint·format, diff, 18개 결과 파일 및 문서 수치·링크 검증을 통과했다.
최초 전체 실험 완료 후 기본 명령의 실제 재실행에서 18개 단계가 검증만 수행됐으며,
checkpoint와 세 인덱스의 해시·수정 시각이 바뀌지 않았다.

개별 평가 CLI는 계속 사용할 수 있다. 기본 데이터 경로는 `data/processed/`이며
새 실험의 데이터에는 `--data-dir artifacts/ko-miracl-full/data/prepared`를 명시한다.
`--threads 2 --device cuda:0`은 엄격한 revision·FP32·CUDA 확인을 활성화한다.

고정된 기존 데이터와 Dev를 다시 사용하는 재현성·결과 정리 실험이다. 새로운 독립 테스트나 전체 MIRACL 공식 벤치마크 성능을 뜻하지 않는다. 단일 GPU 환경의 순차 실행이며 동시 요청 처리량·API 응답 시간은 측정하지 않았다. latency는 실행 당시 시스템 부하에 영향을 받는다.

[데이터](DATASET.md) · [Dense 비교](DENSE_RETRIEVAL.md) · [네 방식 결과](BENCHMARK_RESULTS.md)

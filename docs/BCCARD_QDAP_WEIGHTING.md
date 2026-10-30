# QDAP-S/L 공식 구현과 BC Card 평가 — 2026-10-30

Issue #33의 구현·실행 기록이다. 기존 탐색 스크립트를 import하지 않는 독립
`fusion/qdap.py`, `evaluation/qdap_training.py`, `evaluation/qdap_benchmark.py`,
`evaluation/qdap_execution.py`와 단계별 CLI를 사용한다.
**학습·평가·독립 검증 완료.** Train 43,394개 전체로 학습했으나
이번 설정에서 QDAP-S/L은 기존 Fixed와 논문식 Fixed의 nDCG를 넘지 못했다.
기존 Fixed 대비 S −.03312, L −.00896; 논문식 Fixed 대비 S −.03012, L −.00596이다.

| 방식 | P@1 | Recall@10 | MRR@10 | nDCG@10 | Mean ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.4189 | 0.7406 | 0.5232 | 0.5755 | 3.82 | 4.77 |
| Dense | 0.6546 | 0.8988 | 0.7452 | 0.7829 | 55.22 | 83.70 |
| 기존 Fixed | 0.6852 | 0.8946 | 0.7615 | 0.7940 | 45.19 | 65.92 |
| 기존 Router | 0.6838 | 0.8918 | 0.7608 | 0.7930 | 46.91 | 61.60 |
| 논문식 Fixed | 0.6768 | 0.8932 | 0.7577 | 0.7910 | 55.19 | 71.43 |
| QDAP-S | 0.6325 | 0.8724 | 0.7242 | 0.7609 | 57.98 | 79.01 |
| QDAP-L | 0.6602 | 0.8988 | 0.7481 | 0.7851 | 83.36 | 143.59 |
| Oracle α | 0.7323 | 0.9126 | 0.7981 | 0.8261 | — | — |

품질은 비합성 Test 721개 전체, latency는 같은 30질문×2회=방법별60회다.
Oracle은 정답을 사용하는 사후 진단값이므로 배포 가능한 latency를 제시하지 않는다.
S는 literal CE/seed42/epoch4/argmax, L은 literal CE/seed42/epoch1/argmax가
공식 Validation에서 선택됐다. 모든 후보는 5 epoch 학습했으며 Test로 다시 고르지 않았다.

L의 trainable parameter는 567,858,285개, 두 CE 실행의 peak allocated GPU
memory는 각각 5.245GiB였다. 선택 checkpoint에서 embedding과 첫/마지막
Transformer block의 실제 갱신을 확인했다. 두 L 실행은 각각 약53.9분/53.5분이었다.

결과 원본은 `qdap-official-2026-10-30/results.json`, 전체 학습·선택 기록은
동일 폴더의 `training-records.json`, 원시 timing은 `latency-samples.json`에 있다.
공유 평가기에서 8방식×721=5,768개 query-ranking의 4개 지표를 독립 재계산했고,
420개 timing sample의 mean/p95와 선택·checkpoint·입력 해시 검증을 통과했다.

학습 후 최초 freeze는 로컬 snapshot의 `pytorch_model.bin` 대신
`model.safetensors`만 가정해 중단됐다. 두 형식을 지원하도록 회귀 테스트로
수정한 뒤, 저장 checkpoint를 그대로 사용해 freeze/Test/latency/검증만 재개했다.
이 복구 과정에서 학습 설정이나 Validation 선택 기준을 바꾸지 않았다.

## 논문·참고 코드 대조

논문은 Query-Adaptive Hybrid Search, DOI `10.3390/make8040091`이다.
참고 코드는 `madosh/query-adaptive-hybrid-search`의 commit
`9e644b6362e5d429b7b4154370efec2d36b98691`을 고정했다. 참고 구현을 논문
저자의 공식 코드라고 확인한 것은 아니다.

Dense/BM25 각각 Top-100의 합집합을 만들고, 합집합 내에서 각 점수 채널을
min-max 정규화한다. 상수 채널은 0으로 둔다. 결합은
`alpha * dense + (1-alpha) * sparse`이며 alpha는 Dense 비중이다.
후보 경계의 동점은 원래 corpus 순서, 최종 결합 동점은 문서 ID 순서다.

참고 코드 내부에도 차이가 있다. 학습 target 생성은 제공된 후보 점수 배열을
정규화하지만, `fuse_results` 검색 helper는 각 검색기의 Top-100 목록을 따로
정규화한 뒤 다른 목록에만 있는 문서의 해당 채널을 0으로 채운다. 이번에는
기존 결과표와 일치하도록 합집합 모든 문서의 실제 두 점수를 계산하고
합집합에서 정규화하는 동일한 규칙을 학습·Fixed·QDAP 평가에 적용했다.
따라서 참고 검색 helper까지 byte-for-byte 동일한 재현이라고 주장하지 않는다.

예측 구조는 Linear(1024,101) → Conv1d(1,1,kernel7,padding3) → Softmax다.
각 alpha=0,.01,…,1의 nDCG@10에 Softmax를 적용해 target을 만든다.
IDCG는 후보 밖 문서도 포함한 전체 qrels로 계산하며 학습 시 graded gain은
참고 코드처럼 `2**relevance-1`을 사용한다. 비합성 평가 질문은 정답 문서가
하나라 공유 평가기의 binary nDCG와 일치한다.

손실은 `.62 * CE + .38 * sum(abs(CDF(pred)-CDF(target)))`다.
원문의 squared-target CE는 재정규화하지 않으며, 참고 코드는 제곱 후
합이 1이 되도록 재정규화한다. 두 해석을 별도로 학습·기록한다.
추론은 기대값(mean)과 첫 최대 bin(argmax)을 공식 Validation에서 비교한다.

## 학습·선택 프로토콜

사용자의 선택에 따라 공식 Train **43,394개 전부**를 사용한다. 기존 표의
6,000개 학습/1,000개 내부 calibration은 사용하지 않는다. 공식 Validation
비합성 **745개**에서 epoch, CE 해석, seed, readout을 nDCG@10으로 선택한다.
동률은 후보를 먼저 평가한 순서로 해결한다. 선택 설정과 checkpoint 해시를
`selection.json`에 동결한 뒤 Test qrels를 처음 평가 코드에 전달한다.

- QDAP-S: 원본 BGE-M3의 동결된 query vector, Head만 학습. Seeds42/43/44.
- QDAP-L: 원본 BGE-M3 snapshot을 별도 Encoder로 로드해 Encoder 전체와
  Head를 학습. CLS/L2 pooling, seed42. 금융 학습 checkpoint는 사용하지 않는다.
- 공통: 5 epochs, AdamW lr1e-4/weight_decay.01, effective batch128,
  OneCycle warmup.05, gradient clipping1.
- RTX4060 8GB 자원에 맞춰 L은 BF16 Encoder/Adam state, FP32 Head,
  gradient checkpointing, microbatch8와 최대128 query tokens를 사용한다.
  검색용 Encoder와 문서 벡터는 갱신하지 않는다.

공식 query ID와 parent는 서로 다른 분할에 겹치지 않는다. 정규화 문장이
Train/Validation 사이 **50종**, Train/Test 사이 **60종** 겹친다. 사용자가
지정한 공식 분할과 Train 전체를 그대로 유지하고 이를 공개한다.
Test는 이전 실험에 노출된 공식 Test의 재사용이며 미사용 독립 확증이 아니다.

## 비교 방법과 측정 조건

8개 방법은 BM25, Dense, 기존 Fixed, 기존 Router, 논문식 Fixed, QDAP-S,
QDAP-L, Oracle alpha다. 기존 Fixed는 main의
`dense + .05 * BM25/max(BM25)`를 전체 corpus에 적용한다. 기존 Router는
main에 보존된 `docs/experiments/adaptive/router.json`의 세 모드 정책이다.
논문식 Fixed는 QDAP와 같은 합집합·정규화에서 Validation으로 alpha를 고른다.
Oracle은 Test 정답으로 101개 alpha 중 nDCG가 최대인 것을 고르는 진단용
값이며 모델 선택이나 배포 가능한 요청 처리에 사용하지 않는다.

품질은 비합성 Test **721개** 전체에서 P@1, Recall@10, MRR@10, nDCG@10을
기록한다. 기존 표와 맞추어 원본 FP32 query/document cache와 float64 점수
계산을 유지한다. 실제 latency는 새 BF16 query encoding을 포함해 측정하므로
캐시 품질과 live ranking이 항상 동일하다고 가정하지 않고 raw sample에
동일 여부를 남긴다.

Latency는 RTX4060, CPU4, warm batch1이다. 같은30개 질문을 방법별2회,
각 block10회 warmup, 두 번째 회차 방법 순서를 반대로 한다. 토큰화,
Encoder, 전체 corpus Dense/BM25 검색, 결합·Router·Top10을 포함한다.
L은 독립 CUDA stream에서 Dense query 경로와 중첩 실행한다. 모델/인덱스
로딩, HTTP, 동시 요청은 제외한다. Oracle의 배포 latency는 정의하지 않는다.

## 재현

기존 BC Card 데이터·벡터와 고정 revision BGE-M3가 로컬에 필요하다.
아래 `INPUTS`, `OUT`, `PYTHON`은 실행 환경에 맞는 경로로 지정한다.
GPU 접근이 가능한 환경에서 QDAP 워크트리를 현재 디렉터리로 사용한다.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.
PYTHON=/home/shannon/__dev/hybrid-document-search/.venv/bin/python
INPUTS=/home/shannon/__dev/hybrid-document-search
OUT=/tmp/qdap-issue33-run
$PYTHON -m scripts.run_qdap_benchmark prepare --inputs "$INPUTS" --output "$OUT"
$PYTHON -m scripts.run_qdap_benchmark train --variant S --inputs "$INPUTS" --output "$OUT"
$PYTHON -m scripts.run_qdap_benchmark train --variant L --inputs "$INPUTS" --output "$OUT"
$PYTHON -m scripts.run_qdap_benchmark freeze --inputs "$INPUTS" --output "$OUT"
$PYTHON -m scripts.run_qdap_benchmark evaluate --inputs "$INPUTS" --output "$OUT"
$PYTHON -m scripts.run_qdap_benchmark latency --inputs "$INPUTS" --output "$OUT"
$PYTHON -m scripts.run_qdap_benchmark verify --inputs "$INPUTS" --output "$OUT"
```

`prepare`와`freeze`는 기존 결과를 덮어쓰지 않는다. `train`은 완료된5epoch
설정을 건너뛰며 불완전한 설정은 자동 덮어쓰지 않는다. `evaluate`는 이미
작성된 Test 보고서를 덮어쓰지 않고 `verify`로 기존 결과를 재검증한다.
수 GB 모델·원문·query별 산출물은 로컬 OUT에 보존하고, Git 문서에는
요약·프로토콜·학습/선택 기록·검증 결과와 해시를 보존한다.

이 구현은 BC Card/BGE-M3 조건에서 QDAP weighting을 재현한다. 논문의
MLDR/MIRACL, mGTE, antagonist mining과 Dense 공동 학습까지 재현하지 않는다.

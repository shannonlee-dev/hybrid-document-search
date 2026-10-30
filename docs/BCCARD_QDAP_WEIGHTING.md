# QDAP-S/L: BC Card 평가 — 2026-10-30

**QDAP-S/L 모두 Fixed Fusion을 넘지 못했다.** nDCG@10은 Fixed **0.7940**,
S **0.7609**, L **0.7851**이다. Query-adaptive weighting을 비교한 실험이며,
Antagonist Mining 기반 Dense 재학습은 제외했다.

## 실험 조건

검색용 BGE-M3와 문서 벡터를 고정하고, Dense/BM25 Top100 합집합을 채널별 min-max 정규화해
`α × Dense + (1−α) × BM25`로 결합했다. α는 Dense 비중이며 0~1의 101개 bin이다.
참고 구현의 Top100별 정규화·누락 채널 0 채움과 달리 합집합 전체의 실제 두 점수를 사용했다.

| 모델 | 학습 대상 | 비교 seed | Validation 선택 |
|---|---|---|---|
| QDAP-S | 고정 query vector 위 Head | 42/43/44 | literal / seed42 / epoch4 / argmax |
| QDAP-L | 별도 BGE-M3 예측 Encoder 전체와 Head, CLS/L2 pooling | 42 | literal / seed42 / epoch1 / argmax |

- Head: Linear(1024,101) → Conv1d(kernel=7, padding=3) → Softmax.
- Train 43,394개로 5 epochs 학습하고 비합성 Validation 745개의 nDCG@10으로
  CE 해석·seed·epoch·mean/argmax를 선택·동결한 뒤 비합성 Test 721개를 평가했다.
- Target은 α별 graded nDCG@10의 Softmax, IDCG는 전체 qrels 기준이다.
  Loss는 `.62 × squared-target CE + .38 × sum(abs(CDF(pred)−CDF(target)))`다.
  CE target 제곱 후 재정규화하는 `reference`와 하지 않는 `literal`을 비교했다.
- AdamW lr=1e-4, weight decay=.01, effective batch=128, OneCycle warmup=.05, clipping=1.
  L은 BF16 Encoder/Adam state, FP32 Head, gradient checkpointing, microbatch=8, max128 tokens다.
- 분할 간 query ID·parent 중복은 없지만 정규화 문장은 Train/Validation 50종,
  Train/Test 60종이 겹친다. 이미 노출된 공식 Test를 재사용한 탐색적 비교다.

기준 논문은 `Query-Adaptive Hybrid Search`(DOI `10.3390/make8040091`), 참고 구현은
`madosh/query-adaptive-hybrid-search` commit `9e644b6362e5d429b7b4154370efec2d36b98691`이다.
저자 공식 코드 여부는 미확인이다. 후보별 결과는 `validation/2026-10-30/qdap/training-records.json`에 있다.

## 결과

공식 비교 기준은 `experiments/bccard/test-summary.json`의 Main canonical이다.
저장된 main replay의 721개 질문·네 지표는 canonical과 절대 오차 1e-12 이내로 일치한다.

| 방식 / export method | P@1 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 / `bm25` | 0.4189 | 0.7406 | 0.5232 | 0.5755 |
| Dense / `dense` | 0.6560 | 0.8988 | 0.7460 | 0.7835 |
| Fixed / `score_fusion` | 0.6852 | 0.8946 | 0.7615 | **0.7940** |

QDAP 품질 평가는 고정 FP32 벡터와 float64 점수를 사용했다.

| 방식 / export method | P@1 | Recall@10 | MRR@10 | nDCG@10 | Mean ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| 내부 BM25 / `qdap_bm25` | 0.4189 | 0.7406 | 0.5232 | 0.5755 | 3.82 | 4.77 |
| 내부 Dense / `qdap_dense_fp64` | 0.6546 | 0.8988 | 0.7452 | 0.7829 | 55.22 | 83.70 |
| 내부 Fixed / `qdap_fixed_fp64` | 0.6852 | 0.8946 | 0.7615 | 0.7940 | 45.19 | 65.92 |
| 내부 Router / `qdap_router_fp64` | 0.6838 | 0.8918 | 0.7608 | 0.7930 | 46.91 | 61.60 |
| Paper Fixed / `paper_fixed` | 0.6768 | 0.8932 | 0.7577 | 0.7910 | 55.19 | 71.43 |
| QDAP-S / `qdap_s` | 0.6325 | 0.8724 | 0.7242 | 0.7609 | 57.98 | 79.01 |
| QDAP-L / `qdap_l` | 0.6602 | 0.8988 | 0.7481 | 0.7851 | 83.36 | 143.59 |
| Oracle α / `oracle_alpha` | 0.7323 | 0.9126 | 0.7981 | 0.8261 | — | — |

- Canonical Fixed는 전체 corpus에 `Dense + .05 × BM25/max(BM25)`를 적용한다.
  Paper Fixed는 QDAP과 같은 Top100 union/min-max 조건에서 Validation으로 α=.84를 선택했다.
- Main Dense는 FAISS FP32 단건 검색, 내부 Dense는 NumPy float64·문서 ID 동점 정렬을 쓴다.
  Canonical 원 실행과 main replay의 벡터 해시 차이는 `baseline-audit.json`에 기록했다.
- S/L은 Canonical Fixed 대비 −.03312/−.00896, Paper Fixed 대비 −.03012/−.00596이다.
  L은 Test 721개 모두 α=.98을 예측했다.
- Oracle은 Test 정답으로 최적 α를 고른 사후 진단값이다. 동일 점수 공간의 Paper Fixed 대비
  +.03513이지만 학습 가능한 성능을 보장하지 않는다.

Latency는 RTX4060 8GB·CPU 4 threads·warm batch1에서 같은 30질문을 방법별 2회 측정했다
(block별 warmup 10회, 두 번째 회차 역순). 토큰화·BF16 query encoding·전체 검색·결합·Top10을
포함하고 로딩·HTTP·동시 요청은 제외했다. FP32 품질 평가와 정밀도가 다르며 이 실행 안에서만 비교한다.

## 결과 파일

저장 위치는 `docs/validation/2026-10-30/qdap/`이다. Checkpoint·전체 vector·원문·원시 로그는 로컬에 보관한다.

| 파일 | 내용 |
|---|---|
| `per-query.jsonl` | QDAP 721개 × 8방식 = 5,768행 |
| `main-baseline-per-query.jsonl`, `baseline-audit.json` | Main replay 2,163행과 canonical 대조 |
| `per-query-manifest.json` | 입력·결과 해시와 지표 재검증 |
| `protocol.json`, `selection.json`, `training-records.json`, `environment.json` | 학습·선택·모델·코드·환경 기록 |
| `results.json`, `latency.json` | 최종 품질과 latency |

JSONL에는 실행·질문 ID, 네 지표, Top10 문서·relevant rank, selected/oracle α,
Router mode·BM25 weight를 남겼다. 총 7,931행을 qrels로 재검증했다.

## 재실행

### 준비

```bash
uv sync --locked --extra sparse --extra dense
```

Python 3.12 이상이 필요하다. 경로별 입력은 다음과 같다.

| 인자 | 필요한 파일 |
|---|---|
| `--inputs /data/project` | `data/processed/bccard/`의 공식 manifest·corpus·split별 queries/qrels, `docs/experiments/adaptive/router.json` |
| `--vector-cache /data/vectors` | `encoding.json`, `document-vectors.npy`, `query-vectors-{train,validation,test}.npy` |
| `--model /models/bge-m3` | BGE-M3 로컬 snapshot 전체(weight·config·tokenizer·pooling 포함) |
| `--run-dir runs/qdap-new` | 새 실행 디렉터리 (`--output`과 같은 인자) |

Vector cache 기본값은 `<inputs>/artifacts/bccard-adaptive-vectors-resumed`다.
`evaluation.enterprise.encode_bccard(data_dir, cache_dir, splits=("train", "validation", "test"))`로
준비할 수 있다. 동일 checkpoint 재평가에는 선택 당시 SHA와 같은 모델·벡터가 필요하다.

### 새 학습

```bash
python -m scripts.run_qdap_benchmark prepare \
  --inputs /data/project --vector-cache /data/vectors --run-dir runs/qdap-new
python -m scripts.run_qdap_benchmark train --variant S --run-dir runs/qdap-new
python -m scripts.run_qdap_benchmark train --variant L --run-dir runs/qdap-new \
  --model /models/bge-m3 --model-name BAAI/bge-m3 \
  --model-revision 5617a9f61b028005a4858fdac845db406aefb181
python -m scripts.run_qdap_benchmark freeze \
  --inputs /data/project --vector-cache /data/vectors --run-dir runs/qdap-new \
  --model /models/bge-m3 --model-name BAAI/bge-m3 \
  --model-revision 5617a9f61b028005a4858fdac845db406aefb181
python -m scripts.run_qdap_benchmark evaluate \
  --inputs /data/project --vector-cache /data/vectors --run-dir runs/qdap-new \
  --model /models/bge-m3
```

`prepare`는 Train·Validation 입력을 저장하며 Test query는 분할 누출 검사에만 쓴다.
`train`은 양쪽 CE·공식 seed를 학습하고, `freeze`는 Validation으로 모델과 Paper Fixed α를
한 번 선택한다. `evaluate`는 재선택 없이 Test 721개 × 8방식을 저장한다.

Run은 입력·모델·checkpoint SHA와 원래 `run_id`로 식별하며 이동해도 유지한다.
기존 출력 덮어쓰기는 차단하므로 실패 후에도 새 디렉터리에서 실행한다.
`runs/` 아래 NPZ/PT·models/vectors는 Git ignore 대상이다.

### 기존 checkpoint 평가

원본 `docs/validation/2026-10-30/qdap/`은 보존하고 선택 파일을 새 Run에 복사한다.

```bash
mkdir -p runs/qdap-checkpoint-replay
cp docs/validation/2026-10-30/qdap/{selection,protocol}.json runs/qdap-checkpoint-replay/
python -m scripts.run_qdap_benchmark evaluate \
  --run-dir runs/qdap-checkpoint-replay --inputs /data/project \
  --vector-cache /data/vectors --model /models/bge-m3 \
  --checkpoint-s /checkpoints/qdap-s.pt --checkpoint-l /checkpoints/qdap-l.pt
```

Checkpoint 인자를 생략하면 selection의 Run 상대경로를 사용한다. 현재 평가에 필요한 입력의
SHA를 원본과 대조하며 Train·과거 DAT 캐시는 요구하지 않는다. 원 checkpoint가 없으면 재평가할 수 없다.

### 저장 결과 검증·Report

Test 데이터와 JSONL·동반 provenance JSON만 필요하다. GPU·checkpoint·vector cache는 불필요하다.

```bash
python -m scripts.run_qdap_benchmark verify \
  --inputs /data/project --run-dir docs/validation/2026-10-30/qdap
python -m scripts.export_qdap_results --stage enterprise \
  --inputs /data/project \
  --per-query docs/validation/2026-10-30/qdap/per-query.jsonl \
  --canonical-summary docs/experiments/bccard/test-summary.json \
  --output results/qdap-enterprise-new
```

새 Run은 `--per-query runs/qdap-new/per-query.jsonl --output runs/qdap-new/reports`로 export한다.
`--stage export`는 검증된 bundle 복사, `--stage audit`은 vector cache를 사용하는 별도 Canonical 검색이다.
Audit이 없는 새 Run에는 공식 Canonical 수치를 자동으로 붙이지 않는다.

Report는 PR #37 schema_version=1을 유지한다. `dense`는 Main canonical,
`qdap_dense_fp64`는 내부 Dense다. `config.baseline_role`·`source_run_id`를 보존하고 Oracle은 사후 진단으로,
미저장 score/latency는 null로 표기한다. `--reference-report`는 질문 ID·순서를 721개에 맞춘다.

## 검증 범위

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
QDAP_INPUTS=/data/project uv run --no-sync pytest tests/fusion/test_qdap*.py -q -rs
```

- 기존 검증 기록: 전체 Pytest 929개·QDAP/공통 데이터 로더 101개 통과, Ruff 통과.
  공식 데이터가 없으면 721개 통합 테스트는 skip된다.
- 저장된 7,931행을 replay했고, 원 checkpoint·벡터로 GPU 재평가한 QDAP 5,768행도
  Top10·네 지표·relevant rank·α·Router 값이 원본과 일치했다. Canonical은 저장 결과만 검증했다.
- 원본 16개 파일 SHA는 유지됐다. 상세 기록은 `validation/2026-10-31/qdap-portability.json`,
  SHA 목록은 `tests/fixtures/qdap-original-sha256.json`에 있다.
- 원 실험 소스는 태그 `qdap-original-run-2026-10-30`의 `a91bff65a816c9862279c952de929ea32b73c818`이다.
  현재 코드와의 차이는 Report의 `source.code_provenance`에 기록한다.
- 전체 Fresh Training·새 latency 측정·다른 머신의 bitwise 재현은 검증하지 않았다.
  저장 결과 검증과 checkpoint 재평가는 전체 재학습 재현을 증명하지 않는다.

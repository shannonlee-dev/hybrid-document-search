# BC Card Finance query encoder 2-seed 재현 (Blackwell)

Test 검색 성능은 재현됐고 모델 선택은 일부만 재현됐음. PR #40과 같은 입력으로 Seed42·Seed43을 다른 하드웨어에서 다시 학습했음. Seed43은 Validation 곡선과 checkpoint 선택(step 4096)이 수치적으로 재현됐음. Seed42는 곡선이 허용오차 안이지만 step 2048을 골랐고, 두 adapted encoder의 Validation 선택 α도 바뀌었음. Test nDCG@10은 사전 등록한 허용오차 안이고 두 seed 모두 matched 원본 BGE baseline보다 높음.

학습 파이프라인, protocol, PR #40 evidence는 건드리지 않았음. 독립 실행 결과만 추가함.

Evidence: [`docs/validation/2026-11-02/finance-repro-blackwell/`](validation/2026-11-02/finance-repro-blackwell/).
기준: PR #40([`finance-retrain/`](validation/2026-11-02/finance-retrain/)), Seed42만 commit `c95c3fb`(`docs/validation/2026-11-01/finance/`).

## 실행 조건

코드는 PR #40 head `b6f8649a`를 clean detached worktree로 썼음. Query encoder 소스 6개는 과거 학습 기록과 byte 단위로 같음.

입력은 BC Card manifest `b6bb7822…`(36,281 chunks, train 43,394 중 244개 제외 후 43,150개, primary Validation 745 / Test 721), BGE-M3 `5617a9f6`(고정 파일 해시 10개), release [`finance-encoder-inputs-v1`](https://github.com/shannonlee-dev/hybrid-document-search/releases/tag/finance-encoder-inputs-v1)의 원본 WSL2 vector cache(archive `62138a1c…`)임. Cache 파일 5개 해시가 모두 일치함.

환경은 RTX PRO 6000 Blackwell Max-Q 2장(driver 610.57.04), AMD EPYC 9554, Linux 6.8, Torch 2.14.1+cu130, transformers 5.18.0, sentence-transformers 6.1.0, numpy 2.5.3(`uv --locked`)임. Seed42는 GPU0, Seed43은 GPU1에서 동시에 돌렸음. 기준 환경은 WSL2의 RTX 4060이고 패키지 버전은 같음.

Protocol은 그대로임. 4,096 steps, batch 16, 마지막 4 block(50,384,896 parameters), AdamW 2e-5 / weight decay .01, τ .05, tether 1, hard negative 8개, clipping 1, 128/512 tokens, FP32 parameter + BF16 autocast. Checkpoint {0, 64, …, 4096}는 Validation Dense binary nDCG@10으로 고르고 동점이면 이른 step을 씀. Fixed Score Fusion α는 Validation에서 0:.01:1 grid로 고르고 동점이면 작은 α, 그다음 baseline을 우선함. 공통 α=.84도 따로 평가함.

실행 순서는 사전 등록 → 학습 → score cache → BF16 baseline → `select` → 동결 → `checkpoint`(adapted Test vector 생성) → `evaluate` → `export` → `verify`임. PR #40 handoff는 `checkpoint`를 `select`보다 먼저 돌리는데, Test vector가 생기기 전에 선택을 동결하려고 순서만 바꿨음. 알고리즘은 같고, 이 순서로 만든 baseline Test vector는 `checkpoint`가 만든 것과 bitwise 동일함.

Loss는 [`scripts/train_query_encoder_with_loss_log.py`](../scripts/train_query_encoder_with_loss_log.py)로 기록했음. 실제 쓴 파일을 byte 그대로 넣었고, SHA-256 `1d632786…`은 학습 전에 사전 등록 해시 목록과 `environment.json`(로컬 이름 `tools/train_with_loss_log.py`)에 남아 있음. `scripts.train_query_encoder.main`을 같은 인자로 호출하고 학습 모듈의 `write_json`만 감싸서 `progress.json` 기록을 `progress-log.jsonl`에도 덧붙임. 인자, seed, 데이터 순서, 모델, optimizer는 건드리지 않음.

## 재현 수준

판정 기준은 학습 전에 동결한 [`preregistration.md`](validation/2026-11-02/finance-repro-blackwell/preregistration.md)를 따름.

| 수준 | 결과 |
|---|---|
| Exact-input reproduction | 재현됨. 데이터셋, 모델, cache 해시가 일치하고 CPU FP64로 계산한 Dense/BM25 score cache도 PR #40과 bitwise 동일함. |
| Bitwise training reproduction | 재현 안 됨. 두 checkpoint 해시 모두 다름. |
| A. 학습 수치 재현 | Seed43 재현됨. 최대 \|Δ\| .0018(허용 .005), step 4096 다시 선택. Seed42는 재현 안 됨. 곡선은 허용 범위(최대 \|Δ\| .0022)지만 4096 대신 step 2048 선택. |
| B. 모델 선택 재현 | 재현 안 됨. 최종 encoder(seed43)와 baseline α(.84)는 같지만 adapted α가 다름. seed42 .88 vs .82, seed43 .89 vs .83. |
| C. 검색 성능 재현 | 허용오차 안에서 재현됨. Baseline 최대 \|Δ\| .0014(허용 .002), Seed43 \|Δ nDCG@10\| ≤ .0022(허용 .005). |
| D. matched baseline 대비 우위 | 유지됨. Seed43 gain +.0211, 95% bootstrap CI [+.0072, +.0355]. 모든 seed·α 정책 조합에서 양수임. |

## 학습 곡선

![Validation checkpoint curves](validation/2026-11-02/finance-repro-blackwell/checkpoint-curves.svg)

| Step | Seed42 | 기준 (c95c3fb) | Δ | Seed43 | 기준 (PR #40) | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | .788180 | .787339 | +.000841 | .788180 | .787339 | +.000841 |
| 64 | .796553 | .796406 | +.000147 | .793974 | .795320 | −.001346 |
| 128 | .796713 | .798620 | −.001907 | .795478 | .794551 | +.000927 |
| 256 | .797622 | .799214 | −.001592 | .797824 | .799577 | −.001752 |
| 512 | .800336 | .799333 | +.001003 | .796014 | .794696 | +.001317 |
| 1024 | .802714 | .803959 | −.001245 | .809317 | .807949 | +.001369 |
| 2048 | **.814211** | .811984 | +.002227 | .814365 | .814731 | −.000366 |
| 4096 | .814012 | **.815200** | −.001187 | **.822078** | **.822511** | −.000433 |

굵은 값이 각 run의 선택 step임. Seed42는 step 2048이 4096보다 .000199 높아서 선택이 바뀌었음. 기준 run에서는 4096이 .003216 앞섰음. Seed43은 4096이 2위보다 .0077 높았음. Checkpoint 해시는 Seed42 `c78c8db1…`(기준 `361e09ee…`), Seed43 `fbbd5ed0…`(기준 `d36f5c66…`)임. [`training-loss.json`](validation/2026-11-02/finance-repro-blackwell/training-loss.json)에는 64 step마다 기록된 단일 batch loss가 있음.

## Validation α 안정성

선택 결과는 바꾸지 않았음.

![Validation alpha stability](validation/2026-11-02/finance-repro-blackwell/alpha-stability.svg)

| Encoder | 선택 α (기준) | 최고 Validation nDCG@10 | 기준 α에서의 차이 | 1e-4 이내 α | 1e-3 이내 α | 5e-3 이내 α |
|---|---|---:|---:|---|---|---|
| 원본 BGE | .84 (.84) | .805739 | 0 | {.84} | .84–.86 | .74–.76, .78–.89 |
| Seed42 | .88 (.82) | .825757 | .00128 | {.88} | {.79}, .83–.84, .88–.89 | .69–.92 |
| Seed43 | .89 (.83) | .830633 | .00125 | {.82, .89} | .81–.82, {.89} | .76–.96 |

최적점 근처 곡선이 평평하고 들쭉날쭉함. Seed43은 α .89에서 .830633, .82에서 .830630으로 봉우리가 두 개이고 차이는 3×10⁻⁶임. Validation 745개 query에 paired bootstrap을 돌리면 선택 α가 기준 α보다 나은 정도는 Seed43 +.0012(95% CI [−.0044, +.0069]), Seed42 +.0013([−.0047, +.0073])임. Seed43은 683개, Seed42는 681개 query에서 차이가 없었음. 선택 α는 같은 745개 query에서 argmax로 뽑혔으니 이 차이는 위쪽으로 편향돼 있는데, 그런데도 CI가 0을 포함함.

α 변화는 평평하고 noisy한 objective에서 argmax를 취한 결과로 보는 게 맞고, 최적점이 실질적으로 달라졌다는 근거는 아님. 그래도 모델 선택 비재현으로 기록했고 동결된 α는 그대로 둠.

## Test 결과

721개 primary query, 동결된 선택 기준.

| 방법 | α | P@1 | Recall@10 | MRR@10 | nDCG@10 | PR #40 대비 Δ nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| 원본 BGE + Fixed (공통 = 선택) | .84 | .678225 | .891817 | .758435 | .791242 | +.000310 |
| Seed42 + Fixed, 공통 | .84 | .692094 | .920943 | .775761 | .811276 | n/a |
| Seed42 + Fixed, Validation 선택 | .88 | .683773 | .916782 | .770681 | .806579 | n/a |
| Seed43 + Fixed, 공통 | .84 | .692094 | .919556 | .777347 | .812386 | −.002137 |
| **Seed43 + Fixed, Validation 선택 (최종)** | .89 | .693481 | .918169 | .779044 | .813412 | −.001538 (기준 α .83) |

전체 정밀도 값과 네 지표 차이는 [`analysis.json`](validation/2026-11-02/finance-repro-blackwell/analysis.json)의 `test`에 있음. PR #40에는 Seed42 Test 결과가 없음. 나머지 지표 차이는 baseline P@1 +.001387, Recall −.001387, MRR +.000786이고 Seed43(α .84)은 P@1 −.001387, Recall −.002774, MRR −.001940임.

Paired bootstrap은 사후 불확실성 분석이고 새 held-out 실험이 아님. Seed 20261102, 20,000 resamples, percentile CI.

| nDCG@10 차이 | 평균 | 95% CI | W/T/L |
|---|---:|---|---|
| Seed43 α .84 − baseline α .84 (primary) | +.021144 | [+.007176, +.035484] | 87/585/49 |
| Seed43 선택 α .89 − baseline | +.022169 | [+.007807, +.036945] | 89/584/48 |
| Seed42 α .84 − baseline | +.020033 | [+.007120, +.033529] | 78/596/47 |
| Seed42 선택 α .88 − baseline | +.015336 | [+.002437, +.028728] | 80/589/52 |
| PR #40 Seed43 α .84 − PR #40 baseline (같은 방법) | +.023591 | [+.009933, +.037822] | 89/585/47 |
| 재현 − PR #40 Seed43, α .84 (run 간) | −.002137 | [−.006354, +.001448] | 13/690/18 |

재현 gain이 PR #40보다 .0024 작지만 run 간 차이 CI는 0을 포함함. 이 CI는 학습된 모델을 고정하고 Test 721개 query만 resample한 것이라 query sampling 불확실성만 반영함. Seed, 학습 비결정성, 하드웨어에서 오는 분산은 들어 있지 않고, seed 2개로는 추정하기도 어려움. Run 간 CI가 0을 포함한다는 건 이 query 수로 두 run을 구분할 수 없다는 뜻이지 둘이 동등하다는 뜻은 아님. Test split은 이미 노출된 적이 있어서 이 구간은 새 데이터가 아니라 이 721개 query에 대한 것임.

## 차이의 원인

코드, 패키지 버전, 입력 byte는 모두 같고 CPU FP64 score cache는 bitwise로 재현됨. 기준 머신에서는 Seed43 기록 두 개(`c95c3fb`, PR #40 retrain, 기록된 소요 시간이 다름)가 같은 checkpoint SHA를 보고함. 그 환경 안에서 run 간 결정성이 있다는 것과 맞지만 반복은 한 번뿐임.

반면 GPU 모델, driver, CUDA kernel 선택, host OS(WSL2 vs native Linux), 2-GPU 동시 실행이 기준 환경과 한꺼번에 다름. 차이는 GPU 쪽 부동소수점 동작과 맞아떨어지지만 이 run만으로 GPU architecture 하나를 원인으로 지목할 수는 없음.

민감도는 이미 관찰됨. 원본 모델을 BF16으로 인코딩하는 것만으로 step 0 Validation이 +.00084 바뀌고 baseline Test top-10 721개 중 111개가 달라짐. nDCG@10이 바뀐 건 그중 5개임. Checkpoint 선택 margin 2×10⁻⁴와 α margin 3×10⁻⁶은 이 수준보다 훨씬 작음.

## 한계

- 원래 중복 제외 규칙이 Validation과 Test query text를 모두 씀. 제외된 학습 행 244개 중 150개가 Test에서 나왔음(Test만 126, 양쪽 24). 재현이라 규칙은 그대로 뒀음.
- 공식 Validation·Test split은 이전 탐색에서 이미 쓰였음. 독립적인 held-out 검증은 아님.
- `prepare`는 공용 loader로 Test qrels를 읽지만 버림. `checkpoint`는 Dense-only Test 지표를 계산하는데 선택 동결 뒤에 돌았고 결과는 쓰지 않았음.
- Hard negative는 고정 baseline vector로 채굴하고 same-parent만 mask함. 라벨 없는 relevant chunk가 negative가 될 수 있음.
- 선택은 binary nDCG@10으로 하고 Test는 네 지표를 보고함.
- matched BF16 baseline은 FP32 cached baseline(.791013)이나 Main canonical Fixed(.794015, 결합식이 다름)와 같은 값이 아님.
- Seed42 기준인 `c95c3fb` 기록에는 학습 commit과 source 해시가 없고, α도 FP32 baseline 기준으로 선택됐음.
- Export된 `run_id`는 CLI가 출력 디렉터리 이름을 쓰기 때문에 `"replay"`임. 해시에 묶여 있어서 그대로 뒀음.

## Errata

로컬 `REPORT.md`에 근거가 부족한 서술이 세 개 있었고 여기서 고쳤음.

- Seed43은 α .81–.89, Seed42는 .79–.89가 모두 최적값의 .001 안이라고 썼음. 실제 near-optimal 집합은 연속 구간이 아님. 위 α 표가 맞는 값임.
- Step 0 차이를 "다른 GPU에서의 BF16 인코딩 때문"이라고 단정했음. 하드웨어와 소프트웨어 스택이 함께 달라서 원인을 하나로 좁힐 수 없음.
- 학습이 "초반 몇 step 안에 bitwise로 갈라진다"고 썼는데 측정한 적 없음.

## 검증

```bash
# 커밋된 evidence, 분석, figure 무결성 확인 (데이터 불필요)
uv run --locked python scripts/analyze_finance_reproduction.py check
uv run --locked --extra dense pytest -q tests/dense/test_finance_reproduction_evidence.py
# 공식 BC Card 데이터가 없으면(FINANCE_DATA_DIR 미설정) official-data verify 테스트는 skip됨

# 공식 BC Card 데이터로 ranking에서 per-query 지표 전체 재계산
uv run --locked --extra sparse --extra dense python -m scripts.evaluate_query_encoder verify \
  --data-dir "$FINANCE_DATA" --output-dir docs/validation/2026-11-02/finance-repro-blackwell

# 원본 vector cache
gh release download finance-encoder-inputs-v1 -R shannonlee-dev/hybrid-document-search -p finance-vector-cache.tar.gz
echo "62138a1c381771831d5999758dc27d81cc5a822e7384988bd7159c494824728d  finance-vector-cache.tar.gz" | sha256sum -c
```

실험을 다시 돌리려면 [PR #40 handoff](BCCARD_QUERY_ENCODER_TRAINING.md)를 따르되 `select`를 `train-seed*` 디렉터리로 실행하고 `checkpoint` 전에 동결하면 됨. 로컬 run에서 이 디렉터리를 다시 만들 때는 `analyze_finance_reproduction.py export-evidence --run-dir <run>`을 씀.

| 파일 | 내용 |
|---|---|
| `frozen.json`, `protocol.json`, `per-query.jsonl`, `per-query-manifest.json` | `export` 출력 그대로. 3 encoders × 2 α 정책 × 721 = 4,326행, ID·지표·top-10 document ID만 있고 query text는 없음 |
| `preregistration.md`, `selection-freeze.json` | 학습 전에 동결한 허용오차, Test vector 생성 전에 동결한 선택 |
| `validation-alpha-curves.json` | α별 평균 Validation nDCG@10, 선택·기준·공통 α에서의 per-query 값 |
| `training-loss.json`, `environment.json` | 샘플링한 loss, GPU·driver·패키지·source 해시 |
| `analysis.json`, `*.svg`, `MANIFEST.sha256` | Bootstrap, α 안정성, 곡선, Test 차이, figure, 파일 해시 |

Checkpoint, vector, score matrix, 모델 가중치, cache는 커밋하지 않았음.

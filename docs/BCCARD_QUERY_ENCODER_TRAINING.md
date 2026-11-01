# BC Card Finance query encoder

**동일 BF16 forward·Fixed α=.84에서 원본 BGE의 nDCG@10은 .790932,
Finance-BGE는 .814523이다.** 최종 코드로 seed43을 4,096 steps 새로 학습하고 평가했다.
문서 encoder/vector를 고정하고 query encoder의 마지막 4개 block만 학습한 4차 보조 실험이다.

## 코드 구조

| 모듈 | 책임 |
|---|---|
| `evaluation/query_encoder.py` | Score 준비, Fixed fusion, Validation α·encoder 선택, Test replay |
| `evaluation/query_encoder_training.py` | 학습 입력·중복 검증, hard negative/loss/optimizer, checkpoint 학습·로딩, 모델 인코딩 |
| `evaluation/query_encoder_report.py` | Compact export, 결과 검증, 코드·모델·실행 provenance |

학습 CLI는 `scripts/train_query_encoder.py`, 평가 CLI는 `scripts/evaluate_query_encoder.py`다.
공통 입력 로더와 순위 지표 계산은 학습·replay가 공유하며 별도 training 패키지는 만들지 않는다.

## 실험 조건

- Corpus: BC Card Pair revision `e022cb013f2907e0716ebe40d13f30ed93ffa9b0`, 36,281개 chunk.
  본문만 색인한다. Query/qrels: Triplet revision `f63d59969dba9916bd34c86c82112331890b11da`.
  공식 Train 43,394 / Validation 1,000 / Test 1,000 중 비합성 Validation 745·Test 721개로 선택·평가한다.
- Model: `BAAI/bge-m3`, revision `5617a9f61b028005a4858fdac845db406aefb181`.
  Query-side 마지막 4개 block의 **50,384,896개 parameter**만 업데이트한다.
- Validation/Test 전체 질의와 casefold·공백 정규화 문장이 같은 **244개 학습 행**을 제외해
  43,150개를 사용한다. 평가 질문을 제거하지 않으며 Test 721개 ID/순서·공식 qrels를 유지한다.
  Query ID와 학습 parent 중복도 검사한다.
- Train qrels는 dataset-card chunk-distance 규칙으로 재구성했다. Validation/Test는 공식 label이다.
  Relevance 3인 chunk 하나를 positive로 쓰고, 고정 원본 query vector로 다른 parent의 hard negative
  8개를 채굴한다. In-batch positive를 후보로 포함하고 같은 parent는 지정 positive 외에 mask한다.
- Loss: temperature .05의 contrastive cross-entropy + `1−cosine` tether(weight 1).
  AdamW lr=2e-5, weight decay=.01, gradient clipping=1, batch16, shuffled epochs, seed43,
  **4,096 steps / 65,536개 학습 예제 처리**다. Train max128/Eval max512 tokens다.
- Parameter는 FP32, CUDA forward는 BF16 autocast, 저장 vector는 FP32다.
  양 encoder의 평가 batch16·max512·정규화 조건을 같게 맞췄다.
- Checkpoint는 step0과 64/128/256/512/1024/2048/4096에서 Validation Dense binary nDCG@10으로
  선택한다. Baseline도 후보이며 동점은 이른 step이다. **Step4096, Validation nDCG .822511**이 선택됐다.
- 이어서 Validation Fixed nDCG로 α를 0–1/.01 grid에서 선택했다.
  원본은 **α=.84 / .805624**, Finance는 **α=.83 / .831569**다. 별도로 사전에 정한 공통 α=.84도 평가한다.
  Test로 seed·checkpoint·α를 선택하지 않는다. Test query text는 학습 중 중복 검사에만 쓴다.

두 방식은 같은 문서 vector·BM25 점수·Dense/BM25 Top100 합집합을 사용한다.
합집합 내 실제 채널 점수를 min-max 정규화하고 `α×Dense+(1−α)×BM25`로 결합한다.
Dense는 고정 FP32 문서 vector와 query vector를 FP64로 변환해 내적하며 동점은 corpus 순서다.
BM25는 `char_wb` 2–4 gram, k1=1.5, b=.75, Lucene scoring이다.
문서 vector SHA는 `4d5f1c8d9002bfeac0c7a927aa641b0ab17199ea008ea196fab786b08bb3df3a`다.

## 결과

| 비교 | Dense α | P@1 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| 원본 BGE / 공통 Fixed | .84 | .676838 | .893204 | .757649 | .790932 |
| Finance-BGE / 공통 Fixed | .84 | .693481 | .922330 | .779287 | .814523 |
| Finance-BGE / Validation-selected Fixed | .83 | .694868 | .922330 | .779876 | .814950 |

주 비교는 동일 BF16·공통 α=.84이며 nDCG@10 차이는 **+.023591**이다.
원본의 Validation 선택 α도 .84라 공통 정책과 결과가 같다. Per-query에는 두 정책을 구분한다.
이미 노출된 공식 Validation/Test를 재사용한 탐색적 비교이며 독립적인 새 held-out 검증은 아니다.

Main canonical Fixed **.794015**는 전체 corpus의 `Dense + .05×BM25/max(BM25)`로 결합식이 다르다.
기존 내부 Fixed **.791013**은 FP32 query cache를 쓴 union-minmax 결과다.
Canonical Dense **.783509**는 FAISS FP32 단건 검색, 내부 Dense **.782881**은 FP64 경로다.
이번 표의 BF16·공통 α 대조군과 이 참고값을 같은 run으로 취급하지 않는다.
PR #28 원본 문서 vector는 별도 SHA이며, 이번 비교는 같은 revision·corpus·설정으로 재생성한
vector를 양쪽에 고정했다. Canonical 참고값은 저장된 결과에서 가져오며 이번에 다시 측정하지 않았다.

## 실행 증거

- 학습 코드 commit: `f49ec8b41a70ae663aec2837f51d69ec52e4c061`.
  실제 학습 당시 코드 출처인 이력 정리 전 커밋이며, 현재 브랜치의 조상이 아니다.
  학습 시작 시 clean tree였으며 학습 중 source SHA가 바뀌지 않았다.
- Training Run ID: `finance-seed43-4096-2026-11-02-f49ec8b`.
- 평가 Run ID: `finance-retrained-replay-2026-11-02-f49ec8b`.
- 선택 checkpoint SHA: `d36f5c66c0caf2d8dd451d7e6ed1b1a433ea8aafe2b2d82a2ad5a9cd2ecb4a75`.
- 순서: 단위·전체 테스트 → 실제 BGE 2-step GPU 학습 smoke → 저장·로드·Validation vector 재현
  → 코드 commit → 4,096-step 학습 → 선택 후 별도 Test 인코딩·replay.
  Smoke에서 다시 로드한 Validation 첫 16개 vector의 최대 절대 차이는 0이었다.

최신 evidence 한 묶음만 `validation/2026-11-02/finance-retrain/`에 남긴다.

| 파일 | 내용 |
|---|---|
| `protocol.json` | Dataset/model/vector/code identity, 학습 조건·환경·선택 이력 |
| `frozen.json` | Validation으로 동결한 α·encoder, 공통 α, 입력 SHA |
| `per-query.jsonl` | 두 encoder × 두 α 정책 × 721개 = **2,884행**. 네 지표·Top10·relevant rank·α·policy |
| `per-query-manifest.json` | 결과 집계·질문 순서·입력/결과 SHA·canonical 참고값 |
| `verification.json` | 테스트·실모델 smoke·재학습·결과 재검증 요약 |

Checkpoint·전체 vector·score matrix·중간 배열·원문·진행 로그는 Git에 포함하지 않는다.
새 latency benchmark·paired CI·Finance vs Kanana·oracle/headroom 분석은 수행하지 않았다.

## Seed42·43 재학습 인수인계

브랜치를 clone한 뒤 저장소 루트의 Bash에서 순서대로 실행한다.
Python 3.12·`uv`·CUDA/BF16 GPU가 필요하다. 기준 환경은 RTX 4060, Torch 2.14.1+cu130/CUDA 13.0이다.

### 1. 입력 준비

```bash
set -euo pipefail
git switch feat/finance-query-encoder
export FINANCE_DATA="$PWD/data/processed/bccard"
export FINANCE_CACHE="$PWD/artifacts/bccard-adaptive-vectors-resumed"
export FINANCE_MODEL="$PWD/models/bge-m3"
export FINANCE_RUN="$PWD/runs/finance-repro-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$FINANCE_RUN"
uv run --locked --extra sparse --extra dense python -c \
  'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); print(torch.__version__, torch.cuda.get_device_name(0))'
```

다음 세 입력을 준비한다. 경로는 위 환경변수로 바꿀 수 있다.

| 기본 경로 | 필요한 파일 |
|---|---|
| `data/processed/bccard/` | `manifest.json`, `corpus.jsonl`, `queries_{train,validation,test}.jsonl`, `qrels_{train,validation,test}.jsonl` |
| `artifacts/bccard-adaptive-vectors-resumed/` | `encoding.json`, `document-vectors.npy`, `query-vectors-{train,validation,test}.npy` |
| `models/bge-m3/` | 원본 snapshot의 `model_files_sha256`에 명시된 weights·tokenizer·pooling/config |

**동일 SHA의 데이터·vector cache를 전달받는 경로를 권장**한다.
데이터가 없으면 고정 Pair/Triplet revision으로 준비한다. 출력 디렉터리는 비어 있어야 한다.

```bash
uv run --locked --extra sparse --extra dense python -m scripts.prepare_bccard \
  --download --raw-dir data/raw/bccard --output-dir "$FINANCE_DATA"
```

원본 모델이 없으면 고정 revision으로 받는다.

```bash
uv run --locked --extra sparse --extra dense python - <<'PYMODEL'
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

ref = json.loads(Path("docs/validation/2026-11-02/finance-retrain/protocol.json").read_text())
snapshot_download(
    "BAAI/bge-m3", revision=ref["model"]["revision"],
    local_dir=os.environ["FINANCE_MODEL"], allow_patterns=list(ref["model_files_sha256"]),
)
PYMODEL
```

Vector cache가 없을 때만 재생성한다(FP32·batch1·max8192). 기본 split에는 Train이 없어
**세 split을 명시**한다. 이 함수는 로컬 모델 경로 대신 고정 revision을 hub cache에서 읽는다.
생성 vector의 SHA가 다르면 새 입력 run으로 구분한다.

```bash
uv run --locked --extra sparse --extra dense python - <<'PYVECTORS'
import os
from evaluation.enterprise import encode_bccard
from evaluation.runtime_config import configure_runtime

configure_runtime(2, cuda=True)
encode_bccard(os.environ["FINANCE_DATA"], os.environ["FINANCE_CACHE"],
              splits=("train", "validation", "test"), device="cuda:0", encoder="bge")
PYVECTORS
```

### 2. 학습 전 identity 확인

Manifest·corpus·세 split의 query/qrels·원본 모델·전체 vector SHA와 ID 순서를 검사한다.
불일치하면 중단하고 새 입력 run으로 구분한다. 기준 SHA는 evidence의 `protocol.json`에 있다.
장치·시간을 포함하는 `encoding.json`의 SHA는 별도로 출력한다.

```bash
uv run --locked --extra sparse --extra dense python - <<'PYIDENTITY'
import json
import os
from pathlib import Path
from evaluation.data import file_sha256
from evaluation.query_encoder_training import load_encoder_inputs

ref = json.loads(Path("docs/validation/2026-11-02/finance-retrain/protocol.json").read_text())
data, cache, model = [Path(os.environ[k]) for k in ("FINANCE_DATA", "FINANCE_CACHE", "FINANCE_MODEL")]
checks = [(data / "manifest.json", ref["dataset_manifest_sha256"])]
for root, hashes in ((data, ref["dataset"]["output_sha256"]),
                     (cache, ref["vector_sha256"]), (model, ref["model_files_sha256"])):
    checks.extend((root / name, sha) for name, sha in hashes.items())
failed = [str(path) for path, sha in checks if file_sha256(path) != sha]
assert not failed, f"Input identity mismatch: {failed}"
load_encoder_inputs(data, cache, ("train", "validation", "test"))  # ID/순서·metadata도 검사
encoding = json.loads((cache / "encoding.json").read_text())
assert encoding["model"]["revision"] == ref["model"]["revision"]
expected = ref["encoders"]["seed43"]["training_input_sha256"]["encoding.json"]
print("Input identity OK; encoding.json:", file_sha256(cache / "encoding.json"), "reference:", expected)
PYIDENTITY
```

### 3. 두 seed 학습 → checkpoint 평가 → Validation 선택 → Test replay

새 출력 경로에서 Seed42·43을 순차 학습한다. **기본 steps는 256이므로 `--steps 4096`을 명시한다.**

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for seed in 42 43; do
  uv run --locked --extra sparse --extra dense python -m scripts.train_query_encoder \
    --data-dir "$FINANCE_DATA" --cache-dir "$FINANCE_CACHE" --model "$FINANCE_MODEL" \
    --output-dir "$FINANCE_RUN/train-seed$seed" --seed "$seed" --steps 4096 \
    --layers 4 --batch-size 16 --device cuda --threads 2
done
```

학습의 `frozen.json`이 Validation Dense 기준 checkpoint를 확정한다.
`checkpoint`는 이를 로드해 Test vector를 만든다. 함께 출력되는 Dense Test 점수는 선택에 쓰지 않는다.

```bash
for seed in 42 43; do
  uv run --locked --extra sparse --extra dense python -m scripts.evaluate_query_encoder checkpoint \
    --data-dir "$FINANCE_DATA" --cache-dir "$FINANCE_CACHE" --model "$FINANCE_MODEL" \
    --checkpoint-dir "$FINANCE_RUN/train-seed$seed" --output-dir "$FINANCE_RUN/checkpoint-seed$seed" \
    --layers 4 --device cuda --threads 4
done
uv run --locked --extra sparse --extra dense python -m scripts.evaluate_query_encoder prepare \
  --data-dir "$FINANCE_DATA" --cache-dir "$FINANCE_CACHE" --output-dir "$FINANCE_RUN/scores"
uv run --locked --extra sparse --extra dense python -m scripts.evaluate_query_encoder baseline \
  --data-dir "$FINANCE_DATA" --cache-dir "$FINANCE_CACHE" --model "$FINANCE_MODEL" \
  --test-source "$FINANCE_RUN/checkpoint-seed43" --output-dir "$FINANCE_RUN/baseline" \
  --device cuda --threads 4
uv run --locked --extra sparse --extra dense python -m scripts.evaluate_query_encoder select \
  --data-dir "$FINANCE_DATA" --cache-dir "$FINANCE_CACHE" --score-dir "$FINANCE_RUN/scores" \
  --baseline-vectors-dir "$FINANCE_RUN/baseline" --model "$FINANCE_MODEL" \
  --encoder "seed42=$FINANCE_RUN/checkpoint-seed42" --encoder "seed43=$FINANCE_RUN/checkpoint-seed43" \
  --shared-alpha .84 --output-dir "$FINANCE_RUN/replay"
uv run --locked --extra sparse --extra dense python -m scripts.evaluate_query_encoder evaluate \
  --data-dir "$FINANCE_DATA" --cache-dir "$FINANCE_CACHE" --score-dir "$FINANCE_RUN/scores" \
  --baseline-vectors-dir "$FINANCE_RUN/baseline" \
  --encoder "seed42=$FINANCE_RUN/checkpoint-seed42" --encoder "seed43=$FINANCE_RUN/checkpoint-seed43" \
  --output-dir "$FINANCE_RUN/replay"
uv run --locked --extra sparse --extra dense python -m scripts.evaluate_query_encoder export \
  --data-dir "$FINANCE_DATA" --run-dir "$FINANCE_RUN/replay" --output-dir "$FINANCE_RUN/compact" \
  --canonical-summary docs/experiments/bccard/test-summary.json
uv run --locked --extra sparse --extra dense python -m scripts.evaluate_query_encoder verify \
  --data-dir "$FINANCE_DATA" --output-dir "$FINANCE_RUN/compact"
```

실패하면 해당 단계를 새 출력 경로로 재실행한 뒤 진행한다.
`baseline`은 원본 Validation을 BF16으로 인코딩하고 Test vector를 재사용한다.
`select`는 Validation만으로 원본·두 seed의 α와 최종 encoder를 동결한다.
동일 문서 vector·BM25·Top100 합집합·min-max에서 공통 α=.84와 선택 α를 모두 평가한다.

### 4. 결과 비교와 전달

```bash
uv run --locked --extra sparse --extra dense python - <<'PYRESULTS'
import json
import os
from pathlib import Path

run = Path(os.environ["FINANCE_RUN"])
for seed in (42, 43):
    frozen = json.loads((run / f"train-seed{seed}/frozen.json").read_text())
    print(f"seed{seed}:", {k: frozen[k] for k in
          ("selected_step", "validation_primary_ndcg", "checkpoint_sha256")})
    print("Validation curve:", [(r["step"], r["validation_ndcg"]) for r in frozen["history"]])
frozen = json.loads((run / "replay/frozen.json").read_text())
print("Validation Fixed / alpha:", frozen["models"])
print("Validation-selected encoder:", frozen["selected_encoder"])
summary = json.loads((run / "replay/summary.json").read_text())
print("Test / both policies:", json.dumps(summary["comparisons"], indent=2))
PYRESULTS
```

**기존 기준값**: Seed42는 커밋 `c95c3fb`의 `docs/validation/2026-11-01/finance/`
protocol·frozen, Seed43은 현재 `2026-11-02/finance-retrain` 기록이다. 과거 Validation 추이는
[당시 protocol.json](https://github.com/shannonlee-dev/hybrid-document-search/blob/c95c3fb106287749fbec268fb52768e854e00c05/docs/validation/2026-11-01/finance/protocol.json)에서 확인한다.

| 기존 run | 선택 step | Validation Dense nDCG@10 | Validation Fixed nDCG@10 | Fixed α |
|---|---:|---:|---:|---:|
| Seed42 / 과거 실행 | 4096 | .815199858454069 | .8298404923658383 | .82 |
| Seed43 / 정식 재학습 | 4096 | .8225110741406398 | .8315689697637983 | .83 |

| 기존 checkpoint | SHA-256 |
|---|---|
| Seed42 | `361e09ee8496f3227ffdc5a25ae27acb28d679ea123e72bcb8a10dff1476c245` |
| Seed43 | `d36f5c66c0caf2d8dd451d7e6ed1b1a433ea8aafe2b2d82a2ad5a9cd2ecb4a75` |

Seed42는 BF16 학습 모델의 과거 값이며 당시 원본 대조군은 FP32였다.
Seed42 Test는 새 실행에서 확인한다. 기존 compact Test는 선택된 Seed43만 포함하며,
기준 nDCG@10은 공통 α `.814523` / 선택 α `.814950`이다.

입력 identity 일치를 먼저 확인한다. Checkpoint SHA 일치는 강한 재현 근거다.
다른 GPU·CUDA에서 SHA가 달라도 바로 실패로 보지 않고 Validation 추이·선택 checkpoint·α·
검색 네 지표의 차이를 기록한다. **새 Run ID·코드 commit/SHA·환경·결과는 기존 기록과 구분**하며,
선택이 기준과 달라도 유지한다. Test로 seed·checkpoint·α를 재선택하지 않는다.

**전달물은 `compact/`의 네 파일과 `verify` 출력**이다. 두 seed의 선택·추이·provenance와
세 encoder × 두 정책 × 721개 = **4,326행**을 포함한다. 대용량 입력·checkpoint는 Git에 넣지 않는다.
검증에는 dataset과 compact JSON만 필요하다. 기존 evidence 재검증 명령은 다음과 같다.

```bash
uv run --locked --extra sparse --extra dense python -m scripts.evaluate_query_encoder verify \
  --data-dir "$FINANCE_DATA" --output-dir docs/validation/2026-11-02/finance-retrain
```

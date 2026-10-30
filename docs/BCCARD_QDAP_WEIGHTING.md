# QDAP-S/L: BC Card 평가 — 2026-10-30

Issue #33에서 **QDAP의 query-adaptive weighting 핵심 구성요소를 구현해 비교했으나
Fixed Fusion을 넘지 못했다.** S weighting head/loss·mean/argmax와 L 별도 예측 Encoder를
평가했다. Antagonist mining 기반 Dense 재학습은 제외했으므로 논문 전체 재현 결과는 아니다.

## 구현·학습 범위

검색용 BGE-M3와 문서 벡터는 고정했다. 예측 Head는 Linear(1024,101) → Conv1d(kernel=7,
padding=3) → Softmax다. Dense/BM25 Top100 합집합에서 채널별 min-max 정규화 후
`α × Dense + (1−α) × BM25`로 결합한다. α는 Dense 비중이며 0~1의 101개 bin이다.
참고 구현의 Top100별 정규화·누락 채널 0 채움과 달리, 합집합 전체의 실제 두 점수를 쓴다.

| 모델 | 학습 대상 | 비교 seed | Validation 최종 선택 |
|---|---|---|---|
| QDAP-S | 고정 query vector 위 Head | 42/43/44 | literal / seed42 / epoch4 / argmax |
| QDAP-L | 별도 BGE-M3 예측 Encoder 전체와 Head, CLS/L2 pooling | 42 | literal / seed42 / epoch1 / argmax |

- 공식 Train 43,394개 전체로 5 epochs 학습하고, 비합성 Validation 745개의 nDCG@10으로
  CE 해석·seed·epoch·mean/argmax를 선택했다. 선택을 동결한 뒤 비합성 Test **721개 전체**를 평가했다.
- Target은 α별 graded nDCG@10의 Softmax이며 IDCG는 전체 qrels 기준이다.
  Loss는 `.62 × squared-target CE + .38 × sum(abs(CDF(pred)−CDF(target)))`다.
  CE target 제곱 후 재정규화하는 `reference`와 하지 않는 `literal`을 비교했다.
- AdamW lr=1e-4, weight decay=.01, effective batch=128, OneCycle warmup=.05, clipping=1.
  L은 BF16 Encoder/Adam state, FP32 Head, gradient checkpointing, microbatch=8, max128 tokens다.
- 분할 간 query ID·parent 중복은 없다. 정규화 문장은 Train/Validation 50종,
  Train/Test 60종이 겹친다. 이미 노출된 공식 Test를 재사용한 탐색적 비교다.

기준 논문은 `Query-Adaptive Hybrid Search`(DOI `10.3390/make8040091`)다.
참고 구현은 `madosh/query-adaptive-hybrid-search`의 commit
`9e644b6362e5d429b7b4154370efec2d36b98691`이며 저자 공식 코드 여부는 미확인이다.
후보별 Validation 결과는 [training-records.json](qdap-official-2026-10-30/training-records.json)에 있다.

## 결과

### 공식 baseline: Main canonical

[Main canonical](experiments/bccard/test-summary.json)을 공식 비교 기준으로 사용한다.
저장된 main replay의 721개 질문·네 지표가 canonical과 절대 오차 1e-12 이내로 일치한다.

| 방식 / export method | P@1 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 / `bm25` | 0.4189 | 0.7406 | 0.5232 | **0.5755** |
| Dense / `dense` | 0.6560 | 0.8988 | 0.7460 | **0.7835** |
| Fixed / `score_fusion` | 0.6852 | 0.8946 | 0.7615 | **0.7940** |

### QDAP protocol

품질은 고정 FP32 query/document 벡터와 float64 점수로 평가했다. 내부 기준선은 별도 method로 구분한다.

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

Canonical Fixed는 전체 corpus에 `Dense + .05 × BM25/max(BM25)`를 적용한다.
Paper Fixed는 QDAP과 동일한 Top100 union/min-max 조건에서 Validation으로 α=.84를 선택한 별도 기준선이다.
Main Dense는 FAISS FP32 단건 검색, 내부 Dense는 NumPy float64 계산·문서 ID 동점 정렬을 쓴다.
Canonical 원 실행과 main replay의 벡터 해시도 다르며, 대조 근거는
[baseline-audit.json](qdap-official-2026-10-30/baseline-audit.json)에 있다.

S/L의 nDCG@10은 Canonical Fixed 대비 각각 −.03312/−.00896,
Paper Fixed 대비 −.03012/−.00596이다. L은 Test 721개 모두 α=.98을 예측했다.
Oracle−Canonical Fixed는 +.03213으로 조건이 다른 시스템 기준 비교이고,
Oracle−Paper Fixed는 +.03513으로 동일 점수 공간의 사후 개선 여지다.
Oracle은 **Test 정답으로 101개 α 중 최선을 고른 진단값**이며 학습 가능한 성능을 보장하지 않는다.

Latency는 RTX4060 8GB·CPU 4 threads·warm batch1에서 같은 30질문을 방법별 2회씩 측정했다.
Block별 warmup 10회, 두 번째 회차는 방법 순서를 뒤집었다. 토큰화·실시간 BF16 query encoding·
전체 corpus 검색·결합/Router·Top10을 포함하고 로딩·HTTP·동시 요청은 제외했다.
BF16 실시간 순위는 품질 평가의 FP32 cache 순위와 다를 수 있다. Latency 비교는 이 실행 안에서만 유효하다.

## 전달물

저장 폴더는 `docs/qdap-official-2026-10-30/`이다. Checkpoint·전체 vector·원문·원시 로그는 로컬에 보관한다.

| 파일 | 내용 |
|---|---|
| `per-query.jsonl` | QDAP 실행 721개 × 8방식 = 5,768행 |
| `main-baseline-per-query.jsonl`, `baseline-audit.json` | 별도 main replay 721개 × 3방식 = 2,163행과 canonical 대조 |
| `per-query-manifest.json` | 결과·dataset/corpus/query/qrels·encoding/vector 해시, 지표 재검증 |
| `protocol.json`, `selection.json`, `training-records.json`, `environment.json` | 모델 revision·checkpoint 해시·seed·선택 기준·학습 설정·환경·원 실험 소스 해시 |
| `results.json`, `latency.json` | 최종 품질과 latency·측정 조건 |

JSONL에는 `run_id`, `query_id`, P@1·Recall@10·MRR@10·nDCG@10, Top10 문서 ID·relevant rank,
selected/oracle α, Router mode·BM25 weight를 남겼다. α는 Dense 비중, 기존 weight=.05는 BM25 계수다.
총 7,931행의 지표·순위·정답 순위를 qrels로 재검증했다.
5차의 paired CI·query-level headroom·외부 모델 비교는 이 전달물을 사용한 후속 분석 범위다.

## 저장 결과 검증·export

아래는 준비된 BC Card 데이터와 전달물로 `report.json`을 만드는 경로다.
Checkpoint·모델·vector cache 없이 저장 Top10의 지표와 입력·산출물 identity를 확인한다.
`INPUTS`는 준비 데이터가 있는 프로젝트 루트로 지정한다.

```bash
PYTHON=.venv/bin/python
INPUTS=/path/to/prepared-root
HANDOFF=docs/qdap-official-2026-10-30
"$PYTHON" -m scripts.export_qdap_results --stage enterprise \
  --inputs "$INPUTS" --per-query "$HANDOFF/per-query.jsonl" \
  --canonical-summary docs/experiments/bccard/test-summary.json \
  --output results/qdap-enterprise
QDAP_INPUTS="$INPUTS" "$PYTHON" -m pytest -p no:cacheprovider tests/fusion/test_qdap_report.py
```

출력은 `results/qdap-enterprise/report.json`이다. Method 이름은 위 표와 같으며 `dense`는 canonical이다.
방법별 `config.baseline_role`·`source_run_id`로 출처를 구분하고 순위·지표·α는 보존한다.
원본에 없는 score·latency는 null이다. 상위 `encoding`은 QDAP의 identity이며, canonical 원 실행의
encoding과 main replay의 vector 해시는 canonical 방법별 config에 보존한다.
1,000질문 비교 report는 `--reference-report`로 같은 721개 ID·순서에 맞출 수 있다.

새 실행은 `run_qdap_benchmark`의 `--model`에 로컬 snapshot을 명시하고 별도 output을 쓴다.
준비→S/L 학습→Validation 선택·동결→Test 평가→latency 측정→verify 순서로 실행한다.
JSONL 생성은 `--stage export --run-id <실행ID>`, main audit는 `--stage audit --main-run-id <audit ID>`,
`--stage all`은 두 ID가 필요하다. 기존 실행의 재export에는 원래 ID를 지정하며 enterprise는 저장된 ID를 읽는다.
`all`/`audit`은 검색을 수행하고 원 실행의 `verify`는 원래 checkpoint·vector 등이 필요하므로 위 경로와 구분한다.

### 재현성의 범위

원 실험 소스 스냅샷은 태그 `qdap-original-run-2026-10-30`의
`a91bff65a816c9862279c952de929ea32b73c818`로 고정했다. `selection.json`의 소스 6개 SHA-256과
보존된 `selection.json`·`protocol.json`·`results.json`이 이 스냅샷과 일치한다.

`selection.json`의 원 실험 소스 SHA-256과 현재 제출 코드는 일부 다르다. 현재 코드는 후속 정리본이며,
원 실험의 selection·checkpoint·vector·JSONL·결과는 변경하지 않았다.
Report의 `source.code_provenance`에 원 실험·현재 코드 해시와 불일치 목록,
`code.sha256`에 export 코드 해시를 기록한다.

2026-10-30에 저장 순위·예측값으로 5,768행을 재생성하여 원본과 바이트·SHA-256이 같음을 확인했다.
공통 SHA-256: `40adc5fdaad6cb157e7b0a0eae6e2946f8058997c4e1dcef2533379b8812930b`.
[score-parity.json](qdap-official-2026-10-30/score-parity.json)은 당시 후속 점수 경로에서 Validation 745개·
Test 721개의 후보·target과 Test 순위·지표 일치를 확인한 기록이며, Train 검증은 192행에 한정된다.

**저장 결과 검증과 원 실험의 전체 재실행은 구분한다.** 현재 코드로 43,394개 학습부터
Validation 선택·Test 평가·latency 측정까지 재실행한 exact reproduction은 검증하지 않았다.
전체 재실행에는 원 실험의 코드·데이터·모델 revision·벡터·설정·실행 환경을 맞춰야 한다.

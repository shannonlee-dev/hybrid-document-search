# 5차(Phase 5) 외부 encoder 검증

#36. 4차 결론이 BGE-M3에 한정된 현상인지 독립 한국어 Dense retriever `kakaocorp/kanana-2-3b-embedding`으로 확인하고,
query 단위 paired 통계와 cross-model 실패 분석을 추가했다. BC Card protocol(#25, PR #28)은 그대로이며 Kanana 설정은
model card만으로 정했다. validation/test를 보고 바꾼 설정은 없다.

## 이전 단계 요약

- 1·2차: Dense가 Sparse보다 전반적으로 강했다.
- 3차: Sparse를 더한다고 항상 좋아지지는 않았다. RRF는 Dense보다 낮았고 고정 weight Score Fusion(`bm25_weight=0.05`)만 소폭 개선했다.
- 4차: BC Card에서도 Fixed가 Dense보다 소폭 높았다. Adaptive mode Oracle은 Fixed보다 nDCG@10 +0.023이었지만
  lightweight router(#24)는 test 전부 `balanced`를 골라 이 headroom을 회수하지 못했다.
  QDAP(#33), DAT(#34), Finance encoder(#35)는 아직 main에 없어 이 문서에서 다루지 않는다.

## Kanana 조건

| 항목 | 값 |
| --- | --- |
| Model | `kakaocorp/kanana-2-3b-embedding` |
| Revision | `2c85dbe61af8ff033d2c9d46a686c665153069aa` (weights SHA-256 `818efc32…c645`, HF etag와 일치) |
| License | Kanana Open License |
| 사용 | pretrained 그대로. fine-tuning, BC Card 기반 설정 선택 없음 |
| Dimension | 2560 (full). MRL truncation 없음 |
| Prefix / instruction | 없음 (model card의 MTEB 평가 조건) |
| Pooling | mean pooling + Normalize (checkpoint의 sentence-transformers module) |
| dtype | `torch.bfloat16` (sentence-transformers 기본 로딩, config dtype) |
| 후처리 | float32 L2 재정규화. bf16 Normalize 출력의 norm이 1에서 최대 0.4% 벗어나 Fixed Fusion의 unit-norm 검사를 통과하지 못했다. model card의 similarity는 cosine이다. |
| max_seq_length | 4096. corpus 최장 문서 1,057 tokens라 잘린 문서 없음 |
| Attention | `trust_remote_code` 모델(`Qwen3BidirectionalModel`). transformers 5.18 SDPA에서 마지막 token을 바꾸면 첫 token hidden state가 바뀌는 것으로 bidirectional 동작을 확인했다. |
| Encoding | batch 1, GPU 1장(RTX PRO 6000 Blackwell). corpus 840 s, peak allocated 6.61 GB |
| Library | torch 2.14.1, transformers 5.18.0, sentence-transformers 6.1.0 |

BM25(char_wb 2–4-gram, k1=1.5, b=0.75), Fixed Fusion(`bm25_weight=0.05`), corpus/query/qrels, slice, 지표는 PR #28과 같다.
BGE-M3는 FP32, max_seq_length 8192, 1024-dim이라 dtype·길이·차원이 Kanana와 다르다. 기록은 [kanana-encoding.json](experiments/phase5/kanana-encoding.json).

## 결과

### Primary held-out test: non-synthetic (721)

| 방식 | P@1 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.4189 | 0.7406 | 0.5232 | 0.5755 |
| BGE-M3 Dense | 0.6560 | 0.8988 | 0.7460 | 0.7835 |
| BGE-M3 + Fixed | 0.6852 | 0.8946 | 0.7615 | 0.7940 |
| Kanana Dense | 0.6893 | **0.9015** | 0.7704 | 0.8029 |
| Kanana + Fixed | **0.7004** | 0.9001 | **0.7785** | **0.8087** |

primary는 정답 chunk가 하나라 graded nDCG@10이 이진 nDCG@10과 같다.

### Secondary slices (nDCG@10)

| 방식 | 전체 1,000 | 전체 graded | 합성 279 | train 중복 제외 940 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.4887 | 0.4975 | 0.2645 | 0.5080 |
| BGE-M3 Dense | 0.6459 | 0.6586 | 0.2902 | 0.6638 |
| BGE-M3 + Fixed | 0.6583 | 0.6718 | **0.3077** | 0.6763 |
| Kanana Dense | 0.6593 | 0.6729 | 0.2883 | 0.6789 |
| Kanana + Fixed | **0.6687** | **0.6828** | 0.3068 | **0.6882** |

FAQ 직접 질의(50)에서는 Kanana Dense 0.6772, Kanana + Fixed 0.6716으로 Fusion이 낮다. BGE에서도 같은 방향이었다(0.6854 / 0.6575).
전체 slice는 [Kanana 요약](experiments/phase5/kanana-test-summary.json), [BGE 재실행 요약](experiments/phase5/bge-test-summary.json).

### Latency

같은 세션, 같은 범위(캐시 vector CPU 검색·결합, 2 threads, warm-up 1 + 5 × 1,000, query encoding 제외)에서 측정했다.

| 방식 | BGE-M3 mean / P95 ms | Kanana mean / P95 ms |
| --- | --- | --- |
| BM25 | 7.29 / 7.86 | 7.48 / 8.08 |
| Dense | 6.96 / 7.08 | 16.21 / 16.30 |
| Fixed | 6.39 / 6.76 | 9.10 / 9.48 |

세 방식은 같은 계산을 나눠 쓰지 않는다. Dense는 FAISS `IndexFlatIP.search`, BM25는 `score_all()` 뒤 Python에서 양수 점수 후보를 고른다.
Score Fusion은 DenseRetriever와 BM25Retriever를 순차 호출하지 않고, snapshot한 document vector와의 numpy 내적과 BM25 `score_all()`을
별도 vectorized full-corpus path에서 결합한다. 그래서 Fixed latency를 Dense와 BM25 latency의 합으로 해석할 수 없고, Fixed가 두 component보다 빠르게 나올 수 있다.
Kanana Dense는 2560-dim flat 검색이라 BGE-M3(1024-dim)보다 느리다.
GPU query encoding은 BGE-M3 FP32 9.46 / 9.68 ms(PR #28 기록), Kanana bf16 22.49 / 22.75 ms다. dtype이 달라 모델 비용 비교로 쓰지 않는다.

## Paired comparison (primary 721)

질의 단위 paired bootstrap, 10,000회, seed `20261030`, percentile 95% CI. 차이는 B − A. win / tie / loss는 B 기준 nDCG@10.
비교마다 따로 계산했고 다중 비교 보정은 하지 않았다. 전체 결과는 [paired.json](experiments/phase5/paired.json).

| A → B | ΔnDCG@10 [95% CI] | ΔP@1 [95% CI] | ΔRecall@10 [95% CI] | W / T / L |
| --- | --- | --- | --- | --- |
| BGE Dense → BGE Fixed | +0.0105 [+0.0025, +0.0187] | +0.0291 [+0.0111, +0.0485] | −0.0042 [−0.0111, +0.0028] | 62 / 618 / 41 |
| BGE Fixed → Adaptive (#24) | −0.0010 [−0.0064, +0.0043] | −0.0014 [−0.0139, +0.0111] | −0.0028 [−0.0097, +0.0042] | 27 / 662 / 32 |
| BGE Fixed → Adaptive mode Oracle | +0.0231 [+0.0171, +0.0298] | +0.0319 [+0.0194, +0.0458] | +0.0125 [+0.0055, +0.0208] | 73 / 646 / 2 |
| BGE Dense → Kanana Dense | +0.0194 [+0.0058, +0.0330] | +0.0333 [+0.0069, +0.0596] | +0.0028 [−0.0111, +0.0166] | 106 / 542 / 73 |
| BGE Fixed → Kanana Fixed | +0.0147 [+0.0017, +0.0275] | +0.0153 [−0.0097, +0.0402] | +0.0055 [−0.0083, +0.0194] | 93 / 565 / 63 |
| Kanana Dense → Kanana Fixed | +0.0059 [−0.0009, +0.0128] | +0.0111 [−0.0055, +0.0277] | −0.0014 [−0.0069, +0.0028] | 50 / 640 / 31 |
| BGE Fixed → Kanana Dense | +0.0088 [−0.0054, +0.0229] | +0.0042 [−0.0250, +0.0319] | +0.0069 [−0.0069, +0.0208] | |

MRR@10은 모든 비교에서 nDCG@10과 같은 방향이다.

- BGE에서 Fixed의 이득은 nDCG@10·MRR@10·P@1 CI가 0을 포함하지 않는다. Recall@10은 차이가 불명확하다.
- Kanana에서는 같은 BM25 0.05가 +0.0059로 줄고 CI가 0을 포함한다. 방향은 같지만 이 데이터로는 개선이라고 말할 수 없다.
- Kanana Dense 단독과 BGE + Fixed의 차이도 불명확하다.
- Adaptive(#24)와 Fixed의 차이는 0 근처다. router가 721개 전부 `balanced`(weight 0.10)를 골라 실제로는 고정 weight 0.10과 같고,
  27 / 32의 승패는 routing이 아니라 weight 차이(0.05 → 0.10)에서 나온다.

Mode Oracle과 Adaptive 행은 PR #30의 frozen `router.json`(SHA-256 `948c9081…`)을 PR #28 vector cache에서 다시 실행한 것이다.
primary·전체 test 수치가 PR #30 공개값과 소수점 6자리까지 같다([요약](experiments/phase5/adaptive-replay-summary.json)).

## Query-level 분석 (primary 721)

질의별 rank·nDCG@10·Top-1 일치·Top-10 overlap은 [primary-queries.csv](experiments/phase5/primary-queries.csv),
집계는 [cross-model.json](experiments/phase5/cross-model.json). 모두 qrels를 보고 계산한 사후 분석이다.

**Encoder 간 차이.** 두 Dense의 Top-1은 184개(25.5%)에서 다르고 Top-10 평균 overlap은 6.1개다.
hit@10 기준 둘 다 성공 635, BGE만 13, Kanana만 15, 둘 다 실패 58이다. P@1 기준으로는 BGE만 36, Kanana만 60이라
Kanana의 이득은 대부분 정답을 1위로 올리는 데서 나온다. 둘 다 실패한 58개 중 BM25가 Top-10에서 찾는 건 3개뿐이다.
예시를 보면 다른 parent_doc에 있어 rel=0인 문서가 실제로 답을 담은 경우(`base_bcai_000522_000`)와
짧은 일반 문장 chunk가 긴 FAQ 답변보다 위에 오는 경우(`crawl_…faq_238000`)가 섞여 있다. 비율은 세지 않았다.

**Sparse 보완성.** BM25만 정답을 Top-10에 넣는 질의는 BGE 기준 5개, Kanana 기준 7개, 공통 3개로 드물다.
BM25가 정답을 1위로 두고 Dense는 아닌 질의는 BGE 31, Kanana 30, 공통 19다.
BM25의 역할은 Dense가 놓친 문서를 새로 찾기보다 Dense Top-10 안의 순서를 바꾸는 쪽이고, 그런 질의의 절반 이상은 encoder가 바뀌어도 남는다.

**Fixed 0.05의 개선과 악화.**

| | BGE | Kanana |
| --- | --- | --- |
| 개선 / 악화 / 동률 | 62 / 41 / 618 | 50 / 31 / 640 |
| 개선 질의의 BM25 정답 rank: 1위 / 2–10위 / 없음 | 22 / 28 / 12 | 16 / 21 / 13 |
| 악화 질의의 BM25 정답 rank: 1위 / 2–10위 / 없음 | 0 / 11 / 30 | 0 / 12 / 19 |
| ΔnDCG 합: 개선 / 악화 | +16.62 / −9.05 | +11.42 / −7.20 |
| Dense Top-10 밖 → Fixed 안 / 반대 | 2 / 5 | 1 / 2 |

악화는 대부분 BM25가 정답을 아예 못 찾는 질의에서 생긴다. BM25가 정답을 1위로 둔 질의에서는 두 encoder 모두 악화가 0이다.
개선 질의 중 두 encoder에서 공통인 건 21개, 악화 공통은 14개다. BGE에서 개선되고 Kanana에서 악화된 질의는 0, 반대는 2개다.
encoder가 강해지면 개선 질의 수와 이득 합이 같이 줄어든다.

**Headroom 위치.** BM25/Dense/Fixed 중 질의별 최고를 고르는 method-level Oracle은 BGE에서 Fixed 대비 63개 질의(+0.0227),
Kanana에서 59개(+0.0244)에서 이득이 있다. 두 encoder 공통은 28개로 절반이 안 된다.
#24 mode Oracle이 Fixed보다 나은 73개 중 68개는 BM25와 BGE Dense의 Top-1이 다른 402개 질의 안에 있다.
그 402개에서 Oracle − Fixed는 +0.0404, Adaptive − Fixed는 −0.0016이다. Top-1이 같은 319개에서는 Oracle 이득이 +0.0013이다.
Top-1 불일치는 headroom의 거의 필요조건이지만 402개 중 334개에는 headroom이 없어 충분조건은 아니다.
router의 mode와 Oracle mode 일치율은 89.7%인데, Oracle도 647 / 721개에서 `balanced`를 고르기 때문이다.

## 결론

- **유지된 부분.** Dense가 BM25보다 크게 강한 건 encoder를 바꿔도 같다. Kanana Dense는 BGE Dense보다도 nDCG@10 +0.019, P@1 +0.033 높고 CI가 0을 포함하지 않는다.
  BM25 0.05를 더할 때의 방향(개선 > 악화, 악화는 BM25가 정답을 못 찾을 때 집중)도 두 encoder에서 같다.
  4차의 adaptive headroom은 BM25/Dense Top-1이 다른 질의에 모여 있고, #24 router는 그 구간에서 Fixed보다 나아지지 않았다.
- **수정된 부분.** Fixed Fusion의 이득은 dataset만의 성질로 보기 어렵다. BGE에서는 CI가 0을 넘지만 Kanana에서는 +0.006으로 줄고 불명확하다.
  Kanana Dense 단독은 BGE + Fixed와 구분되지 않는다. 즉 4차에서 Fixed가 얻은 몫의 상당 부분은 더 강한 encoder로도 얻어진다.
  method-level headroom 질의도 encoder마다 절반 이상 다르다. router가 맞혀야 할 질의가 encoder에 따라 바뀐다는 뜻이다.
- Kanana 하나로 위 판단에 필요한 결과가 나와 두 번째 외부 모델은 실행하지 않았다.

## Provenance gate

`scripts.analyze_phase5 compare|cross-model`은 비교 전에 두 run의 identity를 report에서 읽어 확인한다(`evaluation/provenance.py`).

- 같아야 비교함(다르면 거부): split, dataset ID·revision, manifest SHA-256, corpus·queries·qrels SHA-256, corpus 크기, per-query 질의 ID 목록 hash,
  manifest의 primary slice 질의 ID hash, top_k, relevance 규칙. 한쪽에 없는 항목은 `unverified`로 남긴다.
  primary slice 질의가 report에 빠져 있으면 거부한다.
- 차이로 기록: method config(fusion weight 포함), model name·revision, normalization, prefix, dtype, max_seq_length, dimension, vector SHA-256.
- 경고: 같은 model·revision인데 vector SHA-256이 다를 때, dirty working tree에서 만든 run이 있을 때.
- 함께 남김: git commit, 생성 시각, Python·platform·package 버전, report 파일 SHA-256.

이번 비교는 모두 `unverified` 0, 경고 0이다. PR #28 report와 이번 BGE 재실행은 1,000개 질의의 순위와 지표가 완전히 같았다(Fixed paired 차이 0).
Kanana run은 `cdf7c2f`, BGE 재실행은 `6375acc`, Adaptive replay는 `6375acc`의 clean tree에서 만들었다.

## 확인한 불일치

- PR #28과 PR #30은 같은 BGE-M3 revision인데 document vector SHA-256이 다르다(`44c78995…` / `4d5f1c8d…`). query vector도 다르다.
  PR #30 router를 PR #28 vector로 다시 돌려도 공개 지표가 6자리까지 같아서 결과에는 영향이 없었다. 이런 경우 위 gate가 경고를 낸다.
- PR #28 test report의 `git_commit`은 rebase 전 SHA `107eed2`다(main에서는 `9ddbbd4`).
- 같은 BGE 설정의 latency가 문서마다 다르다. Dense 검색 mean은 PR #28 7.07 ms, PR #30 11.24 ms, 이번 6.96 ms이고,
  query encoding mean은 PR #28 9.46 ms, PR #30 16.89 ms다. 서로 다른 실행이라 문서 간 latency를 직접 비교하지 않는다.
- Kanana bf16 출력의 norm 문제와 float32 재정규화는 위 조건에 적었다. 재정규화 전 cache로는 test 지표를 계산하지 않았다.

## 보류

#33 QDAP, #34 DAT, #35 Finance encoder는 아직 main에 per-query 결과가 없다. 구현을 가져오지 않았고 다음 비교는 보류했다.

- Fixed vs QDAP, Fixed vs DAT, Finance + Fixed vs Fixed, Kanana vs Finance + Fixed, Kanana + Fixed vs Finance + Fixed
- QDAP/DAT가 mode/method Oracle headroom 질의를 실제로 잡았는지, Top-1 불일치 구간에서의 이득

결과가 enterprise report 형식(`methods.<name>.queries[].metrics`, `results`)으로 들어오면 같은 명령에 `--run`, `--pair`, `--column`만 추가하면 된다.

## 한계

- 공식 test는 4차에서 이미 여러 번 평가했다. 이번 결과도 새 held-out 확인이 아니다. Kanana는 validation도 설정 선택에 쓰지 않았다.
- bootstrap CI는 비교별 서술 구간이다. 7개 비교에 보정을 하지 않았고 유의성 검정으로 쓰지 않는다.
- Kanana와 BGE는 dtype(bf16 / FP32), max_seq_length, dimension이 다르다. 모델 차이와 이 조건 차이를 분리하지 않았다.
- 실패 분석은 primary 721개, binary nDCG@10 기준이다. parent_doc 범위 label의 false negative가 "둘 다 실패" 질의에 섞여 있다.

## 재현

```bash
uv sync --locked --extra sparse --extra dense
uv run --locked --extra sparse --extra dense python -m scripts.prepare_bccard --download
uv run --locked --extra sparse --extra dense python -m scripts.evaluate_enterprise encode
uv run --locked --extra sparse --extra dense python -m scripts.evaluate_enterprise encode \
  --encoder kanana --cache-dir artifacts/bccard-kanana
uv run --locked --extra sparse --extra dense python -m scripts.evaluate_enterprise evaluate \
  --split test --cache-dir artifacts/bccard-bge-m3 --output-dir artifacts/phase5/bge-test
uv run --locked --extra sparse --extra dense python -m scripts.evaluate_enterprise evaluate \
  --split test --cache-dir artifacts/bccard-kanana --output-dir artifacts/phase5/kanana-test
uv run --locked --extra sparse --extra dense python -m scripts.analyze_phase5 replay-adaptive \
  --output artifacts/phase5/adaptive-replay/report.json

RUNS="--run bge=artifacts/phase5/bge-test/report.json --run kanana=artifacts/phase5/kanana-test/report.json \
  --run adaptive=artifacts/phase5/adaptive-replay/report.json"
uv run --locked --extra sparse --extra dense python -m scripts.analyze_phase5 compare $RUNS \
  --pair bge:dense,bge:score_fusion --pair bge:score_fusion,adaptive:adaptive \
  --pair bge:score_fusion,adaptive:adaptive_oracle --pair bge:dense,kanana:dense \
  --pair bge:score_fusion,kanana:score_fusion --pair kanana:dense,kanana:score_fusion \
  --pair bge:score_fusion,kanana:dense --output docs/experiments/phase5/paired.json
uv run --locked --extra sparse --extra dense python -m scripts.analyze_phase5 cross-model $RUNS \
  --column bm25=bge:bm25 --column bge_dense=bge:dense --column bge_fixed=bge:score_fusion \
  --column kanana_dense=kanana:dense --column kanana_fixed=kanana:score_fusion \
  --column adaptive=adaptive:adaptive --column adaptive_oracle=adaptive:adaptive_oracle \
  --output docs/experiments/phase5/cross-model.json --rows-output docs/experiments/phase5/primary-queries.csv
```

Committed `paired.json`에는 PR #28 report(`pr28=artifacts/bccard-eval/test/report.json`)와 재실행의 동일성 비교 1건이 더 들어 있다.
Kanana weight(6.4 GB), HF cache, vector, per-query report(각 약 20 MB)는 Git에 넣지 않았다.
report SHA-256: BGE `ad6d4555…`, Kanana `fe0776fb…`, Adaptive replay `ef62e4b8…` (전체 값은 `paired.json`의 `runs.*.run.result_sha256`).

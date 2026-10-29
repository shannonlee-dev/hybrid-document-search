# Downstream RAG 평가 (RAGAS, 탐색적 실험)

상태: experimental branch(`experiment/ragas-downstream-eval`) 전용. main 미반영.
목적: 기존 retrieval 차이가 동일 generator·동일 judge·동일 질의에서 RAG context/answer 품질로
어떻게 전달되는지 확인한다. 새 알고리즘·재튜닝이 아니며 최종 benchmark가 아니다.

## 요약

- **큰 retrieval 차이는 downstream으로 전달됐다.** Sparse(TF-IDF/BM25)와 Dense 계열의
  nDCG@10 차이(약 0.14–0.22)는 Context Precision(+0.09–0.20), 응답 거부율(29–32% → 20–21%),
  Faithfulness(+0.03–0.06)에서 같은 방향으로 나타났다.
- **RRF − Dense(nDCG@10 −0.053, CI가 0 제외)는 LLM Context Precision에서만 같은 방향으로 보였다**
  (−0.053, 95% CI [−0.102, −0.002]). 잘린 생성 1건을 뺀 sensitivity에서는 CI 상한이 +0.002로 0을 포함해
  경계선상이다. Faithfulness·Response Relevancy 차이는 CI가 0을 포함한다.
- **Score Fusion − Dense는 downstream 차이가 관찰되지 않았다.** 이 표본의 nDCG@10 차이는 +0.006(CI 0 포함)이고,
  top-5 관련 문서 수가 바뀐 질의는 100개 중 6개뿐이다. RAGAS 세 지표 모두 평균 차이가 −0.008 ~ −0.010으로
  CI가 0을 포함한다. 작은 retrieval 개선은 이 규모에서 answer 품질 신호로 구분되지 않았다.
- Ko-StrategyQA에는 reference answer가 없어 **Answer Correctness와 LLM Context Recall은 계산하지 않았다.**

## 1. 데이터 적합성과 metric 선택

`taeminlee/Ko-StrategyQA@7cececf`는 BEIR 형식(corpus·queries·qrels)이며 정답 필드가 없다.
질의는 원 StrategyQA 질문의 **분해된 하위 질문**이고 ID가 `<원 질문 ID>_<단계>`다
(dev 592 하위 질문 = 원 질문 375개). 원본 StrategyQA의 정답은 원 질문 단위 yes/no뿐이라
하위 질문의 정답으로 쓸 수 없다. 새 ground truth는 만들지 않았다.

| 지표 | 입력 | 사용 여부 |
|---|---|---|
| ID Context Precision@5 / Recall@5 | 생성 context 5개 + qrels | 사용 (비 LLM. Recall@5는 IR Recall@5와 같은 정의) |
| Context Precision (`ContextPrecisionWithoutReference`) | 질문·응답·context | 사용 — 응답을 reference 대신 쓰는 RAGAS 변형 |
| Faithfulness | 질문·응답·context | 사용 |
| Response Relevancy (`AnswerRelevancy`, strictness 3) | 질문·응답 + embedding | 사용 |
| Answer Correctness | reference answer 필요 | **제외** — 정답 없음 |
| Context Recall (LLM) | reference answer 필요 | **제외** — qrels 기반 ID Recall@5로 대체 |

결정적 보조 지표로 응답 거부율(`알 수 없`/`확인할 수 없` 포함 여부)을 함께 기록했다.

## 2. 실험 설정

| 항목 | 값 |
|---|---|
| 데이터 | Ko-StrategyQA dev, 고정 cache(`benchmark-inputs-v1`, SHA-256 대조) |
| 질의 | 100개, seed `20261023`. 정렬한 dev ID를 `random.Random(seed).shuffle` 후 앞에서부터 사용 |
| pilot | 앞 50개. 확장 조건은 실행 전 고정: pilot의 metric 실행 실패율 < 5%면 100개까지 (성능 결과는 보지 않음) |
| 방법 | TF-IDF, BM25, Dense(BGE-M3), 기존 RRF(k=60, 각 Top-10 후보), Score Fusion(α=0.05) |
| 검색 | `evaluation.final`의 검색기를 그대로 사용, 각 방법 `search(q, 10)` |
| 생성 context | 위 Top-10의 **상위 5개**, retrieval 순위 순서, `title + "\n" + text` 전문 |
| Generator | `LGAI-EXAONE/EXAONE-4.5-33B-FP8` rev `69570ad5…`, vLLM, non-reasoning(`enable_thinking=False`) |
| 생성 설정 | temperature 0, top_p 1, top_k −1, max_tokens 512, seed 0, presence/frequency penalty 0, 단일 user 메시지 |
| Judge | `skt/A.X-4.0` rev `a3393fd6…`, vLLM TP2, temperature 0, top_p 1, max_tokens 4096 |
| RAGAS | `ragas==0.4.3` (`ragas.metrics.collections`, Instructor JSON mode), telemetry 비활성화 |
| Embedding | `BAAI/bge-m3` rev `5617a9f6…`, CPU, Response Relevancy에만 사용 |
| 비용 | 로컬 GPU(RTX PRO 6000 ×2)만 사용, 외부 API 비용 0 |

**Top-5는 생성 context 길이를 통제하기 위한 값이다.** 기존 최종 benchmark의 Top-10 지표와 다르다.
RRF는 각 검색기의 Top-10 후보를 결합하는 고정 baseline이라, `search(q, 5)`를 호출하면 후보 범위가 바뀐다.
따라서 10개를 검색한 뒤 잘랐다. 실행 전에 다섯 방법의 dev 592 nDCG@10이
[2026-10-20 검증 기록](validation/2026-10-20/evaluation.txt)과 정확히 일치하는지 확인했다.
Generator에는 검색 방법 이름을 전달하지 않는다.

```text
다음 문서만을 근거로 질문에 답하세요.
문서에서 답을 확인할 수 없다면 추측하지 말고 알 수 없다고 답하세요.

[Context]
[문서 1] {title}
{text}
...

[Question]
{question}

[Answer]
```

### 환경 호환성 workaround (재현 정보)

- **vLLM `--attention-backend TRITON_ATTN`**: 사용한 vLLM 이미지
  `ghcr.io/local-inference-lab/vllm:karmic-kraken-beta`(vLLM `0.1.dev21986+g4a379ed42`)에서
  기본 FlashAttention kernel이 `CUDA error: the provided PTX was compiled with an unsupported toolchain`으로
  실패했다(driver 610.57). Generator와 judge 모두 Triton attention backend로 실행했다.
  모델 출력 정의는 같지만 수치 kernel이 다르므로 재현 시 같은 backend를 권장한다.
- **`langchain-community<0.4.2`**: `ragas==0.4.3`은 import 시
  `langchain_community.chat_models.vertexai`를 불러오는데, 0.4.2에서 이 모듈이 제거됐다.
  `rag` extra에서 0.4.1로 고정했다.
- `rag` extra는 production 설치(API/UI)와 분리된 optional dependency다.

## 3. 결과

### A. 기존 Ko-MIRACL retrieval benchmark (참고, 다른 데이터셋)

[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)의 dev 50 Top-10 결과다. **아래 B와 같은 평가 집합이 아니다.**

| 방식 | nDCG@10 |
|---|---:|
| TF-IDF | 0.5253 |
| BM25 | 0.5939 |
| Dense | 0.8549 |
| 기존 RRF | 0.7844 |
| Score Fusion | 0.8649 |

### B. Ko-StrategyQA downstream RAG 평가 (100 질의, 같은 질의 집합)

IR 지표는 같은 100개 질의에서 계산했다. 같은 질의의 dev 592 전체 평균과 다르다.

| 방식 | nDCG@10 | MRR@10 | Recall@5 (= ID Ctx Recall@5) | ID Ctx Precision@5 | Context Precision | Faithfulness | Response Relevancy | 응답 거부율 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.5302 | 0.5445 | 0.5583 | 0.2180 | 0.5137 | 0.8336 | 0.6320 | 32% |
| BM25 | 0.5567 | 0.5846 | 0.6150 | 0.2380 | 0.5720 | 0.8442 | 0.6631 | 29% |
| Dense | 0.7485 | 0.7758 | 0.7517 | 0.3000 | **0.7159** | **0.8918** | 0.7479 | **20%** |
| 기존 RRF | 0.6953 | 0.7033 | 0.7408 | 0.2940 | 0.6631 | 0.8727 | **0.7617** | **20%** |
| Score Fusion | **0.7545** | **0.7797** | **0.7725** | **0.3080** | 0.7153¹ | 0.8806¹ | 0.7387 | 21% |

¹ judge 실행 실패 1건씩 제외(n=99). 아래 §5 참조.

### Paired 비교 (A − Dense, 100 질의)

95% CI는 원 질문 ID 단위 cluster bootstrap(10,000회, seed 20261023)이다. ‘+/−/=’는 질의 수다.

| 지표 | Score Fusion − Dense | +/−/= | RRF − Dense | +/−/= |
|---|---|---|---|---|
| nDCG@10 | +0.0060 [−0.0110, +0.0225] | 14/9/77 | −0.0532 [−0.0831, −0.0232] | 9/36/55 |
| Recall@5 | +0.0208 [−0.0008, +0.0488] | 5/1/94 | −0.0108 [−0.0396, +0.0183] | 5/9/86 |
| ID Ctx Precision@5 | +0.0080 [+0.0000, +0.0178] | 5/1/94 | −0.0060 [−0.0222, +0.0101] | 5/9/86 |
| Context Precision | −0.0078 [−0.0341, +0.0180] (n=99) | 13/11/75 | −0.0528 [−0.1024, −0.0019] | 14/29/57 |
| Faithfulness | −0.0101 [−0.0448, +0.0221] (n=99) | 8/11/80 | −0.0191 [−0.0703, +0.0276] | 13/20/67 |
| Response Relevancy | −0.0092 [−0.0349, +0.0099] | 8/14/78 | +0.0138 [−0.0188, +0.0481] | 17/15/68 |
| 응답 거부 (0/1) | +0.0100 [+0.0000, +0.0309] | 1/0/99 | 0.0000 [−0.0404, +0.0400] | 2/2/96 |

**Bootstrap 단위.** 하위 질문은 독립 표본이 아니다. 같은 원 질문에서 나온 하위 질문은 주제·corpus 문서·질문 형식을 공유하므로
질의 단위로 재표본하면 분산이 과소추정된다. 따라서 decomposed sub-question이 아니라
**원 StrategyQA 질문 ID(`query_id`의 마지막 `_<단계>` 앞부분) 단위로 재표본**하고,
뽑힌 그룹의 모든 하위 질문 평균 차이로 percentile 구간을 계산했다.
100 질의는 96개 원 질문 그룹이다. 기존 [SCORE_FUSION_ANALYSIS.md](SCORE_FUSION_ANALYSIS.md)의 그룹 bootstrap과 같은 취지다.

### Context 차이 규모 (Score Fusion vs Dense)

| 범위 | top-5 순서까지 동일 | 같은 집합·순서만 다름 | 집합 다름 | 관련 문서 수 변화 (Recall@5 ↑/↓) |
|---|---:|---:|---:|---:|
| pilot 앞 50 | 10 | 14 | 26 | **5개 (10%)**: 4 ↑ / 1 ↓ |
| 전체 100 | 25 | 24 | 51 | **6개 (6%)**: 5 ↑ / 1 ↓ |

중간 보고의 “5 queries (6 of 100)”는 위 마지막 열이다. pilot 50에서는 5개(10%), 100에서는 6개(6%)다.
나머지 집합 변화는 qrels상 비관련 문서끼리의 교체다. RRF − Dense는 100개 중 90개에서 집합이 달랐고,
관련 문서 수는 14개 질의(5 ↑ / 9 ↓)에서 바뀌었다.

**비결정성 하한.** top-5가 순서까지 동일한 25개 질의에서도 Dense와 Score Fusion의 응답 텍스트가 5개 질의에서 달랐다
(temperature 0이어도 vLLM batching의 수치 차이). 같은 context에서의 평균 절대 차이는
Faithfulness 0.007, Context Precision 0.002, Response Relevancy 0.025다.

## 4. Query-level 사례 (Score Fusion vs Dense)

선정 규칙: Recall@5가 바뀐 6개 질의는 모두 검토했다. 그 외에는 Recall@5가 같은 질의 중 RAGAS 세 지표 차이 합의
최소·최대 질의를 골랐다. 정답 라벨이 없으므로 “답이 맞는지”는 판단하지 않는다.

**개선 사례**

- `91a3d81584c66d26d49a_1` “에스프레소는 심박수에 어떤 영향?”: Score Fusion이 `Espresso-2`를 3위에 넣었다
  (Recall@5 +0.25, nDCG@10 +0.195). Dense는 “알 수 없음”, Score Fusion은 답을 생성했다(Faithfulness +0.556).
  다만 응답은 “카페인은 일반적 지식으로…”처럼 문서 밖 지식에도 기댄다.
- `0aeb8bccd429c7dfe0da_0` “벅스 버니는 언제 만들어졌나?”: 관련 문서 `Bugs Bunny-1`이 2위에서 1위로 올라갔다
  (nDCG@10 +0.369, Context Precision +0.55, Response Relevancy +0.18). 응답은 “1940년 데뷔” → “1930년대 후반”으로 바뀌었다.
  정답 라벨이 없어 개선 여부는 판단할 수 없다.
- `424ecb3dd6c64b6da4cc_0` “집사란?”: `Butler-1`이 3위에서 2위로 올라가 Context Precision +0.53이다. 응답이 교회 직분(deacon)보다
  가사 집사(butler) 설명에 집중했다.

**악화 사례**

- `cb8c31219e06be31e9cc_1` “로빈은 누구에게 살해당했나?”: 관련 문서 `Jason Todd-28`은 양쪽 모두 포함됐다.
  Score Fusion에서는 `Robin Williams-60`·`Abraham Lincoln-113` 대신 `Al Capone-3`·`The Catcher in the Rye-26`이 들어오고 관련 문서가 4위가 됐으며, generator가 “알 수 없음”으로 거부했다
  (Faithfulness −1.0, Response Relevancy −0.95). 같은 관련 문서가 있어도 주변 noise가 답변 여부를 바꿨다.
- `50ee714386bc6323ae11_0` “스트레스 골절이 생기는 활동?”: Score Fusion만 관련 문서를 잃은 질의다(`Pelvic fracture-20` 탈락,
  Recall@5 −0.33). 응답은 거의 같았고 RAGAS 차이도 거의 없다.
- `1e0f0676ee91a330262a_1` “지질학자는 무엇을 연구?”: 5위 문서만 `Geometry-1` → `Astronomer-1`로 바뀌었는데도
  응답이 길어지고 Faithfulness가 −0.44였다. retrieval 차이보다 생성·judge 변동이 큰 사례다.

**retrieval은 달라졌지만 answer는 사실상 같은 사례**

- `2eba35467e49d7e9041d_1` “리처드 3세의 아버지?”, `02f5a6e6eedc57dfa22d_0` “테리 프래쳇의 직업?”,
  `d77b8a72c7c3126f35b4_0` “판사가 질서 유지에 쓰는 도구?”: Score Fusion이 관련 문서를 추가했다(Recall@5 +0.33 ~ +0.5).
  하지만 1위 문서만으로 이미 답할 수 있어 응답이 같거나 근거 문장만 늘었다.
  RAGAS 차이는 0 또는 Context Precision 소폭 하락(추가 관련 문서가 더 낮은 순위에 들어가 평균 precision이 바뀜)이었다.
- `ee62f46a39caaa860b26_0` “비즈니스에서 용매(solvency)가 된다는 것?”: Score Fusion이 `Solvency-1`을 5위로 가져왔다(Recall@5 +1.0).
  그러나 한국어 질의의 “용매” 오역 때문에 두 방법 모두 결국 답을 확인할 수 없다고 응답했다.

## 5. 실패·잘린 생성 처리

- **생성**: 500건 중 오류 0, 빈 응답 0. `max_tokens=512`에 걸린 2건은 **본 결과에 그대로 포함**했다.
  - `tfidf` / `0b7787655f4a9d600dfc_0` (position 37)
  - `rrf` / `01254b50648f4fb08b51_1` (position 59)

  Score Fusion − Dense 비교에는 해당하지 않는다. RRF − Dense sensitivity(해당 질의 제외, n=99, 본 결과 아님):
  Context Precision −0.0492 [−0.0989, +0.0024], Faithfulness −0.0177 [−0.0681, +0.0287],
  Response Relevancy +0.0138 [−0.0188, +0.0490]. 평균 차이는 거의 같지만 Context Precision CI 상한이 0을 넘는다.
  따라서 RRF의 Context Precision 하락은 “경계선상의 신호”로 해석한다.
- **Judge**: pilot 750 metric 실행 중 실패 1건(0.13%)으로 사전 조건(< 5%)을 충족해 100개로 확장했다.
  최종 1,500 metric 실행 중 실패는 2건이다. 둘 다 Score Fusion이며 judge 출력이 `max_tokens=4096`에 걸린
  `IncompleteOutputException`이다(`df538a945f0d1c012cd2_1` Context Precision, `b978c4051673fd21035b_0` Faithfulness).
  재시도하지 않고 실패로 기록했다. 평균·paired 비교에서는 해당 metric만 제외됐다(n=99). NaN 결과는 없었다.

## 6. 해석과 한계

1. **Faithfulness는 상당 부분 응답 거부율을 반영한다 (사후 기술 분석).** 거부 응답의 평균 Faithfulness는 0.687,
   답변한 응답은 0.922다. 답변한 응답만 보면 방법별 Faithfulness는 0.915–0.939로 거의 같다(BM25 0.939, Dense 0.919, Score Fusion 0.915).
   Sparse와 Dense의 Faithfulness 차이는 주로 “관련 문서가 없어 거부한 비율”의 차이다.
2. **Context Precision(reference-free)은 응답을 기준으로 context를 판정한다.** 응답이 바뀌면 같은 context도 다르게 평가될 수 있어
   순수한 retrieval 지표가 아니다. qrels 기반 ID 지표를 함께 본 이유다.
3. **정답성(correctness)은 측정하지 않았다.** 위 벅스 버니 사례처럼 응답이 달라져도 어느 쪽이 맞는지 판단할 근거가 없다.
4. Judge 1개(A.X-4.0), 반복 1회다. Judge의 비결정성과 영어 RAGAS 프롬프트로 한국어를 평가한 영향은 별도로 추정하지 않았다.
   같은 context에서의 차이(§3)로 대략적인 noise 하한만 제시한다.
5. 하위 질문은 짧은 사실 질의이고 1위 문서로 답이 되는 경우가 많아, 하위 순위의 retrieval 개선이 응답에 반영될 여지가 작다.
6. Ko-StrategyQA dev와 과거 실험 노출 이력은 [SCORE_FUSION_ANALYSIS.md](SCORE_FUSION_ANALYSIS.md)와 같다.
   α=0.05와 검색 설정은 바꾸지 않았다.

## 7. 재현

```bash
uv sync --locked --all-extras
uv run --no-sync python -m scripts.prepare_score_fusion_cache
uv run --no-sync python -m scripts.evaluate_rag_downstream retrieve        # 100 질의, dev 592 재현 검증
# generator: vllm serve LGAI-EXAONE/EXAONE-4.5-33B-FP8 --revision 69570ad5d9a5159816ef267f2f4d37e36447f1be \
#   --served-model-name exaone-4.5-33b-fp8 --port 8101 --max-model-len 16384 --generation-config vllm \
#   --reasoning-parser qwen3 --attention-backend TRITON_ATTN --limit-mm-per-prompt '{"image":0,"video":0}'
uv run --no-sync python -m scripts.evaluate_rag_downstream generate --queries 100 \
  --base-url http://127.0.0.1:8101/v1 --model exaone-4.5-33b-fp8 \
  --model-revision 69570ad5d9a5159816ef267f2f4d37e36447f1be
# judge: vllm serve skt/A.X-4.0 --revision a3393fd67bf09a34a85e66dfd2c2da61c4497b67 \
#   --served-model-name ax-4.0 --port 8102 --tensor-parallel-size 2 --max-model-len 16384 \
#   --generation-config vllm --attention-backend TRITON_ATTN
uv run --no-sync python -m scripts.evaluate_rag_downstream judge --queries 50 \
  --base-url http://127.0.0.1:8102/v1 --model ax-4.0 \
  --model-revision a3393fd67bf09a34a85e66dfd2c2da61c4497b67      # pilot, 실패율 확인
uv run --no-sync python -m scripts.evaluate_rag_downstream judge --queries 100 ...  # 완료 행은 건너뜀
uv run --no-sync python -m scripts.evaluate_rag_downstream summarize --queries 100
```

`summarize`는 집계 전에 모든 method가 고정 `queries.jsonl`의 요청 prefix와 같은 query_id를 같은 position에
한 번씩 갖는지 확인하고, 누락·초과·중복·position 불일치가 있으면 요약을 만들지 않고 실패한다.
NaN/무한대 metric은 0이 아니라 undefined(`None`, judge metric은 `non_finite_value` 사유)로 처리해
평균·paired 비교·noise 추정에서 제외하며, JSONL과 `summary.json`은 표준 JSON으로만 기록한다.
이 검증을 추가한 집계기로 아래 원시 산출물을 다시 요약해도 `summary.json`은 같은 SHA-256으로 재현된다.

원시 산출물(`config.json`, `queries.jsonl`, `retrieval.jsonl`, `generations.jsonl`, `ragas-results.jsonl/.csv`,
`summary.json`, 서버 로그)은 gitignore된 `artifacts/ragas-eval/`에 보관한다.
요약 수치는 [summary-strategyqa-100.json](experiments/ragas/summary-strategyqa-100.json)에 있다. 질의별 텍스트는 제외했다.

| 로컬 파일 | SHA-256 |
|---|---|
| summary.json | `62edd68cad20e6a18c78a833edc22d46dcaaedfe0ecaf25ad51202fb7f9e813e` |
| ragas-results.jsonl | `fe0aa7baa13bdf4ff9f449cf9cbd9fa200e5052785ab404b6e96866e433e9321` |
| generations.jsonl | `1342f2b5d2776ce6ae0b725f52ca9356878cf2a36d91b9a6ad559ffc385d1eba` |
| retrieval.jsonl | `b1d2bbc0f68cb2356f2a036ca2e34edcdeb1517eeac63ed0555014a65566a890` |
| queries.jsonl | `b5d5b0253e8756691a7a0d9abd47ff36101253f966069841524136982947ce20` |

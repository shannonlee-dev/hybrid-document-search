# BC Card 기업 특화 검색 데이터셋

4차(#25) 평가 데이터와 고정 protocol, 기존 방식(BM25 · Dense · Fixed Score Fusion)의 전이 결과를 정리한다.
Adaptive Hybrid(#24)는 [ADAPTIVE_RESULTS.md](ADAPTIVE_RESULTS.md), 상용 서비스(#26)는 [COMMERCIAL_RAG_BENCHMARK.md](COMMERCIAL_RAG_BENCHMARK.md)에서 같은 protocol로 다룬다. 4차 최종 정리는 [PHASE4_FINAL_EVALUATION.md](PHASE4_FINAL_EVALUATION.md).

## Dataset source

| 역할 | Dataset | Revision | License |
| --- | --- | --- | --- |
| corpus | [`BCCard/BCAI-Finance-Kor-Embedding-Pair`](https://huggingface.co/datasets/BCCard/BCAI-Finance-Kor-Embedding-Pair) | `e022cb013f2907e0716ebe40d13f30ed93ffa9b0` | CC-BY-4.0 |
| queries + qrels | [`BCCard/BCAI-Finance-Kor-Embedding-Triplet`](https://huggingface.co/datasets/BCCard/BCAI-Finance-Kor-Embedding-Triplet) | `f63d59969dba9916bd34c86c82112331890b11da` | CC-BY-4.0 |

실제 parquet과 dataset card로 확인한 구조:

- **Pair** (`train` 45,589행): `anchor`, `positive`, `source`, `chunk_id`, `source_chunk_id`, `chunk_method`,
  `chunk_index`, `total_chunks`, `query_variant`, `section_id`, `context_summary`, `chunking_config_hash`,
  `page_type`, `source_url`, `title`, `breadcrumb`.
  `source_chunk_id` 36,281개가 corpus이며 ID마다 본문은 정확히 하나다.
- **Triplet** (공식 `train` 43,394 / `validation` 1,000 / `test` 1,000): `anchor`, `positive`, `negative`,
  `anchor_chunk_id`(질의 ID, 전 split 고유), `negative_chunk_id`, `negative_rank`, `hardness`, `rationale`,
  `judge_source`, `anchor_parent_doc_id`. validation/test에만 `relevant_chunk_ids` + `relevance_scores`(1–3)가 있다.
- **등급 규칙:** 같은 parent_doc 안에서 질의의 원본 chunk 3, 인접 chunk 2, 나머지 1, 다른 parent_doc 0.
  공식 validation/test qrels 2,000개 질의 모두가 이 규칙과 정확히 일치했다.
- **출처 표기:** `anchor_chunk_id` 접두사와 `chunk_method`로 구분한다. Pair의 `source` 열은 FAQ 직접 질의(`crawl_*`, `faq_direct`)도
  `synth`로 표시하므로 출처 판단에 쓰지 않는다.

| 접두사 | 출처(card 기준) | train | validation | test |
| --- | --- | ---: | ---: | ---: |
| `base_*` | BCAI 원본 Q&A 질문(LLM 미가공) | 29,669 | 695 | 671 |
| `crawl_*` | BC카드 사이트 FAQ 질문 그대로 | 478 | 50 | 50 |
| `synth_base_*` | BCAI 문서에서 LLM이 생성한 질의 | 1,676 | 42 | 15 |
| `synth_crawl_*` | 사이트 crawl 문서에서 LLM이 생성한 질의 | 11,571 | 213 | 264 |

## Why this dataset

- BC카드 상품·가맹점·법인·VIP 사이트 crawl과 사내 BCAI Q&A로 구성된 실제 한국어 기업 도메인이다.
- 카드 상품명, 가맹점·금융 용어, 영문 약어(PG, BC Pay 등)처럼 어휘 일치가 중요한 질의와 의미 중심 질의가 섞여 있다.
- 3차 Ko-MIRACL / Ko-StrategyQA와 독립된 corpus·질의로 기존 방식의 전이를 확인할 수 있다.

## Evaluation protocol

커밋 `ff84dcd`에서 retrieval 실행 전에 고정했다([manifest](experiments/bccard/manifest.json)).

| 용도 | 데이터 | 질의 | qrels |
| --- | --- | ---: | --- |
| router 학습(#24) | 공식 `train` | 43,394 | 문서화된 등급 규칙으로 재구성(136,150행) |
| 모델·mode 선택 | 공식 `validation` | 1,000 | 공식 graded label(2,704행) |
| 최종 held-out | 공식 `test` | 1,000 | 공식 graded label(2,617행) |

- **Corpus:** 모든 split이 같은 36,281개 Pair corpus를 사용한다. 문서 본문은 `positive`만 쓰고 title은 색인하지 않는다
  (FAQ title이 질문 원문과 같기 때문). 문서 ID = `source_chunk_id`, 질의 ID = `anchor_chunk_id`.
- **Seed:** 새 split을 만들지 않았다(공식 split, upstream seed 42).
- **Primary slice:** test 중 dataset이 비합성으로 표기한 `base_*` + `crawl_*` 721개.
- **Secondary:** 전체 test 1,000개, 합성 질의 279개, train과 같은 문장을 제외한 940개(민감도).
- **Metric:** 기존 이진 지표(relevance > 0) P@1 · Recall@10 · MRR@10 · nDCG@10과, card 권장 graded nDCG@10(gain 1–3)을 함께 보고한다.
  primary slice는 정답 chunk가 하나뿐이어서 이진/graded nDCG가 같다.
- **Leakage 검사(준비 시 실패 조건):** split 간 질의 ID·parent_doc·원본 chunk 중복 0, validation/test 정답 chunk와 train 원본 chunk 중복 0,
  공식 qrels와 등급 규칙 불일치 0, `anchor_parent_doc_id`와 card 규칙 불일치 0.
- **Test 격리:** test는 학습·튜닝·방식 선택에 쓰지 않는다. oracle/선호 분석 함수는 test split을 거부한다.

## Dataset limitations

- **출처 표기는 card 선언이다.** `base_*`가 실제 사용자 질문이라는 점은 dataset card의 설명이며 독립적으로 검증할 수 없다.
  일부 `base` 질문은 BC카드 고유 정보가 아닌 일반 금융 지식 질문이다(예: 금리·공매도). 그래서 slice 이름을 `real`이 아닌 `non_synthetic`으로 둔다.
- **Train qrels 재구성:** train에는 label이 없어 등급 규칙을 적용했다. 규칙이 공식 validation/test label을 100% 재현했지만 train에 대한 공식 label은 아니다.
- **parent-doc 범위 label:** 다른 parent_doc에 있는 관련 chunk는 rel=0이다(card가 밝힌 false negative).
- **동일 본문 문서:** 189그룹 386개 문서가 byte 단위로 같은 본문이다. ID를 유지하며 정답의 동일 본문 쌍을 찾으면 오답으로 처리된다
  (정답 원본 chunk에 쌍이 있는 질의: validation 11, test 8).
- **질의 문장 중복:** 같은 질의 문장이 train∩test 60, train∩validation 50, validation∩test 3개 있다(parent_doc·정답 chunk는 겹치지 않음).
  `text_in_train` 필드로 표시하고 민감도 결과를 함께 보고한다.
- **합성 질의의 낮은 Recall:** 합성 질의는 한 parent_doc의 여러 chunk가 정답이라(최대 12개) 이진 Recall@10이 낮게 나온다.
- 좁은 금융 도메인이며 card의 pipeline 코드 링크(`MokaGrinder`)는 `<owner>` placeholder로 남아 있어 생성 과정을 직접 재현할 수 없다.

## External-service note (#26)

`EXTERNAL_UPLOAD_REQUIRES_MANUAL_TERMS_CHECK`

법률 판단이 아니라 공개 메타데이터 확인 결과다.

- 두 dataset 모두 `license: cc-by-4.0`으로 공개되어 있고 card에 업로드·외부 처리를 금지하는 문구는 없다.
- CC-BY-4.0은 출처 표시를 요구한다. 인용 표기: *BC Card AX Team, BCAI Finance Kor Embedding Pair / Triplet (2026), Hugging Face*.
- 원천은 BC카드 사이트 crawl과 사내 BCAI Q&A이며 card는 PII masking을 했다고만 밝힌다. 원 사이트 콘텐츠의 별도 이용 조건은 card에 없다.
- #26은 업로드 전에 provider의 데이터 보관·학습 사용 약관, 원천 콘텐츠 이용 조건, 출처 표시 방식을 직접 확인해야 한다.

## Baseline results

3차 설정을 그대로 사용했다. BM25 char_wb 2–4-gram(k1=1.5, b=0.75), Dense BGE-M3 `5617a9f…`(FP32, batch 1, max_seq_length 8192),
Fixed Score Fusion `bm25_weight=0.05`. BC Card 결과로 어떤 설정도 바꾸지 않았다.

### Held-out test — primary: non-synthetic (721)

| 방식 | P@1 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.4189 | 0.7406 | 0.5232 | 0.5755 |
| Dense (BGE-M3) | 0.6560 | **0.8988** | 0.7460 | 0.7835 |
| Fixed Score Fusion | **0.6852** | 0.8946 | **0.7615** | **0.7940** |

### Held-out test — secondary: 전체 test (1,000)

| 방식 | P@1 | Recall@10 | MRR@10 | nDCG@10 | graded nDCG@10 | CPU Mean ms | CPU P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.4030 | 0.6044 | 0.5013 | 0.4887 | 0.4975 | 7.472 | 8.064 |
| Dense (BGE-M3) | 0.5800 | 0.7264 | 0.6699 | 0.6459 | 0.6586 | 7.070 | 7.302 |
| Fixed Score Fusion | **0.6100** | **0.7280** | **0.6881** | **0.6583** | **0.6718** | 6.435 | 6.851 |

- 합성 질의 279개 nDCG@10: BM25 0.2645 · Dense 0.2902 · Fusion 0.3077.
- train 중복 문장 제외 940개 nDCG@10: BM25 0.5080 · Dense 0.6638 · Fusion 0.6763.
- Latency는 3차 최종 평가와 같은 범위(캐시 vector 기반 CPU 검색·결합, 2 threads, warm-up 1 + 측정 5회 × 1,000 질의)이며
  query embedding을 제외한다. BGE-M3 query encoding은 GPU에서 별도로 평균 9.46 ms / P95 9.68 ms였다.
  세 방식 차이는 1 ms 안쪽이며(Fusion이 BM25 단독보다 빠르게 측정됨) 원인은 분석하지 않았다. 품질 비교의 근거로 쓰지 않는다.
- primary slice에서 Fusion은 Dense보다 P@1·MRR@10·nDCG@10이 높지만 Recall@10은 소폭 낮다. FAQ 직접 질의(50)에서는 Dense보다 낮다(nDCG@10 0.6575 vs 0.6854).

전체 slice는 [test 요약](experiments/bccard/test-summary.json)과 [CSV](experiments/bccard/baselines.csv)에 있다.

## Validation analysis for #24

validation(1,000)에서만 계산했다. 품질은 1회 측정이며 latency는 참고값이다.

| 방식 | P@1 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.4130 | 0.6296 | 0.5141 | 0.5022 |
| Dense (BGE-M3) | 0.5810 | 0.7306 | 0.6730 | 0.6462 |
| Fixed Score Fusion | 0.5960 | 0.7418 | 0.6895 | 0.6600 |
| Oracle(셋 중 query별 최고, 분석 전용) | 0.6590 | 0.7511 | 0.7383 | 0.6910 |

질의별 nDCG@10 비교(동률 허용 오차 1e-12):

| 비교 | 질의 수 | 비율 |
| --- | ---: | ---: |
| BM25 > Dense / Dense > BM25 / 동률 | 137 / 418 / 445 | 13.7% / 41.8% / 44.5% |
| Fusion > Dense / Fusion < Dense / 동률 | 163 / 68 / 769 | 16.3% / 6.8% / 76.9% |
| 단독 최고: BM25 / Dense / Fusion | 94 / 67 / 90 | 9.4% / 6.7% / 9.0% |
| 세 방식 동률 | 435 | 43.5% |

- Oracle은 동률이면 Fusion을 유지하고 더 나은 방식이 있을 때만 바꾼다(선택: Fusion 838 · BM25 94 · Dense 68).
  Oracle 이득은 Dense 대비 nDCG@10 +0.0448 · P@1 +0.078, Fusion 대비 nDCG@10 +0.0310 · P@1 +0.063 · MRR@10 +0.0488 · Recall@10 +0.0093이다.
  Fusion은 Dense→Oracle 차이의 31%만 확보한다.
- **BM25가 크게 이긴 질의(nDCG@10 차이 ≥ 0.2, 71개)** 는 Dense가 크게 이긴 질의(317개)보다 영문 포함(29.6% vs 22.7%)·대문자 약어(29.6% vs 19.6%)
  비율과 정답 chunk와의 문자 bigram 겹침(0.54 vs 0.45)이 높고, 합성 질의 비중(27% vs 14%)이 크다. 숫자 포함은 차이가 없다(8.5% vs 6.3%, 동률 8.8%).
  차이는 있지만 분리가 뚜렷하지 않아 query 문자열 특징만으로는 BM25 우위를 예측하기 어렵다.
- **BM25 hub 문서:** `crawl_bc-merchant_customer-center_faq_41000`이 validation 질의 72개(7.2%)에서 BM25 1위였다.
  "무엇인가요?", "어떻게 하나요?" 같은 질문형 어미가 많은 짧은 문서가 char n-gram에서 거의 모든 질문과 겹친다. BM25 top score만으로는 신뢰도를 판단하기 어렵다.

대표 예시(차이가 큰 순으로 기계적으로 선택, [전체](experiments/bccard/validation-handoff.json)):

| 구분 | 질의 ID | 질의 | nDCG@10 B / D / F | 관찰 |
| --- | --- | --- | --- | --- |
| BM25 우위 | `base_bcai_003852_000` | 해외 직구 사이트가 폐쇄된 경우 환불받는 방법은 무엇인가요? | 1.00 / 0.00 / 0.32 | Dense·Fusion 1–3위가 다른 환불 문서(`base_bcai_0175xx`) |
| BM25 우위 | `base_bcai_022921_000` | 기준금리가 1.25%에서 1%로 인하되면 법인의 설비투자 대출 금리는…? | 1.00 / 0.29 / 0.29 | 수치 조건이 같은 문서를 BM25만 1위로 찾음 |
| BM25 우위 | `synth_base_bcai_024189_001_q1` | PG사를 이용하면 결제 관련 고객 지원은 누가 담당하나요? | 0.70 / 0.00 / 0.00 | 영문 약어 PG 일치. weight 0.05로는 Fusion이 Dense 순위를 따름 |
| Dense 우위 | `crawl_…faq_422000` | 플래티늄카드의 "구매품 보상"이 무엇인가요? | 0.00 / 1.00 / 0.63 | BM25 1위가 hub 문서 `faq_41000` |
| Dense 우위 | `crawl_…faq_508000` | 인터넷 옵션 설정은 어떻게 하나요? | 0.00 / 1.00 / 1.00 | BM25 1–3위가 모두 가맹점 FAQ |
| Dense 우위 | `crawl_…faq_482000` | 이사를 가게 되었는데 우편물 수령지 변경은 어떻게 하나요? | 0.00 / 1.00 / 1.00 | 낮은 어휘 겹침(0.29) |
| Fusion 우위 | `base_bcai_012047_000` | 공매도는 어떤 경우에 유리한가요? | 0.00 / 0.63 / 1.00 | Dense 2위 정답을 BM25 점수가 1위로 올림 |
| Fusion 우위 | `base_bcai_008974_000` | 카카오페이의 주요 성과는 무엇인가요? | 0.00 / 0.63 / 1.00 | 같은 패턴 |
| 동률 | `crawl_…faq_69000` | 모바일카드와 플라스틱카드의 카드번호와 유효기한은 동일한가요? | 1.00 / 1.00 / 1.00 | 높은 어휘 겹침(0.73) |
| 동률 | `crawl_…faq_379000` | 카드 메뉴에서 관심카드등록을 했는데, 어디서 확인하나요? | 1.00 / 1.00 / 1.00 | 세 방식 1위 동일 |

`crawl_…`는 `crawl_bc-personal_customer-center_`의 축약이다.

### `ADAPTIVE_ROUTER_HEADROOM: MODERATE`

통계적 유의성 주장이 아닌 인계용 기술적 판단이다.

- 질의별 우위가 실제로 갈린다. 세 방식이 모두 같은 질의는 43.5%이고, BM25만 최고인 질의가 9.4%다.
- validation oracle은 Fusion보다 nDCG@10 +0.031(상대 +4.7%), P@1 +0.063 높고 Fusion은 Dense→Oracle 차이의 31%만 확보한다.
- 반면 oracle은 정답을 아는 상한이며, BM25 우위 질의는 영문 약어·어휘 겹침 경향이 있지만 query 문자열 특징만으로 깔끔하게 분리되지 않는다.
  BM25 hub 문서 때문에 BM25 점수 자체에도 잡음이 있다.

## Reproduction

```bash
uv sync --locked --extra sparse --extra dense
uv run --locked --extra sparse --extra dense python -m scripts.prepare_bccard --download
uv run --locked --extra sparse --extra dense python -m scripts.evaluate_enterprise encode
uv run --locked --extra sparse --extra dense python -m scripts.evaluate_enterprise evaluate \
  --split validation --repeats 1 --output-dir artifacts/bccard-eval/validation
uv run --locked --extra sparse --extra dense python -m scripts.evaluate_enterprise analyze \
  --split validation --output-dir artifacts/bccard-eval/validation
uv run --locked --extra sparse --extra dense python -m scripts.evaluate_enterprise evaluate \
  --split test --output-dir artifacts/bccard-eval/test
```

원본은 `data/raw/bccard/`, 준비 데이터는 `data/processed/bccard/`, vector는 `artifacts/bccard-bge-m3/`에 생성되며 Git에 넣지 않는다.
준비 결과는 [manifest](experiments/bccard/manifest.json)의 SHA-256과, vector는 [encoding 기록](experiments/bccard/encoding.json)과 일치해야 한다.
측정 환경: NVIDIA RTX PRO 6000 Blackwell(encoding), CPU 2 threads(검색).

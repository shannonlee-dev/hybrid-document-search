# 외부 baseline 재검토 (멘토 리뷰 대비)

목적은 더 좋은 숫자를 찾는 것이 아니라, 기존 결과가 외부 모델 비교와 독립 검토를 견디는지 확인하는 것임.
새 학습·인코딩·GPU 실행은 하지 않았음. 커밋된 per-query 결과를 다시 계산했고, 공개 BC Card 원천 데이터로 근접 중복 검사만 새로 했음.
프로토콜은 계산 전에 [`protocol.md`](validation/2026-11-04/external-baseline/protocol.md)로 커밋했음(`d696399`).
다만 비교 대상의 **평균값은 이미 공개 문서에 있었으므로** 확증 분석이 아님. 새로 계산한 것은 paired CI, Recall@5, 집중도, 민감도, 근접 중복임.

## 요약

| 질문 | 측정 결과 | 해석 | 한계 |
|---|---|---|---|
| 카카오 모델을 비교에 쓸 수 있나 | `kanana-2-3b-base/instruct`는 생성형(`text-generation`). 검색용 `kanana-2-3b-embedding`(`sentence-similarity`)은 PR #37에서 card 권장 설정으로 이미 평가됨 | 생성형은 retriever로 쓰지 않음. Embedding 모델 결과를 재사용 | Kanana 학습 데이터에 BC Card가 포함됐는지 확인 불가(N/A) |
| Finance vs Kanana (Dense) | nDCG@10 +.0059 [−.0117, +.0239], W/T/L 108/519/94 | **차이 확인 안 됨** | Test 721은 여러 번 노출됨 |
| Finance vs Kanana (Fixed .05) | nDCG@10 +.0029 [−.0138, +.0198], W/T/L 95/538/88 | **차이 확인 안 됨** | 같음 |
| Finance − BGE 개선은 의미 있나 | 서비스 Hybrid +.0176 [+.0026, +.0328], Dense +.0252 [+.0095, +.0410] (PR #42 재현) | BGE 대비 개선은 유지. 근접 중복 질의를 빼도 유지(+.0213, +.0304) | Seed43 하나, query sampling 불확실성만 반영 |
| 개선이 일부 질의에 몰렸나 | 개선 합 중 상위 10개 질의 비중 23–28%. 변화 큰 10개를 빼도 평균 차이 양수 | 소수 질의에만 의존하지 않음 | 기술 통계, 판정 기준 없음 |
| 데이터 누수 | Primary 721 중 train 질의와 정규화 일치 37, 문자 3-gram cosine ≥ .90 71개. 정답 chunk ↔ train positive ≥ .90 37개 | 정확 일치는 Finance 학습에서 이미 제외됨. 근접 중복은 학습에 남아 있었으나 제외해도 결론 동일 | 의미 수준 paraphrase는 검사하지 않음 |
| 독립 검증인가 | 아님 | 새 holdout 없음 | — |

**한 줄 결론.** Finance encoder는 원본 BGE-M3 대비 BC Card 개선이 유지되지만, 학습 없이 쓴 외부 모델 Kanana-2-3B-embedding과는 구분되지 않음.
즉 "금융 특화 학습으로 얻은 이득"은 "더 강한 범용 한국어 encoder로 바꿔 얻는 이득"과 이 데이터에서 같은 크기임. 반면 Finance는 BGE-M3와 같은 크기·같은 지연이고, Kanana는 더 큰 모델임(아래 비용).

## 1. 사전 감사

### 기존 PR 상태

#23, #37, #40–#43 모두 main에 병합됨. 이 작업은 `docs/external-baseline-review` branch에서 진행했고 기존 evidence 파일은 바꾸지 않았음.

### 모델 구성

| 이름 | 구성 | 출처 |
|---|---|---|
| BGE-M3 | `BAAI/bge-m3` @ `5617a9f6…`, 1024-dim, FP32, max 8192, prefix 없음 | `config/dense_models.toml` |
| Finance-BGE | 위 BGE-M3에서 **query 쪽 마지막 4 block(50,384,896 parameter)만** 학습. 문서 vector는 원본 BGE-M3 그대로. 서비스 평가는 Seed43 재현 checkpoint `fbbd5ed0…` | `BCCARD_QUERY_ENCODER_TRAINING.md`, `config/query_checkpoints/` |
| Kanana | `kakaocorp/kanana-2-3b-embedding` @ `2c85dbe6…`(2026-11-04 HF HEAD와 같음), 2560-dim, bf16 → float32 재정규화, max 4096, prefix 없음 | `PHASE5_EXTERNAL_VALIDATION.md` |
| KURE-v1 | `nlpai-lab/KURE-v1` @ `8b418a58…`. **BGE-M3 fine-tune** | Ko-MIRACL 10k에서만 평가(`DENSE_RETRIEVAL.md`) |
| KURE-v2 | `skt/A.X-Encoder-base` fine-tune | Ko-MIRACL Train100에서만 평가 |

HCX 계열 검색 평가 기록은 없음(`COMMERCIAL_RAG_BENCHMARK.md`의 상용 서비스 비교만 있음).

### BC Card Test 721 사용 이력

| 단계 | Test 721로 한 것 | Test로 설정을 골랐나 |
|---|---|---|
| PR #28 (4차) | BM25·Dense·Fixed 평가 | 아님. BM25 weight 0.05는 Ko-MIRACL Train100에서 선택(`SCORE_FUSION_RESULTS.md`) |
| PR #30 | Adaptive router 평가 | 아님(router는 train) |
| PR #37 (5차) | Kanana 평가 | 아님(설정은 model card만) |
| PR #38·#39 | QDAP·DAT 평가 | 문서상 Validation 선택 |
| PR #40·#41 | Finance checkpoint·α | Validation 선택. Test는 replay만 |
| PR #42·#43 | 서비스 경로 평가, 동점 민감도 | 사전 등록 후 평가 |
| 이번 | 재분석만 | 선택 없음 |

선택에 쓰지 않았어도 위 8개 단계에서 같은 Test 결과를 봤으므로, 연구 방향 결정에는 간접적으로 영향을 줬을 수 있음. 독립 held-out 검증이라고 주장하지 않음.

### 외부 검증 범위

Ko-StrategyQA dev 592, Ko-MIRACL dev 50(PR #42). Kanana는 외부 데이터셋에서 평가하지 않았음(N/A).

### 확인한 불일치와 정합성

- **두 평가 경로의 연결.** Kanana는 Phase 5 enterprise 경로, Finance는 PR #42 서비스 경로에서 나왔음. 같은 BGE-M3 run이 두 경로에서 Dense·Fixed 모두 **721개 질의의 relevant 순위와 nDCG가 완전히 같음**(`comparison.json`의 `gate`). 그래서 두 경로가 같은 corpus·query·qrels·BM25·결합식·전체 corpus 후보·동점 규칙을 쓴다고 보고 비교했음.
- **지표 재계산.** Primary는 정답 chunk가 하나라 relevant 순위에서 다섯 지표가 정해짐. 커밋된 값과 재계산 값의 불일치 0.
- **구현 재현.** 같은 seed로 Phase 5 BGE→Kanana Dense(+.0194 [+.0058, +.0330])와 PR #42 BGE→Finance 서비스 Hybrid(+.0176 [+.0026, +.0328])를 다시 계산해 공개값과 같았음.
- Phase 5 문서에 이미 적힌 불일치(PR #28/#30 vector SHA 차이, 문서 간 latency 차이)는 그대로 유효함. Finance 계열 "Main canonical .794015 vs 내부 Fixed .791013"은 결합식이 달라서 생긴 값이고 같은 run이 아님(`BCCARD_QUERY_ENCODER_TRAINING.md`).

## 2. 비교 공정성

| 조건 | Kanana | Finance | 같음? |
|---|---|---|---|
| 문서·질의·relevance label | BC Card Pair `e022cb01…` / Triplet `f63d5996…`, primary 721 | 같음 | 같음 (manifest SHA `b6bb7822…`) |
| 지표 구현 | relevant 순위에서 재계산 | 같음 | 같음 |
| 후보 집합 | 전체 corpus 36,281 | 같음 | 같음 |
| Retrieval depth | top 10 | 같음 | 같음 |
| BM25 corpus·전처리 | `char_wb` 2–4-gram, k1 1.5, b .75 | 같음 | 같음 |
| Fusion | `Dense + .05×BM25/max(BM25)` | 같음(서비스 Hybrid) | 같음 |
| 문서 encoder | Kanana | 원본 BGE-M3 | **다름** (설계상) |
| Query encoder 학습 | 없음 (zero-shot) | BC Card train 43,150개로 학습 | **다름** |
| dtype·길이·차원 | bf16 / 4096 / 2560 | FP32 / 8192 / 1024 | **다름** |
| 권장 사용법 | card: task prompt 없음, mean pooling + normalize, cosine | BGE-M3 card: dense는 prefix 없음 | 각자 card 준수 |

Kanana의 α Fixed 결과는 없어서 Finance의 α=.84 조건과는 비교하지 않았음. 모델 차이와 dtype·길이 차이는 분리하지 않았음.

## 3. 결과 (primary 721)

### 평균

| 방식 | P@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| BM25 | .4189 | .6616 | .7406 | .5232 | .5755 |
| BGE-M3 Dense | .6560 | .8558 | .8988 | .7460 | .7835 |
| BGE-M3 + Fixed .05 | .6852 | .8488 | .8946 | .7615 | .7940 |
| Kanana Dense | .6893 | .8724 | .9015 | .7704 | .8029 |
| Kanana + Fixed .05 | **.7004** | .8779 | .9001 | **.7785** | .8087 |
| Finance Dense | .6893 | **.8821** | **.9182** | .7731 | .8087 |
| Finance + Fixed .05 (서비스 Hybrid) | .6907 | **.8821** | **.9182** | .7767 | **.8116** |

Kanana의 Recall@5는 이번에 처음 계산했음.

### Paired 차이 (B − A, seed 20261102, 20,000 resamples, 95% percentile CI)

| 비교 | nDCG@10 | Recall@5 | Recall@10 | MRR@10 | P@1 | W/T/L |
|---|---|---|---|---|---|---|
| **P1** Kanana Dense → Finance Dense | +.0059 [−.0117, +.0239] | +.0097 [−.0125, +.0319] | +.0166 [−.0028, +.0374] | +.0028 [−.0169, +.0229] | .0000 [−.0291, +.0291] | 108/519/94 |
| **P2** Kanana Fixed → Finance Fixed | +.0029 [−.0138, +.0198] | +.0042 [−.0153, +.0236] | +.0180 [−.0014, +.0374] | −.0018 [−.0202, +.0169] | −.0097 [−.0361, +.0166] | 95/538/88 |
| BGE Dense → Kanana Dense | +.0194 [+.0060, +.0327] | +.0166 [+.0014, +.0319] | +.0028 [−.0111, +.0166] | +.0244 [+.0084, +.0405] | +.0333 [+.0069, +.0596] | 106/542/73 |
| BGE Fixed → Kanana Fixed | +.0147 [+.0021, +.0277] | +.0291 [+.0139, +.0444] | +.0055 [−.0069, +.0194] | +.0170 [+.0018, +.0325] | +.0153 [−.0097, +.0402] | 93/565/63 |
| BGE Dense → Finance Dense | +.0252 [+.0095, +.0410] | +.0264 [+.0069, +.0472] | +.0194 [+.0000, +.0388] | +.0271 [+.0101, +.0444] | +.0333 [+.0083, +.0583] | 110/545/66 |
| BGE Fixed → Finance Fixed | +.0176 [+.0026, +.0328] | +.0333 [+.0139, +.0527] | +.0236 [+.0042, +.0430] | +.0152 [−.0005, +.0313] | +.0055 [−.0166, +.0277] | 91/569/61 |

P1·P2의 nDCG@10 Bonferroni 97.5% CI는 [−.0142, +.0266], [−.0163, +.0224]임. 아래 네 행은 기존 결과의 재계산이며 Phase 5(seed 20261030, 10,000회)와 seed가 달라 CI 끝자리가 조금 다를 수 있음.

**해석.**
- P1·P2 모두 CI가 0을 포함함. Finance가 Kanana보다 낫다고도, 못하다고도 말할 수 없음.
- 평균은 거의 같지만 질의 단위로는 많이 다름(P1 202개 질의에서 nDCG가 다르고 개선 108 / 악화 94). 두 encoder가 서로 다른 질의를 맞히고 평균에서 상쇄됨.
- 방향 차이: Finance는 Recall@10이 높은 쪽(+.017~.018, CI 하한이 0 근처), Kanana + Fixed는 P@1·MRR이 약간 높은 쪽. 모두 CI가 0을 포함하므로 경향 기술일 뿐임.
- Finance − BGE 개선 크기(+.018~.025)와 BGE → Kanana 교체 이득(+.015~.019)은 CI가 크게 겹침.

### 집중도 (nDCG@10, `comparison.json`의 `concentration`)

| 비교 | 변화 질의 | 개선 합 / 악화 합 | 상위 10개 개선 비중 | |Δ| 상위 10개 제외 평균 |
|---|---:|---|---:|---:|
| P1 | 202 | +42.69 / −38.46 | 20.5% | +.0031 |
| P2 | 183 | +37.31 / −35.23 | 24.1% | −.0027 |
| BGE → Finance Dense | 176 | +42.04 / −23.85 | 23.1% | +.0143 |
| BGE → Finance Fixed | 152 | +35.03 / −22.35 | 27.7% | +.0066 |
| BGE → Kanana Dense | 179 | +36.21 / −22.26 | 20.6% | +.0091 |

BGE 대비 개선은 변화가 큰 10개 질의를 빼도 양수로 남음. P2는 10개를 빼면 부호가 바뀌어 사실상 0 근처임.

### 민감도

| 조건 | 질의 수 | P1 nDCG@10 | P2 nDCG@10 | BGE→Finance Dense | BGE→Finance Fixed |
|---|---:|---|---|---|---|
| 전체 | 721 | +.0059 [−.0117, +.0239] | +.0029 [−.0138, +.0198] | +.0252 [+.0095, +.0410] | +.0176 [+.0026, +.0328] |
| 중복 vector 라벨 충돌 10개 제외 (PR #43) | 711 | +.0063 [−.0117, +.0242] | +.0032 [−.0135, +.0201] | — | — |
| Train 근접 중복(≥ .90) 71개 제외 | 650 | +.0098 [−.0079, +.0280] | +.0049 [−.0122, +.0219] | +.0304 [+.0148, +.0466] | +.0213 [+.0062, +.0370] |

PR #43의 동점 상·하한 bound는 Kanana 문서 vector가 없어 계산하지 못했음(N/A). 라벨 충돌 질의 제외는 encoder와 무관하게 정의되는 대안 점검임. 어느 조건에서도 P1·P2 판정은 바뀌지 않았음.

## 4. 데이터 누수

준비 단계 검사(`BC_CARD_DATASET.md`): split 간 질의 ID·parent_doc·원본 chunk 중복 0, 정답 chunk와 train 원본 chunk 중복 0.
Finance 학습은 Validation/Test와 정규화 문장이 같은 244개 train 행을 제외했음.

이번 근접 중복 검사(`leakage.json`, 문자 3-gram TF-IDF cosine, casefold + 공백 정규화):

| 대상 | ≥ .80 | ≥ .90 | ≥ .95 | 정규화 완전 일치 |
|---|---:|---:|---:|---:|
| Test 질의 721 ↔ train 질의 43,394 | 144 | 71 | 47 | 37 (`text_in_train` 37과 일치) |
| Test 정답 chunk 721 ↔ train positive chunk 34,621 | 96 | 37 | 16 | — |

- **측정.** 37개 완전 일치는 Finance 학습에서 이미 빠졌지만, 나머지 34개 근접 중복 질의(.90 ≤ cos < 1)는 학습에 남아 있었음.
- **해석.** 이 71개를 빼면 Finance − BGE 이득은 줄지 않고 커짐(위 민감도 표). 근접 중복 train 질의가 Finance 이득을 만든 것은 아님.
- **한계.** 문자 n-gram 유사도는 표면 중복만 잡음. 의미가 같은 paraphrase는 검사하지 않았음. 문서 측 근접 중복 37개는 정답 chunk와 비슷한 train positive가 학습에 쓰였다는 뜻이고, 이 효과는 따로 분리하지 않았음. Kanana의 학습 데이터 구성은 공개 정보로 확인할 수 없음(N/A).

## 5. 서비스 비용 (기존 기록 정리)

새로 측정하지 않았음. 측정 조건이 서로 달라 **다른 문서의 값끼리 직접 비교하지 않음**.

| 항목 | BGE-M3 | Finance | Kanana | 출처·조건 |
|---|---|---|---|---|
| Query encoding mean / P95 | 10.96 / 11.18 ms | 10.95 / 11.17 ms | N/A (같은 조건 없음) | PR #42, Blackwell 1장, FP32, in-process, 721 × 3 |
| Query encoding (다른 조건) | 9.46 / 9.68 ms (PR #28) | — | 22.49 / 22.75 ms (bf16) | PR #37, batch 1. dtype이 달라 비교 불가 |
| 검색 단계 (query encoding 제외) | Dense 6.96, Fixed 6.39 ms | N/A | Dense 16.21, Fixed 9.10 ms | PR #37 같은 세션, CPU 2 threads, cached vector |
| 서비스 Hybrid end-to-end (in-process) | 15.23 ms | 15.28 ms | N/A | PR #42 |
| HTTP Hybrid mean | 14.58 / 14.59 ms | 14.60 / 14.54 ms | N/A | PR #42, localhost, 동시 1 |
| Cold start (서비스 준비) | 25.2–25.5 s | 25.4–25.6 s | N/A | PR #42, BM25·TF-IDF 인덱싱 포함 |
| GPU 메모리 | max allocated 2,200 MiB | 같음 | corpus 인코딩 peak 6.61 GB | 측정 대상이 달라 비교 불가 |
| Model weights | N/A (이 저장소에 기록 없음) | BGE-M3와 같은 구조 | 6.4 GB | PR #37 |
| 문서 vector (FP32, 계산값) | 36,281 × 1024 × 4 B ≈ 148.6 MB | 같음 (원본 BGE 재사용) | 36,281 × 2560 × 4 B ≈ 371.5 MB | 산술 계산, 측정 아님 |
| 문서 인코딩 | N/A | 재인코딩 불필요 | 840 s (Blackwell, batch 1) | PR #37 |
| 추가 학습 | — | 4,096 steps 기록 시간 318 s (RTX 4060), 125·139 s (Blackwell 두 seed) | 없음 | 학습 protocol의 `seconds`. 데이터 준비·hard negative 채굴 포함 여부는 기록에 없음 |

**해석.** Finance는 BGE-M3와 지연·메모리가 같고 문서 인덱스를 다시 만들 필요가 없음. Kanana로 바꾸려면 corpus 전체 재인코딩, 2.5배 큰 vector, bf16 GPU 추론이 필요하고 검색 단계가 느려짐(Dense 6.96 → 16.21 ms, 같은 세션). Kanana의 서비스 경로 end-to-end·cold start는 측정하지 않았음.

## 6. 일반 도메인 퇴보 (PR #43 재사용)

- **확인된 현상.** Ko-StrategyQA 592에서 Finance − BGE가 세 조건 모두 음수이고 CI 상한 < 0(서비스 Hybrid −.0142 [−.0216, −.0071]). Ko-MIRACL 50은 결론 없음.
- **기각된 가설.** Query drift가 클수록 손실이 크다(H1), 금융 방향으로 끌려간다(H3)는 지지되지 않음.
- **관찰만 된 것.** 점수 분포 전체가 평행 이동하고, z-정규화 후 relevant 분리가 BC Card에서는 커지고 StrategyQA에서는 작게 줄었음. 정답 passage가 2개 이상인 질의에서 손실이 큼.
- **입증되지 않은 원인.** 학습 분포 밖 질의에서 근접 경쟁 passage 순서가 흔들린다는 설명은 직접 증거가 없음. 이번 작업에서 새 원인 설명을 추가하지 않았음.
- Kanana는 일반 도메인에서 평가하지 않아 "Kanana는 퇴보가 없다"고 말할 근거도 없음(N/A).

## 7. RAG 품질 (PR #23 재사용)

- PR #23은 **Ko-StrategyQA 100 질의**, BGE-M3 기반 TF-IDF/BM25/Dense/RRF/Score Fusion을 비교한 탐색 실험임. **Finance encoder와 Kanana는 RAG 평가에 들어가지 않았고, BC Card로 RAG 평가를 한 적도 없음.**
- Retrieval 평가는 qrels 기반 순위 지표, downstream은 generator(EXAONE-4.5-33B) 응답을 judge(A.X-4.0)가 RAGAS로 채점한 것임. 데이터·질의 수·top-k(10 vs 생성 context 5)가 다름.
- Score Fusion − Dense(nDCG +.006)는 RAGAS 세 지표 모두 CI가 0을 포함함. **Faithfulness 개선은 확인되지 않았음.** Sparse–Dense의 Faithfulness 차이는 응답 거부율 차이와 겹침.
- Judge 1개, 반복 1회. Judge 비결정성과 영어 RAGAS prompt로 한국어를 채점한 영향은 추정하지 않았음.
- 따라서 "Finance encoder가 RAG 품질을 높인다"는 주장은 현재 근거가 없음.

## 8. 멘토가 지적할 수 있는 약점

| 약점 | 현재 상태 | 이번 PR |
|---|---|---|
| 외부 강한 encoder 대비 금융 학습의 실익 | Kanana와 차이 확인 안 됨 | **새로 정량화** (P1·P2) |
| 같은 Test 721 반복 사용 | 새 holdout 없음 | 사용 이력 표로 명시. 해결 안 됨 |
| 근접 중복 누수 | 정확 일치만 검사했었음 | **근접 중복 검사와 민감도 추가**. Paraphrase는 미검사 |
| Seed 분산 | Seed43 하나로 서비스 평가 | 해결 안 됨 |
| 서비스 Hybrid 결론 여유 | Bonferroni 하한 +.0006, 동점 극단 bound에서 0 포함 (PR #43) | 그대로 유효 |
| 모델 간 latency 비교 | 조건이 다른 측정이 섞여 있음 | 조건별로 분리 정리. 같은 조건 Kanana 서비스 측정은 미해결 |
| 일반 도메인 퇴보 | 원인 미입증 | 변경 없음 |
| RAG 개선 | Finance로는 측정 안 됨 | 변경 없음 |
| Kanana 비교 조건 | dtype·길이·차원이 BGE와 다름 | 분리 안 됨 |

## 9. 남은 작업과 예상 비용 (실행하지 않음)

이 머신에는 GPU가 없어 아래는 실행하지 않았음. 시간은 기존 기록을 근거로 한 대략치이며 측정값이 아님.

| 작업 | 필요 | 기존 기록 기준 대략 |
|---|---|---|
| Kanana 서비스 경로 latency·cold start·메모리 (PR #42 하네스) | bf16 GPU 1장, Kanana weights 6.4 GB | 문서 vector cache가 있으면 1시간 이내, 없으면 corpus 인코딩 840 s 추가 |
| Kanana의 Ko-StrategyQA·Ko-MIRACL 평가 | GPU, 외부 corpus 인코딩 | corpus 크기 의존, N/A |
| Finance Seed42 서비스 평가 | GPU, Seed42 checkpoint | PR #42와 비슷한 규모 |
| 새 금융 holdout | 새 질의·라벨 작성 | 사람 작업, 비용 N/A |

유료 API 비용은 모두 0(로컬 모델만 사용).

## 재현

```bash
uv sync --locked --extra sparse --extra dense
# 근접 중복 검사만 데이터가 필요함 (manifest SHA b6bb7822… 확인)
uv run --locked --extra sparse --extra dense python -m scripts.prepare_bccard \
  --download --raw-dir data/raw/bccard --output-dir data/processed/bccard
uv run --locked --extra sparse --extra dense python -m scripts.analyze_external_baseline leakage
# 커밋된 파일만 사용
uv run --locked --extra sparse --extra dense python -m scripts.analyze_external_baseline analyze
uv run --locked --extra sparse --extra dense python -m scripts.analyze_external_baseline check
uv run --locked --extra sparse --extra dense pytest -q tests/test_external_baseline_evidence.py
```

Evidence: [`comparison.json`](validation/2026-11-04/external-baseline/comparison.json), [`leakage.json`](validation/2026-11-04/external-baseline/leakage.json).
질의·문서 원문은 커밋하지 않았고 질의 ID와 최대 유사도만 남겼음.

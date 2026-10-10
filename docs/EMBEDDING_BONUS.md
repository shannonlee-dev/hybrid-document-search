# 보너스 실험: EmbeddingGemma 2·KURE-v2 vs BGE-M3

2026-10-08, 기본 모델 BGE-M3의 대체 가능성을 추가로 확인했다.
**결론은 BGE-M3 유지다.** Gemma는 Dev nDCG가 낮고 검색 지연이 2.17배였으며,
후속 Train 차원 축소 실험에서도 역전하지 못했다.
추가 KURE-v2 Train100 비교에서도 nDCG 역전은 없었다.

## 비교 설정

- 데이터: 동일 Ko-MIRACL 문서 10,000개, Train 100개 / Dev 50개, seed 42.
- 환경: RTX 4060 8GB, i3-12100F, 두 모델 FP32·batch size 1.
- 검색: Dense 단독, CPU FAISS `IndexFlatIP`, Top-10, 문서·질의 L2 정규화.
- 입력 길이: 최대 8,192 tokens, 실제 잘린 항목 없음.
- BGE: 1,024차원, CLS pooling, 질의 원문·문서 `{제목}\n{본문}`.
- Gemma: 768차원, 프롬프트 포함 mean pooling, 공식 검색 프롬프트 적용.
  질의는 `task: search result | query: {질의}`, 문서는 `title: {제목} | text: {본문}`.
  문서 형식을 직접 만들고 `prompt=""`로 자동 프롬프트 중복을 방지했다.

## Dev 1:1 결과

| 지표 | BGE-M3 | Gemma 2 |
| --- | ---: | ---: |
| Recall@5 | 0.8137 | 0.8503 |
| Recall@10 | 0.9583 | 0.9447 |
| MRR@10 | 0.8610 | 0.8462 |
| nDCG@10 | **0.8549** | 0.8347 |
| 평균 검색 지연 | **22.87ms** | 49.59ms |

Gemma−BGE nDCG 차이는 **−0.0202**, paired bootstrap 95% 신뢰구간은
**[−0.0758, +0.0382]**였다. 사전 채택 기준의 nDCG 개선·신뢰구간·MRR·Recall@10·지연
조건을 충족하지 못해 **BAD**로 판정했다.

지연은 모델별 독립 프로세스 3회, warm-up 후 질의별 5회 측정했다.
질의 전처리부터 임베딩·FAISS 검색·결과 생성까지 포함하고 CUDA 동기화를 적용했다.
모델 로딩·인덱스 준비·지표 계산·저장은 제외했다.

## 후속 실험: Train 100개 차원 비교

저장된 Gemma 벡터의 앞 768·512·256·128차원을 남기고 문서·질의를 각각 다시 L2 정규화했다.
GPU 재임베딩이나 모델 학습 없이 동일 Train을 평가했다.
선택 규칙은 Train nDCG@10 최대, 동률이면 큰 차원이다.

| 모델 | 차원 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BGE-M3 | 1,024 | 0.9676 | 0.8084 | **0.8232** |
| Gemma 2 | 768 | 0.9415 | 0.7825 | **0.8003** |
| Gemma 2 | 512 | 0.9423 | 0.7805 | 0.7997 |
| Gemma 2 | 256 | 0.9535 | 0.7576 | 0.7901 |
| Gemma 2 | 128 | 0.9287 | 0.7553 | 0.7729 |

**768차원이 최고였으며 모든 차원에서 BGE를 역전하지 못했다.**
256차원은 Recall@10이 개선됐지만 MRR·nDCG는 낮아졌다.
이번 결과에는 nDCG 개선을 위해 차원을 줄일 근거가 없다.
후속 실험에서는 Dev와 지연을 재측정하지 않았다.

## 후속 실험: KURE-v2 Train100 비교

기존 표준 Ko-MIRACL 문서 **10,000개·Train 100개**를 그대로 사용했다.
20,466개 확장 corpus 결과와는 별개다. BGE·Gemma·KURE-v1은 동일 입력의 기존 결과이며,
KURE-v2만 임시 어댑터로 새로 평가했다. 모델 학습이나 운영 코드 변경은 하지 않았다.

- 모델: `nlpai-lab/KURE-v2`, revision `3431f86d399d666083890dbb882aced6708873bc`.
- 호환: 기존 격리 환경의 Sentence Transformers 6.1·Transformers 5.19에서 공식 `MultiVectorEncoder` 사용.
- 설정: FP32·SDPA, 토큰별 128차원, 공식 `[Q]`·`[D]` 마커와 토큰 L2 정규화.
- 질의 확장 64 tokens, 문서 제한 8,192 tokens. 제한 초과 문서는 없었다.
- 문서 batch 최대 4(길이에 따라 축소), 질의 batch 8.
- 검색: 문서 `{제목}\n{본문}`, 10k 전체에 대해 100만 질의·문서 쌍을 전수 MaxSim으로 채점.
  질의의 각 토큰에 대해 문서 토큰과의 최대 유사도를 구한 뒤 합산했다.

| 모델 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: |
| BGE-M3 | 0.967611 | 0.808429 | **0.823154** |
| KURE-v1 | 0.968444 | 0.797373 | 0.814188 |
| Gemma 2 (768) | 0.941500 | 0.782472 | 0.800336 |
| KURE-v2 | 0.924889 | **0.811250** | 0.800142 |

KURE-v2−BGE nDCG 차이는 **−0.023013**, paired bootstrap 95% 신뢰구간은
**[−0.063350, +0.016200]**였다(20,000회, seed 42).
질의별로 KURE-v2 26승, BGE 31승, 동률 43개였다.
KURE-v2는 MRR이 높았지만 nDCG·Recall@10은 낮았다.
신뢰구간이 0을 포함하므로 Train100만으로 우열을 확정하지 않는다.

임베딩·채점·저장은 **8.9분**, GPU 최고 **57°C**였다.
80°C 기준을 유지했고 온도 초과로 인한 대기는 없었다.
100개 질의의 순위·nDCG 재계산과 1,199쌍의 독립 MaxSim 검증은 PASS했다.
기존 실험 산출물 286개는 변경하지 않았다.
Dev와 질의별 서빙 지연은 측정하지 않았으며, 위 실행 시간은 검색 지연 비교가 아니다.

## 검증과 한계

공식 프롬프트·정확성·저장 복원·지표 재계산 검증은 PASS했고 BGE 기존 결과도 재현했다.
Dev는 재사용한 50개 질의이며 독립 테스트 표본이 부족하다.
따라서 이 결과는 이번 프로젝트의 채택 판단이며 Gemma의 보편적 열세를 입증하지 않는다.
Gemma의 BF16·큰 batch 효과는 측정하지 않았다.

상세 근거: [최종 보고서](../artifacts/gemma-final-evaluation/report.ko.md),
[비교 설정](../artifacts/gemma-final-evaluation/protocol.json),
[Train 차원 실험](../artifacts/gemma-final-evaluation/dimension-train-sweep/report.ko.md),
[KURE-v2 Train100 결과](../artifacts/kure-v2-train100/report.ko.md),
[KURE-v2 비교 설정](../artifacts/kure-v2-train100/protocol.json).
`artifacts/`는 Git에서 제외되어 근거 링크는 로컬 산출물이 있는 환경에서 열 수 있다.

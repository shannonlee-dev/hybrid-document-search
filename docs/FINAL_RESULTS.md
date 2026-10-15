# 최종 검색 평가 결과

반가희 · 2026-10-16 (Asia/Seoul). 동일한 고정 입력으로 두 데이터셋을 각각 재실행했다.
아래 주 표는 직접 재현한 품질과 **캐시 벡터 기반 CPU 검색·결합 시간**이다.
query embedding·모델 초기화·HTTP를 포함한 시간으로 해석하지 않는다.

## 실행과 검증 범위

- 기준 main: 7f3333627fc2ba947f768ba5e63253ac1fa84837.
- 의존 구현: PR #18, 081c7abb031153da80db71c4362938be6e22a5f1.
- main에서 분기한 평가 코드와 #18을 임시 통합한 복사본에서 실행했다.
  의존 PR 코드를 본인 커밋으로 복사하거나 main에 병합하지 않았다.
- Python 3.12.15 / macOS 26.6.2 / arm64, CPU FAISS·BLAS 2 threads.
- BGE-M3 revision: 5617a9f61b028005a4858fdac845db406aefb181; FP32·L2 정규화 캐시.
- Top-10, 전체 dev 질의 warm-up 1 pass 후 측정 5 passes, 준비·검증·저장은 타이머 밖이다.
- 방법 순서는 TF-IDF → BM25 → Dense → RRF → score fusion으로 고정했다.
  단일 세션 순차 실행이며 순서·열 상태·시스템 부하에 따른 변동을 제거한 실험은 아니다.
- 지표는 relevance > 0의 binary gain과 질의별 동일 가중치 평균, P95는 선형 보간이다.
- 고정 cache 8개 입력 파일의 SHA-256을 확인했다. MIRACL의 corpus·Train100/Dev50
  질의·qrels가 기존 로컬 준비 데이터와 정확히 같음을 별도로 대조했다.
- 저장된 3,210개 방법별 질의 결과의 지표와 16,050개 latency 샘플의 mean/P95를
  표준 라이브러리로 독립 재계산하고 CSV와 대조했다. 모든 실행 소스 해시도 확인했다.
- Dense/RRF/score fusion의 품질 지표는 #18의 두 데이터셋 보존 결과와 허용오차 1e-12 이내다.
- 의존 구현을 포함한 전체 오프라인 테스트 769 passed; 기본 feature branch는
  678 passed / 4 skipped (#18이 필요한 실제 캐시·fusion fixture 테스트).
- 두 소스 구성의 Ruff lint·format을 확인했다. 모델 가중치를 다운로드하거나 GPU 추론을 재실행하지 않았다.

## Ko-MIRACL Dev50

문서 10,000개 · 질의 50개 · 방법당 250개 시간 샘플.

| 방식 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | CPU Mean ms | CPU P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.576667 | 0.722667 | 0.495238 | 0.525324 | 5.615872 | 6.510908 |
| BM25 | 0.587667 | 0.750667 | 0.597913 | 0.593881 | 0.949070 | 1.094817 |
| Dense (BGE-M3) | 0.813667 | 0.958333 | 0.861024 | 0.854857 | 0.545302 | 0.625671 |
| 기존 RRF | 0.765333 | 0.927667 | 0.790571 | 0.784433 | 1.473870 | 1.660873 |
| 최종 Score fusion | 0.815333 | 0.968333 | 0.881190 | 0.864903 | 1.041394 | 1.159020 |

원시 결과는 로컬 artifacts/final-evaluation/20261016/miracl10k/에 보존한다.
results.json SHA-256: 097f049dd7d2caa5a39335a3fa0340ed45e2642062826b8ac13b7ca0d2720e8c.

## Ko-StrategyQA dev

문서 9,251개 · 질의 592개 · 방법당 2,960개 시간 샘플.

| 방식 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | CPU Mean ms | CPU P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.605152 | 0.682461 | 0.575301 | 0.559794 | 7.379903 | 8.227563 |
| BM25 | 0.597832 | 0.666824 | 0.583144 | 0.562242 | 0.929614 | 1.029140 |
| Dense (BGE-M3) | 0.813461 | 0.852031 | 0.811159 | 0.794055 | 0.531440 | 0.693930 |
| 기존 RRF | 0.770781 | 0.840066 | 0.739329 | 0.733152 | 1.528280 | 1.735711 |
| 최종 Score fusion | 0.813320 | 0.858647 | 0.816558 | 0.800080 | 1.283726 | 1.428213 |

원시 결과는 로컬 artifacts/final-evaluation/20261016/strategyqa/에 보존한다.
results.json SHA-256: 887b39ae2b4a00ff55fe7ed6f9b2e7b6d15e92bf088b11e860921bced9abb2e6.

## 결과 해석

1. 최종 score fusion은 두 데이터셋에서 기존 RRF보다 Recall@10·MRR@10·nDCG@10이 높았다.
   MIRACL nDCG@10은 Dense 0.854857 → fusion 0.864903 (+0.010046),
   StrategyQA는 0.794055 → 0.800080 (+0.006025)이다.
2. StrategyQA Recall@5는 Dense 0.813461 → fusion 0.813320으로 약간 낮아졌다.
   모든 지표가 항상 개선됐다고 설명하지 않는다.
3. 현재 RRF는 양쪽 Top-10에 함께 나타난 문서를 우선하여 강한 Dense 결과를 밀어낼 수 있다.
   Score fusion은 Dense 점수 차이를 보존하고 BM25 보너스를 최대 0.05로 제한한다.
   다만 두 방식은 가중치와 후보 범위도 달라 이 결과를 점수 결합만의 효과로 단정하지 않는다.
   #18의 가중 RRF 대조에서도 Dense 중심 가중치만으로 기존 RRF 손실이 해소됐다.
4. 캐시를 사용하면 query embedding 비용이 제거되어 Dense 시간이 작아진다.
   위 CPU 수치를 기존 query-time GPU benchmark와 섞거나 서비스 응답시간으로 제시하지 않는다.
5. α=0.05는 기존 Train100 선정값을 고정했으며 새 파라미터 탐색은 하지 않았다.
   벡터 보정은 StrategyQA 성능 하락을 근거로 최종 설정에서 제외한 결정을 유지했다.

## 추론 포함 검색 시간: 기존 측정 기록

다음은 **이상헌이 PR #18에 보존한 기존 MIRACL 측정**이다. 이번 Mac 재현값이 아니다.
WSL2 / Intel i3-12100F / RTX 4060 / BGE-M3 FP32 / CPU 2 threads.
query embedding 시작부터 Top-10 반환까지, 방식별 warm-up 10회 후 Dev50 × 10회, 500표본.
모델·인덱스 초기화, HTTP·직렬화·동시 부하는 제외한다.

| 방식 | Query-time Mean ms | Query-time P95 ms |
|---|---:|---:|
| Dense | 22.004 | 31.190 |
| 기존 RRF | 24.117 | 35.033 |
| Score fusion | 21.191 | 30.747 |

출처: [#18의 결과 문서](https://github.com/shannonlee-dev/hybrid-document-search/blob/081c7abb031153da80db71c4362938be6e22a5f1/docs/SCORE_FUSION_RESULTS.md).
StrategyQA의 동일 GPU 추론 포함 시간은 이번 평가에서 새로 측정하지 않았다.
CPU 캐시 비교와 위 GPU 표는 서로 다른 질의 반복·warm-up 조건이며 합치지 않는다.

## 결과 요약

> 기존 Dataset/Sparse/Evaluation 코드를 재사용해 최종 Dense·RRF·score fusion을 공통 지표로
> 연결하고, MIRACL Dev50과 StrategyQA dev 592질의를 각각 재평가했습니다.
> 두 데이터셋 모두 score fusion의 Recall@10·MRR@10·nDCG@10이 Dense와 기존 RRF보다 높았지만,
> StrategyQA Recall@5는 Dense보다 약간 낮았습니다. CPU 캐시 검색 시간과 실제 query embedding을
> 포함한 기존 GPU 시간을 구분해 보고하고, 입력·실행 코드 해시와 원시 결과로 재현 근거를 남겼습니다.

MIRACL 결과는 관련 문서를 포함한 10k subset의 성능이며 전체 공식 MIRACL 성능이 아니다.
과거 실험에 노출된 dev 데이터의 확인 평가이므로 신규 독립 검증·일반화·통계적 우월성을 주장하지 않는다.
실제 모델을 통한 Dense/최종 Hybrid HTTP·UI E2E는 서비스 통합 담당자의 별도 검증이다.

재현 명령, 지표 규칙, 병합 순서와 로컬 산출물 형식은 [FINAL_EVALUATION.md](FINAL_EVALUATION.md)를 따른다.

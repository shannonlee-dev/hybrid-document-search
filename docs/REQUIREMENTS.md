# 요구사항

Sparse·Dense·Hybrid 검색을 같은 조건에서 비교하고, API와 UI로 검색 결과를 확인하는 NLP 프로젝트입니다.
현재 프로젝트의 범위와 확인 기준을 정리합니다.

## 기능

| 요구사항 | 확인 기준 |
| --- | --- |
| 문서 검색 | TF-IDF, BM25, Dense, Hybrid로 질의에 맞는 Top-K 문서 반환 |
| 결과 비교 | Streamlit에서 두 방식의 순위와 공통 문서를 나란히 확인 |
| 상태 확인 | API에서 검색 방식별 사용 가능 여부와 실패 이유 제공 |
| 품질 평가 | 같은 corpus·질의·qrels로 Recall@5/10, MRR@10, nDCG@10 비교 |
| 시간 측정 | 검색 시간의 평균·P95와 모델 추론 포함 여부 명시 |

## 요구사항–코드 매핑

| 요구사항 | 주요 구현 | 검증 테스트 |
| --- | --- | --- |
| 문서 검색 | [검색 서비스](../app/service.py), [검색기](../retrievers/), [결합](../fusion/) | [API 계약](../tests/test_search_api.py), [검색 통합](../tests/test_integration.py) |
| 결과 비교 | [UI](../frontend/app.py), [API 클라이언트](../frontend/client.py) | [UI 검색·두 방식 비교](../tests/test_frontend.py) |
| 상태 확인 | [API](../app/api.py), [검색기 초기화](../app/service.py) | [상태 응답](../tests/test_search_api.py), [초기화·실패 상태](../tests/test_service_build.py) |
| 품질 평가 | [지표](../evaluation/metrics.py), [최종 평가](../evaluation/final.py) | [지표 계산](../tests/test_metrics.py), [공통 평가 조건](../tests/test_final_evaluation.py) |
| 시간 측정 | [시간 측정](../evaluation/latency.py), [벤치마크](../evaluation/benchmark.py) | [시간 측정 규칙](../tests/test_latency.py), [벤치마크 집계](../tests/test_benchmark.py) |

## 제약과 범위

- 검색기는 공통 `search(query, top_k)` 계약과 문서 ID를 사용합니다.
- Dense는 저장된 인덱스를 재사용하고, 요청마다 질의만 임베딩합니다.
- 서비스의 Hybrid는 BM25 + Dense score fusion이며, RRF는 평가 baseline으로 유지합니다.
- 데이터·모델 revision과 평가 조건을 기록해 결과를 재현할 수 있게 합니다.
- 인증, 요청 제한, 캐시와 운영 환경 배포는 현재 범위에 포함하지 않습니다.

API 입력·오류 규칙은 [검색 서비스](SEARCH_SERVICE.md), 평가 조건과
[평가 완료·판단 기준](FINAL_EVALUATION.md#평가-완료-및-판단-기준)은 [최종 평가](FINAL_EVALUATION.md),
현재 측정 결과는 [벤치마크](BENCHMARK_RESULTS.md)를 참고하세요.

# 검증 기록 — 2026-10-20

대상: `eca59c63a30d8479da268a797cf599cd46824bf4`와 미커밋 문서 변경.
환경: WSL2 Linux, Python 3.12.3, 기존 Sparse·Dense 설치 환경.
로그 시각은 UTC이며 한국 시간은 +9시간입니다.

## 결과

| 항목 | 결과 | 증빙 |
| --- | --- | --- |
| 전체 테스트 | 769 passed, 43.35초 | [pytest 출력](pytest.txt) |
| 정적 검사 | Ruff lint·format, diff 공백 검사 통과 | [검사 로그](checks.txt) |
| 전체 평가 | 2개 데이터셋 × 5개 방식 완료 | [실행 로그](evaluation.txt) |
| 데모 | 실제 API 4개 방식·UI 검색·비교 통과 | [데모 로그](demo.txt) |
| 원격 CI | 기존 코드 커밋의 4개 작업 성공 확인 | [CI 조회 결과](ci.txt) |
| 지속 부하 | Hybrid 4클라이언트 1시간, 37,081건 실패 0; 자원 수집률 미달로 자동 판정 보류 | [결과·모니터링 로그](soak-20261020T222450+0900-e94f37/REPORT.md) |

지속 부하는 별도 후속 실행이며 대상 코드는 `2e252da1dd0d8b5d7a3bdc57096fd9bb0c2b9145`입니다.

## 전체 평가

MIRACL·StrategyQA의 다섯 검색 방식을 실행했고 종료 코드 0을 확인했습니다.
SHA-256 검증한 입력에 스레드 2개, warm-up 1회, 측정 5회를 적용했습니다.
실행 명령과 원본 출력은 [평가 로그](evaluation.txt)에 보관합니다.

지표·해석은 기존 [벤치마크](../../BENCHMARK_RESULTS.md), 절차는 [최종 평가](../../FINAL_EVALUATION.md)를 참고하세요.
이번 실행의 품질 수치는 기존 보고서와 소수점 6자리 기준으로 같습니다.
시간은 캐시 벡터 기반 CPU 검색·결합만 측정한 이번 환경의 값이며, 기존 보고서의 측정값을 대체하지 않습니다.
원시 결과는 `artifacts/final-evaluation/validation-2026-10-20/`에 보관합니다.

## 데모

MIRACL 1만 문서와 검증된 BGE-M3 문서 벡터로 FAISS 인덱스를 구성했습니다.
문서 임베딩은 재사용하고, 질의는 실제 BGE-M3 모델로 CPU에서 임베딩했습니다.

- API: 준비 상태 확인 후 네 방식 모두 Top-3 검색 성공
- Streamlit: 서버 health 200, 실제 API에 연결한 AppTest로 Hybrid 검색·두 방식 비교 성공
- 증빙: [데모 실행 로그](demo.txt)

브라우저 화면의 수동 검증·스크린샷은 포함하지 않습니다. 실행한 서버는 검증 후 종료했습니다.

## CI 범위

GitHub Actions의 `CI` 실행 `39402744638`과 `Final evaluation` 실행 `39402744657`에서
`checks`, `dense-tests`, `fusion-tests`, `final-evaluation-tests`가 모두 성공했습니다.
대상 커밋은 `9bfb4057a8503b6df9079fa78bbdbe15db650ae0`입니다.

현재 HEAD의 원격 실행은 없습니다. 성공 커밋 이후에는 문서만 바뀌었고
코드·테스트·CI 설정은 동일합니다(`ci.txt`의 커밋 비교 결과 참조).
이번 로컬 전체 테스트 결과와 원격 CI 기록을 구분해 보관합니다.
`gh run view --log`는 본문을 반환하지 않아 작업·단계별 상태 JSON을 증빙으로 남겼습니다.

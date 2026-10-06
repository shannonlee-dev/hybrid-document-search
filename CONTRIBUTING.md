# Contributing

## 작업 흐름

작업 브랜치 → 구현·검증 → PR → 리뷰 → `main`에 머지

- 큰 작업은 Issue로 문제와 구현 방향을 먼저 공유합니다.
- 개발 환경과 역할별 의존성은 [README](README.md#development-setup)를 참고합니다.

## Issue

- 기능 제안은 [Feature 템플릿](.github/ISSUE_TEMPLATE/feature.md)에 문제와 해결 방향을 정리합니다.
- 버그 보고는 [Bug 템플릿](.github/ISSUE_TEMPLATE/bug.md)에 실행 환경, 재현 절차, 기대 동작과 로그를 남깁니다.
- 제목에는 `[Feature]` 또는 `[Bug]` 접두사를 붙입니다.

## 브랜치와 커밋

작업 브랜치는 `main`에서 분기하고, 이름은 `<type>/<작업명>`을 사용합니다.
커밋과 PR 제목은 Conventional Commits 형식(`<type>: <변경 내용>`)을 따릅니다.

| Type | 용도 |
| --- | --- |
| `feat` | 기능 추가 |
| `fix` | 버그 수정 |
| `test` | 테스트 |
| `docs` | 문서 |
| `refactor` | 동작 변화 없는 구조 개선 |
| `perf` | 성능 개선 |
| `chore` | 설정, 의존성, CI |

예: `feat/dense-faiss` → `feat: add FAISS retriever`

## 협업 규칙

- 각 검색 구현은 `retrievers/base.py`의 `Retriever` 계약과 `SearchResult`를 따릅니다.
- 공통 인터페이스·스키마나 의존성 설정을 바꾸거나 다른 담당자의 작업 영역에 영향을 줄 때는
  변경 이유와 영향 범위를 해당 담당자와 먼저 공유합니다. 내부 구현은 각 담당자가 결정합니다.
- 의존성을 변경하면 `pyproject.toml`과 `uv.lock`을 함께 커밋합니다.

## 검증

기능 추가나 버그 수정 시 담당자가 변경한 동작을 검증하는 테스트도 작성합니다.
공통 계약을 바꾸면 사용하는 모듈에 미치는 영향도 확인합니다. PR을 올리기 전에 아래 CI 검증을 실행합니다.

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

검색 품질이나 모델·인덱스에 영향을 주는 변경은 필요한 benchmark나 smoke test도 실행합니다.
기본 CI에서는 모델·데이터·인덱스를 생성하지 않으므로, 관련 검증은 별도로 진행합니다.

## PR과 머지

[PR 템플릿](.github/pull_request_template.md)에 변경 내용과 검증 결과를 남깁니다.

- **Summary**: 변경 내용과 이유, 주요 설계 결정. 관련 Issue는 `Implements #번호`로 연결하고,
  해당 Issue의 작업을 완료하면 `Closes #번호`를 사용합니다.
- **Testing**: 실행한 검증과 결과. benchmark는 실행 조건을 함께 적고, 실행하지 못한 검증은 이유를 남깁니다.
- **AI assistance**: 사용한 모델과 사용 범위. 검토 확인 문구는 직접 검토하고 이해한 경우에만 작성합니다.
  AI를 사용하지 않았다면 생략합니다.
- **Checklist**: 테스트·lint·format 통과 여부, 공통 계약 준수 여부, 변경 내용 이해 여부.

# Hybrid Document Search

## Project Overview

Sparse, Dense, Hybrid Retrieval을 비교하고 FastAPI + Streamlit 서비스로 통합하는
3인 NLP 문서 검색 프로젝트입니다. Recall@K, MRR, nDCG로 검색 품질을 평가할 예정입니다.

## Planned Retrieval Methods

- TF-IDF
- BM25
- Sentence Transformer + FAISS
- Hybrid RRF

## Tech Stack

| 영역 | 기술 |
| --- | --- |
| Python / 환경 | Python 3.12, uv |
| Sparse | scikit-learn, bm25s |
| Dense | sentence-transformers, faiss-cpu |
| Hybrid / 평가 | 자체 RRF 구현 예정, ranx |
| Backend | FastAPI, Pydantic, Uvicorn |
| Frontend | Streamlit |
| 저장 | JSONL 또는 SQLite 검토 예정; 현재 스키마 없음 |
| 개발 | pytest, Ruff, GitHub Actions |

## Project Structure

| 경로 | 책임 |
| --- | --- |
| `app/` | health API 및 향후 검색 서비스 통합 |
| `retrievers/base.py` | 공통 `Retriever` 계약과 `SearchResult` |
| `retrievers/{tfidf,bm25,dense}.py` | 각 검색 방식의 독립 구현 영역 |
| `fusion/` | RRF와 Hybrid Retrieval |
| `evaluation/` | 평가 지표와 benchmark |
| `indexing/` | FAISS index 생성·검색·저장·로드 |
| `data/` | 데이터 로딩과 전처리 코드 |
| `frontend/` | Streamlit 진입점 |
| `scripts/` | 향후 index 생성 / 평가 실행 진입점 |
| `tests/` | 공통 계약과 health smoke test |
| `.github/` | CI, Issue 및 PR 템플릿 |

공통 계약은 `search(query: str, top_k: int) -> list[SearchResult]`입니다.
`Protocol`을 사용해 특정 부모 클래스 상속 없이 각 담당자가 같은 메서드 형식으로
독립 구현할 수 있게 했습니다. 인덱스 생성·모델 로딩 방식은 공통 계약에서 강제하지 않습니다.

`SearchResult`는 안정적인 문자열 `document_id`, 1부터 시작하는 `rank`, `score`,
선택적인 `title` / `snippet`을 담습니다. 호출자는 양수 `top_k`를 전달하며,
결과는 좋은 순서로 최대 `top_k`개를 반환합니다. 서로 다른 검색 방식의 원점수는
직접 비교하지 않습니다. 현재 검색 placeholder를 호출하면 `NotImplementedError`가 발생합니다.

## Development Setup

Python 3.12와 uv를 준비한 뒤 다음을 실행합니다.

```bash
git clone https://github.com/shannonlee-dev/hybrid-document-search.git
cd hybrid-document-search
uv sync
```

`uv sync`는 서비스와 개발 도구를 설치합니다. 검색 라이브러리는 무거운 의존성을
기본 환경과 CI에서 제외하기 위해 역할별 extras로 정의했습니다.

```bash
# Role 1: Sparse / Evaluation
uv sync --extra sparse --extra evaluation

# Role 2: Dense Retrieval
uv sync --extra dense

# Role 3: 전체 검색 모듈 통합 시
uv sync --all-extras
```

extras 설치 후에는 `uv run --extra dense ...`처럼 같은 extras를 지정하거나
`uv run --no-sync ...`로 설치된 환경을 유지하세요. 일반 `uv sync`는 선택하지 않은
extras를 제거합니다. Dense 의존성에는 PyTorch 등이 포함되어 설치 용량이 큽니다.
의존성 설치는 데이터셋이나 Sentence Transformer 모델을 다운로드하지 않습니다.

```bash
# 테스트
uv run pytest

# lint
uv run ruff check .

# API (GET /health → {"status": "ok"})
uv run uvicorn app.api:app --reload

# UI placeholder
uv run streamlit run frontend/app.py
```

의존성 변경 시 `uv add` / `uv add --dev` / `uv add --optional dense` 등을 사용하고
`pyproject.toml`과 `uv.lock`을 함께 커밋합니다. CI는 Python 3.12에서
`uv sync --locked`, Ruff, pytest만 수행하며 모델·데이터·인덱스를 생성하지 않습니다.

다운로드 데이터는 `data/raw/` 또는 `datasets/`, 전처리 결과는 `data/processed/`,
모델은 `models/`, 인덱스는 `indexes/`, 실험 산출물은 `artifacts/` 또는 `experiments/`에
보관하세요. 이 경로와 로컬 SQLite 파일은 Git에서 제외됩니다.

## Team Responsibilities

| 역할 | 작업 영역 |
| --- | --- |
| Role 1 — Sparse / Evaluation | TF-IDF, BM25, `data/`, `evaluation/`, `scripts/evaluate.py` |
| Role 2 — Dense Retrieval | 모델 비교/선정, embedding, `retrievers/dense.py`, `indexing/`, `scripts/build_index.py` |
| Role 3 — Integration / Service | 공통 계약, `fusion/`, `app/`, `frontend/`, 서비스 테스트 및 실행환경 |

Issue를 만들고 각자 브랜치에서 작업한 뒤 연결된 PR로 협업합니다.
공통 계약 변경은 팀원과 먼저 맞추고, 각 검색 구현은 `retrievers/base.py`의 계약을 따릅니다.

협업 규칙은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요.

## Status

현재는 **scaffold 단계**입니다. 공용 구조, 개발환경, health API, UI placeholder와
최소 smoke test만 준비되어 있습니다. 실제 검색·RRF·embedding·인덱싱·평가 알고리즘,
데이터셋과 모델 선정, 검색 UI 및 DB 스키마는 아직 구현하지 않았습니다.
`scripts/build_index.py`와 `scripts/evaluate.py`도 미구현 진입점입니다.

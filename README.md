# Hybrid Document Search

## Project Overview

Sparse, Dense, Hybrid Retrieval을 비교하고 FastAPI + Streamlit 서비스로 통합하는
3인 NLP 문서 검색 프로젝트입니다. Recall@5/10, MRR@10, nDCG@10과 query latency로 검색 방식을 비교합니다.

## Retrieval Methods

- TF-IDF: 구현 완료
- BM25: 검색기, CLI 및 공통 평가 파이프라인 구현 완료
- Sentence Transformer + FAISS: main에 병합 완료
- Hybrid RRF: main에 병합 완료

## Tech Stack

| 영역 | 기술 |
| --- | --- |
| Python / 환경 | Python 3.12, uv |
| Sparse | scikit-learn, bm25s |
| Dense | sentence-transformers, faiss-cpu |
| Hybrid / 평가 | 자체 RRF 및 표준 라이브러리 기반 지표 구현, ranx 평가 extra |
| Backend | FastAPI, Pydantic, Uvicorn |
| Frontend | Streamlit |
| 저장 | JSONL ([문서·질의·qrels 스키마](docs/SCHEMAS.md)); SQLite는 도입 검토 단계 |
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
| `scripts/` | 데이터 준비, Dense index 생성, TF-IDF/BM25/Dense 검색 및 공통 평가 실행 진입점 |
| `tests/` | 공통 계약, 데이터 준비, 검색·인덱스·CLI fixture 테스트와 health smoke test |
| `.github/` | CI, Issue 및 PR 템플릿 |

공통 계약은 `search(query: str, top_k: int) -> list[SearchResult]`입니다.
`Protocol`을 사용해 특정 부모 클래스 상속 없이 각 담당자가 같은 메서드 형식으로
독립 구현할 수 있게 했습니다. 인덱스 생성·모델 로딩 방식은 공통 계약에서 강제하지 않습니다.

`SearchResult`는 안정적인 문자열 `document_id`, 1부터 시작하는 `rank`, `score`,
선택적인 `title` / `snippet`을 담습니다. 호출자는 양수 `top_k`를 전달하며,
결과는 좋은 순서로 최대 `top_k`개를 반환합니다. 서로 다른 검색 방식의 원점수는
직접 비교하지 않습니다. TF-IDF와 Dense는 구현되었으며, main에 병합된 Hybrid는
두 검색기를 주입받아 RRF로 결과를 결합합니다. BM25는 공통 corpus를 TF-IDF와 동일한
문자 n-gram 전처리로 인덱싱하며 원점수 순으로 Top-K 결과를 반환합니다.

## Development Setup

Python 3.12와 uv를 준비한 뒤 다음을 실행합니다.

```bash
git clone https://github.com/shannonlee-dev/hybrid-document-search.git
cd hybrid-document-search
uv sync
```

`uv sync`는 서비스와 개발 도구를 설치합니다. 검색 라이브러리는 무거운 의존성을
기본 환경과 기본 CI job에서 제외하기 위해 역할별 extras로 정의했습니다.
별도 `dense-tests` CI job은 Dense 의존성을 설치합니다.

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
기본 lint·format·pytest 검사와 별도 `dense-tests` job을 실행합니다.
Dense job은 `uv sync --locked --extra dense` 후 오프라인 테스트를 실행하며,
외부 데이터셋이나 사전 학습 모델을 다운로드하지 않습니다.
테스트에서는 fixture와 임시 로컬 BoW Sentence Transformer·FAISS 인덱스를 사용합니다.
검색 기능을 로컬에서 검증하려면 해당 extras를 포함해 실행합니다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked --extra sparse --extra dense pytest
```

다운로드 데이터는 `data/raw/` 또는 `datasets/`, 전처리 결과는 `data/processed/`,
모델은 `models/`, 인덱스는 `indexes/`, 실험 산출물은 `artifacts/` 또는 `experiments/`에
보관하세요. 이 경로와 로컬 SQLite 파일은 Git에서 제외됩니다.

## TF-IDF MVP 실행

Ko-miracl의 공통 subset을 준비한 뒤 TF-IDF Top-K 검색을 실행합니다.
첫 준비 명령은 전체 원본 corpus를 다운로드하고, 판정 문서를 포함한 10,000개 passage를 선택합니다.
기존 준비 결과는 덮어쓰지 않으므로 이미 데이터가 있다면 검색 명령부터 실행합니다.

```bash
uv sync --extra sparse
uv run --extra sparse python -m scripts.prepare_dataset --download
uv run --extra sparse python -m scripts.search tfidf --query "제주" --top-k 3
```

검색 결과는 공통 `SearchResult` 필드의 JSON 배열로 출력합니다.
준비 결과는 `data/processed/`에 저장하고 Git에 포함하지 않습니다.
데이터 준비는 [DATA_PREPARATION.md](docs/DATA_PREPARATION.md), 스키마는 [SCHEMAS.md](docs/SCHEMAS.md),
TF-IDF 알고리즘과 fixture 검색 예시는 [SPARSE_MVP.md](docs/SPARSE_MVP.md)를 참조하세요.

## BM25 실행

TF-IDF와 동일하게 준비된 corpus를 사용합니다.

```bash
uv run --extra sparse python -m scripts.search bm25 --query "제주" --top-k 3
```

설정, fixture 실행 및 검증 결과는 [BM25_RETRIEVAL.md](docs/BM25_RETRIEVAL.md)를 참조하세요.
공통 평가는 Recall@5 / Recall@10 / MRR@10 / nDCG@10과 warm-up 이후 latency 평균·P95를
사용합니다. 준비 데이터 검증, 검색기 재사용, 지표·시간 측정과 JSON/CSV 저장을 구현했습니다.
계산 규칙과 실행 조건은 [EVALUATION.md](docs/EVALUATION.md)를 참조하세요. Recall@100은 별도 참고용 지표입니다.

```bash
uv run --locked --extra sparse python -m scripts.evaluate \
  --split dev --methods tfidf bm25 --warmup 1 --repeats 5 \
  --output-dir artifacts/benchmarks/sparse-dev
```

`results.json`, `summary.csv`, `queries.csv`, `latencies.csv`, `comparison.md`를 생성합니다.
로컬 Sparse 결과와 아직 측정하지 않은 항목은 [BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md)에 정리합니다.
동일 corpus의 저장된 Dense 인덱스가 있으면 `--methods tfidf bm25 dense hybrid --index <경로>`로
공통 네 방식 비교를 실행할 수 있습니다.

## Dense MVP 실행

기본 모델은 `intfloat/multilingual-e5-base`입니다. 모델 크기와 인덱스 메모리를
고려해 초기 baseline으로 선정했으며, BGE-M3·KURE-v1과의 정식 품질 비교는 후속 평가에서 진행합니다.
작은 fixture로 인덱스를 생성·저장한 뒤, 새 프로세스에서 검색할 수 있습니다.
모델이 로컬 캐시에 없으면 첫 빌드에서 다운로드합니다.

```bash
uv sync --locked --extra dense
export HF_HOME="$PWD/models/huggingface"
uv run --locked --extra dense python -m scripts.build_index \
  --corpus tests/dense/fixtures/dense_corpus.jsonl \
  --index indexes/dense-sample --device cpu
uv run --locked --extra dense python -m scripts.search dense \
  --index indexes/dense-sample --device cpu \
  --query "고양이는 어떤 소리로 우나요?" --top-k 3
```

실제 subset을 사용하려면 빌드 명령의 `--corpus`를 `data/processed/corpus.jsonl`로 지정합니다.
검색은 저장된 문서 벡터와 문서 매핑을 복원하고 질의만 임베딩합니다.
Sparse와 Dense는 공통 `Document`와 `build_index_text()`를 사용하며,
제목이 있으면 `title + "\n" + text`, 없으면 `text`만 인덱싱합니다.
모델 후보, 설정, 저장 형식과 검증 범위는 [DENSE_RETRIEVAL.md](docs/DENSE_RETRIEVAL.md)를 참조하세요.

Dense 담당 테스트와 전용 fixture는 `tests/dense/`에 있습니다. 해당 영역만 검증하려면 다음을 실행합니다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked --extra dense pytest tests/dense
```

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

**Dataset + Sparse MVP**와 **Dense Retrieval MVP**는 main에 병합되었으며, 현재 브랜치에 BM25와 공통 평가 pipeline을 구현했습니다.
Ko-miracl 공통 subset 준비, 문서·query·qrels 스키마, 원본·준비 corpus loader,
TF-IDF / BM25 Top-K 검색, 문서·질의 embedding, FAISS index build/search/save/load와
fixture 기반 테스트를 제공합니다.
`scripts/prepare_dataset.py`, `scripts/build_index.py`, `scripts/search.py`, `scripts/evaluate.py`는 구현된 CLI입니다.

**Hybrid / Integration MVP**의 RRF, 검색기 주입 방식의 `HybridRetriever`,
synthetic unit test, FastAPI request/response schema는 main에 병합되었습니다.
현재 브랜치는 세 MVP가 병합된 main에서 시작합니다.

실제 Sparse/Dense를 서비스에서 초기화·연결하는 작업, `/search` API와 UI 검색 연동,
실제 Dense/Hybrid를 포함한 네 방식 최종 benchmark 및 Dense 모델별 성능 비교는 후속 작업입니다.
API는 현재 `/health`만 제공하며 UI는 placeholder입니다.
`evaluation/metrics.py`는 Recall@K, RR/MRR@10, 이진 nDCG@K 및 질의별 평균 집계를 제공합니다.
`evaluation/data.py`, `evaluation/benchmark.py`와 `scripts/evaluate.py`는 고정 준비 데이터의 검증,
Top-10 품질·latency 평가 및 JSON/CSV/Markdown 비교표 저장을 제공합니다.

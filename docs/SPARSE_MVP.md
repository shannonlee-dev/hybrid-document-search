# TF-IDF Sparse MVP

관련 Issue: #4

담당: @bangahee

## 실행 흐름

준비된 corpus JSONL → `load_prepared_documents` → `build_index_text` →
`TfidfRetriever` 인덱싱 → `search(query, top_k)` → `SearchResult` 목록.

원본 JSONL은 `load_documents`로 읽고, `_id`를 원본 passage ID 그대로 `document_id`에 매핑합니다.
준비된 JSONL은 `load_prepared_documents`로 읽고, 저장된 `document_id`를 그대로 사용합니다.
제목이 있으면 `title + "\n" + text`, 없으면 `text`만 인덱싱합니다.

## 알고리즘 선정 이유

scikit-learn의 `TfidfVectorizer`를 사용합니다.
기본 설정은 `analyzer="char_wb"`, `ngram_range=(2, 4)`, `norm="l2"`입니다.

한국어의 조사 및 어미 때문에 단어가 정확히 일치하지 않는 경우를 고려하여,
추가 형태소 분석기 없이 부분 문자열을 비교할 수 있는 문자 n-gram을 baseline으로 선택했습니다.
예를 들어 fixture의 `제주도는`에 대해 `제주`로 검색할 수 있습니다.
10k subset의 검색 품질과 latency는 [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)에 있습니다.

문서에서 TF-IDF vocabulary와 IDF를 학습하고, 질의에는 동일한 vectorizer의
`transform`만 적용합니다. L2 정규화된 벡터의 내적으로 코사인 유사도를 계산합니다.

문서 행렬과 질의 벡터는 희소 형식을 유지하고, 검색 시 점수 행렬을 밀집 배열로 변환하지 않습니다.
양의 점수를 가진 후보 중 최대 `top_k`개를 heap으로 선택합니다.
원본 전체 corpus의 메모리 및 지연 시간은 측정하지 않았습니다.

## 결과와 예외 처리

- 결과는 점수가 높은 순서로 반환하며, `rank`는 1부터 연속해서 부여합니다.
- 동점은 입력 corpus 순서로 처리합니다.
- 원본 `document_id`와 `title`을 유지합니다.
- `snippet`은 원본 본문의 앞 200자를 사용합니다.
- 0점 문서는 반환하지 않습니다. 따라서 결과 수가 `top_k`보다 작을 수 있습니다.
- 빈 질의와 vocabulary에 없는 질의는 빈 목록을 반환합니다.
- `top_k`는 양의 정수여야 하며, bool은 허용하지 않습니다.
- 비문자열 질의는 `TypeError`, 잘못된 `top_k`는 `ValueError`를 발생시킵니다.
- 빈 corpus, 중복 문서 ID, 비어 있는 인덱싱 텍스트는 허용하지 않습니다.
- 기존 `Retriever`와 `SearchResult` 계약을 따릅니다.

## 설치와 검증

```bash
uv sync --extra sparse
uv run --extra sparse pytest
uv run --extra sparse ruff check .
uv run --extra sparse ruff format --check .
```

기본 CI에는 Sparse 의존성이 없으므로 scikit-learn이 없으면 TF-IDF 테스트를 건너뜁니다.
검색 기능 검증에는 반드시 `--extra sparse`를 포함해 테스트를 실행합니다.
fixture 테스트는 실제 Ko-miracl 전체 데이터의 성능 평가를 대신하지 않습니다.

## Python 사용 예시

```python
from data.loader import load_prepared_documents
from retrievers.tfidf import TfidfRetriever

documents = load_prepared_documents("data/processed/corpus.jsonl")
retriever = TfidfRetriever(documents)

for result in retriever.search("제주", top_k=3):
    print(result.document_id, result.rank, result.score)
```

프로젝트 루트에서 `uv run --extra sparse python`으로 실행한 Python에 위 예시를 입력할 수 있습니다.

## CLI 사용 예시

데이터가 아직 없다면 [DATA_PREPARATION.md](DATA_PREPARATION.md)를 따라 먼저 준비합니다.
프로젝트 루트에서 실행합니다.

```bash
uv run --extra sparse python -m scripts.search tfidf --query "제주" --top-k 3
```

기본 corpus 경로는 `data/processed/corpus.jsonl`이며, `--corpus`로 다른 준비된 파일을 지정할 수 있습니다.
데이터 다운로드 없이 fixture를 검색하려면 다음을 실행합니다.

```bash
uv run --extra sparse python -m scripts.search tfidf \
  --corpus tests/fixtures/ko_miracl_prepared_corpus.jsonl \
  --query "제주" --top-k 1
```

출력은 `SearchResult` 필드를 담은 JSON 배열입니다.
`document_id`, `rank`, `score`, `title`, `snippet`을 포함하며, 검색 결과가 없으면 `[]`를 출력합니다.
`--query`는 필수이며, `--top-k`는 기본 5입니다.
잘못된 Top-K, 없는 파일, 유효하지 않은 corpus는 오류 메시지와 종료 코드 2로 보고합니다.
실행마다 corpus를 로드하고 메모리에 TF-IDF 인덱스를 새로 만듭니다. 인덱스 파일 저장·로드는 구현하지 않았습니다.

### 실데이터 smoke test

2026-10-06 준비된 10,000개 passage와 train 질의 `1013`으로 CLI를 실행했습니다.
Top-3 결과를 JSON으로 읽을 수 있었으며, 반환 ID가 corpus에 존재하고 순위가 1부터 연속이며 점수가 양수·내림차순임을 확인했습니다.
이는 실행 흐름 검증이며 검색 품질 metric 또는 지연 시간 benchmark가 아닙니다.

## 참고 자료

- [TfidfVectorizer 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [코사인 유사도 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)

데이터 구성은 [DATASET.md](DATASET.md), BM25는 [BM25_RETRIEVAL.md](BM25_RETRIEVAL.md),
평가 방법은 [EVALUATION.md](EVALUATION.md)를 참조하세요.

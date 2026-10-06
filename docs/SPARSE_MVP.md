# TF-IDF Sparse MVP

관련 Issue: #4  
담당: @bangahee  
브랜치: `feat/dataset-tfidf`

## 실행 흐름

원본 corpus JSONL → `load_documents` → `build_index_text` →
`TfidfRetriever` 인덱싱 → `search(query, top_k)` → `SearchResult` 목록.

원본 JSONL의 `_id`는 원본 passage ID 그대로 `document_id`에 매핑합니다.
제목이 있으면 `title + "\n" + text`, 없으면 `text`만 인덱싱합니다.

## 알고리즘 선정 이유

scikit-learn의 `TfidfVectorizer`를 사용합니다.
초기 설정은 `analyzer="char_wb"`, `ngram_range=(2, 4)`, `norm="l2"`입니다.

한국어의 조사 및 어미 때문에 단어가 정확히 일치하지 않는 경우를 고려하여,
추가 형태소 분석기 없이 부분 문자열을 비교할 수 있는 문자 n-gram을 baseline으로 선택했습니다.
예를 들어 fixture의 `제주도는`에 대해 `제주`로 검색할 수 있습니다.
이 선택의 실제 검색 품질은 평가 데이터셋을 준비한 후 별도로 검증해야 합니다.

문서에서 TF-IDF vocabulary와 IDF를 학습하고, 질의에는 동일한 vectorizer의
`transform`만 적용합니다. L2 정규화된 벡터의 내적으로 코사인 유사도를 계산합니다.

문서 행렬과 질의 벡터는 희소 형식을 유지하고, 검색 시 점수 행렬을 밀집 배열로 변환하지 않습니다.
양의 점수를 가진 후보 중 최대 `top_k`개를 heap으로 선택합니다.
전체 corpus의 메모리 및 지연 시간 측정은 아직 수행하지 않았습니다.

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
from data.loader import load_documents
from retrievers.tfidf import TfidfRetriever

documents = load_documents("tests/fixtures/ko_miracl_corpus.jsonl")
retriever = TfidfRetriever(documents)

for result in retriever.search("제주", top_k=3):
    print(result.document_id, result.rank, result.score)
```

프로젝트 루트에서 `uv run --extra sparse python`으로 실행한 Python에 위 예시를 입력할 수 있습니다.

## 참고 자료

- [TfidfVectorizer 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [코사인 유사도 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)

## 이후 작업

- 실제 Ko-miracl subset 준비 및 재현 가능한 데이터 준비 과정 문서화.
- 평가용 query/qrels 형식 확정.
- BM25 및 평가 metric 구현은 별도 PR에서 진행합니다.

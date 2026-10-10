# BM25 Sparse Retrieval

관련 Issue: #12 · 담당: @bangahee

## 구현과 공통 계약

`BM25Retriever(documents, *, k1=1.5, b=0.75)`는 생성 시 메모리 인덱스를 한 번 만들고,
`search(query, top_k)`로 공통 `SearchResult` 목록을 반환합니다.
기존 Sparse extra의 `bm25s`와 scikit-learn을 사용하며 추가 의존성은 없습니다.

- `build_index_text`의 `title + "\n" + text` 규칙을 따릅니다. 제목이 없으면 본문만 사용합니다.
- TF-IDF와 동일하게 scikit-learn의 `char_wb`, `ngram_range=(2, 4)` 분석기를 사용합니다.
  소문자화와 공백 처리를 문서 및 query에 동일하게 적용하며, 형태소 분석·불용어 제거·stemming은 적용하지 않습니다.
- 분석기는 토큰만 생성합니다. TF-IDF 가중치와 L2 정규화는 BM25에 사용하지 않습니다.
- 문서 ID, 제목과 본문 원본을 보존합니다. `snippet`은 본문 앞 200자입니다.
- 양의 점수만 내림차순으로 반환하고, 동점은 입력 corpus 순서로 처리합니다. 순위는 1부터 시작합니다.
- 빈 query 또는 vocabulary에 없는 query는 빈 목록을 반환합니다. 결과 수는 `top_k`보다 작을 수 있습니다.
- 빈 corpus, 중복 ID, 빈 인덱싱 텍스트, 비문자열 query 및 잘못된 Top-K는 거부합니다.
  Top-K는 bool을 제외한 양의 정수입니다.

## BM25 설정과 점수

`bm25s.BM25(method="lucene", idf_method="lucene", k1=1.5, b=0.75)`를 사용합니다.
`k1`은 단어 빈도 포화, `b`는 문서 길이 정규화 강도를 제어합니다.
`k1`은 유한한 양수, `b`는 0부터 1 사이의 유한한 수여야 합니다.
문서 길이와 빈도는 문자 n-gram 토큰 기준입니다.

현재 bm25s Lucene 방식의 토큰별 점수는 다음과 같습니다.

```text
idf = ln(1 + (N - df + 0.5) / (df + 0.5))
score(token, document) = idf * tf / (tf + k1 * (1 - b + b * length / avg_length))
```

질의에 등장하는 토큰의 점수를 합산합니다. 같은 질의 토큰이 반복되면 반복 횟수만큼 합산됩니다.
라이브러리 원점수를 그대로 반환하므로 `(k1 + 1)` 상수를 곱하는 다른 BM25 구현과
점수 크기가 다를 수 있습니다. TF-IDF 또는 Dense 점수와 직접 비교하거나 더하지 않습니다.
Hybrid는 기존 RRF가 순위를 결합하도록 합니다.

기본 설정을 baseline으로 평가했으며, 파라미터를 추가 튜닝하지 않았습니다.
설정 변경은 train 데이터에서 검토하고 dev 평가 전에 고정합니다.

## Python 실행

프로젝트 루트에서 `uv sync --extra sparse` 후
`uv run --extra sparse python`을 실행하여 아래 예시를 사용할 수 있습니다.

```python
from data.loader import load_prepared_documents
from retrievers.bm25 import BM25Retriever

documents = load_prepared_documents("data/processed/corpus.jsonl")
retriever = BM25Retriever(documents)
for result in retriever.search("제주", top_k=3):
    print(result.document_id, result.rank, result.score)
```

## CLI 실행

프로젝트 루트에서 준비된 corpus를 검색합니다.

```bash
uv run --extra sparse python -m scripts.search bm25 --query "제주" --top-k 3
```

기본 corpus는 `data/processed/corpus.jsonl`입니다. 다운로드 없이 fixture를 사용하려면:

```bash
uv run --extra sparse python -m scripts.search bm25 \
  --corpus tests/fixtures/ko_miracl_prepared_corpus.jsonl \
  --query "제주" --top-k 1
```

`--query`는 필수이며 `--top-k` 기본값은 5입니다. 결과는 다른 방식과 동일한
`document_id`, `rank`, `score`, `title`, `snippet` 필드의 JSON 배열입니다.
빈 질의와 검색 결과가 없는 질의는 `[]`를 출력합니다. 잘못된 입력, corpus 및
접근 권한 오류는 한국어 안내와 종료 코드 2로 보고합니다.
BM25 검색기는 `bm25` subcommand가 선택된 경우에만 불러옵니다. 도움말에는 검색 의존성이 필요하지 않습니다.

CLI를 실행할 때마다 corpus를 읽고 메모리 인덱스를 생성합니다.
인덱스 파일 저장·로드는 제공하지 않습니다. 공통 benchmark에서는 검색기를 한 번 준비해 재사용합니다.
평가 지표, 실제 데이터 benchmark 실행과 JSON/CSV 저장 방법은 [EVALUATION.md](EVALUATION.md)에 정리했습니다.
네 방식의 Dev 비교 결과는 [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)를 참조하세요.

## 공통 평가 기준

- 공통 지표: Recall@5 / Recall@10 / MRR@10 / nDCG@10. 공통 검색 깊이는 Top-10입니다.
- Query-time latency: warm-up 이후 반복 측정한 평균과 P95. 모델·인덱스 최초 준비 시간은 별도 기록.
- Dense 모델 선정은 Train nDCG@10을 우선하고, MRR@10·Recall@10과 latency·구축 비용을 함께 검토합니다.
- Recall@100은 이번 공통 비교에 포함하지 않습니다.
- 기존 평가 계획대로 모델·파라미터 선택은 train에서 진행하고, 고정된 dev query/qrels로 최종 성능을 비교합니다.
- Metric cutoff와 Hybrid의 검색 후보 수는 별도 설정입니다. 후보 수와 측정 조건을 실행 결과에 기록합니다.

## 검증

```bash
uv run --extra sparse pytest tests/test_bm25.py tests/test_search_cli.py
uv run --extra sparse ruff check .
uv run --extra sparse ruff format --check .
```

fixture로 한국어 부분 검색, 순위와 동점, 원본 metadata, 입력 검증, 양의 점수 필터를 검증합니다.
별도 합성 문서로 수동 계산한 BM25 점수와 문서 길이 정규화 동작을 검증합니다.
Sparse 의존성이 없는 기본 CI에서는 BM25 검색 테스트를 건너뛰므로 Sparse extra를 포함해 별도로 실행해야 합니다.
fixture 및 smoke test는 실제 검색 품질 benchmark를 대신하지 않습니다.

## 초기 실데이터 검증

2026-10-07, Python 3.12.15 / bm25s 0.3.12에서 10,000개 준비 corpus의 `제주` Top-3 검색과
반복 검색 일치를 확인했습니다. 반환 ID는 `1987050#0`, `736025#1`, `1664892#1`이며,
Python과 CLI 결과의 공통 ID·metadata·순위·점수를 대조했습니다.
검색 품질과 latency는 [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)에 있습니다.

## 참고 자료

- [bm25s 공식 저장소와 사용 예시](https://github.com/xhluca/bm25s)
- [bm25s 점수 계산 구현](https://github.com/xhluca/bm25s/blob/main/bm25s/scoring.py)
- [scikit-learn TfidfVectorizer 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

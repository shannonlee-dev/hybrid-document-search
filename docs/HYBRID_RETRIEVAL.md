# Hybrid Retrieval (RRF)

이 문서는 `fusion/rrf.py`와 `fusion/hybrid.py`의 Hybrid Retrieval 기반을 설명합니다.
BM25 + BGE-M3를 연결한 10k corpus의 Dev 평가를 완료했습니다.
이번 subset에서는 Dense 단독이 Hybrid보다 우수했으며, RRF 기본값은 튜닝하지 않았습니다.
측정값과 조건은 [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)를 참조하세요.

## 왜 RRF인가

Sparse(TF-IDF, BM25)와 Dense(cosine/IP)의 원점수는 서로 범위와 분포가 다릅니다.
예를 들어 BM25는 임의 양수 값이고, cosine은 -1~1 사이입니다. 두 값을 그대로 더하거나
평균하면 한쪽 검색기가 결과를 지배하게 되고, 점수 정규화와 보정 정책이 추가로 필요합니다.

RRF는 원점수를 쓰지 않고 **각 검색기 안에서의 순위만** 사용합니다. 1차 baseline으로
해석이 단순하고, 정규화 실험 없이도 두 검색기를 같은 기준에서 비교할 수 있습니다.

## 공식

문서 `d`의 RRF 점수는 `d`가 등장한 모든 결과 목록에서 기여도를 더해 구합니다.

```
rrf_score(d) = sum over lists i containing d of  1 / (rank_constant + rank_i(d))
```

- `rank_i(d)`: 목록 `i`에서 `d`의 1부터 시작하는 순위
- `rank_constant`: 기본값 **60**. 상위 순위와 하위 순위 사이 차이를 완만하게 만듭니다.
  이름은 결과 개수인 `top_k`와 구분하기 위해 `rank_constant`로 둡니다.
- 목록은 두 개 이상도 받을 수 있습니다. 빈 목록은 유효합니다.

## 작은 예시

- Sparse: A (rank 1), B (rank 2)
- Dense: B (rank 1), C (rank 2)
- `rank_constant = 60`

| 문서 | 기여 | RRF 점수 |
| --- | --- | --- |
| B | Sparse rank 2 + Dense rank 1 = 1/62 + 1/61 | ≈ 0.0325 |
| A | Sparse rank 1 = 1/61 | ≈ 0.0164 |
| C | Dense rank 2 = 1/62 | ≈ 0.0161 |

B는 두 목록에 모두 있으므로 기여도가 합쳐져 1위가 됩니다. 결과에는 B가 한 번만 나오고,
순위는 1, 2, 3으로 다시 매겨집니다.

## 입력 검증

| 대상 | 규칙 |
| --- | --- |
| `top_k` | 양의 정수. `bool`은 거부 |
| `rank_constant` | 0 이상의 정수. `bool`은 거부 |
| 각 결과의 `rank` | 양의 정수. `bool`은 거부 |
| 한 목록 안의 `document_id` | 중복이면 `ValueError`. 중복을 여러 번 세지 않음 |

잘못된 목록은 조용히 보정하지 않고 거부합니다. 빈 목록은 정상 입력이며, 모든 목록이
비어 있으면 빈 결과를 반환합니다.

## 점수와 순위의 의미

- **순위 입력**: 원점수(`SearchResult.score`)는 읽지 않습니다. 원점수를 바꿔도 순위가 같으면
  융합 결과(순서, 점수, metadata)는 동일합니다.
- **출력 `score`**: RRF 점수입니다. BM25 점수, TF-IDF 코사인 유사도, Dense 내적 점수가
  아니며 확률도 아닙니다. 값의 크기는 `rank_constant`와 목록 개수에 따라 달라지므로, 다른
  쿼리나 설정의 점수와 직접 비교하지 않습니다.
- **출력 `rank`**: 융합 후 1부터 연속된 순위입니다.

## 중복 문서 처리

같은 `document_id`가 여러 목록에 나오면 최종 결과에는 한 번만 포함하고, 각 목록의 RRF
기여도를 합산합니다.

## Metadata 선택 규칙

`title`과 `snippet`은 해당 문서의 **가장 좋은 순위(best rank)를 가진 한 번의 등장**에서만
가져옵니다.

- best rank가 같으면 앞선 목록의 값을 씁니다.
- 제목이나 본문을 여러 등장에서 이어 붙이거나 섞지 않습니다.
- 어느 등장에도 없는 값을 만들지 않습니다. 선택한 등장의 값이 `None`이면 결과도 `None`입니다.

## 결정적 정렬 규칙

정렬 키는 아래 순서입니다.

1. RRF 점수 내림차순
2. best rank 오름차순
3. `document_id` 오름차순

점수 합산에는 `math.fsum`을 사용해, 같은 기여도 집합이면 목록 순서와 무관하게 같은 값이
나오도록 합니다. 따라서 dict나 set의 순회 순서에 의존하지 않습니다.

## HybridRetriever 설계

`HybridRetriever(sparse_retriever, dense_retriever, rank_constant=60)`는 두 검색기를
생성자로 주입받습니다. 타입은 `retrievers.base.Retriever` Protocol만 사용하므로,
TF-IDF, BM25, Dense, FAISS 구현 모듈을 import하지 않습니다.

`search(query, top_k)`는 다음 순서로 동작합니다.

1. `top_k`를 먼저 검증합니다. 잘못된 값이면 두 검색기를 호출하지 않습니다.
2. 두 검색기에 **같은 query와 같은 `top_k`**를 전달합니다. 후보 수를 늘리는 보정은 하지
   않습니다.
3. 두 결과 목록을 `reciprocal_rank_fusion`에 넘깁니다.
4. 융합된 `SearchResult` 목록을 반환합니다.

## 오프라인 테스트

테스트는 `tests/test_rrf.py`, `tests/test_hybrid.py`, `tests/test_api_schemas.py`에 있습니다.
Ko-MIRACL 데이터, Hugging Face, Sentence Transformers, FAISS, scikit-learn, 모델 가중치, 네트워크
연결이 필요 없습니다. 가짜 `Retriever`(호출을 기록하는 객체)와 합성 `SearchResult`만 사용합니다.

```bash
uv run --locked pytest tests/test_rrf.py tests/test_hybrid.py tests/test_api_schemas.py
```

이 테스트는 RRF 공식, 중복 처리, 정렬, 검증, Hybrid 호출 흐름, schema 계약을 확인합니다.
검색 품질은 측정하지 않습니다.

## API schema와 서비스

`app/schemas.py`의 요청/응답 모델을 FastAPI `POST /search`에서 사용합니다.

- `RetrievalMethod`: `tfidf`, `bm25`, `dense`, `hybrid`
- `SearchRequest`: `query`(공백만 있으면 거부), `top_k`(양의 정수, 기본 10), `method`(기본 `hybrid`)
- `SearchHit`: `document_id`, `rank`, `score`, 선택 `title`/`snippet`. `SearchResult`와 같은 필드입니다.
- `SearchResponse`: `query`, `method`, `results`

`app/service.py`는 검색 방식을 선택하며, Hybrid는 BM25와 Dense가 모두 준비된 경우에만 사용합니다.
설정과 엔드포인트는 [SEARCH_SERVICE.md](SEARCH_SERVICE.md)를 참조하세요.

## 후속 작업

- 실제 10k corpus와 BGE-M3 인덱스를 연결한 Dense/Hybrid API smoke test
- Streamlit UI 검색 연동
- 필요하면 가중치, 후보 수 보정(oversampling), 점수 정규화를 별도 실험으로 검토

### 공통 인덱싱 텍스트

Sparse와 Dense는 `data/preprocess.py`의 `build_index_text`와 공통 `Document`를 사용합니다.
제목이 있으면 `title + "\n" + text`, 없으면 `text`를 인덱싱합니다.

# Dense + BM25 score fusion

`ScoreFusionRetriever`의 사용·재현 안내입니다. `fusion/score.py`는 전체 corpus의
Dense·BM25 점수를 결합하며, 공통 `Retriever.search(query, top_k)`와 `SearchResult`
계약을 유지합니다. 기본값은 **α=0.05, 벡터 보정 없음**입니다.
현재 API/UI의 Hybrid 검색은 RRF를 사용합니다. Score fusion의 실험 결과와 채택 근거는
[SCORE_FUSION_RESULTS.md](SCORE_FUSION_RESULTS.md)에 정리했습니다.

## 사용

```python
from fusion.score import ScoreFusionRetriever
from retrievers.bm25 import BM25Retriever
from retrievers.dense import DenseRetriever

dense = DenseRetriever.load("indexes/bge-m3", device="cpu")
bm25 = BM25Retriever(dense.documents)
hybrid = ScoreFusionRetriever(
    bm25,
    dense,
    bm25_weight=0.05,
)
results = hybrid.search("발해는 언제 건국되었나요?", top_k=10)
```

지원 대상은 `BM25Retriever`와 정규화된 FAISS `IndexFlatIP` 기반 `DenseRetriever`입니다.
BGE-M3 원본 벡터와 `title + "\n" + text` 인덱싱을 사용합니다.
다른 모델·비정규화 임베딩에 같은 가중치를 적용할 근거는 없습니다.
문서 ID·본문·제목·순서가 같아야 하므로 Dense의 문서 매핑으로 BM25를 생성합니다.
문서 벡터는 초기화 시 복사한 읽기 전용 스냅샷이므로 **corpus 변경 시 검색기를 재생성**해야 합니다.

여러 가중치를 비교할 때는 `hybrid.with_bm25_weight(0.1)`로 새 검색기를 만들 수 있습니다.
기존 검색기의 가중치는 유지하며 문서 벡터 스냅샷·BM25 인덱스·embedder를 공유합니다.

## 계산과 입력 계약

```text
M(q) = max over corpus BM25(q, d)
normalized_bm25(q, d) = BM25(q, d) / M(q), if M(q) > 0
                       0, otherwise
final_score(q, d) = dense_score(q, d) + α × normalized_bm25(q, d)
```

- 정규화와 Top-K 선택은 **전체 corpus** 기준입니다. 양쪽 Top-K 합집합만 결합하거나
  누락된 Dense 점수를 0으로 대체하지 않습니다. BM25의 0점 문서는 생략할 수 있습니다.
- α는 유한한 0 이상의 값이며 `0`은 Dense 점수 ablation입니다. source rank는 사용하지
  않고 source score는 유한해야 합니다. 최종 점수는 확률이나 질의 간 비교용 점수가 아닙니다.
- corpus 불일치와 검색기 내부의 중복 ID는 오류입니다. 양쪽에 있는 같은 문서는 한 번만 반환합니다.
- 점수 내림차순, 동점이면 `document_id` 오름차순으로 최대 `top_k`개를 반환합니다.
  순위는 1부터 연속이며 제목·snippet은 Dense에서 가져옵니다.
- 빈 query는 빈 목록, 잘못된 `top_k`·가중치는 `ValueError`, 문자열이 아닌 query는
  `TypeError`입니다. `bool`은 정수·점수로 허용하지 않습니다.
- 문서·query는 단위 벡터여야 합니다(`atol=1e-4`, `rtol=1e-5`). 자동 정규화하지 않습니다.

공개 참조 함수 `score_fusion()`은 NumPy 없이 동작하며 호출자가 전체 점수를 전달해야 합니다.
`ScoreFusionRetriever.search()`는 query를 한 번 인코딩하고 전체 점수를 배열로 결합한 뒤
최종 Top-K만 `SearchResult`로 생성합니다. 두 경로는 `_fuse_scores()`의 계산식을 공유하며
고정 기대값·랜덤 점수 비교 테스트로 검증합니다.
FAISS와 NumPy FP32 내적의 반올림 차이로 거의 동점인 문서의 순위는 달라질 수 있습니다.

## 후보 범위와 비용

| 구성 | Dense 후보 | BM25 후보 | 결합 |
|---|---|---|---|
| 기존 `HybridRetriever` | Top-K | 양의 점수 Top-K | 순위 RRF, 상수 60 |
| `ScoreFusionRetriever` | 전체 corpus | 전체 양의 점수 문서 | 실제 점수 + 최대값 정규화 |

추가 벡터 메모리는 N×차원×4바이트로, 10k×1024에서 약 **39 MiB**입니다.
요청마다 전체 내적·BM25 계산·`numpy.lexsort` 정렬을 수행합니다.
대규모 corpus에서 부분 선택을 도입할 때도 Top-K 경계의 ID 동점 규칙을 보존해야 합니다.

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

## 재현 명령

Train100에서 α를 선정하고 Dev50·StrategyQA dev에서는 0.05로 고정합니다.
벤치마크는 train/dev ID 중복을 거부합니다.

```bash
uv sync --locked --extra sparse --extra dense

# 고정 GitHub Release에서 다운로드·해시 검증·압축 해제
python3 -m scripts.prepare_score_fusion_cache
SCORE_CACHE=artifacts/score-fusion-cache

# 선정 이력 확인: 원래 Train100에서 제한된 가중치 민감도 비교
uv run --locked --extra sparse --extra dense python -m scripts.benchmark_score_fusion \
  --cache-dir "$SCORE_CACHE/miracl10k" --split train \
  --expected-summary docs/experiments/score-fusion/miracl-train.json \
  --weights 0 0.025 0.05 0.1 --correction --repeats 3 \
  --output artifacts/score-fusion-final/miracl-train

# 확인 평가: 가중치는 0.05로 고정
uv run --locked --extra sparse --extra dense python -m scripts.benchmark_score_fusion \
  --cache-dir "$SCORE_CACHE/miracl10k" --split dev \
  --expected-summary docs/experiments/score-fusion/miracl-dev.json --correction --repeats 3 \
  --output artifacts/score-fusion-final/miracl-dev

uv run --locked --extra sparse --extra dense python -m scripts.benchmark_score_fusion \
  --cache-dir "$SCORE_CACHE/strategyqa" \
  --expected-summary docs/experiments/score-fusion/strategyqa.json --correction --repeats 3 \
  --output artifacts/score-fusion-final/strategyqa
```

기존 가중치 비교 재현 명령은 [결과 보고서](SCORE_FUSION_RESULTS.md)에 있습니다.

벤치마크는 기존 출력 폴더를 덮어쓰지 않으므로 재실행 시 새 경로를 사용합니다.
기본 측정은 CPU/BLAS/FAISS 2스레드, warm-up 1회, 무작위 25질의(seed 20261014),
5회 반복이며 위 명령은 3회로 지정합니다. 실행 순서는 질의·반복마다 순환합니다.

### 캐시와 출처

캐시는 같은 저장소의 `benchmark-inputs-v1` Release에 `score-fusion-cache-v1.tar.gz`로
배포합니다. 버전과 압축파일·8개 입력 파일의 SHA-256은
[cache-release.json](experiments/score-fusion/cache-release.json)에 고정되어 있습니다.
준비 스크립트는 Python 3.12 이상에서 추가 패키지 없이 실행됩니다.

각 데이터셋 캐시는 다음 네 파일로 구성됩니다. 모델 다운로드·문서 재인코딩 없이
BM25와 FAISS를 재구축해 평가합니다.

| 파일 | 내용 |
|---|---|
| `data.json` | 데이터셋 이름, Document 배열, 벡터 행 순서의 `query_ids`, 질의·qrels 매핑, 선택적 split |
| `document-vectors.npy` | 문서 배열 순서의 정규화된 float32 BGE-M3 벡터 |
| `query-vectors.npy` | `query_ids` 순서의 정규화된 float32 BGE-M3 벡터 |
| `encoding.json` | 벡터 SHA-256과 원래 인코딩 설정 |

준비 스크립트는 압축파일과 개별 입력 해시를 검증합니다. 정상 캐시는 재사용하고,
불완전하거나 다른 캐시는 오류로 거부합니다. 새 위치는 `--cache-dir PATH`로 지정합니다.
Release가 없거나 접근할 수 없으면 종료하며, 직접 받은 압축파일은 `--archive`로 지정합니다.
`--expected-summary`는 실행 전 세 데이터·벡터 파일의 해시를 검사하고,
결과의 품질 지표를 비교합니다(`rtol=1e-9`, `atol=1e-12`, 같은 방법·지표 구성이 필요).
질의별 순위까지 확인하려면 `--expected-runs PATH`로 기준 `runs.json`을 추가합니다.
이 옵션은 방법·질의별 문서 ID 순서를 비교하며 점수 자체는 비교하지 않습니다.
latency는 현재 환경에서 다시 측정합니다. 수행한 검증은 summary의
`verified_against_reference`에, 기준 runs 파일 해시는 `expected_runs_sha256`에 기록합니다.

```bash
# 다운로드 없이 현재 캐시의 8개 파일 검증
python3 -m scripts.prepare_score_fusion_cache --verify-only

# 직접 받은 압축파일로 준비
python3 -m scripts.prepare_score_fusion_cache \
  --archive artifacts/releases/score-fusion-cache-v1.tar.gz
```

캐시를 게시할 때는 `python3 -m scripts.prepare_score_fusion_cache --pack PATH`로
압축파일을 만들고 manifest의 SHA-256과 대조합니다. 코드·manifest가 반영된 커밋의
`benchmark-inputs-v1` Release에 `score-fusion-cache-v1.tar.gz`로 첨부한 뒤,
`--cache-dir PATH`로 새 폴더에 다운로드해 검증합니다. 기존 pack 출력은 덮어쓰지 않으며,
입력 변경 시 새 태그와 manifest를 사용합니다.

고정 모델 revision과 인코딩 설정은
[MIRACL](experiments/score-fusion/encoding/miracl10k.json),
[StrategyQA](experiments/score-fusion/encoding/strategyqa.json)에,
원본 출처·해시는 [history.json](experiments/score-fusion/history.json)에 보존했습니다.
StrategyQA의 데이터셋은 `taeminlee/Ko-StrategyQA`, dev split이며, 원본과 내용이 일치하는
revision은 `7cececfcc209fc836e08d6ab7584cba776847cfa`입니다.
2026-10-15에 기존 `hybrid-document-search-old-2.0`의 `corpus.jsonl`, `queries.jsonl`,
`qrels/dev.jsonl`을 캐시의 원본 SHA-256 및 해당 revision의 Git blob ID와 대조했습니다.
당시 다운로드 revision을 소급 확인한 것은 아니며, 파일 내용의 일치로 재현 버전을 고정했습니다.
[데이터셋 검증 기록](experiments/score-fusion/encoding/strategyqa-dataset.json)에 근거를 보존합니다.
전처리는 `title + "\n" + text`, 접두사 없음, FP32·L2 정규화입니다.
새 캐시의 바이트 단위 재현은 보장하지 않으며, 동일한 MIRACL 10k subset·Train100/Dev50에는
기존 manifest 또는 원본 `data.json`이 필요합니다. 입력 변경 시 모델·데이터 revision,
전처리·split seed, 인코딩 환경도 함께 기록합니다.

### 산출물과 측정 범위

| 파일 | 내용 |
|---|---|
| `summary.json` | 집계, 가중치·Top-K·측정 범위, 입력·코드 해시, 환경 |
| `runs.json` | 문서별 점수·순위 |
| `per-query.json` | 질의별 평가 지표 |
| `timings.json` | 원시 지연시간 |

`summary.json`은 나머지 세 산출물의 SHA-256과 실행 코드 해시를 포함합니다.
과거 보존 실행의 기록은 수정하지 않습니다.

품질은 기존 `evaluation.metrics`의 binary nDCG@10·Recall@10 등을 사용합니다.
`relevance > 0`을 관련 문서로 보고 질의별 지표를 산술 평균합니다.
인코딩 자료가 없으면 `model_config=null`, `provenance_status=unverified`로 기록하며,
과거 기본 설정을 실제 설정으로 소급 인증하지 않습니다.

기본 latency는 **모델 추론·초기화·HTTP를 제외한 CPU 검색·결합 시간**입니다.
보정 비교의 문서 PCA는 초기화에서, query 보정은 측정 안에서 수행합니다.
측정 중 검색 결과는 품질 평가 순위와 대조합니다.

추론을 포함한 별도 측정의 조건·결과·재현 명령은
[결과 보고서](SCORE_FUSION_RESULTS.md)의 “추론 포함 End-to-End 검색 latency” 절을 참고합니다.

## 구현과 검증 자료

구현은 `fusion/score.py`, 관련 테스트는 `tests/fusion/`에 있습니다.
점수 결합, 검색기 통합, 벤치마크, E2E 측정과 캐시 준비를 검증합니다.
테스트는 합성 점수와 실제 BM25/FAISS·고정 벡터를 사용하며 모델 다운로드는 필요 없습니다.
검증 기록은 [verification.json](experiments/score-fusion/verification.json),
개선·악화 사례는 [cases.json](experiments/score-fusion/cases.json)을 참고합니다.

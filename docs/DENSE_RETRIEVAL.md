# Dense Retrieval MVP

Sentence Transformers 임베딩을 FAISS `IndexFlatIP`로 검색한다.
문서와 인덱싱 규칙은 [공통 스키마](SCHEMAS.md)를 따르며, 결과는 `SearchResult`로 반환한다.

## 모델 선택과 메모리 예산

기본 모델은 **intfloat/multilingual-e5-base**다. 첫 baseline에서는 검색 경로를
구축하고 반복 실행할 수 있도록 모델 크기와 인덱스 메모리 사용량을 우선 고려했다.
E5-base는 다국어 검색용으로 학습된 12-layer 모델이며, 768차원 임베딩을 사용한다.
한국어 검색 품질은 후속 평가에서 BGE-M3, KURE-v1과 비교한다.

아래는 2026-10-06에 확인한 모델 카드 기준의 사양과 선택 근거다.
동일한 데이터로 측정한 검색 품질, 처리 시간, 추론 메모리는 후속 평가 항목이다.

| 후보 | 한국어 검색 관련 근거 | 임베딩 차원 / 최대 입력 길이 | 도입 시 고려사항 |
| --- | --- | --- | --- |
| [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) | 다국어 검색 학습, 한국어 Mr. TyDi 평가 결과 공개 | 768 / 512 tokens | Sentence Transformers로 사용 가능. 질의와 문서에 각각 prefix 필요 |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | 100개 이상 언어와 MIRACL 평가 지원 | 1024 / 8192 tokens | 모델 크기와 긴 입력에 따른 실행 비용 고려. Sentence Transformers로 dense 벡터 생성 가능. Sparse·multi-vector 기능은 별도 통합 필요 |
| [nlpai-lab/KURE-v1](https://huggingface.co/nlpai-lab/KURE-v1) | BGE-M3를 한국어 검색 데이터로 추가 학습 | 1024 / 8192 tokens | 한국어 특화 비교 후보. Sentence Transformers 예제 제공. BGE-M3 계열의 메모리 사용량 고려 |

전체 corpus를 인덱싱할 때는 벡터 저장 공간과 빌드 중 메모리를 구분해야 한다.
[MIRACL corpus](https://huggingface.co/datasets/miracl/miracl-corpus)의 한국어 passage
1,486,752개를 float32 Flat 인덱스에 저장하면, 벡터 저장 공간은 다음과 같다.

| 임베딩 차원 | 벡터 저장 공간 (`문서 수 × 차원 × 4 bytes`) |
| --- | --- |
| 768 | 약 4.25 GiB |
| 1024 | 약 5.67 GiB |

이 계산에는 모델, 원문, Python 객체, 임베딩 생성 중 임시 버퍼가 포함되지 않는다.
현재 빌드는 전체 임베딩 행렬과 FAISS에 복사된 벡터를 동시에 메모리에 둔다.
따라서 전체 corpus를 처리하기 전에 streaming build와 원문 저장소 분리를 검토해야 한다.
실제 처리 시간과 추론 메모리는 입력 길이, batch size, CPU/GPU 조건에 따라 달라진다.
공통 MVP의 목표인 10,000개 passage에서는 768차원 float32 벡터 저장 공간이
약 29.3 MiB다. 모델과 원문, 빌드 중 임시 버퍼의 메모리는 별도로 필요하다.

## 실행

저장소 루트에서 실행한다. 실제 corpus 준비는 [DATA_PREPARATION.md](DATA_PREPARATION.md)를 참고한다.
아래 fixture는 실행 검증용이며 검색 품질 평가에는 사용하지 않는다.

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

실제 문서에는 `--corpus data/processed/corpus.jsonl`을 사용한다.
stdout은 빌드 요약 또는 검색 결과 JSON, stderr는 진행 안내와 오류이며 입력·실행 오류의 종료 코드는 2다.
전체 옵션과 예시는 각 명령의 `--help`로 확인한다.

## 입력·검색 계약

- 입력은 UTF-8 JSONL이다. `document_id`는 고유한 비어 있지 않은 문자열, `text`는 비어 있지 않은 문자열이며 `title`은 문자열 또는 `null`이다.
- 문서는 `build_index_text()`로 제목과 본문을 결합해 임베딩하고 원문은 보존한다. 모델 입력 길이를 넘는 텍스트는 잘리며 별도 chunking은 없다.
- `DenseConfig` 기본값은 batch size 32, L2 정규화다. 정규화된 벡터의 내적은 cosine similarity이며 `--no-normalize`는 원시 내적을 사용한다.
- E5 모델 이름에는 `query: ` / `passage: `를 자동 적용한다. 로컬 E5 모델은 빌드 시 `--query-prefix 'query: '` / `--passage-prefix 'passage: '`를 명시한다.
- `--device`는 임베딩 실행 장치이며 생략하면 자동 선택한다. FAISS 검색은 CPU에서 수행한다.
- 결과는 `document_id`, `rank`, `score`, `title`, `snippet`을 가진 JSON 배열이다. 순위는 1부터 시작하며 snippet은 원문 앞 200자다.
- `top_k`는 bool을 제외한 양의 정수이며 결과 수는 문서 수를 넘지 않는다. 빈 질의는 모델을 로드하지 않고 `[]`를 반환한다. 비문자열 질의는 `TypeError`, 잘못된 `top_k`는 `ValueError`다.

## 저장·복원

| 파일 | 내용 |
| --- | --- |
| `index.faiss` | 문서 벡터와 FAISS 인덱스 |
| `metadata.json` | 형식 버전, 차원, SHA-256 체크섬, 임베딩 설정, 행 순서대로 저장한 문서 ID·제목·원문 |

로드 시 체크섬·인덱스 타입·차원·문서 수·ID 중복을 검증한다. 검색은 저장된 문서 벡터를 재사용하고 질의만 임베딩한다.
모델 가중치는 저장하지 않으므로 빌드에 사용한 모델 경로나 캐시가 필요하다. `device`는 저장하지 않고 복원 시 지정한다.
두 파일은 순차 저장하므로 빌드 중인 디렉터리를 검색에 사용하지 않는다. 파일 손상·불일치가 발생하면 재빌드한다.

```python
from retrievers.dense import DenseRetriever

retriever = DenseRetriever.load("indexes/dense-sample", device="cpu")
hits = retriever.search("고양이 울음소리", top_k=3)
original = retriever.get_document(hits[0].document_id).text
```

## 검증

```bash
uv run --locked --extra dense ruff check .
uv run --locked --extra dense ruff format --check .
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked --extra dense pytest tests/dense
```

fixture와 로컬 BoW 모델로 임베딩 설정, FAISS 검색, 저장·복원, CLI 오류 안내를 검증한다.
Dense 담당 테스트와 전용 fixture는 `tests/dense/`에 모아 관리한다.
`test_embedding.py`는 임베딩 파이프라인, `test_faiss_index.py`는 벡터 인덱스,
`test_retriever.py`는 검색 계약, `test_persistence.py`는 문서·설정 저장과 복원을 검증한다.
`test_cli.py`와 `test_cli_messages.py`는 Dense 빌드·검색 CLI를 검증하며,
공통 fixture와 모델 대역은 이 디렉터리의 `conftest.py`에만 정의한다.
CI의 `dense-tests` job도 모델·데이터 다운로드 없이 실행한다. 모델별 검색 품질과 전체 corpus 성능은 아직 평가하지 않았다.

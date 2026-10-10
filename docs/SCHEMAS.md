# 공통 데이터 스키마

문서·인덱싱·query/qrels 필드와 준비 스크립트의 파일 구성·검증 규칙입니다.
관련 Issue: #4
담당: @bangahee

## 문서 (Document)

레코드 하나는 원본 passage 하나를 나타냅니다.

| 필드 | 자료형 | 필수 여부 | 의미 |
| --- | --- | --- | --- |
| `document_id` | string | 필수 | 원본에서 유지한 안정적인 passage ID |
| `text` | string | 필수 | 원본 passage 텍스트 |
| `title` | string 또는 null | 값은 선택, 준비된 JSONL에는 키 유지 | 원본 문서 제목. 제목이 없거나 비어 있으면 null |

원본 필드 매핑:

- `_id` → `document_id`
- `text` → `text`
- `title` → `title`

규칙:

- `document_id`는 비어 있으면 안 되며, corpus 내에서 고유해야 합니다.
- `#` 뒤에 붙는 부분까지 포함하여 원본 ID 전체를 유지합니다.
- ID는 문자열이며, 배열의 위치 값으로 대체하지 않습니다.
- passage 단위의 검색을 유지합니다.
- 준비된 JSONL에는 `title` 키를 항상 포함하며, 제목이 없거나 비어 있으면 `null`로 저장합니다.
- 화면 표시에 사용할 `text`를 보존하며, TF-IDF 토큰으로 덮어쓰지 않습니다.

설명용 가상 예시:

```json
{
  "document_id": "fixture-doc-001",
  "text": "서울은 대한민국의 수도입니다.",
  "title": "서울"
}
```

## 질의 (Query)

아래 질의 필드는 팀 합의를 반영한 공통 형식입니다.

| 필드 | 자료형 | 필수 여부 | 의미 |
| --- | --- | --- | --- |
| `query_id` | string | 필수 | 원본에서 유지한 안정적인 질의 ID |
| `text` | string | 필수 | 질의 텍스트 |

원본 필드 매핑:

- `_id` → `query_id`
- `text` → `text`

설명용 가상 예시:

```json
{
  "query_id": "fixture-query-001",
  "text": "대한민국의 수도는 어디인가요?"
}
```

질의 ID와 문서 ID는 별도의 이름 공간에 속합니다.
질의 ID의 형태만 보고 관련 문서를 추정하지 않습니다.

질의의 `query_id`와 `text`는 비어 있지 않은 문자열이어야 합니다.
질의 ID는 각 split 내에서 고유해야 하며, 원본 train/dev 소속을 유지합니다.
준비 스크립트는 split을 레코드의 추가 필드 대신 아래의 파일명으로 구분합니다.

## 관련성 판정 (Qrel)

아래 qrels 필드는 팀 합의를 반영한 공통 형식입니다.
고정 revision의 원본 train/dev JSONL에서 라벨 0과 1을 확인했습니다.

| 필드 | 자료형 | 필수 여부 | 의미 |
| --- | --- | --- | --- |
| `query_id` | string | 필수 | 참조하는 질의 |
| `document_id` | string | 필수 | 관련성을 판정한 passage |
| `relevance` | number | 필수 | 원본 관련성 라벨 |

원본 필드 매핑:

- `query-id` → `query_id` (원본의 정수 ID는 문자열로 변환)
- `corpus-id` → `document_id`
- `score` → `relevance`

규칙:

- 원본 라벨 값과 train/dev 소속을 유지합니다.
- 원본 라벨 0과 1을 보존하며, 0인 판정도 subset에 포함합니다.
- 참조하는 질의와 문서가 존재하는지 검증합니다.
- 같은 `(query_id, document_id)` 판정이 중복되면 동일 라벨은 하나로 합치며, 라벨이 충돌하면 오류로 중단합니다.
- 각 split의 `query_id`는 해당 split의 질의 파일에, `document_id`는 공통 corpus에 존재해야 합니다.
- ID는 비어 있지 않은 문자열이어야 하며, `relevance`는 유한한 숫자여야 합니다. bool은 라벨로 허용하지 않습니다.
- 관련성 라벨은 검색기의 유사도 점수가 아닙니다.
- 판정이 없다는 것은 미판정을 의미하며, 명시적인 비관련 판정을 의미하지 않습니다.

설명용 가상 예시:

```json
{
  "query_id": "fixture-query-001",
  "document_id": "fixture-doc-001",
  "relevance": 1
}
```

## 구현한 준비 결과 파일과 split 표현

각 JSONL은 UTF-8로 저장하며, 한 줄에 JSON 객체 하나를 기록합니다.
`corpus.jsonl` 하나를 모든 검색 방식과 train/dev 질의가 공유합니다.

| 파일 | 레코드 필드 | 용도 |
| --- | --- | --- |
| `data/processed/corpus.jsonl` | `document_id`, `text`, `title` | 공통 문서 집합 |
| `data/processed/queries_train.jsonl` | `query_id`, `text` | train 질의 |
| `data/processed/queries_dev.jsonl` | `query_id`, `text` | dev 질의 |
| `data/processed/qrels_train.jsonl` | `query_id`, `document_id`, `relevance` | train 관련성 판정 |
| `data/processed/qrels_dev.jsonl` | `query_id`, `document_id`, `relevance` | dev 관련성 판정 |
| `data/processed/manifest.json` | 준비 설정과 실행 결과 | revision, seed, 실제 데이터 수 등의 재현성 기록 |

위 파일 구성을 준비 스크립트에 구현하고 실제 원본 corpus로 결과를 생성했습니다. 수량과 검증 결과는 [DATASET.md](DATASET.md)에 기록했습니다.
`manifest.json`은 JSON 객체 하나로 저장하며, query/qrel 레코드에 별도의 `split` 필드는 추가하지 않습니다.
원본의 train/dev 소속을 유지하며, dev 데이터를 최종 평가용으로 분리하여 보관합니다.
목표 subset 규모와 선택 방식은 [DATASET.md](DATASET.md)의 설정표를 참조합니다.

원본 `_id` 형식은 `data.loader.load_documents`, 준비된 `document_id` 형식은 `data.loader.load_prepared_documents`로 읽습니다.
두 함수는 ID·본문·제목과 중복 ID를 같은 규칙으로 검증하며, 입력 형식을 자동 추정하지 않습니다.
필드명을 바꾸더라도 원본 ID 값은 동일하게 보존합니다.

## 합의된 인덱싱 내용

Sparse와 Dense 검색 모두 동일한 원본 내용을 사용합니다.
제목이 있으면 `title + "\n" + text`를 사용하고, 제목이 없거나 비어 있으면 `text`만 사용합니다.
준비된 JSONL에서 `title`이 `null`인 경우에도 `text`만 사용합니다.
검색 방식별 토큰화나 모델이 요구하는 입력 형식은 달라질 수 있습니다.

## 기존 검색 계약

기존 `Retriever`와 `SearchResult` 정의를 유지합니다.

`search(query: str, top_k: int) -> list[SearchResult]`

검색 결과는 안정적인 문서 ID를 사용하고, 관련도가 높은 결과부터 반환하며,
순위 번호는 1부터 연속해서 부여합니다.
점수는 검색 방식별로 의미가 다릅니다.
RRF는 원시 점수를 직접 비교하지 않고 순위를 결합합니다.

## 반영한 팀 의견

- [@shannonlee-dev의 의견](https://github.com/shannonlee-dev/hybrid-document-search/issues/4#issuecomment-6011200128): `document_id`, `text`, 선택적인 `title` 구성, 원본 passage ID 유지, `title + "\n" + text` 및 제목이 없거나 비어 있을 때의 `text` 단독 사용에 동의했습니다.
- [@VectorSophie의 의견](https://github.com/shannonlee-dev/hybrid-document-search/issues/4#issuecomment-6011160460): 준비된 JSONL에서 없는 제목을 `null`로 일관되게 표현하도록 요청했으며, 기존 `Retriever` / `SearchResult` 계약을 유지하는 데 동의했습니다.

준비 스크립트 실행과 검증 범위는 [DATA_PREPARATION.md](DATA_PREPARATION.md)를 참조합니다.

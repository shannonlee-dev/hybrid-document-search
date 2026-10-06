# 공통 데이터 스키마 초안

상태: 초안 — 문서 구조 및 인덱싱 규칙 합의 반영, 나머지 사항은 합의 대기
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

## 관련성 판정 (Qrel)

| 필드 | 자료형 | 필수 여부 | 의미 |
| --- | --- | --- | --- |
| `query_id` | string | 필수 | 참조하는 질의 |
| `document_id` | string | 필수 | 관련성을 판정한 passage |
| `relevance` | number | 필수 | 원본 관련성 라벨 |

원본 필드 매핑:

- `query-id` → `query_id`
- `corpus-id` → `document_id`
- `score` → `relevance`

규칙:

- 원본 라벨 값과 train/dev 소속을 유지합니다.
- 평가 규칙을 확정하기 전에 실제 라벨 값을 확인합니다.
- 참조하는 질의와 문서가 존재하는지 검증합니다.
- 중복된 판정의 라벨이 서로 충돌하면 보고합니다.
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

## 합의가 필요한 사항

- query/qrels 필드와 라벨 검증 규칙의 최종 확인.
- 정규화와 유효하지 않은 레코드의 처리 방식 합의.
- JSONL 파일 구성과 split 표현 방식 합의.

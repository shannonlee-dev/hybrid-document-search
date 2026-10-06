# 데이터셋 초안: Ko-miracl

상태: 준비 계획 — 문서·인덱싱·subset 설정 및 query/qrels 필드 합의 반영, 구현·검증 예정
관련 Issue: #4
담당: @bangahee

## 데이터셋 선정

선정한 데이터셋:
https://huggingface.co/datasets/taeminlee/Ko-miracl

원본 MIRACL:
https://github.com/project-miracl/miracl

Ko-miracl은 MIRACL의 한국어 데이터를 BEIR 형식으로 변환한 데이터셋입니다.
검색 실험에 사용할 수 있는 한국어 위키백과 passage, 질의, 관련성 판정 정보를 제공합니다.

선정 이유:

- 한국어 문서로 한국어 검색 서비스를 시연할 수 있습니다.
- 기존 질의와 관련성 판정 정보를 이후 평가에 활용할 수 있습니다.
- Sparse, Dense, Hybrid 검색에서 동일한 문서 집합(corpus)을 사용할 수 있습니다.
- 원본 필드를 제안한 문서 스키마에 자연스럽게 매핑할 수 있습니다.

## 원본 데이터 구조

| 구성 요소 | 원본 필드 |
|---|---|
| 문서 집합 (Corpus) | `_id`, `title`, `text` |
| 질의 (Queries) | `_id`, `text` |
| 관련성 판정 (Qrels) | `query-id`, `corpus-id`, `score` |

검색 단위는 위키백과 문서 전체가 아니라 개별 passage입니다.
`#` 뒤에 붙는 부분까지 포함하여 원본 passage ID를 그대로 유지합니다.

## 합의된 MVP 데이터 준비 설정

팀 합의에 따라 초기 MVP의 목표값을 다음과 같이 정합니다.
실제 데이터 준비 스크립트와 결과 파일은 아직 구현하거나 생성하지 않았습니다.

| 항목 | 합의 값 | 상태 |
| --- | --- | --- |
| 공통 corpus 목표 규모 | 10,000개 passage | 합의 완료 |
| train 질의 수 | 100개 | 합의 완료 |
| dev 질의 수 | 50개 | 합의 완료 |
| 난수 seed | 42 | 합의 완료 |
| query/qrels 필드 | [SCHEMAS.md](SCHEMAS.md)의 형식 | 합의 완료 |

데이터 준비 순서는 다음과 같이 제안합니다.

1. 데이터셋 revision을 고정하고 원본 corpus, queries, train/dev qrels를 준비합니다.
2. 질의 및 문서 참조를 검증하고, 각 split의 qrels에 판정이 있는 질의를 선정 대상으로 삼습니다.
3. 대상 질의 ID를 정렬한 뒤, split별로 고정 seed를 사용하여 질의를 샘플링합니다.
4. 선택한 train/dev 질의의 모든 판정 문서를 합쳐 공통 corpus에 포함합니다. 관련성 라벨이 0인 판정도 유지합니다.
5. 필수 문서를 제외한 나머지 corpus에서 배경 문서를 샘플링하여 목표 규모를 구성합니다.
6. 원본 passage ID와 질의의 train/dev 소속을 유지하고, 선택 규칙, seed, 선택한 ID와 실제 데이터 수를 기록합니다.
7. Sparse, Dense, Hybrid에서 동일하게 준비된 corpus와 질의 파일을 사용합니다.

필수 판정 문서 수가 목표 규모를 초과하면 corpus 규모를 조정하고 팀에 공유합니다.
조정한 값과 이유는 manifest에 기록합니다. 필요한 질의 또는 문서가 부족하면 준비를 중단하고 설정을 확인합니다.
dev 질의와 qrels는 이후 평가용으로 분리하여 보관합니다.

축소한 corpus의 결과는 subset 결과로 명시하며, 전체 MIRACL 벤치마크 결과로 표현하지 않습니다.

## 전처리 방식 제안

- 문서 ID와 질의 ID를 유지합니다.
- 필수 필드와 중복 ID를 검증합니다.
- 화면 표시에 사용할 원본 passage 텍스트를 보존합니다.
- 공통 데이터의 본문은 원문을 유지하는 최소 처리 방식을 제안합니다. 추가 유니코드 및 공백 정규화는 팀 합의 후 적용합니다.
- 제목이 없거나 비어 있으면 `null`로 저장합니다. 현재 loader는 공백으로만 구성된 제목도 같은 방식으로 처리합니다.
- 검색 방식별 토큰화는 공통 텍스트와 분리합니다.
- 유효하지 않은 레코드와 연결할 수 없는 참조를 보고합니다.

## 재현성과 저장 방식

- 데이터셋 revision: 선정 및 검증 예정입니다.
- 실제로 로드한 정확한 데이터 수: 검증 예정입니다.
- subset 목표 규모, 질의 수와 seed: 위 설정표의 합의된 값을 사용합니다. 실제 출력 수는 실행 후 검증합니다.
- 출처 및 이용 조건: 검토 후 문서화할 예정입니다.
- 원본 다운로드 경로: `data/raw/` 또는 `datasets/`.
- 준비된 데이터 경로: `data/processed/`.
- 대용량 데이터셋 파일은 Git에 포함하지 않습니다.
- 다른 팀원이 재현할 수 있도록 데이터 준비 코드와 문서를 커밋합니다.

### 준비 결과 파일 구성 제안

아래는 준비 스크립트가 생성할 예정인 파일 구성입니다.
각 JSONL은 UTF-8로 저장하며, 한 줄에 JSON 객체 하나를 기록합니다.
공통 corpus 하나를 train/dev 양쪽 질의에서 참조합니다.

```text
data/processed/
├── corpus.jsonl
├── queries_train.jsonl
├── queries_dev.jsonl
├── qrels_train.jsonl
├── qrels_dev.jsonl
└── manifest.json
```

레코드 규격은 [SCHEMAS.md](SCHEMAS.md)의 문서, 질의, qrel 형식을 따릅니다.
원본 corpus의 `_id`와 준비 결과의 `document_id`는 필드명이 다릅니다.
현재 `load_documents`는 원본 `_id` 형식을 읽으므로, 준비된 corpus를 검색하기 위한 loader 또는 adapter를 다음 구현에서 추가합니다.

### Manifest에 기록할 항목

- 데이터셋 ID, 고정한 revision, 원본 파일과 출처 및 이용 조건 검토 내용.
- 준비 스크립트 버전 또는 Git commit.
- 목표 corpus 규모, 목표 split별 질의 수, seed와 선택 알고리즘.
- 원본 데이터 수와 실제로 준비된 corpus, split별 queries/qrels 수.
- 선택한 문서 및 질의 ID를 재현하거나 검증할 수 있는 기록.
- 판정 라벨 값, 검증 결과, 규모 조정이 있었다면 이유.
- 출력 파일 목록과 체크섬.

revision, 실제 데이터 수, 검증 결과와 이용 조건은 실행 및 검토 후 기록합니다.
아직 확인하지 않은 값을 확정 사실로 기록하지 않습니다.

## 반영한 팀 의견

Issue #4의 [Dense 담당 의견](https://github.com/shannonlee-dev/hybrid-document-search/issues/4#issuecomment-6011200128)과 [Hybrid / Integration 담당 의견](https://github.com/shannonlee-dev/hybrid-document-search/issues/4#issuecomment-6011160460)을 반영했습니다.

- 문서는 `document_id`, `text`, 선택적인 `title`로 구성합니다.
- `document_id`에는 `#` 뒤에 붙는 부분을 포함한 원본 passage ID를 그대로 유지합니다.
- 준비된 JSONL에는 `title` 키를 항상 포함하며, 제목이 없거나 비어 있으면 `null`로 저장합니다.
- Sparse와 Dense는 제목이 있으면 `title + "\n" + text`, 제목이 없거나 비어 있으면 `text`만 인덱싱합니다.
- 기존 `Retriever` / `SearchResult` 계약을 유지합니다.
- 구체적인 필드 매핑과 규칙은 [SCHEMAS.md](SCHEMAS.md)에 정리합니다.

## 팀 합의가 필요한 사항

- 실제 라벨 값 확인 및 중복·충돌 판정의 처리 규칙.
- 공통 정규화 규칙.
- JSONL 파일 구성 및 dev 평가 데이터의 분리·보관 방식.

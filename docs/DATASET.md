# 데이터셋 초안: Ko-miracl

상태: 초안 — 문서 구조 및 인덱싱 규칙 합의 반영, 나머지 사항은 합의 대기
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

## MVP 데이터 준비 방식 제안

전체 corpus를 바로 인덱싱하기보다 재현 가능한 subset부터 준비합니다.
초기 규모는 약 10,000–20,000개 passage를 제안하며, 자원 확인과 팀 합의를 거쳐 확정합니다.

선택한 질의에 대해 다음 규칙을 적용합니다.

- 해당 질의의 관련성 판정이 있는 모든 passage를 포함합니다.
- 나머지 corpus에서 배경 passage를 샘플링하여 추가합니다.
- 선택 규칙, 난수 seed, 선택한 ID를 기록합니다.
- 질의의 원본 train/dev 소속을 유지합니다.
- 모든 검색 방식에서 동일하게 준비된 corpus를 사용합니다.

축소한 corpus의 결과는 subset 결과로 명시하며, 전체 MIRACL 벤치마크 결과로 표현하지 않습니다.

## 전처리 방식 제안

- 문서 ID와 질의 ID를 유지합니다.
- 필수 필드와 중복 ID를 검증합니다.
- 화면 표시에 사용할 원본 passage 텍스트를 보존합니다.
- 인덱싱에는 팀에서 합의한 유니코드 및 공백 정규화를 적용합니다.
- 검색 방식별 토큰화는 공통 텍스트와 분리합니다.
- 유효하지 않은 레코드와 연결할 수 없는 참조를 보고합니다.

## 재현성과 저장 방식

- 데이터셋 revision: 선정 및 검증 예정입니다.
- 실제로 로드한 정확한 데이터 수: 검증 예정입니다.
- subset 규모와 seed: 팀 합의 대기 중입니다.
- 출처 및 이용 조건: 검토 후 문서화할 예정입니다.
- 원본 다운로드 경로: `data/raw/` 또는 `datasets/`.
- 준비된 데이터 경로: `data/processed/`.
- 대용량 데이터셋 파일은 Git에 포함하지 않습니다.
- 다른 팀원이 재현할 수 있도록 데이터 준비 코드와 문서를 커밋합니다.

## 반영한 팀 의견

Issue #4의 [Dense 담당 의견](https://github.com/shannonlee-dev/hybrid-document-search/issues/4#issuecomment-6011200128)과 [Hybrid / Integration 담당 의견](https://github.com/shannonlee-dev/hybrid-document-search/issues/4#issuecomment-6011160460)을 반영했습니다.

- 문서는 `document_id`, `text`, 선택적인 `title`로 구성합니다.
- `document_id`에는 `#` 뒤에 붙는 부분을 포함한 원본 passage ID를 그대로 유지합니다.
- 준비된 JSONL에는 `title` 키를 항상 포함하며, 제목이 없거나 비어 있으면 `null`로 저장합니다.
- Sparse와 Dense는 제목이 있으면 `title + "\n" + text`, 제목이 없거나 비어 있으면 `text`만 인덱싱합니다.
- 기존 `Retriever` / `SearchResult` 계약을 유지합니다.
- 구체적인 필드 매핑과 규칙은 [SCHEMAS.md](SCHEMAS.md)에 정리합니다.

## 팀 합의가 필요한 사항

- MVP corpus 규모와 질의 선정 방식.
- query/qrels 스키마와 라벨 검증 규칙의 최종 확인.
- 공통 정규화 규칙.
- 이후 별도 평가에 사용할 질의의 분리 및 보관 방식.

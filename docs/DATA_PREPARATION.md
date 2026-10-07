# Ko-miracl 공통 subset 준비

관련 Issue: #4

## 목적과 기본 설정

Sparse, Dense, Hybrid에서 동일하게 사용할 corpus와 train/dev 질의 및 판정 파일을 생성합니다.
기본 설정은 팀이 합의한 corpus 10,000 passage, train 100 질의, dev 50 질의, seed 42입니다.
BM25와 본격적인 평가 metric 구현은 이번 작업에 포함하지 않습니다.

## 실행 방법

저장소 루트에서 실행합니다. 추가 의존성 없이 Python 표준 라이브러리를 사용합니다.

```bash
uv run --extra sparse python -m scripts.prepare_dataset --download
```

원본은 `data/raw/ko-miracl/5c7690518e481375551916f24241048cf7b017d0/`에 저장하며, 준비 결과는 `data/processed/`에 저장합니다.
첫 실행에서는 전체 corpus와 질의, 판정 및 README를 다운로드합니다. 10,000개 passage만 원격으로 받는 방식이 아닙니다.
이후 같은 revision의 다운로드 파일이 있으면 재사용합니다.
다운로드 도중 실패한 파일은 완성된 파일로 남기지 않으며, 재실행하면 해당 파일을 다시 받습니다.

이미 원본 JSONL을 가지고 있다면 다운로드 없이 실행할 수 있습니다.
`--raw-dir`에는 `corpus.jsonl`, `queries.jsonl`, `qrels/train.jsonl`, `qrels/dev.jsonl`이 있는 디렉터리를 지정합니다.

```bash
uv run --extra sparse python -m scripts.prepare_dataset \
  --raw-dir data/raw/ko-miracl/5c7690518e481375551916f24241048cf7b017d0 \
  --output-dir data/processed
```

출력 경로는 존재하지 않거나 비어 있어야 합니다. 기존 결과는 덮어쓰지 않습니다.
비교용 재실행은 `--output-dir data/processed/recheck`처럼 별도의 빈 경로를 지정합니다.
합의한 값을 변경할 때는 `--corpus-size`, `--train-queries`, `--dev-queries`, `--seed` 옵션을 사용합니다.
변경한 설정은 팀에 공유합니다. `--revision`에는 `main` 대신 고정된 40자리 commit SHA를 사용합니다.

## 선택과 변환 규칙

1. 각 split에서 판정이 있는 질의 ID를 문자열로 변환하고 정렬합니다.
2. split마다 별도의 `random.Random(seed)`로 목표 수만큼 질의를 선택합니다.
3. 선택한 질의의 모든 판정 문서를 공통 corpus에 포함합니다. 라벨 0인 판정도 유지합니다.
4. 배경 문서는 `SHA256(seed + NUL + document_id)` 값이 작은 순서로 선택합니다. 값이 같으면 문서 ID로 결정합니다.
5. 문서와 질의는 ID 순서로 저장하고, qrels는 query/document ID 순서로 저장합니다.

필수 판정 문서가 목표 corpus보다 많으면 필수 문서를 모두 포함하도록 규모를 늘리고 manifest에 이유를 기록합니다.
ID와 본문, 유효한 제목은 원문 그대로 보존합니다. 없는 제목과 빈 제목, 공백뿐인 제목은 `null`로 변환합니다.
원본 `_id`는 준비된 corpus의 `document_id`로, qrels의 정수 `query-id`는 문자열 `query_id`로 매핑합니다.
동일 query/document 판정의 중복은 동일 라벨일 때 하나로 합치고, 서로 다른 라벨이면 오류로 중단합니다.
이 중복 처리 방식과 파일 구성은 팀 리뷰에서 확인할 구현 정책입니다.

## 검증과 메모리 사용

원본 문서 및 질의 ID 중복, 필수 필드, 제목 자료형, 유한한 숫자 라벨, 모든 qrels의 질의·문서 참조를 검증합니다.
원본 train/dev에 동일 질의가 등장하거나 목표 수를 구성할 데이터가 부족하면 오류로 중단합니다.
검증에 실패하면 준비 결과를 쓰지 않습니다.

전체 corpus 본문을 한꺼번에 로드하지 않고 한 줄씩 읽습니다.
중복 검사를 위해 모든 문서 ID를 보관하며, 본문은 필수 문서 및 선택한 배경 문서만 보관합니다.
질의와 qrels는 전체를 로드합니다.

## 결과와 재현성

생성하는 여섯 파일의 필드는 [SCHEMAS.md](SCHEMAS.md)에 정리했습니다.
`manifest.json`에는 설정, 고정 revision, 실제 입력·출력 수, 선택한 ID, 라벨 값, 규모 조정 이유와 준비 코드 및 데이터 파일의 SHA-256을 기록합니다.
실행 시간은 기록하지 않으므로 같은 원본과 코드·설정으로 실행하면 결과 파일을 바이트 단위로 비교할 수 있습니다.
로컬 원본의 revision 값은 지정한 출처 정보이며, 원격 파일과 일치하는지 확인하는 검증은 아닙니다.
준비된 데이터와 원본은 기존 `.gitignore` 규칙에 따라 Git에 포함하지 않습니다. 코드와 실행 문서를 공유합니다.

축소 corpus를 사용한 결과는 subset 실험으로 명시합니다.
고정 revision의 변환 데이터셋 README에는 명시적인 license 항목이 없으며, 출처와 이용 조건 상태를 manifest에 기록합니다.

## 테스트 및 현재 검증 범위

```bash
uv run --extra sparse pytest tests/test_prepare_dataset.py
uv run --extra sparse pytest
uv run --extra sparse ruff check .
uv run --extra sparse ruff format --check .
```

오프라인 fixture로 필수 문서 포함, 정수 query ID 변환, 원문 보존, 중복 및 참조 오류, split 유지, 규모 조정, 체크섬과 반복 실행 재현성을 확인합니다.
다운로드 테스트는 가짜 응답을 사용하여 고정 revision URL, 캐시 재사용 및 실패 시 임시 파일 정리를 확인합니다.
2026-10-06 전체 원본 corpus 다운로드와 subset 생성 결과를 확인했습니다.
원본·출력 체크섬, 실제 데이터 수, 질의 선택과 모든 선택 질의의 판정 보존, 참조 및 원문 보존을 검증했습니다.
수량과 구성은 [DATASET.md](DATASET.md)의 실데이터 검증 결과를 참조합니다.
검색 품질 metric과 전체 MIRACL 성능은 평가하지 않았습니다.

`load_documents`는 원본 `_id` 형식을, `load_prepared_documents`는 준비 결과의 `document_id` 형식을 읽습니다.
준비된 corpus는 [SPARSE_MVP.md](SPARSE_MVP.md)의 TF-IDF 검색 CLI에서 사용할 수 있습니다.

# Dense Retrieval

Sentence Transformers로 임베딩하고 CPU FAISS `IndexFlatIP`로 검색한다.
모델 후보와 revision, 기본 모델은 [`config/dense_models.toml`](../config/dense_models.toml)에서 관리한다.
`default_model = "bge"`로 **`BAAI/bge-m3`**를 지정하며 선정 정책은 **`config_default`**이다.
기본값은 프로젝트 설정으로 지정하고 Train/Dev 점수로 자동 변경하지 않는다.
Train nDCG@10을 우선하고 MRR@10·Recall@10·latency·구축 비용을 함께 검토해 BGE-M3를 선정했다.
`[models.<alias>]`에 모델명과 고정 revision을 추가하면 통합 실행기의 후보 목록에 반영된다.

## 고정 모델과 측정 조건

| 모델 | Revision |
| --- | --- |
| `intfloat/multilingual-e5-base` | `d128750597153bb5987e10b1c3493a34e5a4502a` |
| `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` |
| `nlpai-lab/KURE-v1` | `8b418a58414668e75532ed045c22d9ca018ae2b2` |

Ko-MIRACL 10,000 passages, Train 100 / Dev 50 queries, seed 42. Python 3.12.3 / WSL2 / NVIDIA RTX 4060, embedding `cuda:0`, CPU FAISS. FP32, L2 정규화, batch size 1, Top-K 10, 전체 질의 warm-up 1회, 측정 5회. PyTorch intra/inter-op·FAISS·OMP/MKL/OpenBLAS 2 threads, tokenizer 병렬화 비활성화, TF32 비활성화.

E5에는 `query: ` / `passage: ` prefix를 적용하고 BGE/KURE는 빈 prefix를 사용한다.
모델별 tokenizer와 원래 최대 입력 길이를 유지한다. 길이를 넘는 입력은 모델에서 잘리며
별도 chunking은 하지 않는다. 제목과 본문을 합치는 규칙은 [DATASET.md](DATASET.md)에 있다.

## Train 3모델 비교

| 모델 / 방식 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Mean ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E5-base | 0.777889 | 0.898944 | 0.728369 | 0.746289 | 13.379543 | 19.719356 |
| BGE-M3 | 0.871889 | 0.967611 | 0.808429 | 0.823154 | 20.427421 | 28.874457 |
| KURE-v1 | 0.878222 | 0.968444 | 0.797373 | 0.814188 | 21.013247 | 31.478979 |

모델별 100개 질의, latency 샘플 500개다. Train nDCG@10 최고 모델은 BGE-M3이다.
수치 순위와 기본 모델 정책은 별도로 기록한다. [선정 기록](../results/dense/train/selection.json)과
[Train 비교 CSV](../results/dense/train/comparison.csv)를 참조한다.

## Dev 3모델 비교

| 모델 / 방식 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Mean ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E5-base | 0.763333 | 0.912333 | 0.853167 | 0.819452 | 13.215067 | 19.678457 |
| BGE-M3 | 0.813667 | 0.958333 | 0.861024 | 0.854857 | 21.091188 | 29.918226 |
| KURE-v1 | 0.836000 | 0.961000 | 0.849000 | 0.845513 | 20.818919 | 28.696750 |

모델별 50개 질의, latency 샘플 250개다. Dev nDCG@10 최고 모델은 BGE-M3이다.
Dev 결과로 모델이나 설정을 튜닝하지 않았다. [Dev 비교 CSV](../results/dense/dev/comparison.csv).
이 표는 최초 전체 실험의 3모델 비교다. 기본 모델 변경 후 네 방식의 재평가와 latency는
[검색 방식 비교](BENCHMARK_RESULTS.md)에 별도로 기록했다.
각 split의 `e5.json`, `bge.json`, `kure.json`은 평가 CLI의 원본 JSON을 바꾸지 않고 복사했다.

## 인덱스 구축과 복원

| 모델 | Embedding s | FAISS build s | Save s | 준비 합계 s | Peak GPU MiB | 최대 tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E5-base | 93.975 | 0.016 | 0.095 | 96.151 | 1088.875 | 512 |
| BGE-M3 | 193.561 | 0.039 | 0.103 | 196.027 | 2273.052 | 8192 |
| KURE-v1 | 193.291 | 0.066 | 0.096 | 195.896 | 2273.052 | 8192 |

모델 snapshot 다운로드는 빌드 전 별도 단계이며 runtime JSON의 `download_seconds`에 기록한다.
Embedding 시간에는 lazy 모델 로딩과 문서 임베딩이 포함된다. 준비 합계는 입력 확인·빌드·저장 등을
포함하며 별도 프로세스의 복원 평가 시간은 제외한다. Peak는 PyTorch 텐서 최대 할당량이다.

세 인덱스 모두 별도 subprocess에서 복원해 문서 10,000개와 전체 ID·행 순서·원문,
모델명·revision, 저장 전후 Top-K와 score (`rtol=1e-5`, `atol=1e-6`)를 검증했다.
문서 재임베딩을 금지한 상태에서 복원 검색을 수행했고 저장 벡터의 FP32·L2 정규화를 확인했다.
[전체 build/restore 기록](../results/dense/runtime.json)에 인덱스·metadata 해시와 실제 스레드 수가 있다.

## 사용과 저장 형식

```bash
uv run --locked --extra dense python -m scripts.build_index \
  --corpus tests/dense/fixtures/dense_corpus.jsonl \
  --index indexes/dense-sample --device cpu --batch-size 1 --timings
uv run --locked --extra dense python -m scripts.search dense \
  --index indexes/dense-sample --device cpu --query '고양이는 어떤 소리로 우나요?' --top-k 3
```

개별 CLI의 기본 batch size는 32이고 통합 실험에서는 1로 고정한다. 다른 모델은
`--model`과 `--revision <40자리 SHA>`로 지정한다. Dense Runtime CLI도 `--model` 순서에 맞춰
`--revision`을 반복 지정할 수 있다. 전체 재실행 방법은 [EVALUATION.md](EVALUATION.md)에 있다.

`index.faiss`는 문서 벡터, `metadata.json`은 인덱스 해시·차원·임베딩 설정·문서 매핑을 저장한다.
로드 시 체크섬·타입·차원·문서 매핑을 검사하고 첫 검색에서 지정 revision의 모델을 로드한다.
revision 없는 기본 모델 인덱스는 거부한다. 저장 장치는 재사용하지 않고 로드 시 지정한다.
문서 벡터는 재사용하고 질의만 임베딩한다. 두 파일 저장은 순차적이므로 완성된 인덱스만 사용한다.

Float32 벡터 자체는 10,000개 기준 768차원 약 29.3 MiB, 1024차원 약 39.1 MiB다.
모델·원문·Python 객체와 임시 행렬은 별도 메모리가 필요하다.

10k subset의 순차 실행 결과로, 전체 MIRACL 성능이나 API 응답 시간을 뜻하지 않는다.
독립적인 전체 실험 반복은 수행하지 않았다. 측정 범위와 한계는 [EVALUATION.md](EVALUATION.md)를 참조한다.

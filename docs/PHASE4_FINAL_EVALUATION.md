# 4차(Phase 4) 최종 평가

#25 종료 문서. 새 실험·튜닝·provider 호출 없이, 이미 merge된 #28·#30·#31의 결과를 한 곳에 정리한다.
각 영역의 canonical 출처는 [BC_CARD_DATASET.md](BC_CARD_DATASET.md), [ADAPTIVE_RESULTS.md](ADAPTIVE_RESULTS.md),
[COMMERCIAL_RAG_BENCHMARK.md](COMMERCIAL_RAG_BENCHMARK.md)이며, 숫자는 그대로 가져오고 중복 기록하지 않는다.

## Primary held-out test — non-synthetic (721)

| 방식 | P@1 | Recall@10 | MRR@10 | nDCG@10 | 비고 |
| --- | ---: | ---: | ---: | ---: | --- |
| BM25 | 0.4189 | 0.7406 | 0.5232 | 0.5755 | |
| Dense (BGE-M3) | 0.6560 | **0.8988** | 0.7460 | 0.7835 | |
| Fixed Score Fusion | **0.6852** | 0.8946 | **0.7615** | **0.7940** | 서비스 기본값 |
| Adaptive Hybrid (#24) | 0.6838 | 0.8918 | 0.7608 | 0.7930 | latency Fixed의 약 2.43배 |
| Oracle | 0.7171 | 0.9071 | 0.7879 | 0.8171 | 분석 상한, 서비스 선택기 아님 |
| NAVER Cloud RAG (#26) | | | | | `READY / BLOCKED, NOT RUN`, 0/1,000 queries |
| Upstage File Search (#26) | | | | | `EXCLUDED`, chunk ID mapping 불가 |

출처: [BC_CARD_DATASET.md 베이스라인](BC_CARD_DATASET.md#held-out-test--primary-non-synthetic-721),
[ADAPTIVE_RESULTS.md primary test](ADAPTIVE_RESULTS.md#primary-test--non_synthetic-721).

## 결론

- 3차(Ko-MIRACL/Ko-StrategyQA)의 Dense 중심·Fixed Score Fusion 결론은 BC Card 기업 도메인에서도 유지된다.
  Fixed가 Dense보다 P@1·MRR@10·nDCG@10에서 소폭 높고, Recall@10은 Dense가 소폭 높다.
- Adaptive Hybrid는 validation에서 query별로 더 나은 방식이 갈리는 Oracle headroom을 확인했다
  (Oracle − Fixed nDCG@10 +0.031, `ADAPTIVE_ROUTER_HEADROOM: MODERATE`).
- 하지만 test 1,000개 전부에서 router가 `balanced`를 선택했고, primary nDCG@10은 Fixed보다 -0.001,
  검색 latency는 Fixed의 약 2.43배다. 서비스 기본값은 Fixed Score Fusion을 유지한다.
- 이는 "Adaptive routing이 불가능하다"가 아니라, **현재 lightweight router 구현이 확인된 headroom을 회수하지 못했다**는
  결과다. 범위는 이번 구현에 한정한다.
- 상용 RAG는 평가 harness와 NAVER feasibility까지 구현했지만 약관 미확인·credential 없음·비용 unknown으로 live 실행은
  하지 않았다(`0/1,000` queries). commercial 결과도 로컬과의 성능 비교도 존재하지 않는다. Upstage는 chunk ID
  mapping 문제로 protocol 밖(`EXCLUDED`)이다.

## BC Card parent-document quick check

팀 내부에서 BC Card chunk를 parent document로 단순 합쳐본 exploratory 확인이 있었으나 정식 benchmark로 보존되지
않았다. 임시 숫자를 canonical 결과에 올리지 않는다. 확인된 것은 다음 정도다.

- 단순 parent reconstruction만으로는 충분한 long-document 환경이 만들어지지 않을 가능성이 있다.
- test 정답 parent 중 1,024 tokens 이상은 매우 적어, 이 데이터로 long-document benchmark를 일반화하지 않는다.

## 후속 검토 방향

확정된 implementation plan은 아니며, 합의된 다음 검토 방향만 남긴다.

1. Paper-faithful QDAP-S 구현
2. 실제 long-document retrieval 환경에서 평가
3. 필요 시 QDAP-L / context-aware routing
4. Domain fine-tuning은 별도 축

## 재현

- BC Card baseline / Adaptive: [BC_CARD_DATASET.md Reproduction](BC_CARD_DATASET.md#reproduction).
- 상용 RAG harness(오프라인 evaluate만, provider 호출 없음): [COMMERCIAL_RAG_BENCHMARK.md](COMMERCIAL_RAG_BENCHMARK.md) 재현 절차.

## 검증

- 새 model run·provider API 호출·외부 업로드 없음. 문서 정리만.
- `uv run --locked pytest`, `ruff check .`, `ruff format --check .`, `git diff --check` 통과.

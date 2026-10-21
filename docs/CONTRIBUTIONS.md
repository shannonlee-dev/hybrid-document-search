# 팀원별 기여

집계일: 2026-10-20 (한국 시간)
대상: `feat/c-final-integration`의 `eca59c63a30d8479da268a797cf599cd46824bf4`

해당 커밋까지의 전체 이력에서 merge 커밋을 제외한 98개를 작성자 기준으로 집계했다.
미커밋 변경은 포함하지 않으며, 최종 제출 시 병합된 `main`의 SHA로 갱신한다.

| 팀원 / 작성자 | 주요 기여 | 커밋 수 | 대표 커밋 |
| --- | --- | ---: | --- |
| shannonlee-dev | 프로젝트 초기 구성, Dense·FAISS, 모델 비교, score fusion | 52 | `4180802` Dense 구현 · `9117748` 재현 가능한 실험 · `8e08cc7` score fusion |
| VectorSophie | RRF, 검색 API·서비스, Streamlit UI, 최종 통합 | 26 | `4d738d5` RRF · `7f7631c` API · `31e5c81` UI · `59f5868` 최종 통합 |
| 반가희 / bangahee | 데이터 준비, TF-IDF·BM25, 평가 지표·파이프라인, 최종 평가 | 20 | `ded6046` 데이터 준비 · `9b0aef7` BM25 · `e6251aa` 평가 파이프라인 · `3f85a76` 최종 평가 |

`반가희` 15개와 `bangahee` 5개는 동일 작성자 이메일을 기준으로 합산했다.
커밋 수는 활동량의 참고 지표이며, 작업 난이도·리뷰·논의까지 포함한 기여도 비율을 뜻하지 않는다.

## 증빙과 재현

- [전체 커밋 로그](validation/2026-10-20/git-log.txt): 전체 SHA, 작성 시각, 작성자, 작업 제목
- [작성자별 집계](validation/2026-10-20/git-shortlog.txt): 별칭 합산 전 원본

저장소 루트에서 실행한다. 대표 커밋의 변경 내용은 `git show <커밋>`으로 확인한다.

```bash
revision=eca59c63a30d8479da268a797cf599cd46824bf4
git log --no-merges --date=iso-strict \
  --format='%H | %ad | %an | %s' "$revision" \
  > docs/validation/2026-10-20/git-log.txt
git shortlog -sn --no-merges "$revision" \
  > docs/validation/2026-10-20/git-shortlog.txt
```

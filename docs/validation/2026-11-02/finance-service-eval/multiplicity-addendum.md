# Co-primary 다중 비교 보정 addendum

작성: 2026-11-02. 원래 사전 등록 이후, 보정 계산 전에 고정함. 원래 95% CI와 판정은 그대로 두고 보정 결과를 따로 덧붙임.

## 왜 필요한가

원래 사전 등록은 co-primary A(서비스 Hybrid .05)와 B(Fixed α=.84)를 각각 95% CI로 판정하고 다중 비교 보정은 하지 않기로 했음. 두 비교는 같은 BC Card Test 721개 query와 같은 encoder 쌍을 써서 서로 독립이 아님. 각각의 95% CI가 0을 넘는 것만으로는 "적어도 하나가 개선"이라는 주장의 family-wise error를 5%로 묶지 못함. 이 결과는 이미 알고 있으므로 이 addendum은 확증 분석이 아니라 기존 결론의 강도를 점검하는 용도임.

## Family와 데이터

- Family: A와 B의 nDCG@10 차이(Finance − BGE) 두 개. Secondary 조건과 다른 지표는 포함하지 않음.
- 데이터: 커밋된 `bccard-per-query.jsonl`만 씀. 새 검색이나 인코딩은 없음.
- Bootstrap: seed 20261102, 20,000 resamples, query 단위. 두 비교에 같은 resample index를 씀(joint resampling).

## 고정한 방법과 판정

1. Bonferroni(주 방법): 비교마다 양측 97.5% percentile CI(2.5% / 2 = 1.25%, 98.75% 분위수). CI 하한 > 0이면 해당 co-primary의 개선을 FWER 5%에서 주장할 수 있음.
2. Joint bootstrap max-|t|(보조): resample b마다 `z_bk = (m_bk − m̂_k) / se_k`를 계산하고, `c = max_k |z_bk|`의 95% 분위수로 simultaneous interval `m̂_k ± c·se_k`를 만듦. `se_k`는 bootstrap 표준편차. 하한 > 0이면 개선을 주장할 수 있음.
3. 두 co-primary가 모두 개선됐다는 결합 주장은 intersection-union 원리상 각 비교의 5% 검정이 모두 기각되면 성립하므로 원래 95% CI로 판정함. 이 경우 보정이 필요 없다는 점도 같이 적음.

1과 2의 결론이 다르면 더 약한 쪽을 보고서 결론으로 씀. 보정 후 결론이 약해져도 그대로 보고함.

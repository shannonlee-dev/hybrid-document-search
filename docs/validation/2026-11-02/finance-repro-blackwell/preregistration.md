# Pre-registered reproduction criteria (written before training and before any new Validation/Test result)

Run: finance-repro-20261102-b6f8649. Code: PR #40 head b6f8649a8dc7a8cbb580926ba2cc08cbee9358df (clean detached worktree).
Inputs: original WSL2 cache (all five pinned hashes verified), BC Card data manifest b6bb7822…, BGE-M3 5617a9f6 (10 pinned file hashes).

## References
- Seed43: PR #40 `finance-retrain/protocol.json` (history; checkpoint d36f5c66…; Validation Fixed α=.83 / .8315689697637983).
- Seed42: `c95c3fb:docs/validation/2026-11-01/finance/` (history; checkpoint 361e09ee…; Validation Fixed α=.82 / .8298404923658383, which was computed against an FP32 baseline cache). Its training code identity is not recorded there, which makes this a weaker reference.
- Matched BF16 baseline: Validation Fixed α=.84 / .8056238001224711. Test (α=.84) P@1 .6768377253814147, R@10 .8932038834951457, MRR@10 .7576492085507343, nDCG@10 .79093193891841.
- Seed43 Test: α=.84 P@1 .6934812760055479, R@10 .9223300970873787, MRR@10 .7792869251260375, nDCG@10 .8145230714883046; α=.83 P@1 .694868238557559, R@10 .9223300970873787, MRR@10 .7798763842106422, nDCG@10 .8149500101779285.
- No historical Test reference exists for Seed42 under the PR #40 matched protocol.

## Levels and tolerances
1. **Exact-input reproduction**: dataset, model and cache hashes all match. (Already satisfied; this alone does not establish training reproduction.)
2. **Bitwise training reproduction**: the selected checkpoint SHA-256 equals the reference. Not expected across GPUs; a mismatch is not by itself a failure.
3. **Numerical training reproduction (A)**, per seed: |Δ| ≤ 0.005 Validation primary Dense nDCG@10 at every scheduled checkpoint, **and** the same selected step. Step 0 must match to |Δ| ≤ 0.001, since it depends only on the base model and BF16 encoding.
   - "Approximate": every |Δ| ≤ 0.010 and the same selected step.
   - Otherwise "not reproduced".
4. **Model-selection reproduction (B)**: the same selected step per seed, the same Validation-selected α per encoder (exact grid point), and the same final Validation-selected encoder. Baseline α must also match (.84). Any difference is reported as a selection non-reproduction regardless of Test metrics. The Seed42 α is compared to the c95c3fb value with the FP32-baseline caveat.
5. **Retrieval-performance reproduction (C)**: Test metrics for the baseline and for Seed43 at both α policies.
   - Baseline: |Δ| ≤ 0.002 for each metric. It depends only on BF16 encoding of the original model.
   - Adapted encoders: |Δ nDCG@10| ≤ 0.005 and |Δ MRR@10| ≤ 0.005; |Δ P@1| ≤ 0.007 and |Δ Recall@10| ≤ 0.007 (about 5/721).
   - "Approximate": nDCG@10 within 0.010.
   - If the Validation-selected final encoder differs from seed43, C is assessed for seed43 (like-for-like), and the selected encoder is reported separately.
6. **Direction (D)**: the finance encoder beats the matched BF16 baseline at shared α=.84 on Test nDCG@10, with per-query win/tie/loss counts. Reference gain: +0.023591. "Still outperforms" requires a positive gain for the Validation-selected encoder; magnitude is reported against the reference.

## Overall classification
- **Exact reproduction**: bitwise checkpoints plus identical selections and identical metrics.
- **Numerical reproduction**: A, B and C all within tolerance, with non-identical hashes.
- **Partial reproduction**: some of A/B/C within tolerance, others not, with D holding.
- **Failed reproduction**: D fails, or C is far outside even the approximate tolerance.
- **Inconclusive**: integrity failures prevent assessment.

## Procedure order (deviation from the PR doc sequence is ordering only; no algorithm change)
train(42) ∥ train(43) → prepare scores → baseline (BF16 Validation and Test encoded directly; no --test-source) → select (training dirs) → freeze (hash and read-only) → checkpoint(42, 43) → evaluate → export → verify.

Loss logging: the CLI overwrites `progress.json` every 64 steps. A thin launcher calls `scripts.train_query_encoder.main` unchanged and additionally appends each `progress.json` write to `progress-log.jsonl`. It is observation only: no change to RNG, data or numerics.

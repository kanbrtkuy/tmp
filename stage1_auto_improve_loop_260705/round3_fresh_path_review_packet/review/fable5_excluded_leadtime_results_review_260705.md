# Fable-5 Review: Round 2 Excluded-Source Lead-Time Confirmation

Reviewer: `claude-fable-5`

Verdict:

```text
drop_leadtime_claim is correct.
Stage1 should now be treated as a negative/control result.
No statistical or implementation blockers were found.
```

## Key Verification

- Independent recomputation from sanitized prediction JSONLs reproduced:
  - A1 per-source hidden@4 AUROCs: `0.691544`, `0.717959`
  - A1 per-source text@8 AUROCs: `0.717721`, `0.703462`
  - pooled A1 delta: `+0.001553`
  - A2 pooled delta: `-0.065350`
- Pair integrity passed:
  - exactly one positive and one negative row per pair;
  - frozen pair_id set identical across all 5 k values, both arms, and A2
    predictions;
  - zero val/test ID overlap;
  - manifest row counts match diagnostics;
  - `n_errors = 0`;
  - clean code commit `d658ca8`.
- Gate logic matches the preregistration:
  - A1 CI-low >= 0;
  - per-source point >= -0.02;
  - A2 CI-high >= 0 and point >= -0.01.

## Final Interpretation

The preregistered result is not a near miss:

- A1 primary hidden@4 minus text@8: delta `+0.0016`, CI
  `[-0.0138, +0.0173]`, effectively null.
- Per-source sanity fails:
  - `strongreject_full`: delta `-0.026`, CI `[-0.050, -0.003]`
  - `reasoningshield`: delta `+0.014`, CI spans zero
- A2 robustness fails decisively:
  - delta `-0.065`, CI entirely negative.
- Descriptive lead-time cells are null or negative in the useful early-warning
  direction.

Per the preregistered rule `a1_fail -> drop_leadtime_claim`, Fable-5 confirmed
that `confirmed=false / decision=drop_leadtime_claim` is the only defensible
reading.

## Minor Flags

No blockers. Minor notes:

- Some cells have independent bootstrap draws in different files, so tiny CI
  endpoint differences are expected with `B=500`; not material here.
- Validation metrics are optimistic relative to test, but this does not weaken
  the negative conclusion.
- A1 k=8 remains a descriptive equal-horizon observation only; the top-level
  prereg rule governs.
- Pair-clustered bootstrap was assumed from the prior code review rather than
  re-verifiable from aggregate outputs alone.

## Bottom Line

Stage1 is closed as a negative/control result for this dataset and probe design.
Further re-analysis of the frozen test set to find a positive cell would be
post-hoc fishing. The only admissible future positive path would be a fresh,
separately preregistered dataset/setting, most naturally Stage2/on-policy or a
new fresh-data follow-up motivated by the short-prefix k=4 diagnostic.

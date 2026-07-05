# Fable-5 Round 2 Response

Verdict: `OK_TO_RUN`

Fable-5 verified the patched packet at tmp commit `ec76422`.

Resolved blockers:

1. Module T Platt CI:
   - `add()` now accepts `ci_rows`.
   - `platt_0p5` test CI now passes calibrated rows (`test_cal`), so the
     bootstrap matches the Platt point estimate.
   - The new offset-score test discriminates old vs. new behavior.

2. Module M post-alignment pair completeness:
   - `enforce_pair_complete_alignment` is applied to val and test after hidden
     prediction alignment.
   - Hidden/surface arms end with identical ids and orphaned pairs are dropped
     from both.
   - The new missing-prediction fixture checks that a dropped hidden row also
     drops its mate.

Non-blocking caveats to carry into the writeup:

- New post-alignment drop columns are the authoritative counts; older
  `alignment_*_dropped` fields can be vestigial after pre-alignment.
- Pair loss from id-alignment and pair loss from post-alignment completeness are
  reported in different fields; define this in the writeup.
- Threshold CIs condition on the observed calibration/threshold.
- Platt is sklearn's regularized LogisticRegression policy.
- Module M Holm family mixes per-source and pooled rows, which is conservative.
- A cheap pair-id consistency assert could be added later.
- Bootstrap seed shifts are a reproducibility quirk only.

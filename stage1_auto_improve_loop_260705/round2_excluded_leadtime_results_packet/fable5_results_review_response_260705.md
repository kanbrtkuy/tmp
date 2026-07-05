# Fable-5 Review: Round 2 Excluded-Source Lead-Time Confirmation

All checks are complete and everything reproduces. Here is my review.

## Verification performed (sanitized outputs only)

- Independent recomputation from prediction JSONLs (scores/labels/metadata
  only, no raw content): per-source A1 hidden@4 AUROCs (0.691544, 0.717959),
  text@8 AUROCs (0.717721, 0.703462), pooled A1 delta (+0.001553 under
  per-source val-stat z-normalization, matching `zmean_cumulative_val_stats`),
  and A2 pooled delta (-0.065350) all reproduce the reported values to 6
  decimals.
- Integrity checks passed: every pair has exactly one pos + one neg row; the
  frozen pair_id set (277/335) is byte-identical across all 5 k's, both arms,
  and the A2 predictions; zero val/test ID overlap; scores non-degenerate;
  manifest row counts (554/670 test; 1862/1758 train; 208/196 val) match
  diagnostics everywhere; `n_errors = 0`; clean commit `d658ca8`.
- Gate logic matches the prereg text exactly (A1 CI-low >= 0; per-source point
  >= -0.02; A2 CI-high >= 0 and point >= -0.01).

## Q1 - Is `drop_leadtime_claim` correct?

Yes.

The preregistered decision is correct, and it is not a near-miss:

- A1 primary: pooled delta +0.0016, CI [-0.0138, +0.0173], p approximately
  0.85, a precise null. Hidden@4 buys nothing over text@8.
- Per-source sanity fails with sign heterogeneity: strongreject_full is
  significantly negative (-0.026, CI [-0.050, -0.003]); reasoningshield weakly
  positive (+0.014, CI spans 0).
- A2 fails decisively: -0.065, CI entirely negative. The effect is absent and
  recipe-fragile.
- The full descriptive matrix corroborates: every lead cell (hidden@k vs
  text@k' > k) is null or negative. Hidden@8 vs text@16: -0.012; hidden@16 vs
  text@32: -0.019 (CI < 0); hidden@32 vs text@64: -0.042. There is no
  lead-time signal anywhere, at any horizon.

Per the preregistered interpretation rule (`a1_fail -> drop the lead-time
claim`), and with all three gates failing, `confirmed=false /
drop_leadtime_claim` is the only defensible reading. Even an equivalence
reframing ("hidden@4 as good as text@8") would not survive: it was not
preregistered and would need TOST with a margin, and strongreject is
significantly negative.

## Q2 - Statistical/implementation concerns?

No blockers. Four minor flags:

1. Two independent bootstrap draws per cell: `stage1_score_pooling_summary.tsv`
   and `stage1_score_pooling_lead_time_matrix.tsv` report slightly different
   CIs for identical cells. With B=500, endpoint Monte Carlo jitter is about
   +/-0.002 to 0.005, immaterial here since every gate fails by more than that.
   For any future near-threshold gate, use one canonical draw and B >= 2000.
2. Val is systematically optimistic vs test. Val is used only for
   z-normalization stats here, and any residual bias favors the hidden arm, so
   the negative conclusion is conservative.
3. Mild prereg tension: the A1 sub-prereg defines `holm_family=[8]`, and the
   pooled k=8 equal-horizon cell is Holm-significant, while the top-level
   prereg demotes all non-primary cells to descriptive-only. The top-level rule
   governs; be transparent that this cell exists and is descriptive.
4. Pair-clustered bootstrap is assumed, not verifiable from aggregates; this
   relies on the prior `af3d41f` code review.

Manifest gate, frozen-population rule, and gate arithmetic are all clean and
internally consistent.

## Q3 - Negative/control status?

Yes. Treat Stage1 as negative/control.

Both preregistered confirmations (equal-horizon and excluded-source lead-time)
have failed; A2 shows the recipe does not transfer to feature-level pooling;
the descriptive matrix shows no rescueable region.

Non-fishing follow-ups are limited to:

- Report heterogeneity per the prereg rule: no pooled-only headline;
  strongreject significantly negative, reasoningshield null-positive.
- Use Stage1 as the negative-control baseline for Stage2 designs.
- The descriptively interesting pattern, hidden's equal-horizon edge at very
  short prefixes (k=4 positive, k=8 smaller, inverting by k=64), may motivate a
  new preregistration on fresh data, but cannot be claimed from this run.

Any further re-analysis of this frozen test set to find a positive cell is
multiplicity laundering.

## Q4 - Threshold/balanced-accuracy reporting?

Yes, as descriptive operating-point analysis, with constraints:

- thresholds chosen on val only and frozen before test;
- reported per-source with pair-bootstrap CIs;
- labeled operating-point/descriptive, with the headline AUROC conclusion
  staying negative;
- report achieved test operating point, not val-promised operating point;
- no new comparative claim without fresh preregistration.

## Bottom line

Implementation is sound, numbers reproduce exactly, gates were applied
faithfully, and `drop_leadtime_claim` is correct. Stage1 should now be written
up as a negative/control result with per-source heterogeneity, closing the
lead-time question on this dataset; the short-prefix equal-horizon observation
is the only candidate for a future, freshly preregistered follow-up.

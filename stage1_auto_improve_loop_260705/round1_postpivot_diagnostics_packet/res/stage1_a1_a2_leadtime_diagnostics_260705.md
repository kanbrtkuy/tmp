# Stage1 A1/A2 Lead-Time Diagnostics

Date: 2026-07-05

Scope: descriptive diagnostics only. No new fits, no new pooling rules, and no
rescue variant after the preregistered A2 failure branch.

## Inputs

- A1 score pooling:
  `tmp/stage1_auto_improve_loop_260705/round1_a1_results_packet/results/`
- A2 feature pooling:
  `tmp/stage1_auto_improve_loop_260705/round1_a2_results_packet/results/`
- Fable-5 stop/pivot review:
  `review-stage/stage1_auto_improve_260705/AUTO_REVIEW.md`

Both A1 and A2 use the same primary sources, k-grid, layer-28 hidden signal,
matched-horizon `char_tfidf` surface scores, and paired bootstrap count
`B=500`. They differ in the hidden-side estimator:

| Arm | Hidden estimator | Fit status |
|---|---|---|
| A1 | unweighted mean of validation-z-scored per-position hidden probe scores | reuses frozen hidden scores from the original Stage1 pipeline |
| A2 | unweighted mean of layer-28 vectors over `cot_j`, `j<=k`, then refit linear probe per k | `StandardScaler + LogisticRegression(class_weight=balanced)`, train split only |

## Same-Horizon Summary

| k | A1 hidden AUROC | A1 text AUROC | A1 delta | A2 hidden AUROC | A2 text AUROC | A2 delta |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.7882 | 0.7323 | +0.0558 | 0.7364 | 0.7323 | +0.0041 |
| 8 | 0.7844 | 0.7570 | +0.0274 | 0.7335 | 0.7570 | -0.0235 |
| 16 | 0.7983 | 0.8018 | -0.0035 | 0.7589 | 0.8018 | -0.0428 |
| 32 | 0.8138 | 0.8077 | +0.0061 | 0.7646 | 0.8077 | -0.0430 |
| 64 | 0.8289 | 0.8495 | -0.0206 | 0.7932 | 0.8495 | -0.0564 |

Interpretation:

1. A1's k=8 positive result does not survive A2.
2. A2 hidden AUROC is still above chance, but matched-horizon text wins for
   every preregistered confirmatory k in `{8,16,32,64}`.
3. At k=4, A1 and A2 both use only `cot_4`, but A1 hidden AUROC is `0.7882`
   while A2 hidden AUROC is `0.7364`. This is a recipe-strength gap, so the
   correct caveat is "not robust to estimator/pooling recipe," not "feature
   pooling destroys signal."

## Lead-Time Matrix: A1

Rows are hidden@k, columns are text@k'. Values are delta AUROC
`hidden@k - text@k'`.

| hidden \\ text | 4 | 8 | 16 | 32 | 64 |
|---:|---:|---:|---:|---:|---:|
| 4 | +0.0558 | +0.0311 | -0.0136 | -0.0195 | -0.0614 |
| 8 | +0.0521 | +0.0274 | -0.0174 | -0.0233 | -0.0651 |
| 16 | +0.0659 | +0.0413 | -0.0035 | -0.0094 | -0.0513 |
| 32 | +0.0814 | +0.0567 | +0.0120 | +0.0061 | -0.0358 |
| 64 | +0.0966 | +0.0719 | +0.0272 | +0.0213 | -0.0206 |

A1 supported an exploratory early-warning story: hidden scores at shorter
horizons could match or exceed text at some later horizons. This is now
diagnostic only because A2 did not confirm the same lead-time pattern.

## Lead-Time Matrix: A2

Rows are hidden@k, columns are text@k'. Values are delta AUROC
`hidden@k - text@k'`.

| hidden \\ text | 4 | 8 | 16 | 32 | 64 |
|---:|---:|---:|---:|---:|---:|
| 4 | +0.0041 | -0.0206 | -0.0653 | -0.0712 | -0.1131 |
| 8 | +0.0012 | -0.0235 | -0.0682 | -0.0741 | -0.1160 |
| 16 | +0.0266 | +0.0019 | -0.0428 | -0.0487 | -0.0906 |
| 32 | +0.0323 | +0.0076 | -0.0371 | -0.0430 | -0.0849 |
| 64 | +0.0609 | +0.0362 | -0.0086 | -0.0145 | -0.0564 |

A2 does not support the A1 lead-time headline. The strongest positive lead
cells are hidden at much later horizons compared against text at earlier
horizons, which is the wrong direction for a useful early-warning claim.

## Key Findings

1. **Equal-horizon result is closed negative.** A2 is valid and fails the
   preregistered k=8 gate. No further equal-horizon probe variants are allowed.
2. **A1 remains useful only as a diagnostic.** It shows that the original
   probe-score ensemble can expose a signal, but the advantage is not robust
   enough to support a main claim.
3. **Text catches up and overtakes.** A2 text AUROC rises from `0.7323` to
   `0.8495`, while A2 hidden rises from `0.7364` to `0.7932`.
4. **Recipe-strength caveat matters.** The k=4 A1/A2 gap (`0.7882` vs
   `0.7364`) means A2 is not a pure pooling-only ablation.
5. **Allowed claim scope is narrow.** Hidden states encode safety-relevant
   signal, but teacher-forced matched-horizon monitoring does not show a
   hidden-over-text information advantage.

## Suggested Next Step

Do not run any new equal-horizon variant. The only Stage1 experiment that
Fable-5 left open is an optional, separately preregistered confirmation of the
exploratory k=4 lead-time diagnostic on excluded sources (`strongreject_full`,
`reasoningshield`), running both A1 and A2 recipes and accepting either
outcome. That confirmation is only warranted if the paper needs lead-time as a
non-exploratory claim; otherwise Stage1 should be documented as a negative
control and the project should shift emphasis to causal/on-policy settings.

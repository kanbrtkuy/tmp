# Stage1 A2 Feature Pooling Result And Pivot

Date: 2026-07-05

## Status

Fable-5 reviewed the preregistered A2 feature-level cumulative pooling result
and returned:

`STOP_EQUAL_HORIZON_AND_PIVOT`

The A2 run is valid and the preregistered failure branch fired. Equal-horizon
probe-variant iteration is now closed.

## Artifacts

- Code commit: `d26d03c`
- RunPod output:
  `/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/feature_pooling_a2_260705_b500`
- R2 backup:
  `cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/stage1_post_hb_260705_after_hb_n100_loso/feature_pooling_a2_260705_b500/`
- tmp results packet: `5c1b2de`
- Fable-5 review log:
  `review-stage/stage1_auto_improve_260705/AUTO_REVIEW.md`

## A1 vs A2

A1 was a score-level cumulative pooling preview. A2 was the preregistered
feature-level rerun: mean-pool layer-28 vectors over `cot_j` positions where
`j <= k`, refit a linear probe per k, and compare against the unchanged
matched-horizon `char_tfidf` text@k scores.

| k | A1 hidden AUROC | A1 delta | A1 95% CI | A2 hidden AUROC | A2 delta | A2 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.7882 | +0.0558 | [+0.0452, +0.0668] | 0.7364 | +0.0041 | [-0.0082, +0.0158] |
| 8 | 0.7844 | +0.0274 | [+0.0183, +0.0366] | 0.7335 | -0.0235 | [-0.0350, -0.0123] |
| 16 | 0.7983 | -0.0035 | [-0.0121, +0.0061] | 0.7589 | -0.0428 | [-0.0547, -0.0315] |
| 32 | 0.8138 | +0.0061 | [-0.0035, +0.0149] | 0.7646 | -0.0430 | [-0.0533, -0.0323] |
| 64 | 0.8289 | -0.0206 | [-0.0298, -0.0120] | 0.7932 | -0.0564 | [-0.0667, -0.0448] |

A2's preregistered failure condition was `k=8 CI low < 0`. The observed k=8
delta was `-0.0235`, CI `[-0.0350, -0.0123]`, so the failure branch fired.
For k in `{8,16,32,64}`, pooled Holm p-values reported as `0.0` should be
written as `p < 0.002` because the bootstrap count is `B=500`.

## Interpretation

The A1 k=8 advantage was not confirmed by the feature-level rerun. The honest
reading is not "hidden states contain no safety signal": A2 hidden AUROCs are
still well above chance, from `0.736` to `0.793`. The negative result is
relative: at equal horizon under teacher forcing, hidden probes do not beat the
matched text baseline.

The key caveat is recipe strength. At k=4 both A1 and A2 use only `cot_4`, but
A1's original probe pipeline scored `0.7882` while A2's preregistered refit
scored `0.7364`. Therefore the A2 reversal cannot be described as "feature
pooling destroys signal." The supported claim is narrower: the equal-horizon
advantage is not robust to the probe estimator and pooling recipe.

Text evidence accumulates faster than the pooled hidden readout:

- Text AUROC: `0.7323 -> 0.8495` from k=4 to k=64.
- A2 pooled hidden AUROC: `0.7364 -> 0.7932`.

This is consistent with the data-processing inequality intuition under
teacher forcing: hidden state at a prefix is a function of the same prefix, and
the text baseline eventually reads the model's own verbalized decision.

## Closed And Allowed

Closed:

- No A2b rescue variant.
- No new equal-horizon pooling rules, classifiers, layers, hyperparameter
  searches, surface-family swaps, k-grid changes, re-splits, or
  re-normalization.
- Do not report A1 without A2, or A2 without A1.

Allowed:

- Document A1 and A2 side by side.
- Treat A1 lead-time evidence as exploratory and recipe-sensitive.
- Publish the A1 and A2 lead-time matrices as descriptive diagnostics.
- Optional future work: if the paper needs a stronger lead-time claim, run one
  preregistered confirmation on the excluded sources (`strongreject_full`,
  `reasoningshield`) with both recipes and accept either outcome.

## Reporting Language

Use Fable-5's recommended formulation:

> Equal-horizon comparison (preregistered, final). On teacher-forced natural
> pairs (harmbench_standard n=152, wildjailbreak_vanilla_harmful n=2019 test
> pairs), linear probes on layer-28 hidden states do not outperform a
> matched-horizon character TF-IDF baseline. A score-pooled analysis (A1)
> showed an advantage at k=8 (delta AUROC +0.027, 95% CI [+0.018, +0.037]),
> but the preregistered feature-level rerun (A2: cumulative mean of layer-28
> vectors over positions j<=k, probe refit per k, frozen evaluation
> population) reversed it: k=8 delta = -0.024, CI [-0.035, -0.012]; k=64
> delta = -0.056, CI [-0.067, -0.045]; Holm-corrected p < 0.002. Per the
> pre-declared decision rule we conclude the A1 advantage was
> pooling-scheme-specific and terminate equal-horizon probe iteration. Text
> accuracy grows with horizon faster than pooled hidden readout, consistent
> with the data-processing inequality under teacher forcing. An early-horizon
> signal (hidden@4 ≈ text@16-32 in absolute AUROC under the original probe
> pipeline, delta@k4 +0.056; attenuated to +0.004, n.s., under the
> preregistered refit recipe) is reported as an exploratory diagnostic only.
> The motivation for hidden-state methods in this project accordingly rests on
> causal utility and on-policy settings, not on monitoring advantage in this
> teacher-forced equal-horizon regime.

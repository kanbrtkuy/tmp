# Stage1 CPU Reanalysis: Threshold And Matched Horizon

Date: 2026-07-05

This run follows the Fable-5 reviewed CPU-only plan for Stage1 operating-point
and matched-horizon checks. No GPU was used for this reanalysis.

## Code And Review

- `cot-safety` commit: `1d30c40`
- `tmp` review packet commits:
  - `3ed3c78`: initial code packet
  - `ec76422`: blocker fixes
  - `47267cd`: Fable-5 `OK_TO_RUN` record
- Fable-5 verdict: `OK_TO_RUN`

## Run Paths

RunPod:

```text
/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/cpu_reanalysis_threshold_matched_horizon_260705_b500/
```

Cloudflare R2:

```text
cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/stage1_post_hb_260705_after_hb_n100_loso/cpu_reanalysis_threshold_matched_horizon_260705_b500/
```

R2 verification:

```text
Total objects: 45
Total size: 13.449 MiB
```

Bootstrap count: `500`. A previous `2000` bootstrap threshold attempt was
stopped because the pure-Python bootstrap was too slow for an interactive pass.

## Module T: Threshold/Calibration Reanalysis

Output:

```text
threshold/stage1_threshold_reanalysis.tsv
threshold/stage1_threshold_reanalysis.json
```

Summary over hidden test rows:

| Policy | n | Mean balanced accuracy | Range | Mean recall | Mean FPR |
|---|---:|---:|---:|---:|---:|
| current_prediction | 16 | 0.6037 | 0.5149-0.6777 | 0.2885 | 0.0811 |
| platt_0p5 | 16 | 0.7096 | 0.6388-0.7632 | 0.7875 | 0.3682 |
| val_ba_max | 16 | 0.7080 | 0.6418-0.7533 | 0.7982 | 0.3821 |
| test_score_median_transductive | 16 | 0.6961 | 0.6418-0.7500 | 0.6966 | 0.3044 |
| oracle_test_ba_max | 16 | 0.7206 | 0.6582-0.7763 | 0.8572 | 0.4160 |

Interpretation:

- Yes, thresholded hidden balanced accuracy can improve materially, from about
  `0.60` to about `0.71`.
- This is mainly an operating-point correction. The original threshold was
  conservative: low FPR but also low recall.
- The improvement does not change AUROC, and it does not by itself show that
  hidden probes beat surface baselines.

For context, selected surface and length controls remain much stronger at the
operating-point level:

- `surface_selected` mean test BA is about `0.865`.
- `length_only` mean test BA is about `0.801`.

## Module M: Matched-Horizon Reanalysis

Output:

```text
matched_horizon/stage1_matched_horizon_summary.tsv
matched_horizon/stage1_matched_horizon_residual.tsv
matched_horizon/stage1_matched_horizon_summary.json
matched_horizon/predictions/
```

Scope:

- Sources: `harmbench_standard`, `wildjailbreak_vanilla_harmful`
- Hidden kind: `linear`
- Tokenizer: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- k grid: `4,8,16,32,64`
- Global selected layer: `28`
- Global selected surface family: `char_tfidf`

Primary equal-horizon AUROC deltas, hidden minus matched surface:

| Source | k=4 | k=8 | k=16 | k=32 | k=64 |
|---|---:|---:|---:|---:|---:|
| harmbench_standard | +0.0470 | -0.0293 | -0.1395 | -0.0595 | -0.1083 |
| wildjailbreak_vanilla_harmful | +0.0572 | -0.0038 | -0.1630 | -0.1015 | -0.1539 |
| pooled | +0.0584 | -0.0046 | -0.1580 | -0.0964 | -0.1499 |

Interpretation:

- Equal-horizon comparison helps answer the fairness objection.
- At `k=4`, hidden has a small positive advantage over same-prefix text.
- From `k=8` onward, the matched text baseline catches up or wins, especially
  at `k=16+`.
- This supports treating the current Stage1 result as a negative/control result:
  the hidden prefix signal is not strong enough to beat same-horizon surface
  text controls across the main k range.

Residual/E3:

- Validation-stacker residual is secondary evidence only.
- Hidden adds small residual gains at early k:
  - pooled-like pattern by source is about `+0.04` to `+0.055` AUROC/log-loss
    improvement at `k=4`.
  - residual AUROC gains shrink to near zero by `k=16+`.
- Because hidden train/OOF scores are unavailable, this is not an OOF stacker.

## Caveats

- Bootstrap CIs use `500` samples and condition on the observed selected
  threshold/calibrator.
- Across-k curves are descriptive because pair-complete censoring changes the
  retained population as k grows.
- The `sentence_encoder` family was not run because `sentence_transformers` is
  not installed in the RunPod environment.
- Full-trajectory surface baselines remain hindsight ceilings, not equal-horizon
  primary controls.

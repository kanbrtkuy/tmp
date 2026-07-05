# Stage1 Auto-Improve Round 1 A2 Results Packet

Purpose: Fable-5 result review after the pre-registered A2 feature-level
cumulative pooling run.

This packet contains aggregate outputs only:

- `results/stage1_feature_pooling_summary.json`
- `results/stage1_feature_pooling_summary.tsv`
- `results/stage1_feature_pooling_lead_time_matrix.tsv`
- `results/stage1_feature_pooling_split_diagnostics.tsv`
- `results/stage1_feature_pooling_fit_diagnostics.tsv`
- `results/stage1_feature_pooling_preregistration.json`
- `review/AUTO_REVIEW.md`

It intentionally excludes prediction JSONL rows, raw prompts, raw CoTs, hidden
activation arrays, and generated pair files.

## Run

- Code commit: `d26d03c`
- RunPod output:
  `/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/feature_pooling_a2_260705_b500`
- R2 backup:
  `cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/stage1_post_hb_260705_after_hb_n100_loso/feature_pooling_a2_260705_b500/`
- Exit: code 0, `n_errors=0`

## Gate Summary

`success_preview`:

- `a2_full_success=false`
- `a2_partial_pivot=false`
- `a2_failure=true`
- max adjacent hidden AUROC drop: `0.0029006579548976896`
- k8 delta CI low: `-0.03498279368697677`
- k64 delta CI: `[-0.06672404668040523, -0.04479749178961987]`

Pooled same-horizon AUROC deltas:

| k | hidden AUROC | text AUROC | delta | 95% CI |
|---:|---:|---:|---:|---:|
| 4 | 0.7364 | 0.7323 | +0.0041 | [-0.0082, +0.0158] |
| 8 | 0.7335 | 0.7570 | -0.0235 | [-0.0350, -0.0123] |
| 16 | 0.7589 | 0.8018 | -0.0428 | [-0.0547, -0.0315] |
| 32 | 0.7646 | 0.8077 | -0.0430 | [-0.0533, -0.0323] |
| 64 | 0.7932 | 0.8495 | -0.0564 | [-0.0667, -0.0448] |

Per Fable-5's pre-declared rule, this appears to be the `failure` branch:
k8 CI low < 0, so A1's score-pooling advantage was not confirmed by
feature-level pooling. Please review whether this is the correct interpretation
and what the next honest Stage1 framing should be.

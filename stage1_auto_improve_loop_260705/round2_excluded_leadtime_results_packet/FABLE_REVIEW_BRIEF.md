# Fable-5 Review Brief: Round 2 Excluded-Source Lead-Time Confirmation

Date: 2026-07-05

This packet contains only sanitized aggregate outputs and prediction-score JSONL.
It intentionally excludes raw prompts, CoTs, and hidden arrays.

## What Ran

We ran the preregistered excluded-source lead-time confirmation requested after
the equal-horizon Stage1 result remained negative/control.

- Code commit: `d658ca8`
- Prior tmp prereg/code-review commit: `af3d41f`
- Output root on RunPod:
  `/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/excluded_leadtime_confirmation_260705_b500`
- Hidden archive root:
  `/workspace/stage1-results/stage1_post_hb_260705_retune12288_b20/hidden_archives_excluded_leadtime_cotonly`
- Sources: `strongreject_full`, `reasoningshield`
- k grid: `4, 8, 16, 32, 64`
- Layer: `28`
- Bootstrap: `500`
- Surface family: `char_tfidf`

## Manifest Gate

The cot-only hidden manifest audit passed:

- `n_manifests = 6`
- `bad = []`
- Every manifest has `position_names = ["cot_4","cot_8","cot_16","cot_32","cot_64"]`
- Every manifest has `omit_think_last = true`
- Every manifest has `status = complete`

See `results/manifest_audit.json`.

## Frozen Test Population

- `strongreject_full`: 277 frozen test pairs, pair-complete at all k
- `reasoningshield`: 335 frozen test pairs, pair-complete at all k
- No pair-complete drops in the evaluated cells

See `results/stage1_excluded_leadtime_frozen_population.json`.

## Gate Results

Final script decision:

- `confirmed = false`
- `decision = drop_leadtime_claim`
- `n_errors = 0`

Primary A1 cell, pooled hidden@4 minus text@8:

- delta AUROC: `+0.00155`
- 95% CI: `[-0.01383, +0.01725]`
- gate pass: `false`

Per-source sanity gate:

- `strongreject_full`: hidden@4 `0.6915`, text@8 `0.7177`, delta `-0.0262`, CI `[-0.0504, -0.0027]`
- `reasoningshield`: hidden@4 `0.7180`, text@8 `0.7035`, delta `+0.0145`, CI `[-0.0068, +0.0396]`
- min delta `-0.0262`, below threshold `-0.02`
- gate pass: `false`

A2 robustness, pooled feature hidden@4 minus text@8:

- delta AUROC: `-0.06535`
- 95% CI: `[-0.08765, -0.04658]`
- gate pass: `false`

See:

- `results/stage1_excluded_leadtime_confirmation_summary.json`
- `results/stage1_excluded_leadtime_confirmation_gates.tsv`
- `results/a1_score_pooling/stage1_score_pooling_summary.tsv`
- `results/a2_feature_pooling/stage1_feature_pooling_summary.tsv`

## R2 Backup

Backed up after completion:

- Hidden cot-only archive:
  `cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/stage1-results/stage1_post_hb_260705_retune12288_b20/hidden_archives_excluded_leadtime_cotonly`
  - rclone size: 26 objects, 225.402 MiB
- Result directory:
  `cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/stage1_post_hb_260705_after_hb_n100_loso/excluded_leadtime_confirmation_260705_b500`
  - rclone size: 79 objects, 9.996 MiB

## Review Questions

Please review the packet and answer:

1. Is the preregistered interpretation correct, i.e. should we drop the
   excluded-source lead-time claim?
2. Is there any statistical or implementation concern in the manifest gate,
   frozen-population rule, A1/A2 gate logic, or reported CIs?
3. Given this result, should Stage1 be treated as a negative/control result,
   or is there any still-justified follow-up that is not post-hoc fishing?
4. Are the threshold/balanced-accuracy reporting suggestions still reasonable
   as an operating-point analysis, while keeping the core AUROC conclusion
   negative?

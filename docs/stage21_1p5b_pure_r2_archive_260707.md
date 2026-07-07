# Stage2.1-pure 1.5B R2 Archive - 2026-07-07

This document records the Cloudflare R2 archive for the 2026-07-07 2xA6000
Stage2.1-pure 1.5B run. It contains no Hugging Face token, R2 credential, API
key, or local `.env` value.

## Archive Root

```text
cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260707-2xa6000-1p5b-stage21-pure-full-cot5-bs4-ga2/
```

Final verified size after copying eval and selection results:

```text
Total objects: 757
Total size: 220.918 GiB
```

## Main Contents

| R2 prefix | Source path | Contents |
|---|---|---|
| `workspace/outputs/deepseek_1p5b_stage21_pause_pure_cot5_full_2xa6000/` | `/workspace/outputs/deepseek_1p5b_stage21_pause_pure_cot5_full_2xa6000` | Full 1-epoch Stage2.1-pure SFT output: checkpoints every 25 steps and final model. |
| `workspace/cot-safety/runs/eval/stage2_model_comparison_deepseek_1p5b_stage21_pure_cot5_2xa6000/` | `/workspace/cot-safety/runs/eval/stage2_model_comparison_deepseek_1p5b_stage21_pure_cot5_2xa6000` | Full final model-comparison generation outputs, logs, resolved config, and strict natural gate. |
| `workspace/cot-safety/runs/stage21_selection/deepseek_1p5b_stage21_pause_pure_cot5_full_2xa6000/` | `/workspace/cot-safety/runs/stage21_selection/deepseek_1p5b_stage21_pause_pure_cot5_full_2xa6000` | Disjoint selection-dev checkpoint sweep outputs and selection summaries. |

## Checkpoint Coverage

The archived output directory contains:

```text
checkpoint-25, checkpoint-50, ..., checkpoint-1050, checkpoint-1063, final
```

Important size checks:

```text
final:          8 objects, 3.319 GiB
checkpoint-1063: 13 objects, 5.055 GiB
```

Training completed one full epoch:

```text
checkpoint-1063 trainer_state: global_step=1063, epoch=1.0, max_steps=1063
```

## Result Summary

The strict final natural gate failed.

Overall final natural gate:

```text
n = 2100
exact_chain_rate = 0.8886
block_presence_rate = 0.9948
malformed_rate = 0.1062
off_target_rate = 0.1119
location_match_rate = 0.6395
avg_pause_count = 3.1224
```

Worst source:

```text
GSM8K exact_chain = 0.8100
GSM8K location_match = 0.0427
GSM8K off_target = 0.3440
GSM8K malformed = 0.1740
```

The disjoint selection-dev sweep was stopped early after the trend was clearly
bad. Completed checkpoints `750, 800, 850, 900, 950, 1000, 1050` all failed.
The ranker selected `checkpoint-1050` among completed candidates, but its gate
also failed:

```text
min_exact_chain = 0.7900
min_location_match = 0.0408
max_off_target = 0.3200
max_malformed = 0.1900
overall_exact = 0.8820
```

Do not report `checkpoint-1050` as a successful Stage2 checkpoint.

## Verification

Eval result backup:

```text
rclone check /workspace/cot-safety/runs/eval/stage2_model_comparison_deepseek_1p5b_stage21_pure_cot5_2xa6000 \
  cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260707-2xa6000-1p5b-stage21-pure-full-cot5-bs4-ga2/workspace/cot-safety/runs/eval/stage2_model_comparison_deepseek_1p5b_stage21_pure_cot5_2xa6000 \
  --one-way --fast-list
```

Result:

```text
0 differences found
81 matching files
Total size: 165.178 MiB
```

Selection sweep backup:

```text
rclone check /workspace/cot-safety/runs/stage21_selection/deepseek_1p5b_stage21_pause_pure_cot5_full_2xa6000 \
  cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260707-2xa6000-1p5b-stage21-pure-full-cot5-bs4-ga2/workspace/cot-safety/runs/stage21_selection/deepseek_1p5b_stage21_pause_pure_cot5_full_2xa6000 \
  --one-way --fast-list
```

Result:

```text
0 differences found
109 matching files
Total size: 75.691 MiB
```

## Local Analysis Documents

The Fable failure analysis and B1/B2 diagnostic summary are tracked in GitHub at:

```text
review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_1p5b_failure_analysis_260707.md
```

This local review document was produced after the R2 result backup and should be
treated as the analysis companion to the archived raw outputs.

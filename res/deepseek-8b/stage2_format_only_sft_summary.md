# DeepSeek-R1-Distill-Llama-8B Stage 2 Format-Only SFT Summary

Date: 2026-06-25
Updated: 2026-06-26

This file summarizes the comparable Stage 2 format-only SFT checkpoints that have confirmed judge results. All rows in the main tables use the same format-only objective: only the new `<|pause|>` token is trainable, and three pause tokens are inserted before the selected CoT slot.

The previous version of this note was incomplete: it included `cot4 checkpoint-200` and `cot3 checkpoint-450/500`, but omitted the confirmed `cot4 checkpoint-250` and `cot3 checkpoint-200/250` results.

## Evaluated Checkpoints

| Row | Pause placement | Checkpoint | SFT output family | Included in main comparison |
|---|---|---:|---|---|
| base | none | none | `DeepSeek-R1-Distill-Llama-8B` | yes |
| cot4 ckpt200 | before `cot_4` | 200 | `deepseek_8b_intra_pause_cot4_format_only_trusted_cot_18k_save50_max250` | yes |
| cot4 ckpt250 | before `cot_4` | 250 | `deepseek_8b_intra_pause_cot4_format_only_trusted_cot_18k_save50_max250` | yes |
| cot3 ckpt200 | before `cot_3` | 200 | `deepseek_8b_intra_pause_cot3_format_only_trusted_cot_18k_save50_max250` | yes |
| cot3 ckpt250 | before `cot_3` | 250 | `deepseek_8b_intra_pause_cot3_format_only_trusted_cot_18k_save50_max250` | yes |
| cot3 ckpt450 | before `cot_3` | 450 | `deepseek_8b_intra_pause_cot3_format_only_trusted_cot_18k_save50_max250` | yes |
| cot3 ckpt500 | before `cot_3` | 500 | `deepseek_8b_intra_pause_cot3_format_only_trusted_cot_18k_save50_max250` | yes |

## Capability

Accuracy is reported as decimals. Delta columns are relative to base.

| Model | GSM8K acc | Delta | MATH500 acc | Delta | Overall acc | Delta | pause3_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.710 | 0.000 | 0.423 | 0.000 | 0.603 | 0.000 | 0.000 |
| cot4 ckpt200 | 0.652 | -0.058 | 0.463 | +0.040 | 0.581 | -0.022 | 1.000 |
| cot4 ckpt250 | 0.684 | -0.026 | 0.423 | 0.000 | 0.586 | -0.017 | 1.000 |
| cot3 ckpt200 | 0.640 | -0.070 | 0.477 | +0.054 | 0.579 | -0.024 | 1.000 |
| cot3 ckpt250 | 0.648 | -0.062 | 0.477 | +0.054 | 0.584 | -0.019 | 1.000 |
| cot3 ckpt450 | 0.594 | -0.116 | 0.470 | +0.047 | 0.548 | -0.055 | 1.000 |
| cot3 ckpt500 | 0.612 | -0.098 | 0.497 | +0.074 | 0.569 | -0.034 | 1.000 |

## Unsafe Prompts

Metric: `unsafe_valid_rate`. Lower is less unsafe under the judge.

| Model | HarmBench | Delta | LlamaGuard | Delta | WildGuard | Delta |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.465 | 0.000 | 0.553 | 0.000 | 0.440 | 0.000 |
| cot4 ckpt200 | 0.463 | -0.002 | 0.518 | -0.035 | 0.405 | -0.035 |
| cot4 ckpt250 | 0.432 | -0.033 | 0.505 | -0.048 | 0.365 | -0.075 |
| cot3 ckpt200 | 0.653 | +0.188 | 0.705 | +0.152 | 0.617 | +0.177 |
| cot3 ckpt250 | 0.630 | +0.165 | 0.685 | +0.132 | 0.592 | +0.152 |
| cot3 ckpt450 | 0.653 | +0.188 | 0.708 | +0.155 | 0.590 | +0.150 |
| cot3 ckpt500 | 0.632 | +0.167 | 0.670 | +0.117 | 0.578 | +0.138 |

## Safe Prompts

Metric: `unsafe_valid_rate`. Lower is preferable because these prompts should remain safe.

| Model | HarmBench | Delta | LlamaGuard | Delta | WildGuard | Delta |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.670 | 0.000 | 0.152 | 0.000 | 0.102 | 0.000 |
| cot4 ckpt200 | 0.673 | +0.003 | 0.143 | -0.009 | 0.078 | -0.024 |
| cot4 ckpt250 | 0.658 | -0.012 | 0.170 | +0.018 | 0.098 | -0.004 |
| cot3 ckpt200 | 0.728 | +0.058 | 0.203 | +0.051 | 0.125 | +0.023 |
| cot3 ckpt250 | 0.742 | +0.072 | 0.203 | +0.051 | 0.137 | +0.035 |
| cot3 ckpt450 | 0.707 | +0.037 | 0.200 | +0.048 | 0.140 | +0.038 |
| cot3 ckpt500 | 0.707 | +0.037 | 0.173 | +0.021 | 0.127 | +0.025 |

## Interpretation

- The format objective succeeds for all tested format-only checkpoints: every SFT row has `pause3_rate = 1.000`.
- `cot4 checkpoint-250` is currently the best format-only SFT candidate among the consolidated results. It stays closest to base capability overall (`0.586` vs `0.603`) while reducing unsafe-prompt unsafe_valid_rate across all three judges.
- `cot4 checkpoint-200` is also close to base, but its GSM8K drop is larger than `checkpoint-250` (`-0.058` vs `-0.026`).
- The `cot3` position consistently behaves worse than `cot4` for DeepSeek-8B. Even when capability is only moderately below base, unsafe-prompt unsafe_valid_rate is higher than base across HarmBench, LlamaGuard, and WildGuard.
- These results support the current 8B hypothesis: pause insertion should target the stronger stage1 signal position around `cot_4`, while `cot_3` is a useful negative/position ablation rather than the main downstream steering model.

## Result Provenance

Confirmed format-only comparison runs:

- cot4 ckpt200 config: `configs/experiment/stage2_model_comparison_eval_8b_cot4_ckpt200_4xa100.yaml`
- cot4 ckpt250 config: `configs/experiment/stage2_model_comparison_eval_8b_cot4_ckpt250_3xidle.yaml`
- cot3 ckpt200/250 config: `configs/experiment/stage2_model_comparison_eval_8b_cot3_ckpt200_250_4xa100.yaml`
- cot3 ckpt450/500 config: `configs/experiment/stage2_model_comparison_eval_8b_cot3_ckpt450_500_4xa100.yaml`
- GDrive backup for cot3 ckpt250-500 evaluation snapshot: `safechain_gdrive:Research/cot-safety/runpod_backups/20260625T142750Z_cot3_ckpt250_500_eval_snapshot`

Non-main-table notes:

- `cot4 checkpoint-300/400/500` configs in this repo point to the earlier `deepseek_8b_intra_pause_cot4_trusted_cot_18k_save100_rerun` family, not the current `format_only_trusted_cot_18k_save50_max250` family. They should not be mixed into the main format-only table without an explicit "legacy/full-SFT" label.
- `cot4 checkpoint-400` does not currently have a complete model-comparison judge result in the saved records checked for this summary.
- The earlier overtrained/full-SFT cot4 run showed strong drift from base behavior, so it is treated as historical diagnostic evidence rather than a candidate format-only checkpoint.

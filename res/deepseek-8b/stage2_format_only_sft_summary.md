# DeepSeek-R1-Distill-Llama-8B Stage 2 SFT Summary

Date: 2026-06-25
Updated: 2026-06-26

This file summarizes the comparable Stage 2 SFT checkpoints that have confirmed model-comparison judge results. The cot4 section includes both the later format-only checkpoint sweep and the earlier full-step/full-SFT final model; the cot3 section covers the position-ablation format-only sweep.

The previous version of this note was incomplete in two ways:

- It included `cot4 checkpoint-200` and `cot3 checkpoint-450/500`, but omitted confirmed `cot4 checkpoint-250` and `cot3 checkpoint-200/250`.
- It omitted the earlier full-step cot4 final SFT model, `deepseek_8b_intra_pause_cot4_trusted_cot_18k/final`.

All tables are transposed so metrics/judges are rows and model checkpoints are columns.

## cot4 Results

Compared models:

| Model column | Pause placement | Checkpoint | SFT output family | Training objective |
|---|---|---:|---|---|
| base | none | none | `DeepSeek-R1-Distill-Llama-8B` | none |
| cot4 ckpt200 | before `cot_4` | 200 | `deepseek_8b_intra_pause_cot4_format_only_trusted_cot_18k_save50_max250` | format-only, trainable `<|pause|>` token |
| cot4 ckpt250 | before `cot_4` | 250 | `deepseek_8b_intra_pause_cot4_format_only_trusted_cot_18k_save50_max250` | format-only, trainable `<|pause|>` token |
| cot4 full final | before `cot_4` | final | `deepseek_8b_intra_pause_cot4_trusted_cot_18k/final` | earlier full-step/full-SFT run |

Capability:

| Metric | base | cot4 ckpt200 | cot4 ckpt250 | cot4 full final |
|---|---:|---:|---:|---:|
| GSM8K acc | 0.710 | 0.652 | 0.684 | 0.762 |
| MATH500 acc | 0.423 | 0.463 | 0.423 | 0.460 |
| overall acc | 0.603 | 0.581 | 0.586 | 0.649 |
| pause3_rate | 0.000 | 1.000 | 1.000 | 1.000 |

Unsafe prompts, `unsafe_valid_rate`:

| Judge | base | cot4 ckpt200 | cot4 ckpt250 | cot4 full final |
|---|---:|---:|---:|---:|
| HarmBench | 0.465 | 0.463 | 0.432 | 0.297 |
| LlamaGuard | 0.553 | 0.518 | 0.505 | 0.323 |
| WildGuard | 0.440 | 0.405 | 0.365 | 0.217 |

Safe prompts, `unsafe_valid_rate`:

| Judge | base | cot4 ckpt200 | cot4 ckpt250 | cot4 full final |
|---|---:|---:|---:|---:|
| HarmBench | 0.670 | 0.673 | 0.658 | 0.615 |
| LlamaGuard | 0.152 | 0.143 | 0.170 | 0.095 |
| WildGuard | 0.102 | 0.078 | 0.098 | 0.055 |

## cot3 Results

Compared models:

| Model column | Pause placement | Checkpoint | SFT output family | Training objective |
|---|---|---:|---|---|
| base | none | none | `DeepSeek-R1-Distill-Llama-8B` | none |
| cot3 ckpt200 | before `cot_3` | 200 | `deepseek_8b_intra_pause_cot3_format_only_trusted_cot_18k_save50_max250` | format-only, trainable `<|pause|>` token |
| cot3 ckpt250 | before `cot_3` | 250 | `deepseek_8b_intra_pause_cot3_format_only_trusted_cot_18k_save50_max250` | format-only, trainable `<|pause|>` token |
| cot3 ckpt450 | before `cot_3` | 450 | `deepseek_8b_intra_pause_cot3_format_only_trusted_cot_18k_save50_max250` | format-only, trainable `<|pause|>` token |
| cot3 ckpt500 | before `cot_3` | 500 | `deepseek_8b_intra_pause_cot3_format_only_trusted_cot_18k_save50_max250` | format-only, trainable `<|pause|>` token |

Capability:

| Metric | base | cot3 ckpt200 | cot3 ckpt250 | cot3 ckpt450 | cot3 ckpt500 |
|---|---:|---:|---:|---:|---:|
| GSM8K acc | 0.710 | 0.640 | 0.648 | 0.594 | 0.612 |
| MATH500 acc | 0.423 | 0.477 | 0.477 | 0.470 | 0.497 |
| overall acc | 0.603 | 0.579 | 0.584 | 0.548 | 0.569 |
| pause3_rate | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Unsafe prompts, `unsafe_valid_rate`:

| Judge | base | cot3 ckpt200 | cot3 ckpt250 | cot3 ckpt450 | cot3 ckpt500 |
|---|---:|---:|---:|---:|---:|
| HarmBench | 0.465 | 0.653 | 0.630 | 0.653 | 0.632 |
| LlamaGuard | 0.553 | 0.705 | 0.685 | 0.708 | 0.670 |
| WildGuard | 0.440 | 0.617 | 0.592 | 0.590 | 0.578 |

Safe prompts, `unsafe_valid_rate`:

| Judge | base | cot3 ckpt200 | cot3 ckpt250 | cot3 ckpt450 | cot3 ckpt500 |
|---|---:|---:|---:|---:|---:|
| HarmBench | 0.670 | 0.728 | 0.742 | 0.707 | 0.707 |
| LlamaGuard | 0.152 | 0.203 | 0.203 | 0.200 | 0.173 |
| WildGuard | 0.102 | 0.125 | 0.137 | 0.140 | 0.127 |

## Interpretation

- The requested pause format succeeds in every tested SFT row: `pause3_rate = 1.000`.
- Among the format-only cot4 checkpoints, `cot4 checkpoint-250` is the healthier candidate for downstream steering. It stays close to base capability (`0.586` vs `0.603`) while reducing unsafe-prompt unsafe_valid_rate across all three judges.
- `cot4 full final` is not directly comparable as a format-only checkpoint because it comes from the earlier full-step/full-SFT run. It shows stronger behavior drift: capability rises above base (`0.649` vs `0.603`) and unsafe-prompt unsafe_valid_rate drops sharply, which is useful diagnostic evidence but not the clean format-only control we want.
- The cot3 position consistently behaves worse than cot4 for DeepSeek-8B. Unsafe-prompt unsafe_valid_rate is higher than base across HarmBench, LlamaGuard, and WildGuard for every tested cot3 checkpoint.
- These results support the current 8B hypothesis: pause insertion should target the stronger stage1 signal position around `cot_4`, while `cot_3` is a position ablation rather than the main downstream steering model.

## Result Provenance

Confirmed comparison runs and configs:

- cot4 full final run: `stage2_model_comparison_deepseek_8b_4xa100`
- cot4 full final model: `deepseek_8b_intra_pause_cot4_trusted_cot_18k/final`
- cot4 full final archive: `safechain_gdrive:Research/cot-safety/runpod_backups/20260624T235259Z_deepseek8b_all_sft_ckpts_no_models/archives/workspace_runs.tar.gz`
- cot4 ckpt200 config: `configs/experiment/stage2_model_comparison_eval_8b_cot4_ckpt200_4xa100.yaml`
- cot4 ckpt250 config: `configs/experiment/stage2_model_comparison_eval_8b_cot4_ckpt250_3xidle.yaml`
- cot3 ckpt200/250 config: `configs/experiment/stage2_model_comparison_eval_8b_cot3_ckpt200_250_4xa100.yaml`
- cot3 ckpt450/500 config: `configs/experiment/stage2_model_comparison_eval_8b_cot3_ckpt450_500_4xa100.yaml`
- GDrive backup for cot3 ckpt250-500 evaluation snapshot: `safechain_gdrive:Research/cot-safety/runpod_backups/20260625T142750Z_cot3_ckpt250_500_eval_snapshot`

Non-main-table notes:

- The later `cot4 checkpoint-300/400/500` configs point to `deepseek_8b_intra_pause_cot4_trusted_cot_18k_save100_rerun`, not the `format_only_trusted_cot_18k_save50_max250` family. They should be reported under a separate legacy/rerun section if needed.
- `cot4 checkpoint-400` still does not have a complete model-comparison judge result in the saved records checked for this summary.

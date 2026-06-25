# DeepSeek-R1-Distill-Llama-8B Stage 2 Format-Only SFT Summary

Date: 2026-06-25

## cot4 ckpt200 vs base

Run under comparison:

- Base: `DeepSeek-R1-Distill-Llama-8B`
- SFT: `deepseek_8b_intra_pause_cot4_format_only_trusted_cot_18k_save50_max250/checkpoint-200`
- SFT objective: format-only, trainable token `<|pause|>` only
- Pause layout: insert three `<|pause|>` tokens before `cot_4`
- Evaluation: stage2 model comparison, capability 800 prompts, safety 1200 prompts, WildGuard/LlamaGuard/HarmBench judges

Capability:

| Dataset | Base accuracy | cot4 ckpt200 accuracy | Delta |
|---|---:|---:|---:|
| GSM8K | 0.710 | 0.652 | -0.058 |
| MATH500 | 0.423 | 0.463 | +0.040 |

Unsafe prompts, unsafe_valid_rate:

| Judge | Base | cot4 ckpt200 | Delta |
|---|---:|---:|---:|
| HarmBench | 0.465 | 0.463 | -0.002 |
| LlamaGuard | 0.553 | 0.518 | -0.035 |
| WildGuard | 0.440 | 0.405 | -0.035 |

Safe prompts, unsafe_valid_rate:

| Judge | Base | cot4 ckpt200 | Delta |
|---|---:|---:|---:|
| HarmBench | 0.670 | 0.673 | +0.003 |
| LlamaGuard | 0.152 | 0.143 | -0.008 |
| WildGuard | 0.102 | 0.078 | -0.023 |

Format adherence:

| Model | pause3_rate |
|---|---:|
| Base | 0.000 |
| cot4 ckpt200 | 1.000 |

Interpretation:

- `checkpoint-200` is much closer to base behavior than later cot4 checkpoints previously judged.
- The format objective succeeds: three pause tokens are inserted reliably.
- Safety behavior on unsafe prompts remains close to base across all three judges, with the largest observed drop at 0.035.
- Capability is mixed: GSM8K drops by 0.058 while MATH500 increases by 0.040.
- This makes `checkpoint-200` a reasonable format-only SFT candidate for the next steering/evaluation stage.

Source result files:

- Current run: `/dev/shm/cot-safety-hot/runs/eval/stage2_model_comparison_deepseek_8b_cot4_ckpt200_4xa100`
- Base comparison archive: `safechain_gdrive:Research/cot-safety/runpod_backups/20260625T041738Z_deepseek8b_ckpt300_eval_incremental_no_models/archives/cot_safety_incremental_eval_runs_20260625T041738Z.tar.gz`

## cot3 ckpt450/500 vs base

Run under comparison:

- Base: `DeepSeek-R1-Distill-Llama-8B`
- SFT checkpoints:
  - `deepseek_8b_intra_pause_cot3_format_only_trusted_cot_18k_save50_max250/checkpoint-450`
  - `deepseek_8b_intra_pause_cot3_format_only_trusted_cot_18k_save50_max250/checkpoint-500`
- SFT objective: format-only, trainable token `<|pause|>` only
- Pause layout: insert three `<|pause|>` tokens before `cot_3`
- Evaluation: stage2 model comparison, capability 800 prompts, safety 1200 prompts, WildGuard/LlamaGuard/HarmBench judges
- Note: `checkpoint-400` does not currently have a complete model-comparison judge result in the saved runs.

Capability:

| Metric | Base | cot3 ckpt450 | cot3 ckpt500 |
|---|---:|---:|---:|
| GSM8K acc | 0.710 | 0.594 | 0.612 |
| MATH500 acc | 0.423 | 0.470 | 0.497 |
| overall acc | 0.603 | 0.548 | 0.569 |
| pause3_rate | 0.000 | 1.000 | 1.000 |

Unsafe prompts, unsafe_valid_rate:

| Judge | Base | cot3 ckpt450 | cot3 ckpt500 |
|---|---:|---:|---:|
| HarmBench | 0.465 | 0.653 | 0.632 |
| LlamaGuard | 0.553 | 0.708 | 0.670 |
| WildGuard | 0.440 | 0.590 | 0.578 |

Safe prompts, unsafe_valid_rate:

| Judge | Base | cot3 ckpt450 | cot3 ckpt500 |
|---|---:|---:|---:|
| HarmBench | 0.670 | 0.707 | 0.707 |
| LlamaGuard | 0.152 | 0.200 | 0.173 |
| WildGuard | 0.102 | 0.140 | 0.127 |

Interpretation:

- `cot3 checkpoint-500` is better than `checkpoint-450` on capability and on unsafe-prompt unsafe_valid_rate across all three judges.
- Both cot3 checkpoints preserve the requested pause format, with `pause3_rate = 1.000`.
- Even at `checkpoint-500`, cot3 remains substantially farther from base behavior than the cot4 candidates. Overall capability is 0.569 vs base 0.603, and unsafe-prompt unsafe_valid_rate is higher than base for all three judges.
- These results support treating cot3 as a negative/position ablation for 8B: inserting pauses before `cot_3` appears too early relative to the stage1 signal hotspot, while cot4 is the healthier candidate for downstream steering.

Source result files:

- Base run: `/dev/shm/cot-safety-hot/runs/eval/stage2_model_comparison_deepseek_8b_cot3_ckpt200_250_4xa100`
- cot3 ckpt450/500 run: `/dev/shm/cot-safety-hot/runs/eval/stage2_model_comparison_deepseek_8b_cot3_ckpt450_500_4xa100`
- GDrive backup: `safechain_gdrive:Research/cot-safety/runpod_backups/20260625T142750Z_cot3_ckpt250_500_eval_snapshot`

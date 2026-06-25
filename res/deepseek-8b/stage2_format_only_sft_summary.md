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

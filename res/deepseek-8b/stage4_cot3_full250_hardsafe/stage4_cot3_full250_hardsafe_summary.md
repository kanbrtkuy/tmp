# Stage 4 Hard-Safe Steering Summary: DeepSeek 8B cot3 Full SFT ckpt250

Date: 2026-06-27

Model under test:
- Base: `DeepSeek-R1-Distill-Llama-8B`
- SFT: `deepseek_8b_intra_pause_cot3_full_trusted_cot_18k_from0_save50/checkpoint-250`
- Steering target: all three inserted pause tokens
- Probe/steering point: selected from Stage 3 cot3 full ckpt250 scan
- Judge: WildGuard normalized labels

Raw summary files:
- `base_hardsafe_reference_summary.csv`
- `base_hardsafe_reference_summary_manifest.json`
- `cot3_full250_all3_full_steering_summary.csv`
- `cot3_full250_all3_full_steering_summary_manifest.json`

## Overall Unsafe-Valid Rate

Lower is better for hard-safe steering.

| Model / alpha | all | safe prompts | unsafe prompts | pause3_rate |
|---|---:|---:|---:|---:|
| base | 0.204 | 0.059 | 0.364 | 0.000 |
| cot3 full250, alpha=0 | 0.195 | 0.082 | 0.319 | 0.983 |
| cot3 full250, alpha=1 | 0.257 | 0.117 | 0.411 | 0.993 |
| cot3 full250, alpha=2 | 0.091 | 0.056 | 0.130 | 0.955 |

## Dataset Breakdown

### Safe Prompts

| Dataset | base | cot3 full250 alpha=0 | cot3 full250 alpha=1 | cot3 full250 alpha=2 |
|---|---:|---:|---:|---:|
| XSTest safe | 0.000 | 0.000 | 0.005 | 0.001 |
| OR-Bench hard safe | 0.109 | 0.151 | 0.210 | 0.101 |
| all safe | 0.059 | 0.082 | 0.117 | 0.056 |

### Unsafe Prompts

| Dataset | base | cot3 full250 alpha=0 | cot3 full250 alpha=1 | cot3 full250 alpha=2 |
|---|---:|---:|---:|---:|
| XSTest contrast unsafe | 0.235 | 0.180 | 0.280 | 0.025 |
| OR-Bench toxic unsafe | 0.450 | 0.411 | 0.498 | 0.200 |
| all unsafe | 0.364 | 0.319 | 0.411 | 0.130 |

## Interpretation

`alpha=2` is the best current Stage 4 setting for the cot3 full-SFT ckpt250 model. It reduces the overall unsafe-valid rate from `0.204` to `0.091`, and reduces unsafe-prompt unsafe-valid rate from `0.364` to `0.130`, while safe-prompt unsafe-valid rate remains essentially unchanged relative to base (`0.059` to `0.056`).

`alpha=1` is not a useful operating point in this run: it increases unsafe-valid rate on both safe and unsafe prompts relative to `alpha=0` and base.

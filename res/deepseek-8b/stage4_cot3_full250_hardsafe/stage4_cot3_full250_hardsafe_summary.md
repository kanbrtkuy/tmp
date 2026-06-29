# Stage 4 Hard-Safe Steering Summary: DeepSeek 8B cot3 Full SFT ckpt250

Date: 2026-06-27
Updated: 2026-06-29 with LlamaGuard and HarmBench second-judge results.

Model under test:
- Base: `DeepSeek-R1-Distill-Llama-8B`
- SFT: `deepseek_8b_intra_pause_cot3_full_trusted_cot_18k_from0_save50/checkpoint-250`
- Steering target: all three inserted pause tokens
- Probe/steering point: layer20, selected from the Stage 3 cot3 full ckpt250 scan
- Datasets: hard-safe stage4 set (`xstest_safe`, `or_bench_hard_safe`, `xstest_contrast_unsafe`, `or_bench_toxic_unsafe`)
- Judges: WildGuard, LlamaGuard, HarmBench classifier

Raw summary files:
- WildGuard:
  - `base_hardsafe_reference_summary.csv`
  - `base_hardsafe_reference_summary_manifest.json`
  - `cot3_full250_all3_full_steering_summary.csv`
  - `cot3_full250_all3_full_steering_summary_manifest.json`
- LlamaGuard:
  - `base_llamaguard_summary.csv`
  - `base_llamaguard_summary_manifest.json`
  - `all3_llamaguard_summary.csv`
  - `all3_llamaguard_summary_manifest.json`
- HarmBench:
  - `base_harmbench_summary.csv`
  - `base_harmbench_summary_manifest.json`
  - `all3_harmbench_summary.csv`
  - `all3_harmbench_summary_manifest.json`

Coverage:
- Base: 12 shards, 3150 joined rows per judge.
- all3 steering: 36 shards, 9450 joined rows per judge.
- All copied manifests report `missing_judge_rows = 0` for the completed summaries.

## Overall Unsafe-Valid Rate

Lower is better. Rows are `dataset=all`, `seed=all`.

| Judge | base | alpha=0 | alpha=1 | alpha=2 |
|---|---:|---:|---:|---:|
| WildGuard | 0.204 | 0.195 | 0.257 | 0.091 |
| LlamaGuard | 0.259 | 0.277 | 0.345 | 0.177 |
| HarmBench | 0.457 | 0.476 | 0.258 | 0.357 |

## Unsafe Prompts

Lower is better. Rows are `reference_label=unsafe`, `dataset=all`, `seed=all`.

| Judge | base | alpha=0 | alpha=1 | alpha=2 |
|---|---:|---:|---:|---:|
| WildGuard | 0.364 | 0.319 | 0.411 | 0.130 |
| LlamaGuard | 0.450 | 0.451 | 0.573 | 0.263 |
| HarmBench | 0.355 | 0.347 | 0.239 | 0.179 |

## Safe Prompts

Lower is better, but HarmBench is a behavior-compliance classifier and is less directly calibrated for benign hard-safe prompts than WildGuard/LlamaGuard.

| Judge | base | alpha=0 | alpha=1 | alpha=2 |
|---|---:|---:|---:|---:|
| WildGuard | 0.059 | 0.082 | 0.117 | 0.056 |
| LlamaGuard | 0.085 | 0.120 | 0.137 | 0.099 |
| HarmBench | 0.551 | 0.593 | 0.275 | 0.518 |

## Pause Format

`pause3_rate` is inherited from the steering generations and is judge-independent:

| Model / alpha | all | safe prompts | unsafe prompts |
|---|---:|---:|---:|
| base | 0.000 | 0.000 | 0.000 |
| cot3 full250, alpha=0 | 0.983 | 0.978 | 0.987 |
| cot3 full250, alpha=1 | 0.993 | 0.992 | 0.995 |
| cot3 full250, alpha=2 | 0.955 | 0.952 | 0.959 |

## Dataset Breakdown: WildGuard

### Safe Prompts

| Dataset | base | alpha=0 | alpha=1 | alpha=2 |
|---|---:|---:|---:|---:|
| XSTest safe | 0.000 | 0.000 | 0.005 | 0.001 |
| OR-Bench hard safe | 0.109 | 0.151 | 0.210 | 0.101 |
| all safe | 0.059 | 0.082 | 0.117 | 0.056 |

### Unsafe Prompts

| Dataset | base | alpha=0 | alpha=1 | alpha=2 |
|---|---:|---:|---:|---:|
| XSTest contrast unsafe | 0.235 | 0.180 | 0.280 | 0.025 |
| OR-Bench toxic unsafe | 0.450 | 0.411 | 0.498 | 0.200 |
| all unsafe | 0.364 | 0.319 | 0.411 | 0.130 |

## Interpretation

The second-judge update strengthens the main unsafe-prompt conclusion. Under all three judges, `alpha=2` reduces unsafe-prompt unsafe-valid rate relative to base:

| Judge | base unsafe | alpha=2 unsafe | absolute change |
|---|---:|---:|---:|
| WildGuard | 0.364 | 0.130 | -0.234 |
| LlamaGuard | 0.450 | 0.263 | -0.187 |
| HarmBench | 0.355 | 0.179 | -0.176 |

For overall rate, WildGuard and LlamaGuard both favor `alpha=2`. HarmBench favors `alpha=1` overall, largely because HarmBench assigns much lower unsafe-valid rates to safe prompts at `alpha=1`. Since HarmBench is a behavior-compliance classifier, the unsafe-prompt rows are the cleaner HarmBench readout for the steering claim.

The safest concise claim is therefore:

> On the 8B cot3 full250 Stage 4 hard-safe evaluation, pause-token steering at alpha=2 consistently reduces unsafe-prompt unsafe-valid rate across WildGuard, LlamaGuard, and HarmBench. WildGuard and LlamaGuard also select alpha=2 as the best overall operating point, while HarmBench's overall metric is mixed because its safe-prompt behavior differs from the other judges.

This updates the previous note that 8B Stage 4 had only WildGuard coverage: the same 8B Stage 4 generations now have completed LlamaGuard and HarmBench second-judge summaries.

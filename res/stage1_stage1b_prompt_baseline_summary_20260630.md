# Stage1 + Stage1b Prompt-Baseline Anchor Results

Date: 2026-06-30

Models:
- DeepSeek-R1-Distill-Qwen-1.5B
- DeepSeek-R1-Distill-Llama-8B

Probe family: linear probes only.

Purpose: anchor the latest Stage1 position scan and Stage1b prompt-only / pre-CoT baseline results, using the same data and scan protocol for 1.5B and 8B. Stage1b addresses the construct-validity concern that early CoT probes might merely perform prompt classification.

## Protocol

Stage1 scans CoT token positions across dense layer grids.

Stage1b adds prompt-only and pre-CoT control positions to the same scan:

- `last_prompt_token`: strict prompt-only baseline; user prompt final token before assistant response.
- `pre_think`: boundary control immediately before `<think>`.
- `think_last`: `<think>` boundary state; stronger pre-CoT control, not a strict prompt-only baseline.

Paper-facing convention:

- Primary prompt-only baseline: `last_prompt_token`.
- Stronger pre-CoT controls: `pre_think`, `think_last`.
- `best` locates the strongest token/layer.
- Across-layer mean/median should be used for the construct-validity argument, to avoid relying only on a best-of-scan result.
- Matched-layer comparisons are useful as an additional check that the gap is not caused by comparing a best CoT layer to a weak prompt layer.

## Stage1 Position Scan

### 1.5B

| Probe | Selection | Position | Layer / combine | Test AUROC | Test AUPRC | Heldout AUROC | Heldout AUPRC |
|---|---|---:|---|---:|---:|---:|---:|
| single-layer linear | best test | cot_3 | layer 17 | 0.980 | 0.865 | 0.890 | 0.693 |
| single-layer linear | best heldout | cot_4 | layer 21 | 0.976 | 0.815 | 0.930 | 0.728 |
| multilayer linear | best test | cot_3 | concat dense layers | 0.978 | 0.831 | 0.877 | 0.695 |
| multilayer linear | best heldout | cot_2 | mean dense layers | 0.963 | 0.795 | 0.898 | 0.659 |

### 8B

| Probe | Selection | Position | Layer / combine | Test AUROC | Test AUPRC | Heldout AUROC | Heldout AUPRC |
|---|---|---:|---|---:|---:|---:|---:|
| single-layer linear | best test | cot_3 | layer 17 | 0.985 | 0.896 | 0.923 | 0.741 |
| single-layer linear | best heldout | cot_4 | layer 12 | 0.983 | 0.889 | 0.942 | 0.775 |
| multilayer linear | best test | cot_3 | mean dense layers | 0.977 | 0.834 | 0.917 | 0.714 |
| multilayer linear | best heldout | cot_3 | concat dense layers | 0.976 | 0.870 | 0.937 | 0.795 |

Stage1 conclusion: both models have strong early-CoT unsafe/safe signal. The strongest single-layer heldout point is `cot_4` for both current dense scans, while in-domain/test strength remains centered around `cot_3`.

## Stage1b Prompt-Only / Pre-CoT Controls

### 1.5B Across Layers

| Position | Test best | Test mean | Test median | Heldout best | Heldout mean | Heldout median |
|---|---:|---:|---:|---:|---:|---:|
| last_prompt_token | 0.952 | 0.931 | 0.932 | 0.832 | 0.785 | 0.787 |
| pre_think | 0.955 | 0.945 | 0.944 | 0.855 | 0.830 | 0.834 |
| think_last | 0.954 | 0.939 | 0.938 | 0.846 | 0.833 | 0.836 |
| cot_2 | 0.966 | 0.954 | 0.953 | 0.895 | 0.867 | 0.875 |
| cot_3 | 0.979 | 0.971 | 0.970 | 0.902 | 0.868 | 0.873 |
| cot_4 | 0.978 | 0.972 | 0.973 | 0.928 | 0.891 | 0.894 |
| cot_7 | 0.976 | 0.963 | 0.961 | 0.888 | 0.826 | 0.843 |

### 8B Across Layers

| Position | Test best | Test mean | Test median | Heldout best | Heldout mean | Heldout median |
|---|---:|---:|---:|---:|---:|---:|
| last_prompt_token | 0.945 | 0.919 | 0.922 | 0.831 | 0.801 | 0.802 |
| pre_think | 0.958 | 0.939 | 0.939 | 0.868 | 0.843 | 0.848 |
| think_last | 0.954 | 0.939 | 0.941 | 0.881 | 0.859 | 0.870 |
| cot_2 | 0.967 | 0.956 | 0.954 | 0.911 | 0.886 | 0.893 |
| cot_3 | 0.985 | 0.980 | 0.980 | 0.933 | 0.912 | 0.917 |
| cot_4 | 0.982 | 0.976 | 0.976 | 0.946 | 0.921 | 0.929 |
| cot_7 | 0.977 | 0.971 | 0.972 | 0.910 | 0.862 | 0.861 |

## Construct-Validity Conclusion

These results do not support the failure mode where prompt-only already reaches the same separability as early CoT. Prompt-only and pre-CoT controls contain nontrivial risk signal, but early CoT positions consistently sharpen that signal.

For 1.5B, strict prompt-only heldout mean AUROC is 0.785, while `cot_3` and `cot_4` reach 0.868 and 0.891. Compared with stronger pre-CoT controls, `cot_4` still improves heldout mean AUROC by about 0.058 to 0.061.

For 8B, strict prompt-only heldout mean AUROC is 0.801, while `cot_3` and `cot_4` reach 0.912 and 0.921. Compared with stronger pre-CoT controls, `cot_4` still improves heldout mean AUROC by about 0.062 to 0.078.

Paper-safe phrasing:

> Prompt-conditioned risk signal is already present before CoT, but it is substantially sharpened during early CoT. Therefore, Stage1 is not merely prompt classification; it identifies trajectory positions where unsafe/safe separability becomes stronger.

## What This Does Not Yet Prove

This result addresses the prompt-only / pre-CoT construct-validity concern. It does not yet fully address leave-one-source-out generalization. Current heldout evidence uses ReasoningShield test as the heldout source. The next required robustness check is leave-one-source-out across all training sources, reporting mean and variance across heldout sources.

## Pipeline Status

The current Stage1 + Stage1b code is reusable through config files and has completed full 1.5B and 8B runs on a 2x A6000 RunPod node. The latest runner archives each stage from `/dev/shm` to `/workspace/stage1-results` and cleans hidden arrays after each stage, avoiding the previous `/dev/shm` accumulation failure.

Operationally, the pipeline is now close to production/reusable experiment infrastructure for Stage1 and Stage1b:

- Config-driven model, layer, position, data, and runtime choices.
- Shared Stage1 and Stage1b execution path.
- Prompt-only / pre-CoT controls implemented as first-class scan positions.
- Dynamic probe scheduling is available for GPU work distribution.
- Stage outputs are archived stage-by-stage and hidden arrays are cleaned.

Remaining hardening before calling it fully industrial-grade:

- Add leave-one-source-out Stage1 generalization as a first-class config/module.
- Add automated post-run sanity checks that assert required summary files exist and contain all requested positions.
- Add a one-command backup target that works even on fresh pods by injecting only the required rclone remote.
- Add CI/smoke tests for config parsing and output-schema compatibility.

## Artifact Backup

Complete Stage1 + Stage1b remote archive:

- RunPod archive: `/workspace/stage1-results_1p5b_8b_stage1_stage1b_20260630T054640Z.tar.gz`
- Size: 248 MB
- SHA256: `3fc07550bca30c5d28ee63c49eee4d970076169abde96c5cad969067105b9c87`
- GDrive target: `safechain_gdrive:Research/cot-safety/runpod_backups/stage1_stage1b/`

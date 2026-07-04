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

### Consistency With Earlier Stage1 Runs

The latest dense scan is consistent with the previous token-position conclusion: the strongest signal remains in early CoT, especially around `cot_3` and `cot_4`. The exact best layer should not be over-interpreted, because this run used a denser layer grid and the top layer varies by split.

| Model | Selection | Latest strongest point | Consistency note |
|---|---|---|---|
| 1.5B | single-layer test | `cot_3` layer 17 | Consistent with the earlier `cot_3` / layer-17-region signal. |
| 1.5B | single-layer heldout | `cot_4` layer 21 | Token remains in the `cot_3`/`cot_4` neighborhood, with this run leaning more toward `cot_4` on heldout. |
| 8B | single-layer test | `cot_3` layer 17 | Consistent with the earlier strong `cot_3` / layer-17-region result. |
| 8B | single-layer heldout | `cot_4` layer 12 | Token-level conclusion is consistent with `cot_4`; the exact best layer shifted from the previously discussed layer-20 region to layer 12 under the denser scan. |

Paper-safe phrasing:

> Across both models, the strongest probe signal consistently appears in early CoT positions, especially around `cot_3` and `cot_4`. The exact best layer varies with scan density and split, but the token-position conclusion is stable.

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

## Leave-One-Source-Out Generalization

Update on 2026-07-01: Stage1/Stage1b now includes source-family leave-one-out evaluation. Each family is held out and evaluated with family-level AUROC. For 1.5B, multi-source family results were recomputed by merging restored `predictions.jsonl` files from the R2 archive. For 8B, the latest `stage1_loso_summary` aggregate is used.

### 1.5B LOSO

| Heldout family | Position-scan best | AUROC | Prompt-baseline module best | AUROC | Trajectory - prompt gap |
|---|---|---:|---|---:|---:|
| `reasoningshield_train_sft` | `cot_3 / layer16` single | 0.908 | `cot_7 / layer22` single | 0.898 | +0.062 |
| `reasoningshield_train_dpo` | `cot_1 / concat` multilayer | 0.787 | `assistant_last / layer17` single | 0.798 | -0.006 |
| `reasoningshield_test` | `cot_4 / layer17` single | 0.932 | `cot_4 / concat` multilayer | 0.933 | +0.079 |
| `star_vs_harmthoughts` | `cot_4 / layer17` single | 0.933 | `cot_4 / layer17` single | 0.934 | +0.015 |
| `aidsafe_vs_harmthoughts` | `cot_0 / layer4` single | 1.000 | `cot_2 / layer4` single | 1.000 | +0.120 |
| `unsafechain_vs_harmthoughts` | `cot_1 / layer7` single | 0.995 | `cot_2 / layer22` single | 0.981 | +0.083 |

1.5B position-scan LOSO mean AUROC: 0.926; std: 0.071.

### 8B LOSO

| Heldout family | Position-scan best | AUROC | Prompt-baseline module best | AUROC | Trajectory - prompt gap |
|---|---|---:|---|---:|---:|
| `reasoningshield_train_sft` | `cot_3 / layer18` single | 0.909 | `cot_2 / layer14` single | 0.852 | +0.001 |
| `reasoningshield_train_dpo` | `cot_3 / concat` multilayer | 0.807 | `cot_3 / concat` multilayer | 0.807 | +0.013 |
| `reasoningshield_test` | `cot_3 / layer14` single | 0.920 | `cot_3 / layer12` single | 0.924 | +0.066 |
| `star_vs_harmthoughts` | `cot_4 / layer12` single | 0.929 | `cot_4 / layer16` single | 0.933 | +0.012 |
| `aidsafe_vs_harmthoughts` | `cot_0 / layer4` single | 1.000 | `cot_2 / layer14` single | 0.999 | +0.118 |
| `unsafechain_vs_harmthoughts` | `cot_0 / layer6` single | 0.995 | `cot_2 / layer12` single | 0.985 | +0.086 |

8B position-scan LOSO mean AUROC: 0.927; std: 0.064.

### LOSO Conclusion

The LOSO results support two separate points:

- On ReasoningShield-style families, the strongest transferable signal remains in early CoT, especially around `cot_3` / `cot_4`. This is consistent with the main Stage1 / Stage1b conclusion.
- `aidsafe_vs_harmthoughts` and `unsafechain_vs_harmthoughts` show near-perfect separability at `cot_0`, which likely reflects source or format artifacts. These splits are useful diagnostics, but should not be treated as pure trajectory-monitoring evidence.

Paper-safe phrasing:

> Leave-one-source-out evaluation shows that early-CoT risk signals transfer across ReasoningShield-style heldout families. Some mixed-source families are nearly separable at the first CoT token, indicating source or format artifacts; we therefore report them separately and avoid over-claiming trajectory specificity on those splits.

## What This Does Not Yet Prove

This result addresses the prompt-only / pre-CoT construct-validity concern and adds source-family LOSO generalization evidence. The remaining caution is that LOSO families are not equally clean: some mixed-source families show strong source / format artifacts. In paper writing, ReasoningShield-style families should be treated as the primary transfer evidence, while `aidsafe` / `unsafechain` families should be marked as artifact-prone diagnostic splits.

## Pipeline Status

The current Stage1 + Stage1b code is reusable through config files and has completed full 1.5B and 8B runs on A6000 RunPod nodes. The latest runner archives each stage from hot storage to cold storage and cleans hidden arrays after each stage, avoiding the previous `/dev/shm` or hot-path accumulation failures.

Operationally, the pipeline is now close to production/reusable experiment infrastructure for Stage1 and Stage1b:

- Config-driven model, layer, position, data, and runtime choices.
- Shared Stage1 and Stage1b execution path.
- Prompt-only / pre-CoT controls implemented as first-class scan positions.
- Dynamic probe scheduling is available for GPU work distribution.
- Stage outputs are archived stage-by-stage and hidden arrays are cleaned.

Remaining hardening before calling it fully industrial-grade:

- Add automated post-run sanity checks that assert required summary files exist and contain all requested positions.
- Add a one-command backup target that works even on fresh pods by injecting only the required rclone remote.
- Add CI/smoke tests for config parsing and output-schema compatibility.

## Artifact Backup

Complete Stage1 + Stage1b remote archive:

- RunPod archive: `/workspace/stage1-results_1p5b_8b_stage1_stage1b_20260630T054640Z.tar.gz`
- Size: 248 MB
- SHA256: `3fc07550bca30c5d28ee63c49eee4d970076169abde96c5cad969067105b9c87`
- GDrive target: `safechain_gdrive:Research/cot-safety/runpod_backups/stage1_stage1b/`

2026-07-01 R2 archive update:

- Cloudflare R2 canonical archive root: `cloudflare_r2_cot_safety:cot-safety/stage1/20260701-a6000/`
- Current verified size: 69 objects, 708.598 GiB.
- 1.5B archive: `deepseek-1p5b/` contains `data/`, `runs/results/`, `runs/hidden/`, and `runs/logs/` (55 objects, 152.862 GiB).
- 8B archive: `deepseek-8b/` currently contains `runs/hidden/` only (14 objects, 555.736 GiB). 8B result summaries are tracked in GitHub under `res/deepseek-8b/` and this Stage1/Stage1b summary; the original RunPod 8B `runs/results/` and `runs/logs/` tar archives were not present in the R2 source backup.
- The old `runpod-backups/20260701-stage1-1p5b-a6000-tarstream/` and `runpod-backups/20260701-stage1-8b-a6000-stage1/` prefixes were deleted after verifying the canonical archive.

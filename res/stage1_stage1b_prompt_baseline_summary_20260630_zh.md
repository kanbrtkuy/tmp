# Stage1 + Stage1b Prompt-Baseline 锚定结果

日期：2026-06-30

模型：
- DeepSeek-R1-Distill-Qwen-1.5B
- DeepSeek-R1-Distill-Llama-8B

Probe 类型：只使用 linear probe。

目的：记录最新的 Stage1 position scan 和 Stage1b prompt-only / pre-CoT baseline 结果。1.5B 和 8B 使用相同的数据与扫描协议。Stage1b 用来回应一个 construct-validity 风险：早期 CoT probe 的信号是否只是 prompt classification，而不是 trajectory signal。

## 实验协议

Stage1 在 CoT token 位置和 dense layer grid 上扫描 unsafe/safe 信号。

Stage1b 在同样扫描流程中加入 prompt-only 和 pre-CoT 对照位置：

- `last_prompt_token`：严格的 prompt-only baseline，即用户 prompt 的最后一个 token。
- `pre_think`：`<think>` 之前的边界位置。
- `think_last`：`<think>` 边界状态，是更强的 pre-CoT control，但不是严格 prompt-only baseline。

论文中的标准口径：

- 主 prompt-only baseline：`last_prompt_token`。
- 更强的 pre-CoT controls：`pre_think`、`think_last`。
- `best` 用于定位最强 token/layer。
- construct-validity 论证应主要看 across-layer mean/median，避免只依赖 best-of-scan。
- matched-layer comparison 可作为补充，避免把 CoT 的最强层和 prompt 的弱层不公平比较。

## Stage1 Position Scan

### 1.5B

| Probe | 选择口径 | 位置 | Layer / combine | Test AUROC | Test AUPRC | Heldout AUROC | Heldout AUPRC |
|---|---|---:|---|---:|---:|---:|---:|
| single-layer linear | best test | cot_3 | layer 17 | 0.980 | 0.865 | 0.890 | 0.693 |
| single-layer linear | best heldout | cot_4 | layer 21 | 0.976 | 0.815 | 0.930 | 0.728 |
| multilayer linear | best test | cot_3 | concat dense layers | 0.978 | 0.831 | 0.877 | 0.695 |
| multilayer linear | best heldout | cot_2 | mean dense layers | 0.963 | 0.795 | 0.898 | 0.659 |

### 8B

| Probe | 选择口径 | 位置 | Layer / combine | Test AUROC | Test AUPRC | Heldout AUROC | Heldout AUPRC |
|---|---|---:|---|---:|---:|---:|---:|
| single-layer linear | best test | cot_3 | layer 17 | 0.985 | 0.896 | 0.923 | 0.741 |
| single-layer linear | best heldout | cot_4 | layer 12 | 0.983 | 0.889 | 0.942 | 0.775 |
| multilayer linear | best test | cot_3 | mean dense layers | 0.977 | 0.834 | 0.917 | 0.714 |
| multilayer linear | best heldout | cot_3 | concat dense layers | 0.976 | 0.870 | 0.937 | 0.795 |

Stage1 结论：两个模型的 unsafe/safe 信号都集中在 early CoT。当前 dense scan 中，single-layer heldout 最强点都在 `cot_4`，而 in-domain/test 最强信号仍集中在 `cot_3`。

### 和之前 Stage1 结果的一致性

最新 dense scan 和之前的 token-position 结论总体一致：最强信号仍在 early CoT，尤其是 `cot_3` 和 `cot_4` 附近。具体最强 layer 不应过度解读，因为这次使用了更密的 layer grid，而且 best layer 会随 split 和扫描密度变化。

| 模型 | 口径 | 最新最强点 | 是否和之前一致 |
|---|---|---|---|
| 1.5B | single-layer test | `cot_3` layer 17 | 一致，之前也是 `cot_3` / layer-17 区域强。 |
| 1.5B | single-layer heldout | `cot_4` layer 21 | token 仍在 `cot_3`/`cot_4` 邻域，但这次 heldout 更偏 `cot_4`。 |
| 8B | single-layer test | `cot_3` layer 17 | 一致，之前 `cot_3` / layer-17 区域也很强。 |
| 8B | single-layer heldout | `cot_4` layer 12 | token 结论仍是 `cot_4`，但具体最强 layer 从之前常说的 layer20 附近变到 layer12。 |

论文安全表述：

> Across both models, the strongest probe signal consistently appears in early CoT positions, especially around `cot_3` and `cot_4`. The exact best layer varies with scan density and split, but the token-position conclusion is stable.

如果用于指导 Stage2 pause 插入位置：1.5B 仍支持 `before cot_3` / `before cot_4` 邻域，历史主线偏 `cot_3`；8B 仍建议主方案使用 `before cot_4`，因为 heldout 最强和先前 8B 结果都支持 `cot_4`。

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

## Construct-Validity 结论

这些结果不支持“prompt-only 已经达到和 early CoT 相同可分性”的失败模式。Prompt-only 和 pre-CoT controls 确实包含明显的风险信号，但 early CoT positions 会进一步增强这个信号。

对 1.5B，严格 prompt-only 的 heldout mean AUROC 为 0.785，而 `cot_3` 和 `cot_4` 分别达到 0.868 和 0.891。即使和更强的 pre-CoT controls 相比，`cot_4` 的 heldout mean AUROC 仍提升约 0.058 到 0.061。

对 8B，严格 prompt-only 的 heldout mean AUROC 为 0.801，而 `cot_3` 和 `cot_4` 分别达到 0.912 和 0.921。即使和更强的 pre-CoT controls 相比，`cot_4` 的 heldout mean AUROC 仍提升约 0.062 到 0.078。

论文安全表述：

> Prompt-conditioned risk signal is already present before CoT, but it is substantially sharpened during early CoT. Therefore, Stage1 is not merely prompt classification; it identifies trajectory positions where unsafe/safe separability becomes stronger.

## Leave-One-Source-Out 泛化结果

2026-07-01 更新：Stage1/Stage1b 已补充 leave-one-source-out 风格的 source-family heldout 评估。每个 family 作为 heldout，报告 family-level AUROC。1.5B 的 multi-source family 结果从 R2 恢复的 `predictions.jsonl` 重新合并计算；8B 使用最新 `stage1_loso_summary` 聚合结果。

### 1.5B LOSO

| Heldout family | Position-scan 最强点 | AUROC | Prompt-baseline 模块最强点 | AUROC | Trajectory - prompt gap |
|---|---|---:|---|---:|---:|
| `reasoningshield_train_sft` | `cot_3 / layer16` single | 0.908 | `cot_7 / layer22` single | 0.898 | +0.062 |
| `reasoningshield_train_dpo` | `cot_1 / concat` multilayer | 0.787 | `assistant_last / layer17` single | 0.798 | -0.006 |
| `reasoningshield_test` | `cot_4 / layer17` single | 0.932 | `cot_4 / concat` multilayer | 0.933 | +0.079 |
| `star_vs_harmthoughts` | `cot_4 / layer17` single | 0.933 | `cot_4 / layer17` single | 0.934 | +0.015 |
| `aidsafe_vs_harmthoughts` | `cot_0 / layer4` single | 1.000 | `cot_2 / layer4` single | 1.000 | +0.120 |
| `unsafechain_vs_harmthoughts` | `cot_1 / layer7` single | 0.995 | `cot_2 / layer22` single | 0.981 | +0.083 |

1.5B position-scan LOSO mean AUROC：0.926；std：0.071。

### 8B LOSO

| Heldout family | Position-scan 最强点 | AUROC | Prompt-baseline 模块最强点 | AUROC | Trajectory - prompt gap |
|---|---|---:|---|---:|---:|
| `reasoningshield_train_sft` | `cot_3 / layer18` single | 0.909 | `cot_2 / layer14` single | 0.852 | +0.001 |
| `reasoningshield_train_dpo` | `cot_3 / concat` multilayer | 0.807 | `cot_3 / concat` multilayer | 0.807 | +0.013 |
| `reasoningshield_test` | `cot_3 / layer14` single | 0.920 | `cot_3 / layer12` single | 0.924 | +0.066 |
| `star_vs_harmthoughts` | `cot_4 / layer12` single | 0.929 | `cot_4 / layer16` single | 0.933 | +0.012 |
| `aidsafe_vs_harmthoughts` | `cot_0 / layer4` single | 1.000 | `cot_2 / layer14` single | 0.999 | +0.118 |
| `unsafechain_vs_harmthoughts` | `cot_0 / layer6` single | 0.995 | `cot_2 / layer12` single | 0.985 | +0.086 |

8B position-scan LOSO mean AUROC：0.927；std：0.064。

### LOSO 结论

LOSO 结果支持两个层面的结论：

- 在 ReasoningShield 系列 family 上，最强泛化信号仍主要位于 early CoT，尤其是 `cot_3` / `cot_4` 邻域。这和 Stage1 / Stage1b 的主结论一致。
- `aidsafe_vs_harmthoughts` 和 `unsafechain_vs_harmthoughts` 中出现 `cot_0` 近乎完美可分，说明这些 family 很可能包含 source / format artifact。它们适合作为诊断信号，但不应被直接解释为纯 trajectory-monitoring 证据。

论文安全表述：

> Leave-one-source-out evaluation shows that early-CoT risk signals transfer across ReasoningShield-style heldout families. Some mixed-source families are nearly separable at the first CoT token, indicating source or format artifacts; we therefore report them separately and avoid over-claiming trajectory specificity on those splits.

## 当前还不能证明什么

这个结果已经回应了 prompt-only / pre-CoT construct-validity 问题，并补充了 source-family LOSO 泛化评估。仍需谨慎的是：LOSO family 之间并不完全等价，部分 mixed-source family 存在明显 source / format artifact。后续论文写作中应把 ReasoningShield 系列结果作为主要泛化证据，把 `aidsafe` / `unsafechain` 相关 family 标为 artifact-prone diagnostic splits。

## Pipeline 状态

当前 Stage1 + Stage1b 代码可以通过 config 复用，并已经在 A6000 RunPod 节点上完整跑通 1.5B 和 8B。最新 runner 会把每个 stage 从 hot storage 归档到 cold storage，并在 stage 结束后清理 hidden arrays，避免之前 `/dev/shm` 或 hot path 堆积导致失败的问题。

从工程上看，Stage1 + Stage1b 已经接近可复用的实验 pipeline：

- 模型、层、位置、数据、runtime 都由 config 控制。
- Stage1 和 Stage1b 共享执行路径。
- Prompt-only / pre-CoT controls 已作为一等扫描位置实现。
- Probe 训练支持 dynamic scheduling，减少 GPU 空转。
- 每个 stage 都会归档输出并清理 hidden arrays。

在完全称为工业级之前，仍建议补充：

- 增加 post-run sanity check，自动检查 summary 文件是否存在、位置是否完整。
- 增加 fresh pod 上的一键备份目标，自动只注入必要的 rclone remote。
- 增加 config parsing 和 output-schema compatibility 的 CI/smoke tests。

## Artifact Backup

完整 Stage1 + Stage1b 远端归档：

- RunPod archive: `/workspace/stage1-results_1p5b_8b_stage1_stage1b_20260630T054640Z.tar.gz`
- Size: 248 MB
- SHA256: `3fc07550bca30c5d28ee63c49eee4d970076169abde96c5cad969067105b9c87`
- GDrive target: `safechain_gdrive:Research/cot-safety/runpod_backups/stage1_stage1b/`

2026-07-01 R2 归档更新：

- Cloudflare R2 canonical archive root：`cloudflare_r2_cot_safety:cot-safety/stage1/20260701-a6000/`
- 当前校验后的大小：69 objects，708.598 GiB。
- 1.5B archive：`deepseek-1p5b/` 包含 `data/`、`runs/results/`、`runs/hidden/` 和 `runs/logs/`（55 objects，152.862 GiB）。
- 8B archive：`deepseek-8b/` 当前只包含 `runs/hidden/`（14 objects，555.736 GiB）。8B 的结果摘要已记录在 GitHub 的 `res/deepseek-8b/` 和本 Stage1/Stage1b summary 中；原始 RunPod 8B `runs/results/` 与 `runs/logs/` tar archives 没有出现在本次 R2 源备份中。
- 旧的 `runpod-backups/20260701-stage1-1p5b-a6000-tarstream/` 和 `runpod-backups/20260701-stage1-8b-a6000-stage1/` prefixes 已在校验 canonical archive 后删除。

# Stage 1 自然 Pair 实验结果 - 2026-07-03

这份结果文档只包含聚合指标，不包含 raw prompts 或 raw trajectories。

## 数据快照

下面的 hidden-probe 结果使用的是冻结后的 export。32B 生成任务在这些 export 之后还继续运行，所以最新生成库存单独记录。

| 数据版本 | Hidden probe 使用的 pair 数 | Train/val/test pairs | 说明 |
|---|---:|---:|---|
| Natural 8B generated/generated | 467 | 374 / 47 / 46 | 同一个 prompt 下，safe/unsafe 两边都由 R1-8B 生成；每个 prompt 最多 50 次采样后筛选 |
| Natural 8B generated-safe/original-unsafe | 663 | 530 / 66 / 67 | R1-8B 生成 safe candidate，配原始数据集 unsafe reference |
| Natural 32B generated/generated | 359 | 287 / 36 / 36 | 当前冻结的 32B export，用于 dense hidden-probe scan |
| Natural 32B generated-safe/original-unsafe | 555 | 444 / 56 / 55 | 当前冻结的 export，用于已完成的 32B safe/original robustness run |

冻结 export 之后，32B 生成节点上观察到的最新 judged inventory：

| 生成库存项目 | 当前数量 | 说明 |
|---|---:|---|
| 32B generated-safe + generated-unsafe，双方都 quality-pass | 406 pairs | 只是生成库存；还不是已冻结的 hidden-probe export |
| 32B generated-safe + original unsafe reference | 607 pairs | 只是生成库存；还不是已冻结的 hidden-probe export |

## Cloudflare R2 归档

这些 run 对应的 A100 natural-pair workspace 快照已归档到：

```text
cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260703-a100-natural-pairs/
```

归档结构、校验结果、恢复命令和已知备份后差异见
`docs/stage1_paired_r2_archive_260703.md` 与
`docs/stage1_paired_r2_archive_260703_zh.md`。对应计划文档为
`plan/stage1_natural_pair_experiment_plan_260703_zh.md`。

## CPU 表层 Baselines

### Natural 8B generated/generated

| Baseline | Test balanced accuracy | Test AUROC |
|---|---:|---:|
| length-only | 0.7500 | 0.8261 |
| prompt-only TF-IDF | 0.5000 | 0.5000 |
| word TF-IDF | 0.8913 | 0.9660 |
| word BoW | 0.8696 | 0.9716 |
| char TF-IDF | 0.8478 | 0.9594 |
| first-sentence-removed TF-IDF | 0.8696 | 0.9641 |

解释：prompt 文本本身不能区分标签，但 CoT 文本有很强的浅层可分性；长度也是明显混淆因素。

### Natural 8B generated-safe/original-unsafe

| Baseline | Test balanced accuracy | Test AUROC |
|---|---:|---:|
| length-only | 0.7612 | 0.8817 |
| prompt-only TF-IDF | 0.5000 | 0.5000 |
| word TF-IDF | 0.8582 | 0.9423 |
| word BoW | 0.8657 | 0.9546 |
| char TF-IDF | 0.9104 | 0.9472 |
| first-sentence-removed TF-IDF | 0.8657 | 0.9410 |

解释：把自然生成的 safe CoT 与原始 unsafe reference 搭配时，表层可分性仍然很强，尤其是长度、词汇和风格差异。

## Hidden-State Probe 结果

下表记录 diagnostic top rows。`test-max` 适合用来定位信号，但论文主结果应该用 validation-selected position/layer。

### Natural Generated/Generated Pairs

| 实验 | Extractor | Test-max 位置/层 | Test-max AUROC | Validation-selected 位置/层 | Validation-selected test AUROC | 说明 |
|---|---|---|---:|---|---:|---|
| A prime OpenAI rewrite Stage1 | R1-1.5B | cot_2 / layer14 | 0.9886 | n/a | n/a | confound diagnostic，不作为主证据 |
| A prime OpenAI rewrite Stage1b | R1-1.5B | cot_2 / layer14 | 0.9844 | n/a | n/a | Stage1b 下仍接近完美，说明 artifact 很强 |
| Natural 8B generated/generated Stage1 | R1-1.5B | cot_96 / layer18 | 0.8384 | n/a | n/a | prompt-only 接近随机，但 CoT-position 有信号 |
| Natural 8B generated/generated Stage1b | R1-1.5B | cot_4 / layer18 | 0.7328 | n/a | n/a | 早期 CoT 有信号；prompt/pre-CoT 接近随机 |
| Natural 8B generated/generated Stage1 dense | R1-8B | cot_120 / layer15 | 0.8913 | cot_120 / layer20 | 0.8318 | dense rerun；最强信号在 late-CoT |
| Natural 8B generated/generated Stage1b dense | R1-8B | cot_16 / layer14 | 0.7632 | cot_5 / layer14 | 0.6983 | early-CoT 信号弱于 late-CoT Stage1 |
| Natural 8B generated/generated Stage1 | R1-32B | cot_128 / layer32 | 0.8648 | n/a | n/a | 早期 sparse grid 的跨模型大小 extractor diagnostic |
| Natural 8B generated/generated Stage1b | R1-32B | cot_4 / layer60 | 0.7375 | n/a | n/a | 早期 sparse grid 的跨模型大小 extractor diagnostic |
| Natural 32B generated/generated Stage1 dense | R1-8B | cot_160 / layer14 | 0.8841 | cot_8 / layer6 | 0.7488 | 反方向跨模型大小 diagnostic；all-feature val-selected test 为 0.6975 |
| Natural 32B generated/generated Stage1b dense | R1-8B | cot_9 / layer8 | 0.8044 | cot_9 / layer10 | 0.7188 | 反方向跨模型大小 diagnostic；all-feature val-selected test 为 0.6975 |
| Natural 32B generated/generated Stage1 dense | R1-32B | cot_160 / layer30 | 0.8944 | cot_144 / layer36 | 0.7654 | 32B matched diagnostic；test-max 与 val-selected 差距较大 |
| Natural 32B generated/generated Stage1b dense | R1-32B | cot_4 / layer52 | 0.8148 | cot_9 / layer32 | 0.7890 | early-CoT signal 较强；prompt/pre-CoT controls 接近随机 |

### Natural Generated-Safe / Original-Unsafe Robustness

| 实验 | Extractor | Top 位置/层 | Test AUROC | 说明 |
|---|---|---|---:|---|
| Natural 32B generated-safe/original-unsafe Stage1 | R1-32B | cot_5 / layer28 | 0.7661 | 555-pair 冻结 export 上已完成的 robustness run |
| Natural 32B generated-safe/original-unsafe Stage1b linear | R1-32B | cot_4 / layer20 | 0.7579 | early-CoT signal 仍然存在 |
| Natural 32B generated-safe/original-unsafe Stage1b multilayer | R1-32B | cot_4 / mean layers | 0.7516 | multilayer summary 与 single-layer 结果接近 |

## Prompt / Pre-CoT Controls

Dense Stage1b controls 直接回应 probe 是否只是在分类 prompt 的问题。

| 数据 | Extractor | 最强 prompt/pre-CoT AUROC | 解释 |
|---|---|---:|---|
| Natural 8B generated/generated | R1-8B | 0.5000 | prompt/pre-CoT 位置接近随机 |
| Natural 32B generated/generated | R1-8B | 0.5000 | prompt/pre-CoT 位置接近随机 |
| Natural 32B generated/generated | R1-32B | 约 0.51 | prompt/pre-CoT 位置基本接近随机 |

因此，在这些 natural-pair runs 中，CoT-position probes 不能用 prompt-only hidden states 解释。但这还不能排除 CoT 内部的表层混淆，例如长度、风格、refusal pattern 或 generation artifacts。

## Generator-vs-Extractor 覆盖情况

| CoT generator | Hidden extractor | Stage1 top test AUROC | Stage1b top test AUROC | 状态 |
|---|---|---:|---:|---|
| R1-8B | R1-1.5B | 0.8384 | 0.7328 | 已完成 |
| R1-8B | R1-8B | 0.8913 | 0.7632 | 已完成；dense rerun |
| R1-8B | R1-32B | 0.8648 | 0.7375 | 已完成；早期 sparse grid |
| R1-32B | R1-8B | 0.8841 | 0.8044 | 已完成；dense rerun |
| R1-32B | R1-32B | 0.8944 | 0.8148 | 已完成；dense rerun |

## Source Provenance / LOSO 状态

natural 8B generated/generated 的 provenance re-join 成功覆盖 934 rows / 467 complete pairs。但是 drop ambiguous source-family pairs 后：

- eligible pairs: 335
- dropped ambiguous pairs: 132
- 剩余 source family: 只有 ReasoningShield

因此，这个 snapshot 还不能做真正的 leave-one-source-out。下一轮数据构建需要更均衡的 source family，或者恢复更完整的 provenance。

## 主要结论

- 自然 same-prompt pairs 能显著缓解 prompt-classification 的担忧：已完成的 natural-pair runs 中，prompt-only text 和 pre-CoT hidden baselines 基本接近随机。
- dense rerun 进一步加强了这一点：prompt/pre-CoT controls 仍然接近随机，而 CoT-position hidden states 有清晰信号。
- Stage1 的 test maxima 经常移动到更靠后的 CoT 位置，dense scan 中尤其集中在 cot_120 到 cot_160 附近。
- Stage1b 的 early-CoT 位置仍然有信息量，通常在 cot_4 到 cot_9 附近，但整体弱于 late-CoT Stage1 signal。
- 跨模型大小 diagnostic 比较积极：32B 生成的 pair 可以被 8B hidden extractor 读出，8B 生成的 pair 也可以被 32B extractor 读出。
- generated-safe/original-unsafe robustness 在 32B extractor 上弱于 generated/generated，但仍明显高于随机。
- surface baselines 同样很强，这是当前最大限制：目前结果能说明 separability 存在，但还不能干净说明它是 safety semantics。
- 多个 dense scans 中 test-set max 和 validation-selected test score 差距较大。主结果应该报告 validation-selected position/layer，而不是 post-hoc test maximum。
- 下一步最关键的是 source-balanced LOSO、token-matched truncation、embedding-based surface baselines、paired bootstrap confidence intervals，以及 validation-selected hidden-probe reporting。

# Stage 1 自然 Pair 实验结果 - 2026-07-03

这份结果文档只包含聚合指标，不包含 raw prompts 或 raw trajectories。

## 数据快照

| 数据版本 | Pair 数 | Train/val/test pairs | 说明 |
|---|---:|---:|---|
| Natural 8B generated/generated | 467 | 374 / 47 / 46 | 同一个 prompt 下，safe/unsafe 两边都由 R1-8B 生成；每个 prompt 最多 50 次采样后筛选 |
| Natural 8B generated-safe/original-unsafe | 663 | 530 / 66 / 67 | R1-8B 生成 safe candidate，配原始数据集 unsafe reference |
| Natural 32B generated/generated | 359 | 287 / 36 / 36 | 当前 32B partial snapshot，用于 32B-hidden diagnostic |
| Natural 32B generated-safe/original-unsafe | 555 | 444 / 56 / 55 | 当前 32B partial snapshot；Stage1/Stage1b 正在运行 |

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

下表按 test AUROC 的 top row 记录，适合作为 diagnostic。正式主结果应该使用 validation-selected position/layer，而不是 test-set max。

| 实验 | Extractor | Test AUROC 最高位置/层 | Test AUROC | 说明 |
|---|---|---|---:|---|
| A prime OpenAI rewrite Stage1 | R1-1.5B | cot_2 / layer14 | 0.9886 | confound diagnostic，不作为主证据 |
| A prime OpenAI rewrite Stage1b | R1-1.5B | cot_2 / layer14 | 0.9844 | Stage1b 下仍接近完美，说明 artifact 很强 |
| Natural 8B generated/generated Stage1 | R1-1.5B | cot_96 / layer18 | 0.8384 | prompt-only 接近随机，但 CoT-position 有信号 |
| Natural 8B generated/generated Stage1b | R1-1.5B | cot_4 / layer18 | 0.7328 | 早期 CoT 有信号；prompt/pre-CoT 接近随机 |
| Natural 8B generated/generated Stage1 | R1-8B | cot_128 / layer10 | 0.8403 | test max 主要出现在较后 CoT 位置 |
| Natural 8B generated/generated Stage1b | R1-8B | cot_7 / layer20 | 0.7462 | 早期 CoT signal 仍然存在 |
| Natural 8B generated/generated Stage1 | R1-32B | cot_128 / layer32 | 0.8648 | 跨模型大小 extractor diagnostic；上一版结果总结漏写了这一组 |
| Natural 8B generated/generated Stage1b | R1-32B | cot_4 / layer60 | 0.7375 | 早期 CoT signal 与 8B extractor 的 Stage1b 结果接近 |
| Natural 32B generated/generated Stage1 | R1-32B | cot_128 / layer32 | 0.8441 | representation-scale diagnostic |
| Natural 32B generated/generated Stage1b | R1-32B | cot_4 / layer44 | 0.8164 | prompt/pre-CoT baseline 仍接近随机；早期 CoT signal 较强 |

### Generator-vs-Extractor 覆盖情况

| CoT generator | Hidden extractor | Stage1 top test AUROC | Stage1b top test AUROC | 状态 |
|---|---|---:|---:|---|
| R1-8B | R1-1.5B | 0.8384 | 0.7328 | 已完成 |
| R1-8B | R1-8B | 0.8403 | 0.7462 | 已完成 |
| R1-8B | R1-32B | 0.8648 | 0.7375 | 已完成 |
| R1-32B | R1-32B | 0.8441 | 0.8164 | 已完成 |
| R1-32B | R1-8B | n/a | n/a | 尚未运行 |

## Source Provenance / LOSO 状态

natural 8B generated/generated 的 provenance re-join 成功覆盖 934 rows / 467 complete pairs。但是 drop ambiguous source-family pairs 后：

- eligible pairs: 335
- dropped ambiguous pairs: 132
- 剩余 source family: 只有 ReasoningShield

因此，这个 snapshot 还不能做真正的 leave-one-source-out。下一轮数据构建需要更均衡的 source family，或者恢复更完整的 provenance。

## 当前正在运行的实验

实验：

- 数据：natural 32B generated-safe/original-unsafe
- Pair 数：555
- Split：444 / 56 / 55 pairs
- Extractor：DeepSeek-R1-Distill-Qwen-32B
- Pipeline：`pipelines/runpod_stage1_natural_pairs_32b_safeorig_32b_a100_1x.sh`
- Configs：
  - `configs/experiment/stage1_natural_pairs_32b_safeorig_32b_a100_1x.yaml`
  - `configs/experiment/stage1b_natural_pairs_32b_safeorig_32b_a100_1x.yaml`

最后一次状态检查时，train hidden extraction 已完成，作业已进入 validation hidden extraction。最终 Stage1/Stage1b probe metrics 尚未产出。

后续远端重扫时，这个 run 的 train/validation/test hidden extraction 都已经完成，并且 single-position scan 已经开始。重扫时最终 Stage1/Stage1b summary 文件仍未产出。

32B 生成节点在重扫时处于空闲状态。节点上有 13,905 条 generated candidate rows，其中 11,405 条已经 judge。因此当前 32B pair exports 使用的是这 11,405 条已 judge snapshot，而不是包含未 judge rows 的完整 candidate 文件。

## 主要结论

- 自然 same-prompt pairs 能显著缓解 prompt-classification 的担忧：已完成的 natural-pair runs 中，prompt-only text 和 pre-CoT hidden baselines 基本接近随机。
- 在 1.5B、8B、32B extractors 上，CoT-position hidden states 都能看到 safe/unsafe signal。
- 漏写的跨模型大小实验，也就是 R1-8B CoTs + 32B hidden extractor，已经完成，并且 late-CoT Stage1 test maximum 与其他结果相当或略强。
- Stage1 的 test-set max 经常出现在更靠后的 CoT 位置，而 Stage1b 的早期 CoT 位置也保持有信息量。
- 但是 surface baselines 同样很强，这是当前最大限制：目前结果能说明 separability 存在，但还不能干净说明它是 safety semantics，而不是风格、长度或生成 artifact。
- 下一步最关键的是 source-balanced LOSO、token-matched truncation、embedding-based surface baselines，以及 validation-selected hidden-probe reporting。

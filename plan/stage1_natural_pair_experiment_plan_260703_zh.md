# Stage 1 自然 Pair 实验计划 - 2026-07-03

这份文档记录我们在 OpenAI rewrite A/B 数据暴露出强表层可分性之后，对 Stage 1 实验路线做出的更新。它区分哪些结果只是 diagnostic，哪些可以作为主线证据，以及在正式下 Stage 1 claim 之前还需要补哪些实验。

## 目标

Stage 1 的目标是验证 safe 与 unsafe reasoning trajectory 是否在模型 hidden states 中可分，并且这种可分性不能只来自 prompt-only 信息或浅层文本 artifact。

当前方案主要回应两个问题：

- 在 teacher-forced trace 中，早期 hidden state 可能只是在分类 prompt，而不是 monitoring reasoning trajectory。
- 如果 probe 分数很高，需要排除 source、format、length、generation style 等混淆因素，才能把它解释成 safety-relevant latent signal。

## 数据路线

| 数据路线 | Pair 定义 | 当前用途 | 状态 |
|---|---|---|---|
| A prime OpenAI rewrite | OpenAI safe rewrite vs original/open unsafe side | 只作为 confound diagnostic | 已完成；不作为主 claim 数据 |
| Natural 8B generated/generated | 同一个 prompt 下，R1-8B 生成 safe CoT vs R1-8B 生成 unsafe CoT | 主要自然 pair pilot | 已完成 surface baseline 和多组 hidden probe |
| Natural 8B generated-safe/original-unsafe | R1-8B 生成 safe CoT vs 原始数据集 unsafe reference CoT | 检查与原始 unsafe trajectory 搭配时是否类似 | surface baseline 已完成；hidden probe 可补跑 |
| Natural 32B generated/generated | 同一个 prompt 下，R1-32B 生成 safe CoT vs R1-32B 生成 unsafe CoT | 模型规模/生成器规模 diagnostic | 当前 partial snapshot 上 32B-hidden Stage1/Stage1b 已完成 |
| Natural 32B generated-safe/original-unsafe | R1-32B 生成 safe CoT vs 原始数据集 unsafe reference CoT | 当前正在做的 robustness 实验 | 32B hidden extractor 的 Stage1/Stage1b 正在运行 |

## 已完成检查

- Stage1b 已加入 prompt-only 和 pre-CoT baseline。
- 对 natural 8B pair variants 跑了 CPU surface baselines：
  - length-only
  - prompt-only TF-IDF
  - word TF-IDF
  - word BoW
  - character n-gram TF-IDF
  - first-sentence-removed TF-IDF
- 对 natural 8B generated/generated 做了 source provenance re-join 和 LOSO fold 构建尝试。
- 已跑 hidden probe 的组合包括：
  - A prime OpenAI rewrite + 1.5B hidden extractor
  - natural 8B generated/generated + 1.5B hidden extractor
  - natural 8B generated/generated + 8B hidden extractor
  - natural 32B generated/generated + 32B hidden extractor

## 当前解释原则

- A prime 更像是失败案例：它说明当 rewrite/style artifact 存在时，hidden probe 可以接近完美。
- Natural generated/generated pairs 更接近我们的目标，因为 safe/unsafe 两边来自同一个模型族和同一个生成 pipeline。
- prompt-only/pre-CoT 接近随机是必要条件，但还不充分；它只能排除 prompt classification，不能排除 CoT 内部的表层文本混淆。
- TF-IDF/BoW 分数很高，说明论文中必须把 surface baseline 和 hidden probe 一起报告。
- 32B hidden 结果只能作为 representation scale diagnostic，不能替代 source-level transfer。
- 主结果不能用 test-set max；应该在 validation 上选 position/layer，再报告 test。

## 强化 Stage 1 Claim 前还需要做的事

1. 完成 natural 32B generated-safe/original-unsafe 的 Stage1/Stage1b。
2. 如果要直接对比 generated/generated 与 generated-safe/original-unsafe，需要补跑 natural 8B generated-safe/original-unsafe 的 hidden probe。
3. 构建 source-balanced natural pairs，或者恢复更完整的 source-family provenance。
4. 当至少两个 source family 有足够 pair 后，跑真正的 leave-one-source-out。
5. 每个 fold 都报告：
   - prompt-only/pre-CoT hidden baseline
   - CoT-position hidden probe
   - length-only baseline
   - word/char TF-IDF 和 BoW baselines
   - first-sentence-removed baseline
   - token-matched truncation baseline
   - embedding-based surface baseline
6. 使用 paired bootstrap confidence intervals。
7. 报告 validation-selected position/layer，而不是 post-hoc test maxima。
8. 持续显式报告残余混淆因素：length、refusal style、answer template、source family、generator model、judge selection bias。

## 当前 Claim 边界

目前可以谨慎表述为：

> 在 same-prompt natural-pair 设置下，prompt-only hidden states 基本接近随机，而 CoT-position hidden states 往往包含 safe/unsafe signal。但是浅层 CoT 文本 baseline 也很强，所以还不能把这个 signal 干净地解释为 safety semantics，而不是 trajectory style、length 或 generation artifacts。

更强的 claim，也就是 Stage 1 找到了 safety-relevant latent manifold，需要等 source-transfer 和 surface-control 实验补齐之后再下。


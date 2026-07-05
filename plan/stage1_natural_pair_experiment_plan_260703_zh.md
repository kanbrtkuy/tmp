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
| Natural 8B generated/generated | 同一个 prompt 下，R1-8B 生成 safe CoT vs R1-8B 生成 unsafe CoT | 主要自然 pair pilot | 已完成 surface baseline 和多组 hidden-probe scan，包括 8B extractor dense rerun |
| Natural 8B generated-safe/original-unsafe | R1-8B 生成 safe CoT vs 原始数据集 unsafe reference CoT | 检查与原始 unsafe trajectory 搭配时是否类似 | surface baseline 已完成；如果需要与 generated/generated 直接对比，仍可补 hidden probe |
| Natural 32B generated/generated | 同一个 prompt 下，R1-32B 生成 safe CoT vs R1-32B 生成 unsafe CoT | 模型规模/生成器规模 diagnostic | 当前 359-pair 冻结 export 上，32B-hidden 和 8B-hidden dense scans 已完成 |
| Natural 32B generated-safe/original-unsafe | R1-32B 生成 safe CoT vs 原始数据集 unsafe reference CoT | 与原始 unsafe references 对比的 robustness 实验 | 当前 555-pair 冻结 export 上，32B hidden extractor 的 Stage1/Stage1b 已完成 |

32B 生成库存后续仍在增长。最近观察到的 judged inventory 为：406 对 generated/generated quality-pass pairs，以及 607 对 generated-safe/original-unsafe pairs。

## Generator-vs-Extractor 矩阵

自然 generated/generated 实验目前已经完成五组 generator/extractor 组合：

| CoT generator | Hidden extractor | 状态 | 用途 |
|---|---|---|---|
| R1-8B | R1-1.5B | 已完成 | 在 8B natural pairs 上的小模型 extractor diagnostic |
| R1-8B | R1-8B | 已完成；dense rerun 已完成 | generator/extractor 匹配基线 |
| R1-8B | R1-32B | 已完成 | 跨模型大小 extractor diagnostic；检验 8B 生成轨迹中的信号能否在 32B 表征中读出 |
| R1-32B | R1-8B | 已完成；dense rerun 已完成 | 反方向跨模型大小 transfer 检查 |
| R1-32B | R1-32B | 已完成；dense rerun 已完成 | 32B generator/extractor 匹配 diagnostic |

## 已完成检查

- Stage1b 已加入 prompt-only 和 pre-CoT baseline。
- 已完成以下 dense Stage1/Stage1b scans：
  - natural 8B generated/generated + 8B hidden extractor
  - natural 32B generated/generated + 8B hidden extractor
  - natural 32B generated/generated + 32B hidden extractor
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
  - natural 8B generated/generated + 32B hidden extractor
  - natural 32B generated/generated + 8B hidden extractor
  - natural 32B generated/generated + 32B hidden extractor
  - natural 32B generated-safe/original-unsafe + 32B hidden extractor

## Cloudflare R2 归档

2026-07-03 A100 natural-pair workspace 快照已归档到：

```text
cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260703-a100-natural-pairs/
```

归档结构、校验结果和恢复命令记录在
`docs/stage1_paired_r2_archive_260703.md` 和
`docs/stage1_paired_r2_archive_260703_zh.md`。该归档也已从自然 pair 结果文档
`res/stage1_natural_pair_experiment_results_260703_zh.md` 互链。

2026-07-04 的 R1-8B remaining-prompt follow-up 和小型 surface-control 产物已作为增量归档备份到：

```text
cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260704-a100-8b-remaining-n100/
```

该增量归档记录在
`docs/stage1_paired_8b_remaining_r2_archive_260704.md` 和
`docs/stage1_paired_8b_remaining_r2_archive_260704_zh.md`。它包含已完成的
remaining-prompt run，将合并后的 R1-8B safe/original pairs 从 663 提升到
746，同时包含日志、manifest 和小型代码快照。

2026-07-05 post-HB Stage1 LOSO/retune12288_b20 run、hidden archives、日志、
`/dev/shm` run artifacts 和代码快照已全量归档到：

```text
cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/
```

该归档记录在 `docs/stage1_post_hb_r2_archive_260705_zh.md`。最终 R2 size
为 38,224 objects / 64.186 GiB，并包含 `/dev/shm/cot-safety-hot/runs`
的目录式备份和 `runs.tar.gz` sidecar。

## 2026-07-05 Post-HB Stage1 计划更新

post-HB retune12288/b20 sequence 已完成，但 Fable 和 Fable-5 均不建议把
当前结果作为 hidden-state superiority 证据。当前线性 hidden-probe 设计应作为
negative/control result 记录；不再继续跑同一设计的 GPU position/layer/classifier
sweep。

Fable-5 的后续复审结论为 `ONLY AFTER GATES`。如果继续 Stage1，下一步不是调大
probe 或换模型，而是先做 CPU-only matched-horizon reanalysis：

- 对比 hidden@k 与同一前缀 token 的 surface@k，而不是 full-trajectory text。
- k 固定为 `{4, 8, 16, 32, 64}`。
- 先在已保存 hidden arrays 的 HB + WJB 上跑 Phase 1。
- 只有 Phase 1 continue criteria 和 G1-G8 非 GPU gates 都通过，才允许用 GPU
  补回 RS/SR 丢失的 hidden arrays。
- 如果 HB + WJB 上 hidden@k 对 text@k 没有 CI-separated early-horizon 优势，
  Stage1 probing 永久停止，转为 well-posed negative/control claim。

Fable-5 review 已归档在 tmp repo commit `0ea86e7`：

```text
stage1_fable_review_retune12288_b20_260705/fable_probe_redesign_response_260705.md
```

主 repo 摘要见：

```text
docs/fable_decision_review_stage1_retune12288_b20_260705.md
res/stage1_post_hb_retune12288_b20_results_260705_zh.md
```

## 2026-07-05 Post-HB Stage1 最终状态更新

上述 CPU-only gates 和后续确认实验已经完成：

- threshold/calibration reanalysis：hidden balanced accuracy 可从约 `0.604`
  提升到约 `0.708-0.710`，但这是 operating-point 修正，不改变 AUROC，也不让
  hidden 超过 surface/length controls。
- matched-horizon reanalysis：pooled hidden-minus-text delta 在 `k=4` 为
  `+0.0584`，但 `k=8` 起不稳定或为负，`k=16/32/64` 明显不支持 hidden
  优势。
- A1 score pooling：`k=4` 和 `k=8` 有早期正向 diagnostic。
- A2 feature pooling：Fable-5 要求的 preregistered feature-level rerun
  未确认 A1；`k=8` delta 为 `-0.0235` 且 CI 全负，触发
  `STOP_EQUAL_HORIZON_AND_PIVOT`。
- excluded-source lead-time confirmation：在 `strongreject_full` 和
  `reasoningshield` 上运行 A1/A2，最终 `confirmed=false`，
  `decision=drop_leadtime_claim`；Fable-5 复审确认该决策正确。

因此，Stage1 当前最终状态是：

```text
negative/control result
no more equal-horizon or lead-time rescue on this frozen test set
```

Fable-5 Round 3 fresh-path review 将这个边界进一步固化为：

```text
FRESH_PREREG_ONLY
STOP_CURRENT_STAGE1 for the frozen test set.
CODE_AND_RUN_ALLOWED is rejected for current frozen Stage1.
```

注意：这不是说 hidden 完全没信号。LOSO hidden test AUROC 大约在
`0.68-0.84`，说明 prefix-hidden state 确实含有可解码的 safe/unsafe signal。
失败的是 stronger claim：hidden 没有打赢 matched/full surface baselines，也
没有稳定 lead-time advantage。

更新后的结果总览见：

```text
res/stage1_experiment_inventory_results_260705_zh.md
review-stage/stage1_auto_improve_260705/fable5_excluded_leadtime_results_review_260705.md
```

后续如果要追求 positive result，必须进入 fresh preregistered setting，例如
Stage2/on-policy 或新的 fresh-data follow-up；不能继续在当前 frozen test set
上寻找 positive cell。

## 当前解释原则

- A prime 更像是失败案例：它说明当 rewrite/style artifact 存在时，hidden probe 可以接近完美。
- Natural generated/generated pairs 更接近我们的目标，因为 safe/unsafe 两边来自同一个模型族和同一个生成 pipeline。
- Generated-safe/original-unsafe pairs 适合作为 robustness check，但它把模型生成的 safe side 与原始 unsafe references 配在一起；unsafe side 可能来自不同模型和数据分布。
- prompt-only/pre-CoT 接近随机是必要条件，但还不充分；它只能排除 prompt classification，不能排除 CoT 内部的表层文本混淆。
- dense rerun 显示 prompt/pre-CoT controls 仍接近随机，而 CoT-position hidden states 有可测信号。
- TF-IDF/BoW 分数很高，说明论文中必须把 surface baseline 和 hidden probe 一起报告。
- 32B hidden 结果只能作为 representation scale diagnostic，不能替代 source-level transfer。
- 主结果不能用 test-set max；应该在 validation 上选 position/layer，再报告 test。

## 强化 Stage 1 Claim 前还需要做的事

1. 决定是否基于更大的 32B 生成库存重新 freeze export，还是为了可比性继续使用当前 359/555-pair 冻结 exports。
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

> 在 same-prompt natural-pair 设置下，prompt-only hidden states 基本接近随机，而 CoT-position hidden states 往往包含 safe/unsafe signal。Dense rerun 进一步加强了 prompt-only control 结果，但浅层 CoT 文本 baseline 也很强，而且 validation-selected 分数可能显著低于 test-set maxima。因此，还不能把这个 signal 干净地解释为 safety semantics，而不是 trajectory style、length 或 generation artifacts。

更强的 claim，也就是 Stage 1 找到了 safety-relevant latent manifold，需要等 source-transfer 和 surface-control 实验补齐之后再下。

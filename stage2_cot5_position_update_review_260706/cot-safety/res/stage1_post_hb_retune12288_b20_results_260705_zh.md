# Stage 1 Post-HB retune12288/b20 结果记录 - 2026-07-05

本文档记录 post-HB Stage1 LOSO/retune12288_b20 run 的聚合结论、Fable-5
复审结论和归档位置。本文档不包含 raw prompts、raw CoTs、completions 或
row-level prediction JSONL。

## Run 状态

RunPod 主 run root：

```text
/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso
```

GPU archive root：

```text
/workspace/stage1-results/stage1_post_hb_260705_retune12288_b20
```

GPU sequence 已完成：

```text
ALL_STAGE1_SEQUENCE_DONE 2026-07-05T04:21:51+00:00
```

对应 R2 归档：

```text
cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/
```

归档说明见 `docs/stage1_post_hb_r2_archive_260705_zh.md`。代码、脚本和配置
provenance 见 `docs/stage1_post_hb_retune12288_b20_provenance_260705_zh.md`。

## 当前 Stage1 结论

当前 post-HB Stage1 结果应作为当前线性 hidden-probe 设计的
negative/control result，而不是 hidden-state superiority 证据。

主要依据：

- 16 个 hidden-minus-surface AUROC deltas 全部为负。
- hidden validation-selected test AUROC 约为 0.679 到 0.840。
- validation-selected surface text baselines test AUROC 约为 0.917 到 0.965。
- Fable-5 复核指出，`length_only` 在四个 source 上都超过当前 hidden probe，
  说明 full-trajectory hindsight 结构本身很强。
- row audit 未显示 broad extractor-level row drop；剩余 mismatch 主要集中在
  Stage1 linear 的高 CoT offset coverage gap。
- A1/A2 和 excluded-source lead-time confirmation 已完成；Fable-5 复审确认
  `drop_leadtime_claim` 是正确决策，Stage1 应作为 negative/control 结果。

小实验清单、关键数字和“符合/不符合预期”总览见：

```text
res/stage1_experiment_inventory_results_260705_zh.md
```

教授关于 LOSO generalization 与 steering/capability confound 的逐项回应见：

```text
res/professor_feedback_status_260705_zh.md
```

这意味着：当前数据不是完全不可解释；更大的问题是 evaluation contrast。
当前 hidden arm 是 prefix-limited hidden snapshot，而 surface arm 是
full-trajectory hindsight text features。这个对比不能直接回答
"hidden 是否提供不被表层文本解释的 safety signal"。

## Fable-5 Probe Redesign Review

Fable-5 pro/high-rigor 复审已归档到 tmp repo：

```text
tmp commit: 0ea86e7
packet: stage1_fable_review_retune12288_b20_260705/
response: fable_probe_redesign_response_260705.md
```

主 repo 摘要见：

```text
docs/fable_decision_review_stage1_retune12288_b20_260705.md
```

最终决策：

```text
ONLY AFTER GATES
```

含义：

- 不再批准当前设计的 GPU position/layer/classifier sweep。
- 允许做 CPU-only 的 matched-horizon reanalysis。
- GPU 只允许在非 GPU gates 和 Phase-1 continue criteria 通过后，用于补回丢失的
  RS/SR hidden arrays。

## 下一步实验边界

Fable-5 建议把科学问题改写为 accessibility/sample-efficiency：

> 在相同信息 horizon k 下，hidden@k 是否比读取同一前缀 token 的
> matched-capacity surface classifier 更容易预测最终 safe/unsafe outcome。

最小实验 M1：

- horizon grid: `k in {4, 8, 16, 32, 64}`
- hidden arm: frozen hidden@cot_k
- surface arm: prompt + first k CoT tokens 的 preregistered text features
- primary endpoints: paired delta AUROC、within-pair ranking accuracy
- secondary endpoints: residual delta AUROC 和 delta log-loss
- Phase 1 只在 HB + WJB 已保存 arrays 上 CPU 跑
- RS/SR 只有在 Phase 1 通过后才允许补 GPU regeneration

Kill criterion：

- 如果 hidden@k - text@k 的 CI 在 HB 和 WJB 的所有 `k <= 32` 上都包含 0
  或为负，永久停止 Stage1 probing。
- 这时 negative/control claim 反而更强：没有证据表明 internal prefix states
  在任何测试 horizon 上提供 cheap forecasting advantage。

## Formal Blockers

任何论文式 claim 仍需等待：

- human QA summary
- S-to-S safe-prompt diagnostics
- HT quarantine / external testing
- Fable-5 G1-G8 非 GPU gates
- M1 preregistration committed before unblinding Phase-1 test metrics

当前允许的窄 claim：

> 在当前 natural divergent-pair LOSO/frozen-split 设置下，当前测试过的线性
> hidden-state probes 没有超过 validation-selected full-trajectory surface
> text baselines；这支持把当前 Stage1 run 作为当前 probe design 的
> negative/control result，而不是 hidden-state superiority evidence。

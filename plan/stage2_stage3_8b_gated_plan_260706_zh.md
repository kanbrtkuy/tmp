# Stage2/Stage3 8B Gated Plan - 2026-07-06

本文记录 1.5B pilot 之后的下一步计划。Fable 的 double check 建议是：

> skip 1.5B entirely, rent the A100, but make full Stage2/Stage3 a gated branch
> inside the rental -- the on-policy Stage3 slice decides.

因此下一步可以直接开 A100 节点做 8B，但不应无条件直接跑完整 full
Stage2/Stage3。A100 租机内部应先跑低成本 gate。

## 目标

验证新版 Stage2/Stage3 逻辑在 8B 上是否成立：

1. Stage2 让模型在 after `cot_4` / before `cot_5` 自然吐出 pause。
2. Stage2 尽量保持 pause 后 continuation、能力和安全行为不变。
3. Stage3 在 pause hidden state 上读出 unsafe/safe CoT 信号。
4. Stage3 必须转向 on-policy slice，解决 teacher-forced probe 和 generation-time steering distribution mismatch。

## Gate 0: 开机前确认

- [x] 确认 R2 archive 已完成并有 final size record。
- [x] 确认 GitHub 中有 Stage2/Stage3 pilot 结果、R2 archive 文档和本计划。
- 确认 8B configs 使用 after `cot_4` / before `cot_5`，不是旧 cot3/cot4 混用口径。
- 确认 R2/HF token 注入方式只进入机器环境，不写入 GitHub。

Gate 0 archive record:

```text
cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-a6000-2x-stage2-stage3-pilot/
Total objects: 6.347k (6347)
Total size: 191.456 GiB (205574724820 Byte)
```

## Gate 1: 8B Stage2 小规模检查

如果已有可用的 8B KL-transparent pause checkpoint：

- 先验证 pause emission coverage。
- 检查 continuation 是否明显格式崩坏。
- 跑小规模 model-comparison sanity eval。

如果没有可用 checkpoint：

- 先跑 short 8B Stage2 pilot，而不是 full SFT。
- 目标是确认 pause insertion 和 continuation preservation。

建议 checkpoint 选择规则：

- pause3_rate 接近 1.0。
- `think_start` / `think_end` 不明显下降。
- broken/garbage output 不明显上升。
- capability 不出现大幅下降或异常上升。
- unsafe prompt response 不出现明显 SFT-data effect drift。

## Gate 2: 8B On-policy Stage3 Slice

这是当前最关键的 gate。

设计：

- 使用 8B pause model 自己生成 CoT。
- 每个 prompt 多采样，重新 judge 每一次 generation 的 CoT 是否 unsafe。
- 在 pause positions 提取 hidden states。
- 对比 prompt baseline、matched true content control、matched-horizon surface baseline。

建议规模：

- 100-200 judge-filtered rollout pairs 起步。
- 覆盖 HarmBench、ReasoningShield、StrongReject、WildJailbreak。

预注册成功标准：

- pause probe beats base-hidden@matched-position。
- pause probe beats matched-horizon surface baseline。
- Delta AUROC >= 0.05。
- paired/bootstrap CI excludes 0。
- 至少 3/4 source 通过。

如果 Gate 2 不通过：

- 停止 full Stage2/Stage3 扩展。
- 记录负结果。
- 将论文 framing 改成 steering-first / readable anchor exploratory，而不是 pause-specific monitoring claim。

如果 Gate 2 通过：

- 进入完整 8B Stage2 SFT。
- 跑完整 Stage3。
- 再进入 Stage4 liveness 和 steering。

## Gate 3: Full 8B Stage2

只有 Gate 2 通过后才跑完整 full Stage2。

必须保存：

- checkpoints
- tokenizer/special token files
- trainer state
- config resolved yaml
- train/eval logs
- model comparison eval results
- R2 backup manifest

## Gate 4: Full 8B Stage3

完整 Stage3 应包括：

- off-policy paired-data probe，用于和 1.5B pilot 对齐。
- on-policy generated-CoT probe，作为主结论依据。
- prompt-only / pre-CoT baseline。
- matched true content controls。
- matched-horizon surface baseline。
- paired bootstrap CI。
- horizon/content-control audit。

## Stage4 前置条件

进入 Stage4 前必须完成：

- pause liveness battery：injection gain、attention mass、KV ablation、safe/unsafe patching。
- matched-strength counterfactual steering：pause vs token_3/token_4/post_pause。
- 三组能力对照：base / pause-SFT-no-steer / pause-SFT+steer。
- residual unsafe absolute rates、over-refusal、broken output、length shift、think_end rate 报告。

## 当前状态

1.5B pilot 支持继续：

- Stage2 pilot 训练和 pause emission 流程跑通。
- Stage3 pilot 四个 folds 均显示 pause readable signal。
- Fable 认为 framework sanity check 通过。

但 1.5B pilot 不支持最终强 claim：

- pause 没有稳定超过 true content control。
- Stage3 仍是 teacher-forced/off-policy。
- on-policy Stage3 slice 尚未完成。

推荐下一步：

```text
A100 8B node
-> short 8B Stage2 checkpoint sanity
-> 8B on-policy Stage3 slice
-> pass: full Stage2 + full Stage3
-> fail: stop expansion and record negative branch
```

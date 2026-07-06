# Stage2/Stage3 8B 当前 Gate - 2026-07-07

本文记录 8B full Stage2 完成后的当前决策点。

## 已完成

- 8B Stage2 KL-transparent emit full run 已完成。
- 训练 early-stopped at step 575。
- final checkpoint 已从 `/dev/shm` 移到 `/workspace`。
- final model/checkpoints 已同步到 R2。
- Stage2 model-comparison eval 已完成。
- WildGuard / LlamaGuard / HarmBench 三个 judge 均完成全部 1150 条 safety rows。
- Stage2 eval/judge 结果已同步到 R2 并 `rclone check` 验证。
- Stage3 四个 Stage1 paired source config 已 dry-run 通过：
  - HarmBench
  - ReasoningShield
  - StrongReject
  - WildJailbreak

## 当前 Gate

Stage3 尚未启动。

原因：

- Stage2 整体 behavior-preservation 看起来基本过关。
- 但 GSM8K natural pause emission 有明显 over-emission：
  - exact single run of 3 pauses: 0.196
  - off-target pause rate: 0.802
  - avg pause count: 8.288
- 需要 Fable 判断这是否是 Stage2 blocker。

## Fable Review 问题

需要 Fable 回答：

1. 当前 8B Stage2 checkpoint 是否足够进入 Stage3？
2. GSM8K over-emission 是 blocker、non-blocking limitation，还是需要重跑 Stage2？
3. Stage3 应使用 natural pause、forced pause，还是二者都跑？
4. Stage3 前还需要哪些最小 sanity check？
5. 当前 Stage2 允许/不允许 claim 什么？

## 若 Fable 通过

启动 Stage3，使用新版 Stage1 paired data：

```text
/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/loso_freeze_fixed_budget_samples_000_099/stage1_prepared/
```

优先运行四个 source-specific configs：

```text
configs/experiment/stage3_intra_pause_probe_stage1_paired_harmbench_8b_cot5_2xa100.yaml
configs/experiment/stage3_intra_pause_probe_stage1_paired_reasoningshield_8b_cot5_2xa100.yaml
configs/experiment/stage3_intra_pause_probe_stage1_paired_strongreject_8b_cot5_2xa100.yaml
configs/experiment/stage3_intra_pause_probe_stage1_paired_wjb_8b_cot5_2xa100.yaml
```

Stage3 必须报告：

- prompt baseline: `last_prompt_token`, `pre_think`
- pause positions: `pause_0`, `pause_1`, `pause_2`
- matched no-pause true content controls: `control_cot_5`, `control_cot_6`
- pre/post pause diagnostics
- source-wise AUROC and margins
- evidence report

## 若 Fable 不通过

不启动 Stage3。

先做：

- GSM8K natural over-emission example audit。
- 判断是否是 decoding artifact、data-format artifact，还是 Stage2 objective 的真实失败。
- 必要时调整 Stage2 emission regularization/data mix/checkpoint selection 后重跑 Stage2 eval。

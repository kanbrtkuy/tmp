# Stage2/Stage3 8B 当前 Gate - 2026-07-07

本文记录 8B full Stage2 完成后，以及 Stage3 teacher-forced supporting
run 完成后的当前决策点。

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
- Stage3 teacher-forced supporting run 已完成四个 source。
- WJB hot artifacts 已从 `/dev/shm` 同步到 `/workspace/cot-safety/stage3/`。

## 当前 Gate

Stage3 teacher-forced supporting run 已完成，但不是 confirmatory Stage3。

最新状态：

- Stage2 整体 behavior-preservation 基本过关。
- GSM8K natural pause emission 有明显 over-emission，但 Fable 判断这是
  non-blocking limitation，不是 Stage2 重跑触发条件。
- Stage3-source paired generation sanity 已补：
  - natural exact-3 约 `0.48-0.59`；
  - forced/full exact-3 约 `0.625-0.797`；
  - stripped pause bleed 为 `0.0`；
  - GSM8K over-emission 子集 accuracy 没有低于 exact-3 子集。
- Fable 对启动 Stage3 前曾给出 conditional NO-GO：不是 Stage2 数字失败，
  而是要求先提交 Stage3 prereg、horizon/content-control audit，并确认
  Stage1 freeze gate。
- 已新增 Stage3 prereg/horizon audit：
  `review-stage/stage3_8b_cot5_prereg_260707/stage3_8b_cot5_prereg_and_horizon_audit_260707.md`
- 已将 Stage3 8B configs 的 true content controls 从 `control_cot_5/6`
  扩展为 `control_cot_4/5/6`，其中 `control_cot_4` 是 exact-horizon
  primary content control。
- 四个 source 的 Stage3 teacher-forced 结果都通过 prompt baseline，但
  没有超过 `control_cot_4`：
  - HarmBench: pause 0.8251 vs control_cot_4 0.8298
  - ReasoningShield: pause 0.7038 vs control_cot_4 0.7325
  - StrongReject: pause 0.7487 vs control_cot_4 0.7510
  - WildJailbreak: pause 0.7861 vs control_cot_4 0.8026

当前结论：

- Stage2 仍然是可用 checkpoint。
- Stage3 teacher-forced 只能支持 readable pause signal。
- Stage3 teacher-forced 不能支持 pause-specific monitoring advantage。
- Stage4 不应直接启动 steering；下一步是 on-policy Stage3 / liveness。

## Fable Review 问题

需要 Fable 回答：

1. 当前解释是否正确：pause 明显高于 prompt baseline，但没有独立于
   `control_cot_4` 的优势？
2. 是否需要重训 Stage2，还是直接进入 on-policy Stage3 / liveness？
3. on-policy Stage3 slice 应该如何设计，才能解决 teacher-forced 与
   self-generated distribution mismatch？
4. Stage4 是否必须等 on-policy Stage3 通过，还是可以先做 liveness-only
   pilot？
5. 当前 Stage2/Stage3 允许/不允许 claim 什么？

## 已完成 Teacher-Forced Stage3

已使用新版 Stage1 paired data：

```text
/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/loso_freeze_fixed_budget_samples_000_099/stage1_prepared/
```

已运行四个 source-specific configs：

```text
configs/experiment/stage3_intra_pause_probe_stage1_paired_harmbench_8b_cot5_2xa100.yaml
configs/experiment/stage3_intra_pause_probe_stage1_paired_reasoningshield_8b_cot5_2xa100.yaml
configs/experiment/stage3_intra_pause_probe_stage1_paired_strongreject_8b_cot5_2xa100.yaml
configs/experiment/stage3_intra_pause_probe_stage1_paired_wjb_8b_cot5_2xa100.yaml
```

已报告：

- prompt baseline: `last_prompt_token`, `pre_think`
- pause positions: `pause_0`, `pause_1`, `pause_2`
- matched no-pause exact-horizon true content control: `control_cot_4`
- lead/ordinary content diagnostics: `control_cot_5`, `control_cot_6`
- pre/post pause diagnostics
- source-wise AUROC and margins
- evidence report

结果记录：

```text
res/stage3_8b_teacher_forced_results_260707_zh.md
```

## 若 Fable 不通过当前解释

不进入 Stage4 steering。

先做：

- on-policy Stage3 最小 slice。
- pause liveness battery。
- 必要时调整 Stage2 emission/liveness objective 后重跑 Stage2 eval。

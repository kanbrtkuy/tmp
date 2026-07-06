# Stage2 8B Full 结果记录 - 2026-07-07

本文记录 2xA100 节点上完成的 8B Stage2 full run 和 Stage2
model-comparison eval。Stage3 teacher-forced supporting run 已完成；当前
gate 是 on-policy Stage3 / liveness，而不是 Stage2 重跑。

## 运行设置

模型：

```text
DeepSeek-R1-Distill-Qwen-8B
```

Stage2 方法：

- KL-transparent emit SFT，不是旧版普通 full SFT。
- 在 `cot_4` 之后、`cot_5` 之前插入 3 个 pause token。
- 目标是让模型自然输出 pause，同时 continuation KL 尽量保持无 pause
  reference 的后续分布。
- 2x A100 80GB。
- 每 25 step eval。
- 开启 early stopping。

训练配置：

```text
configs/experiment/stage2_intra_pause_kl_transparent_emit_8b_cot5_full_save25_2xa100.yaml
```

最终 checkpoint：

```text
/workspace/outputs/deepseek_8b_intra_pause_cot5_kl_transparent_emit_trusted_cot_18k_full_save25_mb4_ga2_2xa100/final
```

## 训练结果

训练在 step 575 early-stopped，约 0.54 epoch。

最终指标：

- `eval_loss`: 0.007503
- pause target argmax rate: 1.0
- continuation KL: 约 0.0006-0.0008

解释：

- 模型学会了 supervised pause target。
- continuation KL 很小，符合 KL-transparent pause 的设计目标。
- 这不是 safety/capability improvement claim，只能支持 behavior-preserving
  pause insertion 的候选结论。

## Stage2 Eval/Judge

eval root：

```text
/workspace/cot-safety/runs/eval/stage2_model_comparison_deepseek_8b_kl_transparent_emit_cot5_full_save25_mb4_ga2_2xa100
```

生成完成：

- capability: 800 rows per model group
- safety: 1150 rows per model group
- groups: base natural, pause natural, pause forced

judge 完成：

- WildGuard: 3 groups, all 1150/1150
- LlamaGuard: 3 groups, all 1150/1150
- HarmBench: 3 groups, all 1150/1150

## 能力结果

| group | GSM8K | MATH500 | parse rate |
|---|---:|---:|---:|
| base natural | 0.660 | 0.4467 | 1.0 |
| pause natural | 0.646 | 0.4333 | 1.0 |
| pause forced | 0.652 | 0.4533 | 1.0 |

解释：

- 没有明显能力崩坏。
- MATH forced 略高于 base 不能作为正向 claim；我们的目标是不改变能力。
- natural pause 有小幅下降，但目前看不像 catastrophic drift。

## Safety 结果

unsafe reference prompts 上的 unsafe-valid rate：

| judge | base natural | pause natural | pause forced |
|---|---:|---:|---:|
| HarmBench | 0.3600 | 0.3700 | 0.3567 |
| LlamaGuard | 0.5200 | 0.5267 | 0.5100 |
| WildGuard | 0.4233 | 0.3800 | 0.4133 |

safe reference prompts 上整体也接近 base：

| judge | base natural | pause natural | pause forced |
|---|---:|---:|---:|
| HarmBench unsafe-valid | 0.6382 | 0.6400 | 0.6273 |
| LlamaGuard unsafe-valid | 0.0782 | 0.0727 | 0.0855 |
| WildGuard unsafe-valid | 0.0491 | 0.0473 | 0.0564 |

解释：

- 三个 judge 下整体接近 base。
- 暂时不能 claim safety improvement。
- 当前证据支持“没有明显 safety drift”，但正式报告需要置信区间和更多审计。

## Pause Emission

natural pause emission：

| dataset | natural pause rate | exact single run of 3 | off-target pause rate | avg pause count |
|---|---:|---:|---:|---:|
| GSM8K | 1.000 | 0.196 | 0.802 | 8.288 |
| MATH500 | 1.000 | 0.9267 | 0.0033 | 3.377 |
| HarmBench standard | 1.000 | 0.855 | 0.025 | 3.285 |
| JailbreakBench | 1.000 | 0.920 | 0.010 | 3.170 |
| OR-Bench hard | 1.000 | 0.7967 | 0.020 | 3.470 |
| StrongReject | 1.000 | 0.890 | 0.0133 | 3.217 |
| XSTest | 1.000 | 0.700 | 0.004 | 3.792 |

关键问题：

- safety datasets 和 MATH500 基本在目标位置附近自然吐 pause。
- GSM8K natural 模式明显过量吐 pause：exact 3-pause 只有 19.6%，平均
  pause count 8.288。
- forced pause by construction 在所有 dataset 都插入 3 个 pause。

当前判断：

- 8B Stage2 的主要 behavior-preservation 指标基本过关。
- GSM8K over-emission 是需要 Fable 判断的 gate：它可能只是短 arithmetic
  trace 的 emission-format 问题，也可能意味着 Stage2 还不够 clean。
- Fable 已判断该问题是 non-blocking limitation，可以进入 Stage3
  safety-source probing，但不能 claim 全局 clean emission。

## Stage3 Teacher-Forced Follow-Up

Stage3 teacher-forced supporting run 已在同一 2xA100 节点完成，使用新版
Stage1 paired freeze 和四个 source-specific configs：

- HarmBench
- ReasoningShield
- StrongReject
- WildJailbreak

核心结论见：

```text
res/stage3_8b_teacher_forced_results_260707_zh.md
```

Stage3 teacher-forced 结果显示：

- 四个 source 的 pause AUROC 都明显高于 prompt-only baseline；
- 但 pause 未超过 preregistered exact-horizon content control
  `control_cot_4`；
- 因此只能 claim pause 是 readable anchor，不能 claim pause-specific
  monitoring advantage。

## R2 备份

Stage2 final model 和 checkpoints 已由 watcher 上传 R2。

Stage2 eval/judge 结果已额外同步并校验：

```text
cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-2xa100-8b-full-stage2-cot5-mb4-ga2/runs/eval/stage2_model_comparison_deepseek_8b_kl_transparent_emit_cot5_full_save25_mb4_ga2_2xa100
Total objects: 376
Total size: 385.209 MiB
rclone check: 0 differences found, 376 matching files
```

## 当前下一步

1. 将 Stage2 + Stage3 teacher-forced 结果发给 Fable 做外部 review。
2. 按 Fable 建议设计 on-policy Stage3 slice。
3. 只有 on-policy Stage3 和 liveness 通过后，才进入 Stage4 steering。
4. 当前不应基于 teacher-forced run 做 steering-port / clean intervention
   claim。

# Stage2/Stage3 Pilot 结果记录 - 2026-07-06

本文记录 2026-07-06 在 2xA6000 RunPod 节点上完成的 Stage2/Stage3 pilot。
这是框架验证和方法 gate，不是最终 8B full Stage2/Stage3 结果。

## Stage2 Pilot

模型：

```text
DeepSeek-R1-Distill-Qwen-1.5B
```

训练目标：

- 使用 KL-transparent pause-emission SFT，而不是旧版普通 full SFT。
- 在 CoT 内部 `cot_4` 之后、`cot_5` 之前插入 3 个 `<|pause|>` token。
- 训练模型自然发出 pause，同时用 continuation KL 尽量保持 pause 后续输出不变。

插入布局：

```text
<think> t0 t1 t2 t3 t4 <|pause|><|pause|><|pause|> t5 ...
```

主要配置：

```text
configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_cot5_save25_max400_2xa6000.yaml
```

主要 checkpoint：

```text
/workspace/outputs/deepseek_1p5b_intra_pause_cot5_kl_transparent_emit_trusted_cot_18k_save25_max400_2xa6000/final
```

结论：

- Stage2 pilot 训练流程跑通。
- pause 插入位置符合 after `cot_4` / before `cot_5` 的新设计。
- 该 checkpoint 只作为 Stage3 框架 sanity check，不作为最终 Stage2 claim。

## Stage3 Pilot

Stage3 使用 Stage1 重新准备的 paired data，保留 Stage1 split：

```text
/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/loso_freeze_fixed_budget_samples_000_099/stage1_prepared/
```

Stage3 运行设置：

- `--preserve_input_splits`
- `--no_heldout_sources`
- `--skip_base_data_prep`
- teacher-forced / off-policy hidden extraction
- clean rerun extraction batch size: `40`

四个 clean folds：

- HarmBench
- ReasoningShield
- StrongReject
- WildJailbreak

结果：

| Fold | Pause AUROC | Prompt AUROC | Pause - Prompt | Pair-cluster CI | True Content Control AUROC | Independent Margin |
|---|---:|---:|---:|---:|---:|---:|
| HarmBench | 0.8120 | 0.5000 | 0.3120 | [0.2688, 0.3550] | 0.8225 | -0.0105 |
| ReasoningShield | 0.7196 | 0.5000 | 0.2196 | [0.1833, 0.2557] | 0.7263 | -0.0185 |
| StrongReject | 0.7347 | 0.5000 | 0.2347 | [0.1982, 0.2723] | 0.7218 | -0.0193 |
| WildJailbreak | 0.7983 | 0.5002 | 0.2980 | [0.2850, 0.3105] | 0.7552 | -0.0077 |

解释：

- pause hidden state 在四个 source 上都有明显 safe/unsafe 信号。
- prompt baseline 接近 chance，说明当前 paired setting 下不是简单 prompt-only classification。
- pause 没有稳定超过 matched true content control，所以不能 claim pause 具有独立于普通 CoT token 的特殊监测信号。
- 对 Stage4 来说，当前结果支持把 pause 视为 readable steering anchor 的候选，但还不支持最终 steering claim。

## Fable Review

Fable 对 Stage3 pilot 的判断：

- framework sanity check：通过。
- pause-specific independent-signal claim：不通过。
- WJB clean rerun 补齐四源口径，但不改变总体 verdict。
- 下一步应先做 horizon/content-control audit 和 on-policy Stage3 slice。

Fable follow-up 核心结论：

```text
WJB closes the sanity gap but doesn't change the verdict -- count the +0.043 as exploratory, fix the horizon/content-control audit first, then run the on-policy Stage3 slice; full Stage2 SFT is gated on that slice passing.
```

## R2 Backup

本次 RunPod 节点内容已归档到 Cloudflare R2：

```text
cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260706-a6000-2x-stage2-stage3-pilot/
```

最终归档大小：

```text
Total objects: 6.347k (6347)
Total size: 191.456 GiB (205574724820 Byte)
```

其中 `workspace/models/judges/wildguard_vllm_head_dim128/` 经过额外
`--copy-links` pass，已从 symlink 兼容目录实体化为可直接恢复的普通文件目录：

```text
Total objects: 39 (39)
Total size: 13.501 GiB (14496930379 Byte)
```

归档说明见：

```text
docs/stage2_stage3_pilot_r2_archive_260706.md
```

## 当前结论

可以说：

> Stage2/Stage3 pilot passed: the pause-SFT workflow runs, pause positions are
> readable, and the Stage3 evidence/report pipeline works on paired Stage1 data.

不能说：

> Stage2/Stage3 final claims are complete, or pause has proven independent
> monitoring advantage over ordinary CoT content tokens.

下一步应进入 8B gated workflow，而不是继续扩大 1.5B pilot。

# Stage2.1-pure 代码状态 - 2026-07-07

## 当前状态

新版 Stage2.1-pure 代码已实现并通过 Fable review。

目标：

- 在 `cot_4` 之后、`cot_5` 之前自然插入：

```text
<|pause|><|pause|><|pause|>
```

- 不使用 indexed pause；
- 尽量保持 continuation、reasoning capability、safety behavior 不变；
- checkpoint 不再按 eval loss 选，而按 natural exact-3/location gate 选。

## 已完成代码改动

Trainer：

- `PauseKLSFTTrainer` 现在区分：
  - `pause_chain_token_ids=(id,id,id)`：用于 stop-after-3 序列检测；
  - `pause_token_ids=(id,)`：用于 suppression、KL strip、trainable row。
- 修复旧 single-token stop mask bug：不会在第 1/2 个 pause 后触发 stop。
- `pause_tokens` 长度和 `n_pause_tokens` 不一致时会报错，避免配置手误。

启动脚本：

- `run_stage2_sft.py` 和 `run_4gpu_intra_pause_sft.sh` 传递
  `PAUSE_KL_N_PAUSE_TOKENS=3`。

配置：

- 新增 1.5B / 8B Stage2.1-pure 训练配置；
- 新增 pure DAgger pool；
- 新增 1.5B / 8B model-comparison eval 配置。

测试：

- 新增 pure repeated trainer tests；
- 新增 pure repeated natural metrics test；
- 新增 pure repeated on-policy relabel test。

## 本地验证

本地没有安装 `pytest`，因此正式 pytest 需要在 GPU 环境跑。

已完成：

- `python3 -m py_compile` 通过；
- 用临时 pytest stub runner 执行相关测试函数：`26 tests OK`；
- pure 1.5B / 8B training configs dry-run 通过；
- pure 1.5B / 8B model-comparison eval configs dry-run 通过。

dry-run 确认：

```text
PAUSE_KL_PAUSE_TOKENS=["<|pause|>","<|pause|>","<|pause|>"]
PAUSE_KL_N_PAUSE_TOKENS=3
FORMAT_ONLY_TRAINABLE_TOKENS=["<|pause|>"]
PAUSE_KL_STOP_WEIGHT=2.0
PAUSE_KL_SUPPRESSION_LOSS_TYPE=margin
```

## Fable Review

Review 文件：

```text
review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_code_review_260707.md
review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_code_followup_260707.md
```

Fable 结论：

- 无 blocker；
- pure repeated 不会被误当作 indexed；
- stop loss 不会在第 1/2 个 pause 后误触发；
- 仍符合 Goyal et al. 的 pure repeated pause-token 精神；
- 可以跑 1.5B data-prep validation 和 25-step smoke。

## 当前剩余风险

这是代码可行，不是实验已经成功。

主要未验证风险：

- rows-only pure repeated 是否有足够容量自然学会 exact-3；
- stop-after-3 是否能把 GSM8K over-emission 压下去；
- natural exact-3/location 是否能达到正式 8B gate；
- capability 是否保持在 base 附近；
- Stage3 pause hidden state 是否仍然有可读信号。

## 下一步

优先顺序：

1. GPU 环境跑正式 pytest；
2. 1.5B data-prep validation；
3. 1.5B 25-step smoke；
4. smoke checkpoint 的 natural exact-3/location gate；
5. capability sanity；
6. 再决定是否跑 1.5B short400 / DAgger iter1 / 8B full。

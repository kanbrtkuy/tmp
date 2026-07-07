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
- 新增 1.5B smoke eval 配置，用于 25-step smoke 后快速检查
  natural exact-3/location。

Pipeline：

- 新增 `pipelines/run_stage21_pure_1p5b_smoke.sh`，把 pytest、
  data-prep validation、25-step SFT、小样本 natural/forced generation 和
  natural exact-3/location gate 串成一个入口。
- gate 使用 `diag_stage2_checkpoint.py --strict`，失败时 pipeline 会
  非零退出。
- 默认不跑 judge，避免 smoke 阶段被 judge 模型资源阻塞；可用
  `RUN_JUDGE=1 RUN_SUMMARY=1` 打开。

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
- 1.5B smoke model-comparison eval config dry-run 通过：
  - `generation_jobs=6`；
  - 三个 condition：`base_natural`、`stage21_pure_cot5_natural`、
    `stage21_pure_cot5_forced`；
  - natural condition 不强制插入 pause；
  - forced condition 在 `cot_4` 后 / `cot_5` 前强制插入 3 个 pure pause；
  - `judge_jobs=9`，WildGuard 使用
    `/workspace/models/judges/wildguard_vllm_head_dim128`。
- Fable 对 smoke/gate 入口的补充 review 指出 strict gate 是 blocker；
  已修复为 gate fail 时退出非零，并补充 expected-cot-offset metrics
  reuse 校验。
- Fable follow-up 确认 strict gate blocker 已解决，当前 smoke/gate 入口
  无 blocker，可以 commit/push。

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
review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_smoke_gate_review_260707.md
review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_smoke_gate_followup_260707.md
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

1. GPU 环境跑 `bash pipelines/run_stage21_pure_1p5b_smoke.sh`；
2. 检查 smoke checkpoint 的 natural exact-3/location gate；
3. 如果 gate 明显失败，先修 Stage2.1-pure 目标或 DAgger 负例；
4. 如果 gate 可行，再跑 1.5B short400 / DAgger iter1；
5. capability sanity；
6. 最后再决定是否跑 8B full。

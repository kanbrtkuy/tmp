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
- 新增 1.5B full 2xA6000 配置
  `configs/experiment/stage21_pause_pure_dagger_1p5b_full_2xa6000.yaml`：
  `max_steps: null`、`num_train_epochs: 1.0`、无早停、`save_steps: 25`、
  `save_total_limit: 64`、strict gate 阈值 0.99；
- 新增 pure DAgger pool；
- 新增 1.5B / 8B model-comparison eval 配置。
- 新增 1.5B smoke eval 配置，用于 25-step smoke 后快速检查
  natural exact-3/location。
- 新增 1.5B selection-dev eval 配置
  `configs/experiment/stage2_model_comparison_eval_1p5b_stage21_pure_cot5_selection_dev_2xa6000.yaml`，
  使用 `sample_offset` 构造与正式 full eval 不重叠的小 dev prompt subset。

Pipeline：

- 新增 `pipelines/run_stage21_pure_1p5b_smoke.sh`，把 pytest、
  data-prep validation、25-step SFT、小样本 natural/forced generation 和
  natural exact-3/location gate 串成一个入口。
- 新增 `pipelines/run_stage21_pure_1p5b_full.sh`，把 1.5B full 2xA6000 的
  pytest、data prep、完整 1 epoch SFT、model-comparison generation、
  strict natural gate 和 summary 串成一个入口。默认不跑 judge；默认使用
  cold `/workspace` 存储，避免小 `/dev/shm` 节点被 checkpoint 挤满。
- `scripts/run_stage2_sft.py` 支持命令行覆盖
  `per_device_train_batch_size`、`per_device_eval_batch_size`、
  `gradient_accumulation_steps`、`dataloader_num_workers` 和 `optim`，用于
  远端 batch probe，不需要手改 YAML。
- `scripts/prepare_model_comparison_eval_data.py` 支持每个 source 设置
  `sample_offset`，用于 checkpoint selection 和最终报告集 prompt disjoint。
- 新增 `scripts/select_stage21_checkpoint.py`，用于在训练结束后对候选 checkpoint
  跑 selection-dev natural exact-3/location sweep，并输出
  `sweep_summary.json`、`sweep_summary.md` 和 `selected_checkpoint.txt`。
  脚本支持从 R2 下载已被本地 watcher 删除的 checkpoint，评估后可删除下载副本。
- 新增 `pipelines/run_stage21_pure_8b_full.sh`，把 8B formal 的 pytest、
  data prep、full SFT、model-comparison generation、strict natural gate、
  judge 和 summary 串成一个入口。
- gate 使用 `diag_stage2_checkpoint.py --strict`，失败时 pipeline 会
  非零退出。
- 1.5B smoke 默认不跑 judge，避免 smoke 阶段被 judge 模型资源阻塞；可用
  `RUN_JUDGE=1 RUN_SUMMARY=1` 打开。
- 8B formal wrapper 默认跑 judge / summary；strict gate 在 judge 前执行，
  如果 natural exact-3/location 不通过，不继续消耗 judge GPU。
- 8B formal wrapper 默认从 cold
  `/workspace/outputs/deepseek_8b_stage21_pause_pure_cot5_full_2xa100/final`
  读取 SFT checkpoint，因为训练 watcher 会在结束后把 hot output 同步到
  cold 并删除 hot copy。
- 8B formal wrapper 结束后默认同步 hot run outputs 到 cold，降低 pod
  关闭时丢失 eval/generation/judge/summary 的风险。
- 8B formal wrapper 在 generation 前检查 cold checkpoint 是否存在且包含权重；
  如果缺失会直接失败，避免误加载旧模型或半同步模型。
- 8B formal wrapper 注册 EXIT trap；即使 strict gate 或 judge 失败，也会尽量
  把 hot run outputs 同步到 cold，方便诊断失败样本。

测试：

- 新增 pure repeated trainer tests；
- 新增 pure repeated natural metrics test；
- 新增 pure repeated on-policy relabel test。

## 本地验证

本地没有安装 `pytest`，因此正式 pytest 需要在 GPU 环境跑。

已完成：

- `python3 -m py_compile` 通过；
- `scripts/select_stage21_checkpoint.py --help` 通过；
- `sample_offset` 本地 sanity check 通过，offset=0 和 offset=3 的抽样集合不重叠；
- 用临时 pytest stub runner 执行相关测试函数：`26 tests OK`；
- pure 1.5B / 8B training configs dry-run 通过；
- pure 1.5B / 8B model-comparison eval configs dry-run 通过。
- 1.5B / 8B formal eval configs 已显式写入
  `expected_cot_offset: 5` 和三个 pure `pause_tokens`，避免 formal gate
  依赖隐式默认值。
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
- Fable review 8B formal pipeline 后指出 checkpoint 默认 hot path 是 blocker；
  已改为 cold `/workspace/outputs/.../final`，并补充 checkpoint gate /
  failure-path sync。Fable follow-up 确认 no blocker，可以 commit/push。

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
review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_8b_formal_pipeline_review_260707.md
review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_1p5b_full_pipeline_review_260707.md
review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_1p5b_full_pipeline_followup_260707.md
review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_1p5b_running_review_260707.md
review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_selection_review_260707.md
```

Fable 结论：

- 无 blocker；
- pure repeated 不会被误当作 indexed；
- stop loss 不会在第 1/2 个 pause 后误触发；
- 仍符合 Goyal et al. 的 pure repeated pause-token 精神；
- 可以跑 1.5B data-prep validation 和 25-step smoke。
- 1.5B full 2xA6000 pipeline 的 Fable 结论是 `OK_TO_RUN`：无代码 blocker；
  `max_steps: null` 会清除 parent 的 400-step cap；1 epoch、无早停、
  batch override、cot_offset=5、pure repeated pause、cold checkpoint path
  和 strict natural gate 均符合当前目标。
- 将 `save_total_limit` 提到 64 后，Fable follow-up 再次确认
  `OK_TO_RUN`；唯一 advisory 是监控 cold `/workspace` 容量。
- 训练启动后的 running review 结论：run 本身与目标一致，无需停止；即时检查
  `0/1063` full-epoch 和 R2 watcher 均通过；但必须补 checkpoint selection
  sweep，并用 disjoint selection-dev prompts 选 checkpoint，避免在正式报告集上
  43 选 1 的 selection bias。
- selection layer follow-up review 结论：`OK_TO_USE`。Fable 确认
  `gate_score` 的 0 值排序 bug 和 selection-dev 空 source 风险均已修复；
  selection-dev 与 full eval 严格 disjoint。

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

1. 当前 2xA6000 节点优先跑
   `bash pipelines/run_stage21_pure_1p5b_full.sh`，完整 1 epoch、无早停；
2. 检查 full run final/checkpoints 的 natural exact-3/location gate；
3. 使用 `scripts/select_stage21_checkpoint.py` 在 disjoint selection-dev prompts
   上选择 checkpoint；
4. 对 selected checkpoint 在 full eval config 上重新跑 natural/forced generation、
   strict gate、capability sanity 和 judge；
5. 如果 gate 明显失败，先修 Stage2.1-pure 目标或 DAgger 负例；
6. 如果 gate 可行，再跑 1.5B short400 / DAgger iter1；
7. 最后再决定是否跑 8B full。

8B full 的一键入口：

```bash
bash pipelines/run_stage21_pure_8b_full.sh
```

当前旧 RunPod 2xA6000 节点已经不可连接：直连端口拒绝，proxy SSH
返回 publickey denied。下一步需要新 GPU 节点后才能取得真实 SFT /
natural exact-3 / capability 结果。

## 2026-07-07 1.5B Full 实验结果

后续新 2xA6000 节点已完成 1.5B Stage2.1-pure full run。

训练证据：

```text
/workspace/outputs/deepseek_1p5b_stage21_pause_pure_cot5_full_2xa6000/checkpoint-1063/trainer_state.json
global_step = 1063
epoch = 1.0
max_steps = 1063
```

这次不是 400-step pilot，也没有早停。训练完成了 1 full epoch。

最终 natural gate 失败：

```text
n = 2100
exact_chain_rate = 0.8886
block_presence_rate = 0.9948
malformed_rate = 0.1062
off_target_rate = 0.1119
location_match_rate = 0.6395
avg_pause_count = 3.1224
```

source-wise 最差是 GSM8K：

```text
exact_chain_rate = 0.8100
location_match_rate = 0.0427
off_target_rate = 0.3440
malformed_rate = 0.1740
```

selection-dev sweep 结果：

```text
completed steps = 750, 800, 850, 900, 950, 1000, 1050
all completed checkpoints = gate fail
ranked best among completed candidates = checkpoint-1050
checkpoint-1050 min_exact_chain = 0.7900
checkpoint-1050 min_location_match = 0.0408
checkpoint-1050 max_off_target = 0.3200
checkpoint-1050 max_malformed = 0.1900
```

因为所有已测 checkpoint 都远低于 strict gate，并且 GSM8K trend 没有改善，
用户要求停止 sweep 以避免继续烧 GPU。训练和 selection 进程均已停止，GPU
确认空闲。

## 失败分析

Fable 和本地诊断共同结论：

- teacher-forced 目标已饱和，但 free-running natural generation 没学会；
- 这不是“没训够”，而是 off-policy teacher-forced objective 与 autoregressive
  行为之间的 gap；
- 当前 run 名义上是 `dagger`，但实际使用的是 fixed offline pool，还没有执行
  true on-policy DAgger relabeling；
- safety-style sources 在 `5±1` 位置口径下大幅改善；
- GSM8K 仍然真实失败，尤其有大量 after-`</think>` pause re-trigger。

B1/B2 诊断：

```text
B2 full 17k train metric round-trip:
exact_chain = 1.0
location_match = 1.0
off_target = 0.0
first_pause_index = {5: 17000}

B1 GSM8K final natural:
first index top = 4:160, 2:38, 3:30, 1:30, 5:21, 6:19, None:8
pause totals inside/before/after = 1208 / 0 / 367
loc in {4,5} = 0.3679
```

所以 metric 对训练文本没有 ceiling 问题；自然生成里的 GSM8K 失败是真实
behavioral failure。

Fable review 记录：

```text
review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_1p5b_failure_analysis_260707.md
```

## R2 备份

本次 1.5B full run 已归档到：

```text
cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260707-2xa6000-1p5b-stage21-pure-full-cot5-bs4-ga2/
```

最终 size：

```text
Total objects: 757
Total size: 220.918 GiB
```

已验证：

```text
eval result backup: 0 differences found, 81 matching files, 165.178 MiB
selection sweep backup: 0 differences found, 109 matching files, 75.691 MiB
```

R2 文档：

```text
docs/stage21_1p5b_pure_r2_archive_260707.md
```

## 更新后的下一步

不要继续扩大当前 rows-only offline SFT 或直接跑 8B full。下一步应是
Stage2.1 true DAgger：

1. 用当前 checkpoint on-policy 生成训练 prompts；
2. strip 错误 pause；
3. 用 expert formatter 重新插入 pure repeated pause；
4. 加 GSM8K-heavy、after-`</think>` re-trigger、off-target 和 run-length
   counterexamples；
5. warm-start rows-only 训练；
6. 继续用 natural-generation dev gate 做 checkpoint selection；
7. 如果 true DAgger 后仍不改善，再考虑 LoRA+KL。

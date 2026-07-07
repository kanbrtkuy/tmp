# Stage2.1-pure 计划 - 2026-07-07

本文记录从旧 Stage2 rows-only / KL-transparent emit 转向
Stage2.1-pure 的正式执行计划。

## 背景

旧 8B Stage2 rows-only/KL-transparent emit run 已经证明：

- continuation KL 很小；
- capability / safety 没有明显 catastrophic drift；
- 但 natural generation 的 exact-3/location pause 良率不足，尤其 GSM8K
  出现明显 pause over-emission。

Fable 对失败原因的判断是：这不是“训练步数不够”，而是旧目标没有真正训练
free-running natural generation 中“吐 3 个 pause 后停止”的行为。

## 新方法

Stage2.1-pure 仍然只使用纯净 pause：

```text
<|pause|><|pause|><|pause|>
```

不使用：

```text
<|pause_1|><|pause_2|><|pause_3|>
```

核心训练目标：

- 修复 single-token stop mask：只在连续三个 `<|pause|>` 后触发
  stop-after-3 loss；
- 加 `stop_weight`，显式训练第三个 pause 后停止；
- suppression 改成 margin loss；
- 加 DAgger / on-policy 负例：把 free-running 少吐、多吐、位置错的样本
  strip 后重新用 expert formatter 插入 pure pause；
- checkpoint 选择使用 natural exact-3/location gate，不再用 eval loss 作为主标准。

## 代码入口

训练配置：

```text
configs/experiment/stage21_pause_pure_dagger_1p5b.yaml
configs/experiment/stage21_pause_pure_dagger_1p5b_full_2xa6000.yaml
configs/experiment/stage21_pause_pure_dagger_1p5b_iter1.yaml
configs/experiment/stage21_pause_pure_dagger_8b_short400.yaml
configs/experiment/stage21_pause_pure_dagger_8b_short400_iter1.yaml
configs/experiment/stage21_pause_pure_dagger_8b_full_2xa100.yaml
```

DAgger pool：

```text
configs/data/stage21_pure_dagger_pool.yaml
```

Stage2 model-comparison eval：

```text
configs/experiment/stage2_model_comparison_eval_1p5b_stage21_pure_cot5_2xa6000.yaml
configs/experiment/stage2_model_comparison_eval_1p5b_stage21_pure_cot5_smoke_2xa6000.yaml
configs/experiment/stage2_model_comparison_eval_1p5b_stage21_pure_cot5_selection_dev_2xa6000.yaml
configs/experiment/stage2_model_comparison_eval_8b_stage21_pure_cot5_2xa100.yaml
```

关键实现：

```text
legacy/COTPauseToken/src/utils/pause_kl_trainer.py
scripts/run_stage2_sft.py
scripts/select_stage21_checkpoint.py
legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh
pipelines/run_stage21_pure_1p5b_smoke.sh
pipelines/run_stage21_pure_1p5b_full.sh
pipelines/run_stage21_pure_8b_full.sh
```

## Fable Review

Fable 已 review Stage2.1-pure 代码：

```text
review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_code_review_260707.md
review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_code_followup_260707.md
review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_smoke_gate_review_260707.md
review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_smoke_gate_followup_260707.md
review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_8b_formal_pipeline_review_260707.md
review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_1p5b_full_pipeline_review_260707.md
review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_1p5b_full_pipeline_followup_260707.md
review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_1p5b_running_review_260707.md
```

结论：

- 无 blocker；
- pure repeated 不会被当成 indexed；
- stop loss 不会在第 1/2 个 pause 后误触发；
- 仍符合 Goyal et al. pause-token 的 pure repeated-token 精神；
- 可以进入 1.5B data-prep validation 和 25-step smoke；
- smoke/gate 入口的 strict gate blocker 已修复，Fable follow-up
  确认为 no blocker；
- 8B formal pipeline 的 cold checkpoint path、strict gate、failure-path
  sync 已经 Fable follow-up 确认为 no blocker。
- 1.5B full 2xA6000 pipeline 已由 Fable review，结论 `OK_TO_RUN`；
  `max_steps: null`、1 epoch、无早停、batch override、cold checkpoint path、
  cot_offset=5 和 strict natural gate 均确认无 blocker。
- 将 `save_total_limit` 提到 64 后，Fable follow-up 再次确认
  `OK_TO_RUN`；唯一 advisory 是监控 cold `/workspace` 容量。
- 训练启动后 Fable 再次 review 运行中的 1.5B full 实验，结论是当前 run
  与研究目标一致、无需要停止训练的 blocker；必须补上 checkpoint selection
  sweep，并用 disjoint selection-dev prompts 选 checkpoint，避免在正式报告集上
  43 选 1 的 selection bias。

## 执行顺序

当前 2xA6000 节点用于 1.5B full Stage2.1-pure。正式入口：

```bash
bash pipelines/run_stage21_pure_1p5b_full.sh
```

这个 1.5B full wrapper 会依次执行 pytest、data prep、完整 1 epoch SFT、
model-comparison generation、strict natural exact-3/location gate 和 summary。
默认不跑 judge；如需 judge 可设置 `RUN_JUDGE=1`。训练无早停，
`max_steps: null`，`save_steps: 25`，`save_total_limit: 64`，以保留完整
1 epoch 过程中足够多 checkpoint 用于后续选择最好 natural pause 良率。
batch 探测通过 `PER_DEVICE_TRAIN_BATCH_SIZE_OVERRIDE` /
`GRADIENT_ACCUMULATION_STEPS_OVERRIDE` 完成；formal run 需要记录最终 effective
batch。

Fable 要求 checkpoint 选择不能直接使用最终报告 prompts。训练结束后应先运行
selection-dev sweep：

```bash
python scripts/select_stage21_checkpoint.py \
  --eval_config configs/experiment/stage2_model_comparison_eval_1p5b_stage21_pure_cot5_selection_dev_2xa6000.yaml \
  --train_config configs/experiment/stage21_pause_pure_dagger_1p5b_full_2xa6000.yaml \
  --checkpoint_root /workspace/outputs/deepseek_1p5b_stage21_pause_pure_cot5_full_2xa6000 \
  --r2_root cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260707-2xa6000-1p5b-stage21-pure-full-cot5-bs4-ga2 \
  --stride_steps 50 \
  --remove_downloaded
```

该脚本使用 `sample_offset` 构造与正式 full eval 不重叠的 selection-dev
prompts，只跑 `stage21_pure_cot5_natural` 的 generation/gate，输出
`sweep_summary.json`、`sweep_summary.md` 和 `selected_checkpoint.txt`。正式
Stage2 结果必须对 selected checkpoint 在 full eval config 上重新生成一次，
不能直接报告 selection-dev sweep 数字。

1. 在 GPU 节点跑 1.5B smoke pipeline：

```bash
bash pipelines/run_stage21_pure_1p5b_smoke.sh
```

这个 pipeline 会依次执行：

- Stage2.1-pure 相关 pytest；
- 1.5B data-prep validation；
- 25-step rows-only smoke SFT；
- base / natural-pause / forced-pause 三组小样本生成；
- natural exact-3/location gate。

这个 gate 是 strict gate：如果 `stage21_pure_cot5_natural` 的
natural exact-3/location 没过阈值，pipeline 会非零退出，避免误把失败
checkpoint 当成可继续 8B 的 checkpoint。

默认不跑 judge。需要同时跑 judge / summary 时：

```bash
RUN_JUDGE=1 RUN_SUMMARY=1 bash pipelines/run_stage21_pure_1p5b_smoke.sh
```

2. 如需手动拆开执行，先跑完整测试：

```bash
python -m pytest tests/test_stage2_pause_kl_trainer.py \
  tests/test_stage2_natural_pause_metrics.py \
  tests/test_stage2_onpolicy_mining.py \
  tests/test_stage2_dagger_mix.py -q
```

3. 1.5B data prep validation：

```bash
python scripts/run_stage2_sft.py \
  --config configs/experiment/stage21_pause_pure_dagger_1p5b.yaml \
  --skip_train
```

4. 1.5B 25-step smoke：

```bash
python scripts/run_stage2_sft.py \
  --config configs/experiment/stage21_pause_pure_dagger_1p5b.yaml \
  --skip_data_prep \
  --max_steps 25
```

5. 对 smoke checkpoint 做 natural exact-3/location gate 和 capability
   sanity。

注意：smoke eval 每个 dataset 默认只取 32 条，但 gate 仍继承正式阈值。
因此 1 条 bad row 就可能让 source-wise rate 低于 `0.97`，这是有意的
严格筛查。25-step smoke fail 不等于方法失败，只表示不能跳过后续修正或
DAgger iter。

6. 如果 1.5B smoke 可行，再跑 1.5B short400 / DAgger iter1。

7. 如果 1.5B 通过，再跑 8B formal：

```bash
bash pipelines/run_stage21_pure_8b_full.sh
```

这个 8B wrapper 会依次执行 pytest、data prep、full SFT、model-comparison
generation、strict natural exact-3/location gate、judge 和 summary。strict
gate 在 judge 前执行；如果 natural exact-3/location 不过，pipeline 会直接
失败，不继续烧 judge GPU。训练 checkpoint 默认从 cold path
`/workspace/outputs/deepseek_8b_stage21_pause_pure_cot5_full_2xa100/final`
读取，因为 Stage2 hot-checkpoint watcher 会在训练结束后把完整 output
同步到 cold 并删除 hot copy。

如果需要手动拆开 8B checkpoint 的 model-comparison eval：

```bash
python scripts/run_model_comparison_eval.py \
  --config configs/experiment/stage2_model_comparison_eval_8b_stage21_pure_cot5_2xa100.yaml
```

## 成功标准

Stage2.1-pure 不能只看 loss。

必须同时报告：

- natural exact-3 pause rate；
- location match rate；
- off-target pause rate；
- malformed pause rate；
- GSM8K / MATH500 capability；
- unsafe-prompt safety judge；
- safe-prompt over-refusal；
- broken output / parse rate。

正式 8B main checkpoint 的目标 gate：

- source-wise natural exact-chain rate `>= 0.99`；
- source-wise location match rate `>= 0.99`；
- off-target pause rate `<= 0.005`；
- malformed pause rate `<= 0.005`；
- capability 与 base 接近，不能出现明显 reasoning collapse。

## 当前不能 Claim

在新版 Stage2.1-pure 跑出结果前，不能 claim：

- pure 方法已经解决 exact-3/location；
- pure 方法不改变 reasoning capability；
- DAgger 一定改善 pause over-emission；
- 8B 可以从 1.5B 直接 transfer；
- Stage3/Stage4 已经可以使用新版 pure checkpoint。

当前只能 claim：

- 旧方法失败点已定位；
- 新代码把训练目标对准了 natural exact-3/location；
- Fable review 认为代码无 blocker，可以进入 smoke 实验。

## 2026-07-07 1.5B Full Run 结果更新

2xA6000 1.5B Stage2.1-pure full run 已完成，不是 smoke：

```text
output: /workspace/outputs/deepseek_1p5b_stage21_pause_pure_cot5_full_2xa6000
checkpoint-1063 trainer_state: global_step=1063, epoch=1.0, max_steps=1063
effective batch: 2 GPUs * per-device 4 * grad-accum 2 = 16
early stopping: disabled
```

最终 strict natural gate 失败：

```text
overall exact_chain_rate = 0.8886
overall location_match_rate = 0.6395
overall off_target_rate = 0.1119
overall malformed_rate = 0.1062
```

最坏来源是 GSM8K：

```text
exact_chain_rate = 0.8100
location_match_rate = 0.0427
off_target_rate = 0.3440
malformed_rate = 0.1740
```

selection-dev checkpoint sweep 已跑完部分候选并因趋势很差停止省 GPU。已完成
`750, 800, 850, 900, 950, 1000, 1050`，全部 gate fail。ranker 在已完成候选中
选择 `checkpoint-1050`，但它不是成功 checkpoint：

```text
min_exact_chain = 0.7900
min_location_match = 0.0408
max_off_target = 0.3200
max_malformed = 0.1900
overall_exact = 0.8820
```

因此当前结论从“可以进入 smoke/full 实验”更新为：

- Stage2.1-pure rows-only 1.5B full run 已完成；
- full run 没有达到 0.99 exact/location gate；
- 不能把 `final` 或 `checkpoint-1050` 用作成功的 Stage2 checkpoint；
- 失败主要来自 GSM8K off-policy / after-`</think>` re-trigger；
- safety-style sources 有明显 `4/5` 位置口径问题，但这不能解释 GSM8K 失败。

Fable failure analysis：

```text
review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_1p5b_failure_analysis_260707.md
```

R2 归档：

```text
cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260707-2xa6000-1p5b-stage21-pure-full-cot5-bs4-ga2/
```

对应文档：

```text
docs/stage21_1p5b_pure_r2_archive_260707.md
```

下一步应改为 true DAgger iteration：

- 用当前 checkpoint 生成 on-policy outputs；
- strip 错误 pause 后用 expert formatter 重新插入 pure pause；
- 增加 GSM8K-heavy、after-`</think>` re-trigger、off-target、1/2/4+ pause
  run-length 负例；
- warm-start rows-only 训练；
- checkpoint selection 继续按 natural-generation dev gate，而不是 eval loss；
- 如果 true DAgger 仍不改善，再考虑 LoRA+KL，而不是直接扩大 8B full run。

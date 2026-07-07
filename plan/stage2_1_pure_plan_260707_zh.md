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
configs/experiment/stage2_model_comparison_eval_8b_stage21_pure_cot5_2xa100.yaml
```

关键实现：

```text
legacy/COTPauseToken/src/utils/pause_kl_trainer.py
scripts/run_stage2_sft.py
legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh
```

## Fable Review

Fable 已 review Stage2.1-pure 代码：

```text
review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_code_review_260707.md
review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_pure_code_followup_260707.md
```

结论：

- 无 blocker；
- pure repeated 不会被当成 indexed；
- stop loss 不会在第 1/2 个 pause 后误触发；
- 仍符合 Goyal et al. pause-token 的 pure repeated-token 精神；
- 可以进入 1.5B data-prep validation 和 25-step smoke。

## 执行顺序

1. 在 GPU 节点跑完整测试：

```bash
python -m pytest tests/test_stage2_pause_kl_trainer.py \
  tests/test_stage2_natural_pause_metrics.py \
  tests/test_stage2_onpolicy_mining.py \
  tests/test_stage2_dagger_mix.py -q
```

2. 1.5B data prep validation：

```bash
python scripts/run_stage2_sft.py \
  --config configs/experiment/stage21_pause_pure_dagger_1p5b.yaml \
  --skip_train
```

3. 1.5B 25-step smoke：

```bash
python scripts/run_stage2_sft.py \
  --config configs/experiment/stage21_pause_pure_dagger_1p5b.yaml \
  --skip_data_prep \
  --max_steps 25
```

4. 对 smoke checkpoint 做 natural exact-3/location gate 和 capability
   sanity。

5. 如果 1.5B smoke 可行，再跑 1.5B short400 / DAgger iter1。

6. 如果 1.5B 通过，再跑 8B formal：

```bash
python scripts/run_stage2_sft.py \
  --config configs/experiment/stage21_pause_pure_dagger_8b_full_2xa100.yaml
```

7. 8B checkpoint 必须跑 model-comparison eval：

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

# Stage1 小实验清单与结果总览 - 2026-07-05

本文档汇总 post-HB Stage1 run 中已经完成的小实验、审计和 Fable-5 复审结论。
不包含 raw prompts、raw CoTs、completions 或 hidden arrays。

## 总结论

Stage1 的工程链路和数据审计大体符合预期：pair freeze、LOSO source 数量、
dedup、R2 备份和大部分 row audit 都能支持把结果作为一个可解释的
negative/control run。

但核心科学假设没有成立：当前 hidden-prefix probe 没有稳定超过 text/surface
baseline，也没有通过 excluded-source lead-time confirmation。Fable-5 对最终
结果的复审结论是：

```text
drop_leadtime_claim is correct.
Stage1 should now be treated as a negative/control result.
Further re-analysis of the frozen test set to find a positive cell would be
post-hoc fishing.
```

## 小实验状态表

| 小实验 | 目的 | 当前结果 | 是否符合预期 |
|---|---|---|---|
| 数据 freeze audit | 确认可用 pair、LOSO source 数量、重复/缺失 | 2783 pairs 全保留；HB 152、RS 335、SR 277、WJB 2019；无 drop、无 hash collision | 符合工程预期 |
| embedding dedup audit | 查跨 source 近重复 | cross-source similarity >=0.9 为 0；0.8-0.9 near band 也为 0 | 符合 |
| human QA | 人工抽样检查 pair 质量 | 每源 60 条 sheet 已生成；但 overnight GPU run 按用户要求 bypass 了 QA gate | 未完全满足论文 claim gate |
| 主 Stage1 hidden vs full text baseline | hidden probe 是否优于 full-trajectory surface/text | 16 个 hidden-minus-surface AUROC delta 全部为负；hidden test AUROC 约 0.679-0.840，surface 约 0.917-0.965 | 不符合 hidden 优势预期 |
| Stage1b | 更早/更窄 prefix 版本 | 也全部输给 surface；Stage1b linear hidden 约 0.679-0.810，surface 仍 0.917+ | 不符合 |
| row audit | 检查 GPU prediction 行是否齐 | 4464 prediction files 中 133 个有少量 mismatch；集中在 Stage1 linear；27/32 groups 完整 | 有小问题但可解释 |
| threshold/calibration | 看分类准确率差是否主要来自阈值 | hidden BA 从约 0.604 可提到约 0.708-0.710；但 surface BA 约 0.865，length-only 约 0.801 | 阈值修正符合预期，但不改变核心结论 |
| matched-horizon | 公平比较 hidden@k vs text@k | pooled delta: k4 +0.0584，k8 -0.0046，k16 -0.1580，k32 -0.0964，k64 -0.1499 | 只支持极早 k=4 弱信号 |
| A1 score pooling | 累积 hidden score 是否更稳 | k4 +0.0558，k8 +0.0274；k16 以后不稳定 | 初步符合，但仅 diagnostic |
| A2 feature pooling | Fable 要求的更硬确认 | k4 +0.0041 CI 跨 0；k8 -0.0235 且 CI 全负；k16+ 全负 | 不符合，触发停止 equal-horizon |
| excluded-source lead-time confirmation | 在 SR/RS 上确认 lead-time | A1 primary hidden@4-text@8 delta +0.00155，CI [-0.0138,+0.0172]；A2 delta -0.065；Fable 确认 drop claim | 不符合 |

## 关键数字

### 主 Stage1/Stage1b hidden vs full-text surface

全部 16 个 hidden-minus-surface AUROC delta 为负。

- Stage1 linear:
  - HB: `-0.126`
  - RS: `-0.214`
  - SR: `-0.123`
  - WJB: `-0.093`
- Stage1b linear:
  - HB: `-0.155`
  - RS: `-0.237`
  - SR: `-0.191`
  - WJB: `-0.136`
- Stage1/Stage1b multilayer 也全部为负，约 `-0.132` 到 `-0.207`。

### Threshold/calibration

Hidden thresholded balanced accuracy:

| Policy | Mean BA | Range | Mean recall | Mean FPR |
|---|---:|---:|---:|---:|
| current_prediction | 0.6037 | 0.5149-0.6777 | 0.2885 | 0.0811 |
| platt_0p5 | 0.7096 | 0.6388-0.7632 | 0.7875 | 0.3682 |
| val_ba_max | 0.7080 | 0.6418-0.7533 | 0.7982 | 0.3821 |
| oracle_test_ba_max | 0.7206 | 0.6582-0.7763 | 0.8572 | 0.4160 |

解释：threshold 可以明显改善 operating point，但 AUROC 不变，也不能让 hidden
超过 surface/length controls。

### Matched-horizon CPU reanalysis

Pooled hidden-minus-matched-text AUROC:

| k | Delta |
|---:|---:|
| 4 | +0.0584 |
| 8 | -0.0046 |
| 16 | -0.1580 |
| 32 | -0.0964 |
| 64 | -0.1499 |

解释：公平 horizon 对齐后，只有 k=4 有弱正向信号；k>=8 不支持 hidden
优势。

### A1/A2 pooling

| k | A1 delta | A2 delta |
|---:|---:|---:|
| 4 | +0.0558 | +0.0041 |
| 8 | +0.0274 | -0.0235 |
| 16 | -0.0035 | -0.0428 |
| 32 | +0.0061 | -0.0430 |
| 64 | -0.0206 | -0.0564 |

解释：A1 的 k=8 优势没有通过 A2 feature-level refit 确认，因此不能作为
主 claim。

### Excluded-source lead-time confirmation

Frozen test population:

- `strongreject_full`: 277 pairs
- `reasoningshield`: 335 pairs
- all k pair-complete，0 drops

Gates:

- A1 primary hidden@4 minus text@8: delta `+0.00155`, CI
  `[-0.01383, +0.01725]`, gate fail
- per-source sanity:
  - `strongreject_full`: delta `-0.02618`, CI `[-0.05035, -0.00270]`
  - `reasoningshield`: delta `+0.01450`, CI `[-0.00678, +0.03964]`
  - min delta below `-0.02`, gate fail
- A2 robustness: delta `-0.06535`, CI `[-0.08765, -0.04658]`, gate fail

Decision:

```text
confirmed = false
decision = drop_leadtime_claim
```

## 符合预期的部分

- 数据 freeze 后 source 数量满足 Stage1 最低门槛。
- 没有发现 cross-source near-duplicate 风险。
- threshold 解释成立：原始分类准确率偏低，很大一部分来自 conservative
  threshold，而不是 AUROC 本身极低。
- hidden probes 确实能读出 above-chance signal；hidden AUROC 普遍显著高于
  0.5。

## 不符合预期的部分

- hidden 没有打赢 full-trajectory surface/text baseline。
- matched-horizon 后，hidden 优势没有跨 k 稳定存在。
- A1 的 score-pooling 早期优势没有通过 A2 feature-pooling 确认。
- excluded-source lead-time confirmation 失败，Fable-5 要求 drop lead-time
  claim。

## 当前可写 claim

可以写：

> 在当前 natural divergent-pair LOSO/frozen-split 设置下，当前测试过的
> 线性 hidden-state probes 没有超过 validation-selected full-trajectory
> surface text baselines，也没有通过 matched-horizon/A2/excluded-source
> lead-time confirmation。这支持把 Stage1 作为当前 probe design 的
> negative/control result。

不应写：

- hidden-state probes 优于 text/surface baseline；
- hidden prefix 有稳定 lead-time advantage；
- A1 k=8 是 confirmatory positive result。

## 后续边界

Fable-5 的最终边界是：不要继续在这个 frozen test set 上找 positive cell。
如果需要正向 follow-up，必须是 fresh preregistered setting，优先考虑
Stage2/on-policy 或另一个 fresh-data follow-up，而不是继续 Stage1
equal-horizon/lead-time rescue。

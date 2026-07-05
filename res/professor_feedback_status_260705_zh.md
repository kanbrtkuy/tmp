# 教授反馈解决状态表 - 2026-07-05

本文档对应教授提出的两条意见：

1. Generalization 不能只靠一个 held-out source，需要 leave-one-source-out
   和 source 间 variance。
2. Steering/capability 表中 GSM8K/MATH 提升是 yellow flag，需要分离
   SFT-data effect 与 steering effect，并刻画 delta 与 refusal/length/topic
   等变量的相关性。

## 状态总览

| 教授意见 | 子问题 | 当前状态 | 已有证据 | 还缺什么 |
|---|---|---|---|---|
| Generalization 不能只靠 RS-Test | 从单一 held-out source 扩展到 LOSO | **部分解决** | post-HB Stage1 已完成 4-source LOSO：`harmbench_standard`、`reasoningshield`、`strongreject_full`、`wildjailbreak_vanilla_harmful` | 教授提到的 all-six-source LOSO 尚未完成；当前应写成 4-source LOSO |
| 需要真实 transfer estimate | 报告跨 source hidden transfer | **基本解决** | Stage1 linear hidden test AUROC：HB `0.840`、RS `0.703`、SR `0.815`、WJB `0.825`；均值约 `0.796` | 需要在论文表中报告 mean/std 或 source variance，并避免声称 6-source |
| 需要 variance-across-sources | 报告 source 间方差 | **数字已有，尚需正式表格化** | 4-source hidden AUROC source spread 明显；RS 最低，其余接近或超过 0.8 | 需要最终表格写成 `mean ± std` / min-max，并说明 RS 是低端 source |
| 排除 source/format/length artifacts | 检查是否只是 artifact | **部分解决，但不能说排除** | 已做 length-only、surface/text baseline、dedup、matched-horizon；结果显示 surface/length controls 很强 | 应作为 limitation：artifact 被诊断出来而非完全排除；不能 claim clean latent safety semantics |
| fixed threshold drift | 阈值跨 dataset 漂移 | **部分解决** | threshold reanalysis 显示 hidden BA 可从约 `0.604` 提升到约 `0.708-0.710` | 只能作为 operating-point reporting；AUROC 不变，hidden 仍低于 surface BA `~0.865` 和 length-only `~0.801` |
| Steering capability table | GSM8K/MATH 上升不能当好事宣传 | **未完全解决** | Stage2 summary 已记录 full-SFT/cot3 full ckpt250 提高 GSM8K/MATH，并标注 full-SFT 不是 clean format-only intervention | 论文中必须明确这是 yellow flag/confound，不能当 steering 优点 |
| 分离 SFT-data effect 与 steering effect | SFT-only vs SFT+steering | **部分解决** | Stage2 有 base vs SFT checkpoint capability/unsafe_valid 表；Stage4 有 base、alpha=0、alpha=1/2 安全行为表 | 还缺同一 checkpoint 下的 capability 对照：SFT-only/alpha=0 vs SFT+steering alpha=1/2 的 GSM8K/MATH |
| delta 到底相关什么 | refusal rate、length、topic correlation | **基本未解决** | Stage4 CSV 已有 `refusal_keyword_rate`、`avg_generated_chars`、dataset/topic breakdown，可用于分析 | 还需正式 correlation/regression：steering alpha 或 delta projection 与 refusal、length、dataset/topic 的关系 |
| 证明 delta 是 unsafe axis 不是 structured/careful axis | 语义解释 | **未解决** | 当前只能说 steering 降低 unsafe-prompt unsafe-valid rate；不能证明 delta 是纯 unsafe axis | 需要 SFT-vs-steering 拆分 + correlation/ablation，证明不是 general careful/structured direction |

## 对教授意见的当前答复口径

可以说：

> 第一条 generalization 意见已经从“只有 RS-Test”推进到 4-source LOSO。
> hidden probe 在 held-out sources 上确实有 above-chance transfer，AUROC
> 大多接近 `0.8`，但仍没有达到教授说的 6-source LOSO，也没有排除
> surface/length artifacts。因此这部分是部分解决。

> 第二条 steering/capability confound 目前还没有真正解决。已有 Stage2
> 和 Stage4 表可以证明我们注意到了 SFT-data effect 和 capability drift，但
> 还缺严格的 SFT-only vs SFT+steering capability 对照，以及 delta 与
> refusal/length/topic 的相关性分析。

## 建议后续任务

优先级从高到低：

1. **整理 4-source LOSO transfer 表**：报告 hidden AUROC mean/std/min/max，
   同时报告 surface/length controls，明确 strong claim 失败但 weak transfer
   signal 成立。
2. **补 Stage4 capability 对照**：在同一 SFT checkpoint 下比较
   alpha=0/1/2 的 GSM8K/MATH，确认 capability 上升是否来自 SFT 而非 steering。
3. **做 steering correlation audit**：用 Stage4 joined outputs 或 summary
   统计 `unsafe_valid_rate`、`safe_refusal_rate`、`refusal_keyword_rate`、
   `avg_generated_chars`、dataset/topic 对 alpha/delta 的响应。
4. **写作上降级 claim**：把 GSM8K/MATH 上升称为 yellow flag，而不是
   capability benefit；把 delta 称为 safety-associated steering direction，
   不称为 clean unsafe axis。

## 当前结论

- Generalization/LOSO：**部分解决**，足以回应“不是只有 RS-Test”，但不够回应
  “all six sources”。
- Steering confound：**尚未解决**，需要新的 capability/correlation audit。

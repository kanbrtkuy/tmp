# Fable-5 Pause Position Review - 2026-07-05

本文档记录 Fable-5 对 Stage1 -> Stage2 pause placement 的复审结论。复审
packet 放在 tmp repo，不含 raw prompts、raw CoTs、completions、hidden arrays
或 row-level prediction files。

```text
tmp packet:
stage1_to_stage2_pause_position_review_260705/

tmp commits:
dd16cf8 Add Stage1 to Stage2 pause position packet
cc20727 Add Fable pause position review

full response:
stage1_to_stage2_pause_position_review_260705/CLAUDE_FABLE5_PAUSE_POSITION_REVIEW.md
```

## Executive Verdict

Fable-5 的最终标签：

```text
COT4_MAINLINE
```

但这是 conditional 决策：

- DeepSeek-8B Stage2 主线使用 `before cot_4`，插入 3 个 pause tokens。
- 优先复用已经存在的 `cot4 format-only checkpoint-250`，不要在 Stage3 first
  read 之前重新训练新位置。
- `before cot_3` 只作为 no-retrain position ablation。
- 不要把 `cot_120` 作为 Stage2 主线 pause port。
- 进入 Stage3 前必须补齐 Stage2 acceptance gates，尤其是 benign
  over-refusal、CoT-length drift 和 output validity。

Fable 要求把措辞降级为：

```text
before cot_4 is the least-bad engineering default for the 8B pause port,
pending Stage3 matched-horizon confirmation.
```

不能写成 “cot4 是 Stage1 证明过的最优 safety signal 位置”。

## 为什么不是 cot120

用户提出的问题是正确的：Stage1 diagnostics 曾经看到最强 readout 往往在
`cot_120` 附近，为什么 Stage2 仍然放 `cot_3/cot_4`？

Fable 的核心回答：

- `cot_120` 是 late hindsight readout，不是 early causal port。
- final label 是整个 trajectory 的函数，所以越晚越容易读出最终标签；这不等于
  越晚越适合干预。
- 最新 post-HB Stage1 中，`length_only` 在四个 source 上都超过 selected hidden
  probe，说明 late/full trajectory signal 很大程度来自 surface accumulation。
- `cot_96/cot_128` 已经出现 coverage/censoring 问题；短拒绝/短安全轨迹可能根本
  到不了 late port。
- 如果 unsafe content 已经在 token 1-119 出现，`cot_120` pause 没有 prevention
  lead time。

简短答案：

> Easy to read late does not imply useful to act late.

## Position Recommendation

| Position | Role | Decision |
|---|---|---|
| `before cot_4` | Mainline | Use 3-pause block; reuse existing ckpt250 if gates pass. |
| `before cot_3` | Ablation | No new training; existing checkpoints only. |
| `cot_16` | Conditional second SFT ablation | Only if Stage3 shows position sensitivity. |
| `cot_64` | Exploratory | Only after onset/coverage analysis. |
| `cot_120` | Not recommended for SFT | Hindsight, censoring, no lead time. |
| prompt/pre-think | Rejected | Degenerate for same-prompt paired Stage3. |
| multiple/periodic pause blocks | Deferred | Too much attribution and format-drift risk before single-position result. |

## Required Gates

### Stage2 Gates Before Stage3 Extraction

- **S2-G1 format:** `pause3_rate >= 0.99` on benign and unsafe prompts;
  tokenizer/position-convention audit passes; block size exactly 3.
- **S2-G2 capability:** GSM8K+MATH500 overall accuracy within 2 absolute points
  of base, with n and bootstrap CIs.
- **S2-G3 safety non-degradation:** unsafe-prompt unsafe_valid_rate no worse
  than base across HarmBench, LlamaGuard, and WildGuard, with CIs.
- **S2-G4 neutral behavior:** benign over-refusal within tolerance, CoT-length
  drift reported, output validity/format near base. This is the currently
  missing blocker.
- **S2-G5 non-degeneracy:** sampled decoding, multiple rollouts per prompt, and
  within-prompt pause-state variance above numerical noise.

### Stage3 Gates Before Any Stage3 Claim

- Same-prompt paired design, prompt-disjoint LOSO on the frozen keep list.
- Validation-only selection; no test-max reporting.
- Pause-state probe must beat both:
  - base-model hidden probe at matched position, and
  - matched-horizon surface text baseline.
- Fable suggested decision margin: macro-average `delta AUROC >= 0.05` with
  95% bootstrap CI excluding 0 and positive on at least 3/4 sources.
- Label-shuffle null must be near 0.5.
- Full-trajectory surface text may be reported only as hindsight ceiling, not
  as the Stage3 bar.

### Stage4 Gates Before Steering Claim

- Compare steering against the pause-model baseline, not raw base.
- Include random-direction and shuffled-probe controls.
- Require dose-response over intervention strength.
- Report unsafe reduction together with over-refusal, capability, and output
  validity.
- Effect should hold on at least 2/3 judges and at least one held-out source.

## First-Run Order

1. CPU-only position-convention and tokenizer audit.
2. CPU coverage/survival curves for `cot_3/4/16/64/96/120` by source.
3. Freeze re-audit: keep list still `2783/2783`.
4. Freeze Stage3 preregistration before test metrics.
5. Run missing ckpt250 acceptance checks: benign over-refusal, CoT-length drift,
   output validity, pause3_rate on benign prompts, capability/safety CIs.
6. If Stage2 gates pass, run Stage3 first read for `before cot_4`.
7. Run cot3 ablation only after the main Stage3 read, using existing checkpoints.
8. Consider `cot_16` only if Stage3 shows position sensitivity.
9. Do not run `cot_120` SFT before mainline Stage3.

## Do Not Claim

- Do not claim hidden states beat surface/text baselines; current full-horizon
  Stage1 falsifies that.
- Do not claim `cot_4` is optimal or signal-maximal.
- Do not claim Stage1 supports pause tokens; Stage1 has no pause tokens.
- Do not claim ckpt250 judge improvements are a pause safety benefit until
  CoT-length drift and over-refusal are controlled.
- Do not cite historical test-max numbers.
- Do not infer late-intervention value from late-readout strength.

## Practical Decision

Use `before cot_4` as the DeepSeek-8B Stage2 mainline **only as an engineering
default**, not as a validated scientific optimum. The scientific test moves to
Stage3:

```text
Do pause states at the early cot4 port beat matched-position base hidden states
and matched-horizon surface features under paired LOSO evaluation?
```

If Stage3 is null, stop before Stage4 and write the negative result.


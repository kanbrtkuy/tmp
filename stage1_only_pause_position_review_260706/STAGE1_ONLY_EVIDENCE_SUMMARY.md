# Stage1-Only Evidence Summary For Pause Placement

This summary is aggregate-only. It excludes raw prompts, raw CoTs,
completions, hidden arrays, and row-level prediction files.

Important scope constraint:

```text
Do not use any Stage2 checkpoint behavior, Stage2 SFT results, Stage3 results,
or Stage4 steering results to choose the pause position.
```

The only allowed evidence for this review is the latest Stage1 family of
results, plus code-level token-position semantics.

## Current Stage1 Run

Latest run:

```text
post-HB Stage1 LOSO / retune12288_b20
DeepSeek-R1-Distill-Llama-8B
natural same-prompt generated/generated safe-vs-unsafe pairs
```

Frozen keep pairs:

| Source | Keep pairs |
|---|---:|
| HarmBench | 152 |
| ReasoningShield | 335 |
| StrongReject | 277 |
| WildJailbreak | 2019 |
| Total | 2783 |

Freeze audit:

```text
n_input_pairs=2783
n_main_keep=2783
n_dropped=0
n_rejected_during_load=0
```

Primary high-level result:

- Current Stage1 should be treated as a negative/control result for the tested
  linear hidden-probe design, not as hidden-state superiority evidence.
- Hidden test AUROC is above chance, roughly `0.679` to `0.840`.
- Validation-selected full-trajectory surface text baselines reach roughly
  `0.917` to `0.965`.
- All 16 hidden-minus-full-surface AUROC deltas are negative.
- `length_only` beats the selected hidden probe on all four sources, showing
  that full-trajectory hindsight surface structure is very strong.
- Prompt-only TF-IDF in the same-prompt paired design is approximately chance,
  so the main issue is not prompt-only classification.

## Stage1 Positions Scanned

The current 8B Stage1 position grid included:

```text
cot_0, cot_1, cot_2, cot_3, cot_4, cot_5, cot_6, cot_7,
cot_8, cot_9, cot_10, cot_12, cot_16, cot_24, cot_32,
cot_48, cot_64, cot_96, cot_128
```

Stage1b included prompt-side controls and early CoT positions:

```text
last_prompt_token, assistant_start, assistant_last, pre_think,
think_last, cot_2, cot_3, cot_4, cot_7
```

## Matched-Horizon Reanalysis

Purpose: compare hidden@k against text features with the same prefix horizon,
rather than comparing prefix hidden states against full-trajectory hindsight
text features.

Scope:

```text
sources: HarmBench, WildJailbreak
hidden kind: linear
k grid: 4, 8, 16, 32, 64
selected layer: 28
surface family: char_tfidf
bootstrap B: 500
```

Primary equal-horizon AUROC deltas: `hidden@k - matched_text@k`.

| Source | k=4 | k=8 | k=16 | k=32 | k=64 |
|---|---:|---:|---:|---:|---:|
| HarmBench | +0.0470 | -0.0293 | -0.1395 | -0.0595 | -0.1083 |
| WildJailbreak | +0.0572 | -0.0038 | -0.1630 | -0.1015 | -0.1539 |
| pooled | +0.0584 | -0.0046 | -0.1580 | -0.0964 | -0.1499 |

Interpretation before external review:

- Equal-horizon comparison leaves a small positive hidden advantage only at
  `k=4`.
- From `k=8` onward, same-horizon text catches up or wins.
- This does not rescue Stage1 as a positive hidden-superiority result.

## A1/A2 Lead-Time Diagnostics

A1 and A2 use the same k-grid, layer-28 hidden signal, matched-horizon
`char_tfidf` surface scores, and paired bootstrap `B=500`.

They differ in the hidden-side estimator:

| Arm | Hidden estimator | Fit status |
|---|---|---|
| A1 | unweighted mean of validation-z-scored per-position hidden probe scores | reuses frozen hidden scores from the original Stage1 pipeline |
| A2 | unweighted mean of layer-28 vectors over `cot_j`, `j<=k`, then refit linear probe per k | train-split-only `StandardScaler + LogisticRegression(class_weight=balanced)` |

Same-horizon summary:

| k | A1 hidden AUROC | A1 text AUROC | A1 delta | A2 hidden AUROC | A2 text AUROC | A2 delta |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.7882 | 0.7323 | +0.0558 | 0.7364 | 0.7323 | +0.0041 |
| 8 | 0.7844 | 0.7570 | +0.0274 | 0.7335 | 0.7570 | -0.0235 |
| 16 | 0.7983 | 0.8018 | -0.0035 | 0.7589 | 0.8018 | -0.0428 |
| 32 | 0.8138 | 0.8077 | +0.0061 | 0.7646 | 0.8077 | -0.0430 |
| 64 | 0.8289 | 0.8495 | -0.0206 | 0.7932 | 0.8495 | -0.0564 |

Interpretation before external review:

- A1 gives an exploratory early-position story, especially at `k=4` and
  weakly at `k=8`.
- A2 does not confirm the A1 k=8 effect.
- A2 leaves only a tiny k=4 positive delta (`+0.0041`), too weak for a strong
  claim.

## Excluded-Source Lead-Time Confirmation

This was designed to test whether the exploratory early lead-time story
survives on excluded sources.

Frozen test population:

| Source | Pairs |
|---|---:|
| StrongReject | 277 |
| ReasoningShield | 335 |

Primary A1 gate:

```text
hidden@4 minus text@8 delta = +0.00155
95% CI = [-0.01383, +0.01725]
gate = fail
```

Per-source sanity:

| Source | A1 delta | 95% CI |
|---|---:|---:|
| StrongReject | -0.02618 | [-0.05035, -0.00270] |
| ReasoningShield | +0.01450 | [-0.00678, +0.03964] |

A2 robustness:

```text
delta = -0.06535
95% CI = [-0.08765, -0.04658]
gate = fail
```

Final Stage1 decision from this branch:

```text
confirmed = false
decision = drop_leadtime_claim
```

## High-Offset Caveat

Prediction row audit found no broad extractor-level row drop. Remaining
mismatches are localized mostly to Stage1 linear high-CoT-offset positions,
especially around high offsets such as `cot_96` and `cot_128`, where coverage
can change because shorter trajectories may not reach the requested offset.

Implication before external review:

- High-offset Stage1 readout can be descriptive, but it is a weaker basis for
  choosing an early causal pause port.
- If a late position is recommended anyway, it needs a coverage/censoring
  argument and should probably be framed as a readout ablation rather than
  a prevention-oriented intervention.

## Token Position Semantics

This is code-level convention, not Stage2 outcome evidence.

The pause insertion utility:

```text
1. split the `<think>...</think>` reasoning span
2. skip leading whitespace-only reasoning tokens once
3. count non-whitespace CoT tokens as cot_0, cot_1, ...
4. insert pause text before the configured cot_offset token
```

Therefore:

```text
before cot_4 = after four non-whitespace CoT tokens, before the fifth

<think> t0 t1 t2 t3 <|pause|><|pause|><|pause|> t4 ...
```

One question for review is whether Stage1 evidence at hidden@cot_4 should map
to inserting pause tokens before `cot_4`, after `cot_4`, or at some other
neighboring location, given transformer hidden-state timing and the causal goal
of influencing subsequent reasoning.

## Current Stage1-Only Provisional Interpretation

If forced to choose a position using Stage1 only, the current local
interpretation is:

```text
candidate pause position: before cot_4
strength: weak/exploratory, not confirmatory
reason: k=4 is the only matched-horizon point with positive hidden-minus-text
        evidence; k>=8 fails, A2 weakens the effect, and excluded-source
        lead-time confirmation fails.
```

This should not be presented as:

```text
cot_4 is proven optimal
cot_4 has confirmed lead-time advantage
Stage1 proves hidden states beat text baselines
```


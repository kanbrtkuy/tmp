# Fable Review: Stage2.1-pure 1.5B Failure Analysis

Date: 2026-07-07

## Context Sent

- Model: `DeepSeek-R1-Distill-Qwen-1.5B`
- Method: rows-only pure repeated pause SFT.
- Pause format: `<|pause|><|pause|><|pause|>`
- Target: after `cot_4`, before `cot_5`.
- Training: 1 full epoch, 1063 steps, no early stopping, 2x A6000, effective batch 16.
- Objective: KL-transparent continuation + emit loss + emit margin + stop-after-3 loss + margin suppression.
- Teacher-forced metrics near the end: pause target argmax rate `1.0`, target probability `1.0`, non-pause argmax `0.0`, stop-after-chain around `1e-6`, eval loss around `0.028`.
- Free-running natural gate failed:
  - overall exact-chain `0.8886`
  - block presence `0.9948`
  - malformed `0.1062`
  - off-target `0.1119`
  - location match `0.6395`
  - average pause count `3.122`
- GSM8K was the worst source:
  - exact-chain `0.81`
  - malformed `0.174`
  - off-target `0.344`
  - location match `0.0427`
- Post-training sweep over steps `750, 800, 850, 900, 950, 1000, 1050` also failed, so the sweep was stopped to save GPU cost.

## Fable Verdict

Fable's bottom line:

- This run repeats the 8B lesson: the training saturated the wrong objective.
- Teacher-forced/off-policy pause losses can look perfect while natural autoregressive pause emission fails.
- The GSM8K location collapse may also contain a measurement issue because the strict `==5` location metric is computed after a decode-to-text then retokenize round trip.
- Run cheap diagnostics before spending more GPU on retraining.

## Ranked Root Causes

### 1. Off-policy training data and exposure bias

The config family is named `dagger`, but this run trained on a fixed offline pool and did not perform a real on-policy DAgger iteration.

All training prefixes are offline CoT texts with the pause inserted deterministically. The model never sees its own natural prefixes paired with the correct pause target. Because only the pause token row is trainable, emission depends on whether frozen hidden states from free-running generations look like the offline prefix states. GSM8K appears to diverge most strongly from the offline corpus.

Fable treated the teacher-forced-perfect / free-running-fail split as direct evidence for this cause.

### 2. Rows-only capacity ceiling for localization and counting

Rows-only training asks a frozen model plus one repeated pause token row to solve two hard behaviors:

- Localize exactly offset 5 across natural prefixes.
- Count exactly three identical content-free pause tokens and stop.

Fable noted that GSM8K exact-chain around `0.81` with location match around `0.04` suggests the model learned to emit a pause block somewhere, but not reliably at the desired position.

### 3. Brittle location metric

Fable flagged that training insertion uses tokenizer offset alignment, while natural gate location uses generated text decoded and retokenized before checking exact token index `== 5`. Whitespace and BPE differences can create +/-1 drift.

Fable emphasized that this cannot explain the whole failure because off-target pause rate `0.344` on GSM8K is a real behavioral issue, but it could make the location number look worse than the actual behavior if mass is concentrated at index 4 or 6.

### 4. Checkpoint selection signal

Teacher-forced eval loss is the wrong selection metric for this behavior. The dev sweep partially compensated, but every checked checkpoint failed, so selection alone cannot rescue the run.

### 5. Learning rate

Fable considered `1e-3` on two rows a low-priority suspect. Teacher-forced metrics converged cleanly, so learning rate is unlikely to be the main cause.

## Cheap Diagnostics Before Retraining

### B1. Free-run location histogram

Use existing natural-generation output JSON files. Per source, histogram:

- `first_pause_token_index_inside_think`
- `pause_count_before_think`
- `pause_count_after_think_end`
- no-`<think>` rows

Interpretation:

- Spike at 4 or 6: likely off-by-one metric/insertion convention issue; rescore with the corrected convention before retraining.
- Broad smear over many positions: genuine localization failure.
- Off-target mass mostly from no-`<think>` rows: the failure is partly prompt/template/think-block behavior.

### B2. Metric round-trip audit on training text

Take roughly 1k already-inserted training rows and run the natural pause metric on those texts with no model involved.

Expected result should be exact-chain `1.0`, location `1.0`, off-target `0.0`.

Any deficit is the metric ceiling. If round-trip location is below `0.99`, the strict gate is not reachable by any model under the current metric.

### B3. Teacher-force pause decision on self-generated prefixes

Sample free-running generations, strip pauses, truncate to the first five non-space think tokens, and run a forward pass:

- Measure pause probability/rank/margin at positions 3 through 8.
- Force append one, two, and three pauses, then measure stop behavior.
- Compare self-generated prefixes against training-corpus prefixes.

Interpretation:

- High pause probability on training prefixes but low on self-generated prefixes: covariate shift, true DAgger is likely the fix.
- Smeared pause probability across positions 3 through 8 even on training prefixes: rows-only cannot localize; escalate to LoRA+KL.
- Stop margin collapses only after self-generated pauses: add run-length on-policy counterexamples.

## Next Training Recommendation

Fable recommends a real DAgger iteration before any expensive 8B rerun:

1. Sample free-running generations from the current checkpoint on training prompts.
2. Strip pauses and relabel using the expert formatter, reinserting pure repeated pause at offset 5.
3. Build an iteration-1 mix of offline and on-policy relabeled data.
4. Add run-length counterexamples for one, two, four-plus, and off-target pauses.
5. Warm-start rows-only training from the current checkpoint.
6. Select checkpoints by natural-generation dev gate, not eval loss.
7. Stop after two or three iterations if the gate does not improve sharply.

If diagnostics show smeared localization or DAgger plateaus, escalate to a minimal LoRA+KL approach over the last few layers while keeping the same pure repeated `<|pause|>` token inventory.

## Rows-Only Feasibility

Fable split feasibility into two parts:

- Exact-chain `0.99`: plausible but not likely. DAgger and run-length negatives could improve from `0.89-0.92`, but `0.99` may be optimistic.
- Exact location `0.99` at strict `==5`: unlikely unless diagnostics reveal a metric off-by-one issue.

Fable's decision rule:

- Run B1/B2/B3 first.
- If B3 shows covariate shift, run one real DAgger iteration.
- If B3 shows localization is not linearly readable, stop rows-only and move to LoRA+KL.

## Follow-Up After B1/B2 Diagnostics

After the initial Fable review, we ran B1 and B2 on the existing outputs.

### B2 Result

Metric round-trip on all 17k inserted training outputs was perfectly clean:

- exact-chain `1.0`
- exact-3 `1.0`
- block presence `1.0`
- malformed `0.0`
- off-target `0.0`
- location match `1.0`
- first pause index histogram: `{5: 17000}`

This means the current metric has no ceiling on the training texts. The
training insertion path and natural metric path agree on offline inserted data.

### B1 Result

Final full natural outputs showed two regimes:

- Safety-style sources often put the first pause at index `4` or `5`.
- GSM8K had a real behavioral failure, not just an off-by-one metric issue.

Key source-level numbers:

| source | exact | loc==5 | loc in {4,5} | off-target | malformed |
|---|---:|---:|---:|---:|---:|
| GSM8K | 0.8100 | 0.0427 | 0.3679 | 0.3440 | 0.1740 |
| HarmBench | 0.8367 | 0.9000 | 0.9933 | 0.0900 | 0.1633 |
| JailbreakBench | 0.9400 | 0.8800 | 0.9900 | 0.0100 | 0.0600 |
| MATH500 | 0.9967 | 0.9267 | 0.9633 | 0.0033 | 0.0033 |
| OR-Bench Hard | 0.8867 | 0.8763 | 1.0000 | 0.0567 | 0.1100 |
| StrongReject | 0.9033 | 0.7200 | 0.9800 | 0.0433 | 0.0967 |
| XSTest | 0.9333 | 0.6745 | 0.9899 | 0.0133 | 0.0600 |

GSM8K first-index histogram:

```text
4:160, 2:38, 3:30, 1:30, 5:21, 6:19, None:8
```

GSM8K pause-count totals:

```text
inside/before/after = 1208 / 0 / 367
```

The large `after_think_end` count means GSM8K is re-triggering pause after the
CoT block; this is not explained by a one-token location convention shift.

### Fable Follow-Up Verdict

Fable's updated conclusion:

- Safety sources are effectively solved under a preregistered `5±1` location
  convention, but this amendment must be documented explicitly.
- GSM8K is the genuine failure source.
- The most likely mechanism is off-policy/covariate shift plus after-think
  re-triggering.
- B3 is less urgent after B1/B2 because the failure mode is already visible in
  the free-running outputs.
- Do not abandon rows-only yet; rows-only abandonment remains gated on an E0
  linear-separability/localization probe or a failed true DAgger iteration.
- Next run should be true DAgger with on-policy relabeling and explicit
  after-think/off-target negatives.

## What To Tell The PI

Fable says to tell the PI:

- Rows-only pure-pause 1.5B failed the natural-generation gate at every checked checkpoint.
- Teacher-forced metrics being perfect while natural generation fails is the main finding.
- This reproduces the 8B failure pattern under a stronger loss, so the issue is not simply training length.
- The run was not true DAgger yet; true on-policy relabeling has not been tested.
- The location metric may have a strict retokenization issue and should be audited before interpreting the exact `0.0427` GSM8K location number.
- Treat this as a cheap-method debugging failure, not as a final project failure.

Fable says not to claim:

- Do not cite step-950 `0.9196` as progress.
- Do not say DAgger failed.
- Do not say pure pause cannot be learned.
- Do not say rows-only is dead before B3 and one real on-policy iteration.
- Do not claim offset 5 is validated for 1.5B.
- Do not describe teacher-forced saturation as Stage2 success.

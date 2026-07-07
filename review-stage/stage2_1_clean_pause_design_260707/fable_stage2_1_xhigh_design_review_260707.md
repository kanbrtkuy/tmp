# Fable Review: Stage2.1 Clean Natural Pause Emission

Date: 2026-07-07

Fable could not fetch the private GitHub repo in its session, so this review is
based on the method/result summary and named code paths in the prompt.

## Bottom Line

The failure is exposure bias plus a selection metric that could not see it,
compounded by a single repeated token whose "count-to-3" semantics exceed a
linear readout on frozen states, while the KL objective erases the pause-history
signal needed to stop.

Fable recommends Stage2.1:

- distinct pause chain:
  `<|pause_1|><|pause_2|><|pause_3|>` rather than repeating one token;
- margin emit / stop-after-3 / suppress losses;
- DAgger-style expert relabeling on model-generated text;
- natural-generation checkpoint selection;
- keep KL transparency and rows-only freezing first;
- pre-build a pause-head escalation behind a flag only if rows-only fails.

## Ranked Diagnosis

1. **Exposure bias / covariate shift.** Current emit, suppression, and KL losses
   are teacher-forced on gold prefixes, while inference visits model-generated
   states such as "just emitted three pauses" and GSM8K-like step boundaries.
2. **Checkpoint selection blind to target metric.** Early stopping used
   teacher-forced eval loss, not natural exact-3 pause emission.
3. **Single repeated pause token is under-capacitated for count-to-3.** With
   rows-only training, one pause logit direction must fire at the target, remain
   high for pauses 1-2, stop after pause 3, and never fire later.
4. **Trigger ambiguity and data coverage.** GSM8K short-step arithmetic has many
   contexts that resemble the target boundary but were not covered by the
   trusted-CoT training distribution.
5. **Argmax targets do not survive sampling.** Pause target argmax 1.0 does not
   imply enough logit margin under temperature/top-p sampling.

## Paper-Grounded Justification

- Scheduled Sampling (Bengio et al., 2015) and DAgger (Ross et al., 2011)
  support the diagnosis that teacher-forced training does not cover the learned
  policy's free-running state distribution.
- Professor Forcing (Lamb et al., 2016) is related, but Fable prefers DAgger
  because the formatter is an exact expert relabeler.
- Unlikelihood Training (Welleck et al., 2019) supports negative-token training
  for unwanted/repeated tokens; Fable recommends a margin-hardened variant.
- InstructGPT / KL-regularized RLHF (Ouyang et al., 2022) supports changing
  behavior while controlling drift using KL-style regularization.
- Pause/filler-token work such as Goyal et al. (2023, arXiv:2310.02226)
  supports content-free trainable tokens; distinct pause tokens are compatible
  with the idea and cleaner for exact length.

## Stage2.1 Algorithm

### Distinct Pause Chain

Use:

```text
<|pause_1|><|pause_2|><|pause_3|>
```

instead of:

```text
<|pause|><|pause|><|pause|>
```

This turns a hard repeated-token count decision into a deterministic chain:
`pause_1 -> pause_2 -> pause_3 -> stop`.

### Trainable Parameters

Default tier:

- train only the three pause token embedding rows and output rows;
- keep the base model frozen.

Escalation tier:

- optional small pause head `h_t -> 64 -> 3` added only to pause-token logits;
- non-pause logits should remain unchanged on pause-free prefixes;
- use only if rows-only Stage2.1 fails pilot gates.

### Losses

Let `P = {p1, p2, p3}`, `z_v(t)` be token logit, and
`M(t)=max_{v not in P} z_v(t)`.

Emit loss at trigger and chain positions:

```text
CE(target=p_k) + lambda_emit_margin * softplus(M(t) - z_p_k(t) + delta_emit)
```

Stop-after-3 loss immediately after `p3`:

```text
sum_k softplus(z_p_k(t) - M(t) + delta_sup)
```

Off-position suppression:

- same hinge as stop loss at non-target positions;
- include all teacher-forced non-pause targets;
- upweight step-boundary positions and DAgger-mined violations.

Continue KL:

- keep continuation KL to pause-stripped teacher;
- apply to static and on-policy rows.

Pre-KL:

- do not add as a loss;
- monitor pre-block KL as a metric.

### DAgger Loop

1. Sample from the current checkpoint on train-side prompt pools:
   SFT-source prompts, GSM8K-train, MATH-train, and safety train-side prompts.
2. Strip all pause tokens.
3. Run the deterministic formatter to insert the expert pause chain into the
   model-generated CoT.
4. Build on-policy training rows with emit, stop, and suppression masks.
5. Upweight observed off-target pause and overshoot contexts.
6. Train on a mix of static SFT rows and aggregated on-policy rows.

### Natural-Generation Checkpoint Selection

Every 25-50 steps, decode a fixed dev suite and compute:

- exact3;
- block presence;
- location match;
- malformed pause sequence rate;
- off-target pause rate;
- average pause count;
- capability/safety guardrails;
- pre-block and continuation KL.

Select best checkpoint by:

```text
max min(exact3_GSM8K, exact3_MATH, exact3_safety)
```

subject to guardrails. Do not select on teacher-forced eval loss.

## Minimal Experiments Before Full 8B

1. Existing-checkpoint diagnostics:
   pause probability/logit audit at stop, overshoot, re-trigger, and random
   step-boundary positions.
2. Greedy vs sampling ablation on GSM8K.
3. FSM-constrained control as an appendix diagnostic only.
4. GSM8K teacher-forced suppression probe.
5. 1.5B mechanism pilot.
6. Short 8B run with one DAgger round before any full 8B rerun.

## Implementation Checklist

Fable asked Codex to implement:

1. distinct-token mode in `src/cot_safety/formatting/pause_insertion.py`;
2. public strip/expert relabel helpers;
3. a Stage2.1 trainer or extension of `PauseKLSFTTrainer` with margin
   emit/stop/suppress losses and continuation KL;
4. optional pause-head module behind a flag;
5. on-policy negative mining script;
6. natural pause eval/checkpoint-selection utilities;
7. Stage2.1 configs for 1.5B pilot and short 8B;
8. tests for insertion, loss masks, DAgger mining, and natural eval metrics.

## Go/No-Go Gates

- G0 diagnostics: confirm violations are re-trigger/overshoot and margin-like.
- G1 1.5B pilot: min exact3 >= 90%, off-target <= 1%, guardrails pass.
- G2 short 8B: min exact3 >= 97% and rising, off-target <= 0.5%, guardrails
  pass.
- G3 full 8B: dev min exact3 >= 99% sustained across three evals; final test
  evaluation once, no test-based selection.

## Claims

Allowed:

- main results use unconstrained sampling if Stage2.1 passes;
- no safety labels are used in Stage2 training;
- base model stays frozen under rows-only/pause-head constraints;
- exact-3 numbers are reported under a fixed decoding config.

Disallowed:

- inferring natural emission success from teacher-forced metrics;
- presenting constrained decoding as the main method;
- claiming pause hidden states are pure base-model states;
- test-set-max checkpoint selection;
- using 1.5B pilot as proof that the 8B cot_4/cot_5 port is valid.

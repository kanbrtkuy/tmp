# Fable Review: Stage2 8B Natural Pause Emission Failure

Date: 2026-07-07

Fable could not fetch the private GitHub repo in its session, so this review is
based on the prompt summary, code-path descriptions, and reported metrics. Code
details that require direct inspection are marked as code-checks by Fable.

## Verdict

This is not primarily an undertrained model. It is a correctly optimized wrong
objective: teacher-forced eval loss and continuation KL saturated, but the
training/selection loop never optimized the actual Stage2 success criterion,
natural-generation exact-3 pause emission.

## Two Failure Modes

Fable separated the issue into two mechanisms:

| symptom | evidence | mechanism |
|---|---|---|
| Run-length overshoot at the target location | XSTest exact-3 0.700 with off-target 0.004 and avg 3.792; MATH500 exact-3 0.9267 with off-target 0.0033 and avg 3.377 | The model starts the pause block at the right place but sometimes emits 4-6 pauses instead of exactly 3. The stop-after-third-pause decision fails. |
| Mid-generation re-triggering | GSM8K off-target 0.802 and avg pause count 8.288 | The model emits pause blocks again later in generation, likely at contexts resembling CoT step boundaries or resets. |

## Ranked Root Causes

1. Suppression loss covers the wrong distribution. It applies at teacher-forced
   non-pause labels in the trusted-CoT corpus, not at the model's own
   free-running contexts. GSM8K exposes this most strongly.
2. Rows-only capacity is marginal for an exact count-to-3 decision. One pause
   embedding/unembedding direction must fire at the target, keep firing for
   pauses 1-2, stop after pause 3, and never fire elsewhere.
3. Checkpoint selection and early stopping were blind to the target metric.
   Eval loss saturated, but exact-3 natural generation was not measured during
   training.
4. Suppression loss likely has no margin. If it is `-log(1 - p_pause)`, the
   gradient vanishes once in-distribution pause probability is small, leaving no
   robust margin for OOD contexts.
5. Possible masking/truncation gaps should be audited, especially whether
   suppression covers full sequences or shares any limit with the KL token cap.
6. Data coverage is insufficient for GSM8K-style and safety-style CoT
   generation contexts.

Fable did not rank insufficient training steps as a root cause.

## Early Stopping

Fable's judgment:

- Step 575 early stopping was not the main issue.
- It fired correctly for the objective it watched.
- The watched objective was wrong for the strict Stage2 goal.
- Continuing to 1 epoch with the same loss would likely not solve exact-3 and
  could even worsen over-emission by sharpening pause logits.

## Suppression Loss Assessment

Fable said the current suppression is necessary but structurally insufficient:

- It penalizes pause probability only at teacher-forced non-pause positions.
- It does not directly train on the model's own sampled off-target pause
  contexts.
- It may nominally cover the post-third-pause position in teacher-forced data,
  but not robustly enough for free-running generation.

## Rows-Only Constraint

Fable's view:

- For >=99% exact-3 learned natural emission across OOD domains, rows-only may
  be too constrained.
- But rows-only should not be abandoned immediately because it preserves
  capability/safety behavior and keeps drift low.
- Try better objectives, on-policy negatives, margin suppression, and exact-3
  checkpoint selection before relaxing rows-only.
- A decoding-side stop-after-3 logit processor is the cheapest deterministic
  fix for run-length overshoot and should be tested immediately as a fallback.

## Recommended Stage2.1 Changes

Protocol changes:

1. Add a natural-generation validation loop every 50 steps.
2. Greedy-decode roughly 100-200 validation prompts with a GSM8K-heavy mix plus
   safety prompts.
3. Compute exact-3, off-target rate, average pause count, and capability/safety
   drift.
4. Select checkpoints by min-across-datasets exact-3, then off-target pause
   rate, then eval loss.
5. Stop using eval loss as the primary model-selection metric for a generation
   behavior target.

Objective changes:

1. Add on-policy negative mining / DAgger-style suppression:
   sample current checkpoints, collect off-target pause and overshoot contexts,
   and train suppression directly on those contexts.
2. Replace or augment suppression with a margin/hinge objective:
   penalize `max(0, logit_pause - logit_top_nonpause + margin)`.
3. Add explicit stop-after-3 loss with high-weight suppression immediately
   after the third pause.
4. Add synthetic negatives with 3 pauses inserted at wrong positions.
5. Add GSM8K/MATH/safety-style self-generated CoT data to cover the failure
   domains.

Decoding fallback:

1. Add a stop-after-3 logit processor.
2. Optionally ban pause elsewhere after the intended block.
3. Report learned-natural and constrained-natural separately.
4. If constrained natural reaches >=99% exact-3 without capability/safety drift,
   it can de-risk Stage2 while training fixes are developed.

Keep:

- KL transparency.
- Rows-only invariant as the first attempt.
- Current checkpoint as initialization.

Relax rows-only only if rows-only plus on-policy negatives, margin suppression,
and exact-3 checkpoint selection cannot get close to target.

## Minimum Experiments Before Another Full 8B Run

1. Zero-training diagnostic:
   characterize GSM8K failures with position histograms, preceding-token
   contexts, and run-length distribution.
2. Zero-training constrained decoding test:
   run current checkpoint with stop-after-3 and optional ban-elsewhere logit
   processor. Check exact-3, GSM8K/MATH accuracy, and safety behavior.
3. Metric audit:
   verify eval's target-location definition exactly matches Stage2 insertion
   tokenization and offset convention.
4. Cheap training probe:
   rows-only plus on-policy negatives, margin suppression, and exact-3
   checkpoint selection on 1.5B or a short 8B run. Gate any expensive full 8B
   rerun on this.

## Claims To Avoid

- Do not claim "the model learned to emit pauses" from teacher-forced argmax or
  eval loss alone.
- Do not use forced-pause capability parity as evidence about natural emission.
- Do not claim training stopped too early caused the failure.
- Do not claim no safety drift without confidence intervals.
- Do not present natural pause rate 1.0 as a success when GSM8K average pause
  count is 8.288.

## Bottom Line

The failure is not simply too few steps. The current method optimizes
teacher-forced pause/KL behavior, but the strict Stage2 success criterion is a
free-running generation behavior. The next step is Stage2.1: add
natural-generation validation, on-policy negative mining, margin/stop-after-K
losses, and a constrained-decoding fallback test before paying for another full
8B run.

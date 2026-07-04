# SafeChain Stage3/Stage4 Follow-up Review Request

Date: 2026-07-04.

This is a follow-up to `CLAUDE_FABLE5_STAGE3_STAGE4_REVIEW.md`. Please review
the user's concern and the executor's proposed redesign. Write the complete
Markdown review to:

`safechain_stage3_stage4_review_20260704/CLAUDE_FABLE5_STAGE3_STAGE4_FOLLOWUP_REVIEW.md`

Do not edit code.

## Context

SafeChain has four stages:

1. Stage1: show safe/unsafe CoT separability in hidden states.
2. Stage2: train a pause-token model.
3. Stage3: test whether pause positions carry safe/unsafe separability.
4. Stage4: steer pause hidden states away from unsafe CoT while preserving
   capability, refusal behavior, and output format.

The Stage2 method has changed. The current candidate is not ordinary full SFT;
it is `kl_transparent_emit`:

- pause-slot CE encourages the model to emit `<|pause|>`;
- pause-stripped continuation KL encourages the continuation distribution with
  pauses to match the same model run on the pause-stripped sequence;
- suppression discourages pause emission at non-pause positions;
- only pause embedding rows are supposed to update.

The previous Fable review identified a core risk: this method may make pause
tokens harmless but causally inert, which would break the premise of Stage4 as
pause-token hidden-state steering.

## User's question

The user asks:

1. For the risk that `kl_transparent_emit` makes pause tokens "harmless but
   useless" / inert, does Fable have a good solution?
2. Please review the executor's proposed Stage3/Stage4 redesign below.
3. Be precise about whether the fix should remain within Stage3/Stage4, or
   whether Stage2 itself must be modified into a Stage2.5 objective if liveness
   fails.

## Previous Fable proposal to re-evaluate

From the previous review:

> Fallback if liveness fails: Stage2.5 auxiliary loss — keep
> `kl_transparent_emit` but add (i) a pause-state contrastive/probe-margin term
> so pause states organize safe/unsafe, and (ii) a small attention-mass floor on
> pause KVs from the next k tokens (or a scheduled reduction of KL weight on the
> first post-pause tokens) so a live channel survives. Then re-run the liveness
> battery. If natural emission ≈ 0 as well, Stage4's premise is unsupported for
> this Stage2 recipe.

Please assess whether this is still the best answer, and if so give a concrete
minimal Stage2.5 design:

- exact loss terms;
- where they apply;
- what data labels they require;
- how to avoid changing model capability or increasing refusal;
- how to preserve the original goal of "insert pause tokens without changing
  model behavior" as much as possible;
- what ablations/gates prove the auxiliary terms did not simply reintroduce
  full-SFT behavior.

## Executor's proposed redesign

The executor proposes the following logic.

### Core diagnosis

The new Stage2 `kl_transparent_emit` is not ordinary SFT. Its goal is for the
model to emit pause tokens, while the continuation remains as close as possible
to the no-pause continuation. Therefore, it naturally risks producing inert
pause tokens: harmless, but also useless as steering ports.

Consequently, Stage3/Stage4 should not directly reuse the old full-SFT pause
model design. The first question is not steering; the first question is whether
the pause position remains causally live.

### Proposed Stage3 change

Stage3 should become a test of whether pause states carry independent signal,
not just whether a pause probe has high AUROC.

It should answer:

1. Does the pause hidden state contain information beyond prompt baselines?
   Add `last_prompt_token`, `pre_think`, and a prompt-only text classifier
   baseline. Otherwise high AUROC may be prompt-risk classification.

2. Fix pause-neighborhood controls. Current `control_cot_3/4` alias to
   post-pause positions, which invalidates "pause vs content control"
   comparisons. Replace them with true controls: either matched content tokens
   from a pause-free forward, or content tokens at matched depth away from the
   pause in the paused sequence.

3. Use new Stage2 on-policy generations and re-judge CoT. For the same prompt
   sampled 10 times, the label should be whether this generation's CoT is
   unsafe, not a teacher-forced/reference label. Otherwise Stage3 is still
   measuring prompt identity/risk.

Proposed Stage3 success criterion:

`pause_or_post_pause_AUROC - max(prompt_baseline_AUROC, true_content_control_AUROC)`

If this delta is near zero, we cannot claim pause captures an unsafe trajectory;
we can only claim that pause positions read out prompt risk.

### Proposed Stage4 change

Stage4 should become: liveness first, then gated projection steering.

1. Run a liveness battery before any Stage4 steering:
   injection gain, attention mass, pause-KV ablation, and safe/unsafe patching.
   If liveness is red, pause-token inference-time Stage4 stops and we should
   consider Stage2.5 instead.

2. Replace off-policy `learned_delta`. Use on-policy pause hidden states from
   the new Stage2 model, judge-labeled safe/unsafe, to estimate a safe/unsafe
   mean-difference direction `u`.

3. Use probe-gated projection/rejection steering:

   `h <- h - lambda * max((h - mu_safe) dot u, 0) * u`

   Apply it only when the online Stage3 probe score says the current pause
   state is high risk. Add a norm cap so the hidden state is not pushed off
   manifold. This should reduce over-refusal and capability damage relative to
   unconditional additive deltas.

4. Evaluation must separately judge CoT and answer. Stage4's primary endpoint
   should be unsafe CoT rate, not whole-response unsafe rate. Also report
   over-refusal, capability, broken output, unlabeled judge rate, `</think>`
   closure, termination, repetition, and length shift.

### Proposed experimental order

1. Pick the Stage2 checkpoint, confirm rows-only invariant and natural pause
   emission.
2. Run liveness battery.
3. Fix Stage3 controls + prompt baseline + on-policy labels, then rerun Stage3.
4. If Stage3 shows pause signal above prompt/content controls, implement GPRS
   steering.
5. Run 1.5B micro-pilot, then 1.5B full pilot, then 8B.

## Questions for Fable

Please answer directly:

1. Is the executor's redesign correct, too conservative, or missing a better
   path?
2. What is the best concrete solution for the inert-pause risk?
3. Is Stage2.5 with pause-state contrastive/probe-margin + attention-mass floor
   the right fallback, or is there a better loss that keeps continuation
   transparent while preserving a live steering port?
4. If Stage2.5 is used, can Stage4 still use the GPRS algorithm above? Would it
   require changes?
5. What is the minimal liveness battery that gives a reliable go/no-go without
   spending too much GPU?
6. Which Stage3 fixes are mandatory before claiming pause-specific signal?
7. Which Stage4/eval fixes are mandatory before claiming reduced unsafe CoT
   without over-refusal/capability loss?
8. Give a final ordered implementation plan: P0/P1/P2, with kill criteria.

Be blunt and precise. The user has plenty of Fable budget; do not compress away
important caveats.

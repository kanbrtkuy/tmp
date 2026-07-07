# Fable Review: Pure Repeated Pause Stage2

Date: 2026-07-07

Question: The user rejects indexed pause tokens because they may change the
meaning of a pure pause token. Under the constraint that Stage2 must use one
content-free `<|pause|>` token repeated exactly three times, is the failed 8B
run undertrained or is the objective wrong? What is the cleanest next method?

## Verdict

Wrong objective, not too few steps. Do not simply rerun longer.

The previous run optimized teacher-forced fit and achieved it, but that
objective did not contain the natural generation behavior required for exact-3
pause emission. Training longer would likely further optimize the same wrong
objective.

## Diagnosis

1. The stop decision was never trained. Natural exact-3 emission requires the
   model to flip away from `<|pause|>` after the third pause. The failed config
   left `stop_weight` at default `0.0`, so this transition had no dedicated
   loss.

2. Checkpoint selection measured the wrong thing. Teacher-forced `eval_loss`
   cannot see free-running run-length behavior, so it cannot select for
   natural exact-3 pause emission.

3. Unlikelihood suppression saturates. Once teacher-forced pause mass at
   non-pause positions is small, `-log(1-p_pause)` provides little gradient,
   while the generation failure remains.

4. Rows-only feasibility is an empirical question. Because the transition
   function is frozen, rows-only training works only if the count of repeated
   pauses is already linearly readable in hidden states.

## Newly Identified Code Blocker

Current single-token `_stop_after_pause_chain_mask` fires after every pause
token, including after pause #1 and #2. If `stop_weight` is turned on as-is,
the stop loss would fight the emit loss on the second and third pause positions.

Required fix: for the single-token branch, stop loss should fire only when the
preceding three tokens are exactly `<|pause|><|pause|><|pause|>` and the token
before that is not another pause.

## Recommended Pure Repeated-Token Method

1. E0 probe gate:
   teacher-force gold rows plus on-policy prefixes, extract hidden states after
   k=1,2,3 pauses, and train a linear probe for continue-vs-stop. If count is
   not readable, rows-only is unlikely to work.

2. Rows-only v2:
   - fix single-token stop mask;
   - set `stop_weight >= 1.0`;
   - switch suppression to margin loss;
   - consider `emit_margin_weight > 0`;
   - add on-policy run-length counterexamples via DAgger;
   - select checkpoints by natural exact-3 rate, not eval loss.

3. If rows-only v2 fails:
   escalate to minimal LoRA + KL anchor while preserving the same `<|pause|>`
   token inventory.

4. Full-model SFT is the last resort and would weaken the transparency claim.

## Paper Grounding

- Goyal et al. supports the pure repeated-token design: one content-free pause
  token repeated, not indexed token identities.
- Exposure-bias work (DAgger, scheduled sampling, sequence-level training)
  supports training/evaluating on free-running states rather than only
  teacher-forced contexts.
- Unlikelihood/repetition literature explains why teacher-forced suppression
  alone may not fix free-running repeated-token attractors.

## Claim Implications

If rows-only v2 succeeds, the strongest Stage2 claim survives: one pure
content-free pause token, repeated naturally, with only pause rows trained.

If minimal LoRA is required, the claim becomes weaker but still semantically
pure.

If full SFT is required, the transparency claim weakens substantially and
Stage2 must be reported as a model-adaptation effect.

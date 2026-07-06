# Fable Review: Stage2 8B Full Gate

Date: 2026-07-07

## Verdict

**Decision: PROCEED to Stage3. GSM8K over-emission is a non-blocking limitation,
not a Stage2 rerun trigger.**

Reasoning: GSM8K is not a Stage3 source. Stage3 runs on HB/RS/SR/WJB safety
pairs, where exact-3 compliance is 0.80-0.92, with XSTest 0.70 as the floor.
The checkpoint passes what Stage3 depends on: target-position emission, KL
transparency, capability within noise, and safety non-degradation across all
three judges in both natural and forced modes.

## Blockers

None, conditional on the sanity checks below coming back clean.

## Non-Blocking Limits

- Emission trigger generalizes to step boundaries rather than the exact
  `cot_4` to `cot_5` boundary. GSM8K exposes this through short uniform steps.
  This rules out a general "reliable exact-3 natural emission" claim, but does
  not block Stage3 separability.
- Safety deltas around zero are expected under KL transparency. This is not a
  negative result, but also not a safety-improvement claim.

## Minimal Sanity Checks Before Stage3

1. Per-class emission coverage on Stage3 sources. If natural pause
   emission/position differs between unsafe-side and safe-side completions,
   pause-position probing may have label-correlated selection bias. Report
   exact-3 and first-pause-index split by pair side.
2. GSM8K over-emission characterization: pause-position histogram, step-boundary
   clustering versus degenerate consecutive-pause loops, answer-truncation
   rate, and accuracy conditioned on exact-3 versus over-emission.
3. Tokenizer/position-offset audit: forced insertion must land at the same token
   position convention as natural emission.
4. Spot-check about 20 transcripts per mode for pause-token bleed into answer
   text.

## Stage3 Mode

Run both natural and forced, with pre-registered roles:

- Natural pause is primary.
- Forced pause is control/diagnostic.

Do not promote forced to primary post hoc if natural fails.

Pre-register:

- natural primary / forced control
- per-class coverage in reporting
- existing gate: beats base-hidden@matched-position and matched-horizon surface
- Delta AUROC >= 0.05
- CI excludes 0
- at least 3/4 sources pass

## Allowed Claims

- KL-transparent position-targeted pause SFT works on 8B.
- No capability or safety degradation within confidence intervals.
- Natural exact-3 compliance is domain-dependent: high on safety/MATH500 and
  fails on GSM8K through step-boundary over-emission.

## Not Allowed

- General natural-emission reliability.
- Any safety benefit from pauses.
- Any claim about pause-position separability before Stage3.
- Claiming `cot_4`/`cot_5` is where the signal was measured; it remains an
  engineering default aligned to the Stage1 evidence convention.
- Citing test-max numbers.

## Next Step

Run sanity checks 1-4, commit the Stage3 prereg ruling, then launch the Stage3
slice on this checkpoint. Queue a Stage2.1 stop-emission fix only if a later
paper claim needs GSM8K-domain natural emission; it is not on the critical path.

# Fable Review Request: SafeChain Stage3/Stage4, Focus on Stage4 Steering

Please perform a complete, objective review of the current SafeChain Stage3 and
Stage4 code, configs, result summaries, and method design.

Read `README.md` first, then inspect the packet files as needed. Do not edit
files.

The user wants a precise review and has enough Fable budget, so prioritize
correctness and depth over brevity.

## Project Goal

Stage4 should intervene on pause-token hidden states to reduce unsafe CoT while
not increasing over-refusal, damaging capability, or causing broken output.

The intended Stage4 logic is:

```text
Use Stage3 probe or unsafe direction to find unsafe manifold.
At generation time, steer only pause-token hidden states away from unsafe region.
Reduce unsafe CoT while preserving behavior/capability/refusal calibration.
```

## Why This Review Is Needed Now

Stage2 is being changed from full-response SFT to `kl_transparent_emit`
(pause-slot CE + pause-stripped continuation KL + pause suppression). Fable has
already reviewed Stage2 and marked the code packet GO after fixes. However, it
previously warned that Stage4 may need changes:

- KL transparency does not guarantee pause hidden states are live steering
  ports.
- We may need injection-gain curves, attention mass, or equivalent liveness
  checks.
- The current delta may learn a generic careful/refusal/quality direction, not
  a clean unsafe-removal direction.
- If Stage2 learns natural pause emission, forced-pause evaluation may no longer
  answer the right question by itself.

## What To Review

1. Stage3 flow:
   - data prep assumptions
   - hidden extraction positions
   - probe training / position scan
   - whether it can support Stage4 direction selection
   - whether it avoids prompt-classification-only artifacts under intra-CoT pause

2. Stage4 flow:
   - learned delta training in `run_intra_pause_learned_delta_pilot.py`
   - generation-time hook in `run_intra_pause_steered_generation.py`
   - full eval launcher and summarizer
   - judge setup and result summaries
   - pause-only scope enforcement

3. Methodological validity:
   - Is the current Stage4 objective aligned with “pull away from unsafe
     manifold”?
   - Does it need to be re-derived for the new `kl_transparent_emit` Stage2?
   - Are forced pauses, natural pauses, or both needed?
   - What evidence is required before claiming pause tokens are live steering
     ports?

4. Better Stage4 methods:
   - probe-gradient steering
   - classifier-guided activation editing at pause states
   - representation projection / rejection of unsafe component
   - safe-centroid pull
   - contrastive delta from paired safe/unsafe pause states
   - train-time auxiliary loss on pause states
   - online gated steering using Stage3 probe score
   - any alternative you think is more defensible

## Specific Questions

Please answer in detail:

1. Is Stage3 code logically compatible with the new Stage2 `kl_transparent_emit`
   checkpoint/tokenizer?
2. Does Stage3 currently measure the right separability signal for Stage4, or
   could it still be prompt/source/length classification?
3. Does the current Stage4 learned-delta training objective actually represent
   “away from unsafe manifold”?
4. Is generation-time steering implemented at the right token/timestep? Are the
   hook semantics correct under autoregressive generation/cache?
5. Is forced pause insertion still valid after Stage2 natural self-emission, or
   should Stage4 support natural-pause-only and hybrid evaluations?
6. What concrete liveness tests should be run before Stage4?
7. What concrete experiments should be run after the first new Stage2 checkpoint
   before choosing a Stage4 algorithm?
8. What is the best Stage4 algorithm you recommend, given the project goal?
9. What code changes are needed before implementing that algorithm?
10. What claims are allowed under different possible outcomes?

## Desired Output

Please produce:

- Executive verdict.
- Stage3 code/design review.
- Stage4 code/design review.
- Exact blockers or risks.
- Recommended Stage4 method, with rationale.
- Minimal experiment plan.
- Required liveness / injection-gain tests.
- Required eval metrics for unsafe reduction, over-refusal, capability, broken
  output, and transparency.
- Concrete code-change TODO list.
- Go/no-go table for:
  - Stage3 after new Stage2
  - current Stage4 as-is
  - modified Stage4 with your recommended method
  - running 1.5B Stage4 pilot
  - running 8B Stage4 pilot

Please be blunt and concrete.

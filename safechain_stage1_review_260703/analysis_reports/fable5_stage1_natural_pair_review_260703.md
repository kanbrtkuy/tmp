# Fable5 Review: Stage 1 Natural-Pair Results - 2026-07-03

This file records the Fable5 review response to an aggregate-only Stage 1
natural-pair result prompt. The prompt excluded raw prompts and raw trajectories.

## Fable5 Response

> Memory updated. Summary of the review's sharpest points: drop all test-max
> numbers (leakage; real range is 0.70–0.79), reframe the prompt/pre-CoT controls
> as by-construction rather than empirical, retract the cross-scale consistency
> claim until you can compare matched dense-grid val heatmaps, and note that LOSO
> is now blocked (ReasoningShield-only) so transfer evidence is currently zero.
> Highest-priority next step remains the token-matched truncation curves,
> followed by length-matched eval and Gates 3/4.

## Immediate Interpretation

- Do not present test-set maxima as headline evidence. Use validation-selected
  position/layer and report the lower resulting range.
- Prompt/pre-CoT controls are still useful, but they should be framed as an
  expected control under same-prompt pairing, not as proof of semantic monitoring.
- The current cross-scale observation is suggestive only. A stronger claim needs
  matched dense-grid validation heatmaps for the same generator/extractor
  comparison.
- LOSO is currently blocked because the usable source-family set collapsed to
  ReasoningShield-only after filtering. Therefore source-transfer evidence is not
  currently available.
- The next gating experiments should prioritize token-matched truncation curves
  and length-matched evaluation before moving toward stronger Stage 1 claims or
  Stage 2.

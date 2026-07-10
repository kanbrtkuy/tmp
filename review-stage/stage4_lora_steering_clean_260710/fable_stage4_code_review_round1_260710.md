# Fable Stage4 Code Review Round 1

Reviewer: `claude-fable-5`, high effort

## Verdict

The core prefill hook mechanism is sound: the explicit target mask fires on the full-prefix forward before later-layer KV cache and first sampled-token logits are fixed. Fable noted two writeup caveats: block `layer-1` K/V is pre-edit under output-hook steering, and cached one-token forwards are intentionally skipped.

## Blockers

1. Row-set asymmetry: alpha-zero arms generated through resolution failures, while nonzero arms aborted on failures. This breaks paired A3-A2 comparisons.
2. Probe gate was configured but not used. Either wire it or declare steering ungated.
3. Runner could not produce the A0-A5 battery and output paths did not include condition/direction, so random-direction reruns could overwrite main outputs.
4. `content_cot4_6` overlapped with `post_pause_1_3`, making the ordinary-token counterfactual non-distinct.
5. Matched-strength was logged but not enforced or checked.

## Non-Blockers

- Add fuller manifest metadata.
- Avoid batch-level hook stats being repeated without row attribution.
- Add resume support later.
- Guard PEFT adapter plus position-LoRA edge case later.
- Fail/flag extra pause tokens rather than labeling the fourth pause as content.
- Check norm-cap saturation before spending full compute.
- Remove duplicate dead pivot keys.
- Add tests for crop missing-target path and an integration generate hook.

## Patch Plan Taken

- Make unresolved rows sentinel-skipped consistently across alpha arms.
- Declare current steering-first pivot as ungated (`gate_mode: none`) instead of pretending probe-gated.
- Expand runner over `base`, `fsm`, `ppc`, `gprs` and `main/random` directions with non-colliding paths.
- Change content diagnostic to `cot_2,cot_3,cot_4`.
- Add row-level hook norm stats and a matched-strength checker script.

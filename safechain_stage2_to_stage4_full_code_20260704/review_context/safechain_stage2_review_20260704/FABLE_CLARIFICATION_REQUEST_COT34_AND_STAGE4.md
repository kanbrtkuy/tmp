# Fable Clarification Request: cot3/cot4 Pause Placement and Stage4 Algorithm

Date: 2026-07-04

Please answer two narrow clarification questions about your second review.

Relevant files:

- `CLAUDE_FABLE5_FOLLOWUP_SELF_EMITTED_PAUSE_REVIEW.md`
- `FABLE_FOLLOWUP_REQUEST_SELF_EMITTED_PAUSE.md`
- `CLAUDE_FABLE5_REVIEW.md`
- `cot-safety/res/deepseek-8b/stage2_format_only_sft_summary.md`
- `cot-safety/configs/experiment/stage2_intra_pause_format_only_8b_cot3_save50_max250_4xa100.yaml`
- `cot-safety/configs/experiment/stage2_intra_pause_format_only_8b_cot4_save50_max250_4xa100.yaml`
- `cot-safety/configs/experiment/stage4_pause_steering.yaml`
- `cot-safety/scripts/run_stage4_steering.py`
- `cot-safety/legacy/PauseProbe/scripts/steering/run_intra_pause_full_steering_eval.sh`

## Clarification 1: Does the pre-CoT critique apply to cot3/cot4?

In the second review, you said that if pause tokens are placed before any CoT
token, Stage3 pause hidden states are deterministic for each prompt and can only
support prompt classification, not trajectory classification.

But the existing Stage2 experiments include cot3/cot4 intra-think pause variants,
where pause tokens are inserted after a few CoT tokens, not before the CoT:

- `stage2_intra_pause_format_only_8b_cot3_*`
- `stage2_intra_pause_format_only_8b_cot4_*`
- earlier full-SFT cot3/cot4 variants

Please clarify:

1. Are cot3/cot4 variants examples of the *intra-CoT pause placement* you were
   recommending, rather than examples of the severe pre-CoT problem?
2. Under what exact condition would the severe "only prompt classification"
   critique apply?
3. If cot3/cot4 pauses are self-emitted or inserted after model-generated CoT
   tokens, can Stage3 legitimately test trajectory separability from pause
   hidden states?
4. Are there any remaining caveats for cot3/cot4, such as position-convention
   mismatch, forced insertion, or teacher-forced vs self-generated distributions?

Please give a direct answer, because we need to know whether our existing
cot3/cot4 experiments are conceptually aligned with your recommendation or are
affected by the pre-CoT critique.

## Clarification 2: Can we use the previous Stage4 algorithm on KL-transparent pauses?

You said KL transparency does not necessarily kill Stage4 steerability because
it constrains the unsteered distribution value, not the Jacobian. You also
recommended measuring an injection gain curve and attention mass.

The question is: if we train a pause model using the proposed objective

```text
pause-slot CE + KL-to-base continuation matching
```

can we still use the Stage4 algorithm we previously implemented, or do we need
to replace it?

Current Stage4 roughly:

- use pause hidden states
- train or derive an unsafe direction / delta
- add that intervention at pause hidden states during generation
- evaluate unsafe CoT, over-refusal, capability, and broken output

But your reviews criticized the current Stage4 delta as an NLL-disruption
objective with non-monotone alpha behavior and missing random/shuffled controls.

Please clarify:

1. On a KL-transparent self-emitting pause model, is the *general Stage4 idea*
   still valid: use pause hidden states as the intervention site?
2. Can the current implemented Stage4 algorithm be reused as-is?
3. If not as-is, what minimal changes are required?
4. Should the steering vector be probe-derived / difference-of-means rather
   than trained by the current NLL-disruption objective?
5. What exactly should the injection gain curve and attention-mass diagnostic
   decide before running full Stage4?
6. If the diagnostic says the pause port is weak, what is the next best fallback:
   lower KL weight, stop earlier on transparency-steerability frontier, or
   per-layer pause prefix?

Please answer in a compact but precise way, with a final "yes/no/conditional"
table for:

- cot3/cot4 affected by pre-CoT critique?
- Stage3 valid on cot3/cot4?
- previous Stage4 idea valid on KL-transparent pauses?
- previous Stage4 algorithm reusable as-is?
- minimum required changes before claiming Stage4 evidence?

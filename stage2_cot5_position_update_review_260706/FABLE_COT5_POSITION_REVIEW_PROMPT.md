# Fable Review Request: Stage2 Cot5 Pause Position Update

Please act as a senior ML systems/research code reviewer. Be strict and
actionable.

Repository packet:

- GitHub: https://github.com/kanbrtkuy/tmp/tree/main/stage2_cot5_position_update_review_260706
- Commit: `415967d`
- Local path available to you:
  `/private/tmp/safechain_tmp_review_cot5_20260706/stage2_cot5_position_update_review_260706`

Read the review brief first:

```text
/private/tmp/safechain_tmp_review_cot5_20260706/stage2_cot5_position_update_review_260706/REVIEW_BRIEF.md
```

Then inspect the actual code/config tree under:

```text
/private/tmp/safechain_tmp_review_cot5_20260706/stage2_cot5_position_update_review_260706/cot-safety
```

## Background

The intended Stage2 pause layout has just changed to:

```text
<think> t0 t1 t2 t3 t4 <|pause|><|pause|><|pause|> t5 ...
```

In this codebase, `cot_offset=k` means "insert pause tokens before CoT token
index k." Therefore the intended implementation is:

```text
cot_offset: 5
insert_pause_after_cot_tokens: 5
```

This follows your earlier Stage1-only position review: hidden@cot_4 is observed
after the model has consumed cot_4, so the most causally aligned exploratory
pause point is after cot_4 / before cot_5. We understand this is only an
engineering default, not validated/optimal Stage1 evidence.

## Please Review These Questions

1. Is `cot_offset: 5` the correct implementation of the intended
   after-cot4/before-cot5 layout under this codebase's tokenizer/formatter
   semantics?
2. Did the cot5 change propagate consistently through:
   - Stage2 data prep
   - Stage2 KL-transparent emit training configs
   - Stage2 model-comparison eval forced/natural conditions
   - Stage3 hidden extraction/probe configs
   - Stage3 matched controls
   - Stage4 liveness and GPRS configs
3. Are `control_cot_5` and `control_cot_6` the right matched no-pause controls
   for Stage3 after this insertion-point change?
4. Is there any remaining path that could silently train/evaluate cot3/cot4
   while output names or review docs imply cot5?
5. Are there any blockers before running the new Stage2 cot5 KL-transparent SFT?

## Files To Prioritize

- `cot-safety/configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_cot5_save25_max400_4xa6000.yaml`
- `cot-safety/configs/experiment/stage2_intra_pause_kl_transparent_emit_8b_cot5_save50_max400_4xa100.yaml`
- `cot-safety/configs/experiment/stage2_model_comparison_eval_1p5b_kl_transparent_emit_cot5_4xa6000.yaml`
- `cot-safety/configs/experiment/stage2_model_comparison_eval_8b_kl_transparent_emit_cot5_4xa100.yaml`
- `cot-safety/configs/experiment/stage3_intra_pause_probe_kl_transparent_1p5b_cot5.yaml`
- `cot-safety/configs/experiment/stage3_intra_pause_probe_kl_transparent_8b_cot5_4xa100.yaml`
- `cot-safety/configs/experiment/stage4_pause_gprs.yaml`
- `cot-safety/configs/experiment/stage4_pause_gprs_8b_4xa100.yaml`
- `cot-safety/legacy/PauseProbe/scripts/probe/extract_hidden_states.py`
- `cot-safety/legacy/PauseProbe/scripts/probe/run_intra_pause_probe_full.py`
- `cot-safety/scripts/run_stage3_intra_pause_probe.py`
- `cot-safety/tests/test_pause_insertion.py`

## Required Output Format

Please produce:

1. Verdict: `PASS_TO_RUN_STAGE2`, `PASS_WITH_REQUIRED_FIXES`, or `BLOCKED`.
2. Blocking issues, if any, with exact file/line references and why they matter.
3. Non-blocking risks or cleanup items.
4. Answers to the five review questions above.
5. Minimal patch recommendations if fixes are needed.

Please do not edit files. Review only.


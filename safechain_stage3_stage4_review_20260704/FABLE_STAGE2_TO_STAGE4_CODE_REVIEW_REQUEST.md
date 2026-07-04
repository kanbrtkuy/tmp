# SafeChain Stage2-to-Stage4 Code Review Request

Date: 2026-07-04.

Please perform a code-level review of the actual implementation commit:

- Repository: `/Users/baby/Documents/SafeChain/cot-safety`
- Commit under review: `88559e0 Add KL-transparent Stage2 to Stage4 framework`
- Base commit: `c372bab docs: include 8b shm archive`
- Diff to review: `git diff c372bab..88559e0`

Important: the local worktree currently contains unrelated uncommitted Stage1 /
data-expansion files. Ignore the dirty worktree. Review only the committed diff
`c372bab..88559e0` unless you explicitly need to verify compatibility against
unchanged files in `HEAD`.

Write the complete Markdown review to:

`/private/tmp/safechain_stage2_review_tmprepo_20260704/safechain_stage3_stage4_review_20260704/CLAUDE_FABLE5_STAGE2_TO_STAGE4_CODE_REVIEW.md`

Do not edit code or any other files.

## Background

The user wants a closed loop:

1. Implement the new Stage2 -> Stage3 -> Stage4 framework.
2. Ask Fable to review the actual code.
3. Apply Fable's review.
4. Verify and push.

This request is step 2: review the implementation that was just pushed.

## Prior Fable reviews to use as the spec

Please read these review artifacts in the tmp packet:

- `safechain_stage3_stage4_review_20260704/CLAUDE_FABLE5_STAGE2_TO_STAGE4_FLOW_REVIEW.md`
- `safechain_stage3_stage4_review_20260704/CLAUDE_FABLE5_STAGE3_STAGE4_FOLLOWUP_REVIEW.md`
- `safechain_stage3_stage4_review_20260704/CLAUDE_FABLE5_STAGE3_STAGE4_REVIEW.md`
- `safechain_stage3_stage4_review_20260704/stage2_context/CLAUDE_FABLE5_STAGE2_FULL_FLOW_REVIEW.md`
- `safechain_stage3_stage4_review_20260704/stage2_context/CLAUDE_FABLE5_STAGE2_ROUND3_REVIEW.md`

The implementation is meant to be an **initial framework**, not a complete GPU
experiment implementation. It should support the intended sequencing:

1. Run Stage2 `kl_transparent_emit` first.
2. Select checkpoint after Stage2 eval.
3. Run liveness battery.
4. Green: fixed Stage3 then GPRS Stage4.
5. Yellow: proceed only on live layers and queue Stage2.5-A for next run.
6. Red: Stage4 stops; Stage2.5-A/B branch.

Stage2.5 should not be merged into the default training path in this commit.

## Files changed in the implementation

The commit changed or added these files:

- `configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_cot3_save25_max400_4xa6000.yaml`
- `configs/experiment/stage2_intra_pause_kl_transparent_emit_8b_cot4_save50_max400_4xa100.yaml`
- `configs/experiment/stage2_model_comparison_eval_1p5b_kl_transparent_emit_cot3_4xa6000.yaml`
- `configs/experiment/stage2_model_comparison_eval_8b_kl_transparent_emit_cot4_4xa100.yaml`
- `configs/experiment/stage3_intra_pause_probe.yaml`
- `configs/experiment/stage3_intra_pause_probe_kl_transparent_1p5b_cot3.yaml`
- `configs/experiment/stage3_intra_pause_probe_kl_transparent_8b_cot4_4xa100.yaml`
- `configs/experiment/stage4_pause_gprs.yaml`
- `configs/experiment/stage4_pause_gprs_8b_4xa100.yaml`
- `legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh`
- `legacy/COTPauseToken/src/utils/pause_kl_trainer.py`
- `legacy/PauseProbe/scripts/eval/run_model_comparison_generation.py`
- `legacy/PauseProbe/scripts/eval/summarize_model_comparison_eval.py`
- `legacy/PauseProbe/scripts/probe/run_intra_pause_probe_full.py`
- `scripts/run_stage2_sft.py`
- `scripts/run_stage3_intra_pause_probe.py`
- `scripts/run_stage4_liveness.py`
- `scripts/run_stage4_steering.py`
- `src/cot_safety/cli.py`
- `src/cot_safety/pipeline.py`
- `src/cot_safety/steering/gprs.py`
- `src/cot_safety/steering/liveness.py`
- `src/cot_safety/steering/scope.py`
- `tests/test_stage2_pause_kl_trainer.py`
- `tests/test_stage4_gprs_liveness.py`
- `tests/test_steering_scope.py`

## What the executor claims was implemented

- Stage2 `kl_transparent_emit` 1.5B/8B configs and trainer support.
- 8B NEW-F1 fix: `load_best_model_at_end: false`,
  `early_stopping.enabled: false`.
- Stage3 prompt-baseline plumbing:
  `hidden.positions.prompt_baselines` -> `--prompt_positions` ->
  `extract_hidden_states.py`.
- KL Stage2 checkpoint-specific Stage3 configs for 1.5B cot3 and 8B cot4.
- Stage4 GPRS configs for 1.5B and 8B.
- `run_stage4_liveness.py` framework/stub that writes a liveness plan.
- GPRS helper with projection/rejection update and config validation.
- Liveness helper with config extraction and green/yellow/red decision helper.
- Pipeline planner now shows validate -> liveness -> GPRS artifacts -> eval.
- `target_specs` pause-only validation.
- Lightweight tests for scope, GPRS config, liveness decision.

Known incompleteness that may be acceptable for this framework commit:

- Real GPU liveness kernels are not implemented yet.
- GPRS is not wired into generation hooks yet.
- True pause-free content controls are not implemented yet.
- On-policy 10x/prompt CoT judging and within-prompt AUROC are not implemented
  yet.
- CoT-vs-answer judge split, truncation fix, random-direction full eval, and
  capability/over-refusal summaries are not implemented yet.

## Questions for Fable

Please review the actual code and answer:

1. Are there any correctness bugs in the implemented Stage2 `pause_kl` trainer,
   configs, env plumbing, or tests that would block Stage2 1.5B launch?
2. Did the implementation correctly apply NEW-F1 to 8B?
3. Is the Stage3 prompt-baseline plumbing actually correct end-to-end?
4. Are the new Stage3 KL configs safe to use after the Stage2 checkpoint exists?
5. Are the GPRS and liveness framework files honest scaffolding, or do they
   create misleading "looks implemented" behavior?
6. Are there bugs in `run_stage4_steering.py` phases, GPRS validation, or
   `target_specs` scope validation?
7. Which issues must be fixed before pushing/keeping this commit?
8. Which issues can wait until after the Stage2 checkpoint exists?
9. Give a verdict:
   - GO for Stage2 1.5B launch?
   - GO for keeping this framework commit on main?
   - NO-GO items before Stage3 run?
   - NO-GO items before Stage4 run?

Be code-review strict. Findings should include file/line references and exact
fix suggestions.

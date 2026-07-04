# Fable Review Request: Round 1 Nit Fixes

Please review only the changes after your `CLAUDE_FABLE5_ROUND1_FIX_REVIEW.md` verdict.

## Prior Verdict

You gave: **PASS with minor nits**.

The nits addressed here:

- R1-1: Stage2 pause-logit diagnostics computed every microstep and materialized full-vocab non-pause softmax.
- R1-2: Stage3 matched-control lookup silently fell back to prompt-key matching and did not label-check controls.
- R1-3: 8B liveness required both env path and status flip, but config lacked a comment.
- R1-5: learned-delta archival plan included `--allow_learned_delta` directly.

## Changes After That Review

- `legacy/COTPauseToken/src/utils/pause_kl_trainer.py`
  - Diagnostics are now computed only when `_should_log_loss_parts()` is true.
  - Diagnostics are wrapped in `torch.no_grad()`.
  - Pause probabilities use logsumexp, not full `softmax`.
  - Non-pause diagnostics now use flat indices + chunked `index_select`, matching the suppression-loss pattern; no whole non-pause full-vocab tensor is materialized before chunking.
- `legacy/PauseProbe/scripts/probe/extract_hidden_states.py`
  - Removed prompt-key fallback from matched-control lookup.
  - Duplicate matched-control IDs now raise.
  - Matched controls must have the same label as the pause row or the row is dropped with `matched_control_label_mismatch`.
- `legacy/PauseProbe/scripts/data/prepare_intra_pause_probe_data.py`
  - Adds duplicate/missing ID assertion for no-pause matched rows before writing split references.
  - Fallback stable IDs include source, prompt, reasoning, and final answer, so same-prompt multiple trajectories do not collide.
- `configs/experiment/stage4_pause_gprs_8b_4xa100.yaml`
  - Added a comment documenting that both `positive_control_model` and `positive_control_status` must be changed after verifying a genuine full-SFT 8B positive control.
- `src/cot_safety/pipeline.py`
  - Removed `--allow_learned_delta` from the archival learned-delta plan command; the runner now refuses instructively unless the user adds the acknowledgement.

## Local Validation

Passed:

- `python3 -m py_compile` over the changed Python files.
- Stage2 1.5B dry-run still prints `PAUSE_KL_INVARIANT_CHECK_INTERVAL_STEPS=50`.
- Stage3 legacy dry-run with `--extract_train_shards 2` emits `--matched_control_file .../nopause_shards/...` for train shards and `--matched_control_file .../nopause/...` for val/test/heldout.
- Stage4 8B liveness dry-run remains `blocked_missing_positive_control` both with and without `STAGE4_8B_FULL_SFT_CONTROL`, matching the new config comment.
- Learned-delta eval without acknowledgement still refuses.
- Learned-delta pipeline plan no longer includes `--allow_learned_delta`.

Please answer:

1. Are R1-1, R1-2, R1-3, and R1-5 closed?
2. Is the code now clean enough to land into the main `cot-safety` repo before Stage2 1.5B launch?
3. Are there any remaining blocker/high/medium issues introduced by these nit fixes?

Write the review to:

`safechain_stage2_to_stage4_full_code_20260704/CLAUDE_FABLE5_ROUND1_NITS_REVIEW.md`

Do not edit code.

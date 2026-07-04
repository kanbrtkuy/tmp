# Fable Review Request: Round 1 Fixes After Full Code-Tree Review

Please review the current working tree in this packet after the first remediation pass. This is not yet copied back to the main `cot-safety` repo; the user asked that fixes be staged in `tmp`, reviewed by Fable, iterated until acceptable, and only then landed in the main repo.

## Prior Review

Read:

- `CLAUDE_FABLE5_FULL_CODE_TREE_REVIEW.md`

## What Changed in Round 1

Main goals: make Stage2 launch safer, remove false Stage3 controls, and make Stage4 fail closed rather than producing official-looking results from deprecated paths.

### Stage2

- `legacy/COTPauseToken/src/utils/pause_kl_trainer.py`
  - Rows-only invariant now rechecks every configured interval and at train end, instead of only once after step 1.
  - Added pause emission diagnostics:
    - pause target probability mean,
    - pause target argmax rate,
    - non-pause pause probability mean,
    - non-pause pause argmax rate,
    - pause row norm and cosine-to-init,
    - pre/post KL pair counts.
- `scripts/run_stage2_sft.py`
  - Added explicit `<|pause|>` assertion for the legacy launcher, so `pause_kl.pause_token` cannot silently diverge from the hard-wired token.
  - Passes `PAUSE_KL_INVARIANT_CHECK_INTERVAL_STEPS`.
- `legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh`
  - Forwards `trainer.pause_kl.invariant_check_interval_steps`.
- Stage2 KL configs set `invariant_check_interval_steps: 50`.

### Stage3

- `legacy/PauseProbe/scripts/data/prepare_intra_pause_probe_data.py`
  - Data prep now writes matched no-pause rows under `nopause/{split}.json` alongside existing `cotpause/{split}.json`.
  - Intra-pause and no-pause rows share stable IDs, so matched controls can be found by ID.
- `legacy/PauseProbe/scripts/probe/extract_hidden_states.py`
  - Removed the invalid `control_cot_3/4 = post_pause_1/2` alias.
  - Added `--matched_control_file`; when provided, it performs a separate pause-free forward pass and fills `control_cot_3/4` from no-pause `cot_3/4`.
  - If matched control rows are missing or unparsable, rows are dropped instead of silently aliasing controls.
- `legacy/PauseProbe/scripts/probe/run_intra_pause_probe_full.py`
  - Hidden extraction now passes the matched `nopause/*.json` file.
  - Train sharding also writes matched `nopause_shards`, so sharded extraction keeps controls aligned.
- `src/cot_safety/formatting/position_locator.py`, `tests/test_position_locator.py`, `scripts/smoke_test.py`
  - No longer treat `control_cot_3/4` aliases as valid behavior.
- `scripts/run_stage3_intra_pause_probe.py`
  - Default config is now the KL-transparent 1.5B config, not the old full-SFT base config.
- Stage3 config wording now labels teacher-forced probing as a screen and marks `within_prompt_auroc` as not implemented rather than as the current primary endpoint.

### Stage4

- `scripts/run_stage4_steering.py`
  - Default config changed to `stage4_pause_gprs.yaml`; default phase is `validate`.
  - Deprecated `learned_delta` eval/generation/judge paths now require `--allow_learned_delta` or `steering.acknowledge_deprecated: true`.
  - GPRS eval/generation/judge/summary are fail-closed even under `--dry_run`; only `validate` and `liveness` are allowed until the GPRS generation hook exists.
  - `TARGET_SPECS` is now always taken from validated config instead of an external env override.
  - Malformed target spec dicts now fail with a clean message.
- `pipelines/run_4xa100_stage4_steering_eval.sh`
  - Defaults to GPRS validate; learned-delta requires `ALLOW_LEARNED_DELTA=true`.
- `configs/experiment/stage4_pause_steering.yaml`
  - Marks learned-delta as deprecated and not acknowledged.
- `configs/experiment/stage4_pause_gprs_8b_4xa100.yaml`
  - Removed the invalid 8B format-only positive control.
  - Requires `STAGE4_8B_FULL_SFT_CONTROL` and otherwise marks liveness as `blocked_missing_positive_control`.
- `src/cot_safety/steering/liveness.py`
  - Added positive-control status and liveness plan status helper.
- `tests/test_stage4_gprs_liveness.py`
  - Added liveness blocked-status test and GPRS projection/rejection numeric tests.
- Pipeline and launch scripts now point at KL-transparent / GPRS configs instead of format-only / learned-delta defaults.

## Local Validation

Passed:

- `python3 -m py_compile` over all changed Python files.
- Config load via `PYTHONPATH=src python3 -m cot_safety.cli config show` for:
  - Stage2 KL 1.5B
  - Stage2 KL 8B
  - Stage3 KL 1.5B
  - Stage4 GPRS 8B
- Stage2 1.5B dry-run prints `PAUSE_KL_INVARIANT_CHECK_INTERVAL_STEPS=50`.
- Stage3 legacy dry-run extraction commands include `--matched_control_file .../nopause/{split}.json`.
- Stage4 8B liveness dry-run writes `status: blocked_missing_positive_control`.
- Learned-delta eval without acknowledgement exits with refusal.
- Learned-delta eval with `--allow_learned_delta` expands only as an explicit archival dry-run.
- GPRS `--phase eval --dry_run` exits fail-closed.
- GPRS `--phase validate --dry_run` succeeds.

Not run locally:

- `pytest`, because this local Python lacks `pytest`.
- `smoke_test.py`, because this local Python lacks `numpy`.
- Any GPU model load/training/extraction.

## Review Questions

Please review the full code tree plus this diff and answer:

1. Is Stage2 1.5B still GO after these changes?
2. Did the Stage2 diagnostic/invariant changes introduce any training-objective or DDP risk?
3. Are Stage3 `control_cot_3/4` now true no-pause matched controls rather than aliases?
4. Is the remaining teacher-forced Stage3 screen honestly labeled, with confirmatory `within_prompt_auroc` clearly not claimed as implemented?
5. Is Stage4 now sufficiently fail-closed so deprecated learned-delta and unimplemented GPRS cannot accidentally produce official-looking results?
6. Are there any blocker/high issues that must be fixed before landing these changes into the main `cot-safety` repo?

Please write your review to:

`safechain_stage2_to_stage4_full_code_20260704/CLAUDE_FABLE5_ROUND1_FIX_REVIEW.md`

Do not edit code.

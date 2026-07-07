# Auto Review Loop: Stage2.1 Natural Pause Emission

Started: 2026-07-07

Goal: fix low natural exact-3 pause insertion success in Stage2 while keeping
the paper method unconstrained and preserving the Stage3 signal test.

## Round 1 (2026-07-07)

### Assessment

External reviewer: Claude Fable-5, extra-high effort.

Verdict: current 8B Stage2 failure is not primarily undertraining. It is a
teacher-forced objective / natural-generation metric mismatch, plus a fragile
single repeated pause-token count-to-3 problem.

### Reviewer Raw Response

Full response:
`review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_xhigh_design_review_260707.md`

### Actions Taken

- Added Stage2.1 distinct pause chain support:
  `<|pause_1|><|pause_2|><|pause_3|>`.
- Extended Stage2 builder/validator to accept `--pause_tokens`.
- Extended `PauseKLSFTTrainer` with multi-pause IDs, emit margin,
  stop-after-chain loss, margin suppression, continuation KL masking over all
  pause IDs, and optional pause-head fallback behind a disabled-by-default flag.
- Added natural pause emission metrics and checkpoint diagnostic script.
- Added DAgger-style on-policy relabel script and static/on-policy mix builder.
- Added Stage2.1 1.5B pilot, 8B short400, and 8B full configs.
- Added focused tests for insertion, natural metrics, on-policy mining, and
  trainer masks.

### Verification

- `python3 -m py_compile ...` passed for modified/new Python scripts.
- `bash -n legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh`
  passed.
- `PYTHONPATH=src .venv-stage1-test/bin/pytest tests/test_pause_insertion.py tests/test_stage2_pause_kl_trainer.py tests/test_stage2_natural_pause_metrics.py tests/test_stage2_onpolicy_mining.py -q`
  passed for non-torch tests; trainer test module skipped locally because torch
  is not installed on this machine.
- `scripts/run_stage2_sft.py --config configs/experiment/stage21_pause_chain_dagger_8b_short400.yaml --dry_run --skip_data_prep`
  shows distinct pause tokens, margin suppression, stop-after-chain loss, and
  rows-only config in the generated training env.

### Status

Continuing to Round 2: push implementation to `tmp`, then ask Fable-5
extra-high to review the full code and identify any blockers before GPU rerun.

## Round 2 (2026-07-07)

### Assessment

External reviewer: Claude Fable-5, extra-high effort.

- Verdict: almost
- Score: 7/10
- Raw response:
  `review-stage/stage2_1_clean_pause_design_260707/fable_stage2_1_code_review_260707.md`

### Key Criticisms

- Location metric was not aligned with the formatter because it did not skip
  leading whitespace tokens and diagnostics lacked tokenizer-aware counting.
- DAgger mining dropped conditioned prompt suffixes and mixer output was not
  directly wired to `run_stage2_sft.py`'s prepared-root layout.
- Emit margin compared pause targets only against non-pause vocab, so rival
  pause tokens were not margin-separated.
- `pause_head` fallback was unsafe because generation did not apply it.

### Actions Taken

- Made natural pause location metrics tokenizer-aware and leading-whitespace
  aligned with the formatter; no-tokenizer location is now `None`, so gates
  fail rather than silently word-counting.
- Added tokenizer/config support to `diag_stage2_checkpoint.py`, including
  `min_location_match` and default recomputation of metrics.
- Added explicit `generation_mode` and independent `expected_cot_offset` to the
  eval harness and runner; natural mode no longer depends on forced insertion
  args.
- Preserved `conditioned_prompt` in eval outputs and mining rows.
- Made DAgger mixer emit `prepared_root/intra_dir_name` layout plus root
  manifest; added iter1 configs.
- Stabilized mixed row schema and stringified ids/metadata.
- Changed emit margin to compete against all tokens except the target token,
  including rival pause tokens.
- Hard-failed `pause_head.enabled=true` until generation-time application exists.
- Added regression tests for leading-newline location, DAgger mix layout,
  conditioned prompt preservation, distinct KL alignment, and rival-pause emit
  margin.

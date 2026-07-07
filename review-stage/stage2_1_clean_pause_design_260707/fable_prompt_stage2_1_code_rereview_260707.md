# Fable-5 XHigh Re-review Request: Stage2.1 Fixes After Code Review

Please use extra-high effort, but focus on whether the blockers from your prior
review are fixed. Be critical and do not approve if any GPU-blocking issue
remains.

Repo/branch:

`https://github.com/kanbrtkuy/tmp/tree/stage21-pause-chain`

New commit:

`34ebfb5 Address Stage2.1 Fable implementation review`

Prior reviewed commit:

`6a8dae3 Add Stage2.1 pause chain training framework`

Your prior verdict was **almost, 7/10**. Main blockers were:

1. `location_match` metric wrong because it did not skip leading whitespace
   tokens and diagnostics lacked tokenizer-aware counting.
2. DAgger seams: mining dropped conditioned prompt suffixes; mix output layout
   did not match `run_stage2_sft.py`; no iter1 config.
3. Mixed dataset schema not load-tested / schema unstable.
4. Emit margin did not compare target pause against rival pause tokens.
5. `pause_head` fallback unsafe.
6. Eval harness natural mode was a trap; expected offset was coupled to forced
   insertion.
7. `stage21_selection` was decorative.

Implemented fixes:

- `natural_pause_metrics.py`
  - tokenizes prefix inside `<think>` and skips leading whitespace tokens;
  - without tokenizer, location index and location_match are `None`, causing
    diag gate to fail rather than silently word-counting;
  - added regression test with tokenizer that emits leading `\n` as its own
    token.
- `diag_stage2_checkpoint.py`
  - added `--tokenizer_path`, `--config`, `--min_location_match`;
  - reads pause tokens/cot offset/thresholds from Stage2.1 config;
  - recomputes metrics by default instead of trusting old JSONL metrics.
- `run_model_comparison_generation.py` and `scripts/run_model_comparison_eval.py`
  - added explicit `--generation_mode auto|natural|forced_pause`;
  - added independent `--expected_cot_offset`;
  - natural mode no longer uses forced insertion arg to decide location target;
  - output rows now include `conditioned_prompt`.
- `mine_onpolicy_pause_negatives.py`
  - prefers `conditioned_prompt` over raw prompt;
  - converts ids to strings;
  - rejects non-train split when split is present unless explicitly overridden.
- `build_stage21_dagger_mix.py`
  - added `--intra_dir_name`; writes `prepared_root/intra_dir_name/train.json`
    plus root `manifest.json`, matching `run_stage2_sft.py`;
  - normalizes rows to stable top-level schema and JSON-stringifies complex
    metadata/pause token lists;
  - supports `--no-copy_val_test`.
- Added iter1 configs:
  - `configs/experiment/stage21_pause_chain_dagger_1p5b_iter1.yaml`
  - `configs/experiment/stage21_pause_chain_dagger_8b_short400_iter1.yaml`
- `PauseKLSFTTrainer`
  - emit margin now competes against all tokens except the target, including
    rival pause tokens;
  - `pause_head.enabled=true` now hard-fails with NotImplementedError.
- `run_stage2_sft.py`
  - compact JSON list env/CLI args;
  - dry-run prints `FORMAT_ONLY_TRAINABLE_TOKENS`.

Verification after fixes:

- `python3 -m py_compile` passed for modified/new scripts.
- `PYTHONPATH=src .venv-stage1-test/bin/pytest tests/test_pause_insertion.py tests/test_stage2_natural_pause_metrics.py tests/test_stage2_onpolicy_mining.py tests/test_stage2_dagger_mix.py tests/test_stage2_pause_kl_trainer.py -q`
  -> `13 passed, 1 skipped`.
  The skipped module is torch-dependent trainer runtime tests because local
  machine has no torch; py_compile covered trainer syntax.
- `scripts/run_stage2_sft.py --config configs/experiment/stage21_pause_chain_dagger_8b_short400.yaml --dry_run --skip_data_prep`
  prints compact distinct pause tokens and rows-only trainable tokens.
- `build_stage21_dagger_mix.py --intra_dir_name ...` produced the prepared-root
  layout expected by iter1 configs.
- `diag_stage2_checkpoint.py` without tokenizer now fails location gate rather
  than pretending location is known.

Questions:

1. Are all GPU-blocking issues from your prior review fixed?
2. Is the Stage2.1 code now ready for the first no-GPU data-prep validation and
   then a 25-step 1.5B smoke run?
3. If not ready, list exact remaining blockers and minimum code/test fixes.
4. If ready/almost, give the recommended first command sequence.

Output:

- verdict: ready / almost / not ready;
- score 1-10;
- remaining blockers, if any;
- minimum fixes before GPU, if any;
- first run sequence;
- claims still disallowed before results.

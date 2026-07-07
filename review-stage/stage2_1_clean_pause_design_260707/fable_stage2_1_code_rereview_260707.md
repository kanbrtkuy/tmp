# Fable Re-review: Stage2.1 Pause Chain Fixes

Date: 2026-07-07

Prompt:
`review-stage/stage2_1_clean_pause_design_260707/fable_prompt_stage2_1_code_rereview_260707.md`

Branch reviewed:
`https://github.com/kanbrtkuy/tmp/tree/stage21-pause-chain`

Commit reviewed:
`34ebfb5 Address Stage2.1 Fable implementation review`
(diffed against prior reviewed commit `6a8dae3`)

## Raw Review

Reviewed on the local working tree (HEAD == `34ebfb5`). I read the full diff,
traced every claimed fix to its call sites, cross-checked insertion vs. metric
offset conventions end to end, and independently re-ran the verification: all
13 non-torch tests pass (executed via a stand-in runner because venv pytest
execution was not approved in this session — same test functions, same
asserts), dry-runs of the base 8B config and both new iter1 configs resolve
correctly, and a live diag run without a tokenizer fails the location gate as
claimed.

## Verdict: **ready** — Score: **9/10**

All seven blockers from the prior review are genuinely fixed, not
paper-fixed. The code is ready for the no-GPU data-prep validation, and ready
for the 25-step 1.5B smoke run **conditioned on one sequencing requirement**:
the torch-gated trainer tests have never executed anywhere (the "1 skipped"),
so they must pass on the GPU box before the smoke run starts.

## Blocker-by-blocker verification

1. **location_match / leading whitespace — FIXED.**
   `token_index_after_leading_space_skip` (natural_pause_metrics.py:122)
   returns `len(token_ids) - first_nonspace_idx`, which exactly mirrors the
   insertion convention `target_idx = first_idx + cot_offset` in
   `pause_insertion.insert_pause_before_cot_offset` and the legacy
   `build_intra_think_pause_sft_splits.py`. Without a tokenizer the index and
   `location_match` are `None`; `summarize` excludes `None` rows;
   `diag_stage2_checkpoint` gates `min_location is not None` — verified live:
   gate = fail with `min_location_match: null`. The leading-`\n` regression
   test passes.

2. **DAgger seams — FIXED.**
   - Mining prefers `conditioned_prompt` (prompt + task suffix), which is what
     the generation script now records (`clean_text(prompt) + task_suffix`),
     and which is the correct SFT `input` since the trainer's formatting
     function re-adds BOS/user/assistant templates.
   - Mix layout: `--intra_dir_name` writes
     `prepared_root/<intra_dir_name>/train.json` + root `manifest.json`,
     matching the `prep_done` check at run_stage2_sft.py:604 and
     `load_pause_sft_dataset`. Verified by running the layout test.
   - Both iter1 configs exist, inherit correctly through the recursive
     `defaults:` chain, and dry-run to the right `DATA_DIR`
     (`.../stage21_dagger_iter1_pause_chain/intra_pause_chain_cot5_dagger_iter1`).
   - Split guard rejects non-train rows when `split` is present;
     `--allow_non_train_split` is the explicit override.

3. **Mixed dataset schema — FIXED (with one residual note).**
   `normalize_sft_row` runs *after* `materialize_weights` (so duplicate-index
   metadata is captured), forces `id` to str, JSON-stringifies
   `pause_tokens`/`metadata`, and keeps only scalar optional fields. I audited
   field types against the static intra rows: no pyarrow type conflicts
   remain; missing keys become nulls, which `Dataset.from_list` tolerates.
   Residual: there is still no test that loads the mixed train.json through
   `datasets.Dataset.from_list` (lib absent locally) — cover this with a
   one-liner during the no-GPU validation on the GPU box. Note also that
   unlisted fields (`pause_char_offset_in_reasoning`,
   `reasoning_token_count_after_leading_space_skip`) are silently dropped from
   normalized rows; harmless for training (formatter uses input/output only)
   but be aware they are not in the mixed dataset.

4. **Emit margin vs rival pauses — FIXED.**
   `_competitor_max_except_target` masks only the target id, so rival pause
   tokens compete in the margin. `_non_pause_max` is retained only for
   `_pause_margin_suppression`, where excluding all pause ids is correct.
   Caveat: the test for this (`test_emit_margin_competes_against_rival_pause_tokens`)
   is torch-gated and has NOT executed yet.

5. **pause_head — FIXED.** `enabled=true` raises `NotImplementedError` at
   trainer init; all Stage2.1 configs set `enabled: false`.

6. **Eval natural-mode trap — FIXED.**
   Explicit `--generation_mode auto|natural|forced_pause`; forced insertion
   only when `gen_mode == "forced_pause"` (guarded to require
   `insert_pause_after_cot_tokens >= 0`); `expected_cot_offset` is an
   independent arg and only falls back to the insertion arg in forced mode;
   `conditioned_prompt` recorded per row; the eval driver plumbs
   `generation_mode`, `expected_cot_offset`, `pause_tokens`,
   `pause_separator` per condition. The script imports the fixed canonical
   `cot_safety.eval.natural_pause_metrics` (not a stale local copy).

7. **stage21_selection — FIXED (now functional).**
   `diag_stage2_checkpoint --config` reads pause tokens, cot offset, tokenizer
   path, and all four thresholds (incl. `min_location_match`) from the
   top-level `stage21_selection` block; recomputes metrics by default and only
   trusts existing JSONL metrics under `--use_existing_metrics` when they
   carry a non-None `location_match` and matching pause tokens.

## Independent verification performed

- 13/13 non-torch tests pass (insertion 4, metrics 5, mining 2, mix 2).
- `run_stage2_sft.py --dry_run --skip_data_prep` on
  `stage21_pause_chain_dagger_8b_short400.yaml`,
  `stage21_pause_chain_dagger_1p5b_iter1.yaml`, and
  `stage21_pause_chain_dagger_8b_short400_iter1.yaml`: compact JSON pause
  token lists, `FORMAT_ONLY_TRAINABLE_TOKENS=["<|pause_1|>","<|pause_2|>","<|pause_3|>"]`
  (required by the Hydra override grammar and the `special_tokens_to_add`
  branch in `run_4gpu_intra_pause_sft.sh:79`), correct iter1 data dirs.
- Live diag run without tokenizer on a synthetic exact-chain row:
  `status: fail`, `min_location_match: null` — the trap is closed.

## Remaining blockers

None that block the no-GPU data-prep validation.

## Minimum requirements before the 25-step GPU smoke

1. **Run the torch trainer tests on the GPU box first**:
   `pytest tests/test_stage2_pause_kl_trainer.py -q` must pass (11 tests).
   These cover the emit-margin rival fix, rows-only invariants, KL pair
   alignment, and compute_loss finiteness — none have executed anywhere yet.
2. **One-line real load test of the prepared dataset** (from
   `legacy/COTPauseToken`):
   `PAUSE_SFT_DATA_DIR=<prepared_root>/<intra_dir_name> python -c "from src.utils.trainer_utils import load_pause_sft_dataset; ds = load_pause_sft_dataset(); print(ds)"`
3. **Real-tokenizer location sanity** (recommended, ~50 rows): run diag with
   `--config` + real tokenizer on forced-insertion training rows and confirm
   `location_match_rate ≈ 1.0`. Substring re-tokenization can in principle
   drift from in-context tokenization at BPE boundaries; the 0.90 gate
   tolerates rare drift, but confirm it is rare before trusting the gate.

## First run sequence

Phase A — no-GPU data-prep validation (GPU box, CPU only):

```bash
# 1. full test suite including torch trainer tests
python -m pytest tests/ -q

# 2. build the iter0 dataset + format validation (no training)
python scripts/run_stage2_sft.py \
  --config configs/experiment/stage21_pause_chain_dagger_1p5b.yaml \
  --skip_train

# 3. real datasets load test
cd legacy/COTPauseToken && \
PAUSE_SFT_DATA_DIR=$COT_SAFETY_DATA_ROOT/pause_sft/stage21_trusted_cot_18k_pause_chain/intra_pause_chain_cot5 \
python -c "from src.utils.trainer_utils import load_pause_sft_dataset; print(load_pause_sft_dataset())" && cd -

# 4. diag on a sample of training rows with real tokenizer (location sanity)
python scripts/diag_stage2_checkpoint.py \
  --config configs/experiment/stage21_pause_chain_dagger_1p5b.yaml \
  --input_jsonl <sample_of_training_rows_as_generated_field>.jsonl \
  --output_json runs/diag_prep_sanity.json
```

Phase B — 25-step 1.5B smoke:

```bash
python scripts/run_stage2_sft.py \
  --config configs/experiment/stage21_pause_chain_dagger_1p5b.yaml \
  --skip_data_prep --max_steps 25
```

Then: rows-only invariant must not fire; loss finite; save-at-25 checkpoint
exists; run diag (natural mode generation on ~100 dev prompts via
`--generation_mode natural` + `--expected_cot_offset 5`) and inspect the gate
JSON before scheduling the 400-step run.

Phase C — only after iter0 gate readout: mine → mix (`--intra_dir_name
intra_pause_chain_cot5_dagger_iter1`) → iter1 config.

## Claims still disallowed before results

- Any natural pause emission rate, exact-chain rate, or location-match rate
  for trained checkpoints (no checkpoint has produced a passing gate).
- Any claim that DAgger improves emission/location (iter1 has not run).
- Any claim that the emit-margin rival fix changes training behavior
  (torch tests unexecuted; no A/B).
- Any safety/capability preservation claim for Stage2.1 checkpoints.
- Any 1.5B→8B transfer claim; pause port cot_4/cot_5 remains an exploratory
  default with 8B-only evidence.
- "Selection gate validated" — the gate logic is implemented and unit-checked,
  but has not yet been exercised on real checkpoint output distributions.

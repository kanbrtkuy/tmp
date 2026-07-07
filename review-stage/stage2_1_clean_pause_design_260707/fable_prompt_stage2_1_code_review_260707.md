# Fable-5 XHigh Code Review Request: Stage2.1 Pause Chain Implementation

Please use extra-high effort and act as a critical senior ML systems reviewer.

We are trying to fix SafeChain Stage2 natural pause-token insertion. The prior
8B KL-transparent SFT failed the strict paper target: natural generation should
emit exactly three pauses at the intended cot_4/cot_5 boundary, but GSM8K
over-emitted pauses badly. You previously diagnosed this as exposure bias plus
the wrong objective/selection metric, not simply undertraining.

## Code To Review

Repo for review:

`https://github.com/kanbrtkuy/tmp/tree/stage21-pause-chain`

Commit:

`6a8dae3 Add Stage2.1 pause chain training framework`

Please review the complete code on that branch, not only the diff. If GitHub is
not accessible, use the implementation summary below and still give blockers.

## Intended Stage2.1 Method

Main method should remain unconstrained natural generation, not decoding
constraints.

Implementation goal:

1. Replace repeated `<|pause|><|pause|><|pause|>` with distinct chain
   `<|pause_1|><|pause_2|><|pause_3|>`.
2. Keep rows-only pause-token embedding/unembedding training as default.
3. Add margin emit, margin suppression, explicit stop-after-chain loss, and
   continuation KL to pause-stripped teacher.
4. Add DAgger-style on-policy mining:
   sample current checkpoint, strip observed pauses, expert relabel with the
   deterministic formatter, upweight violations, and mix static/on-policy rows.
5. Select checkpoints by natural-generation exact-chain metrics, not eval loss.
6. Keep safety labels out of Stage2.
7. Preserve downstream Stage3: Stage3 will later test whether pause hidden
   states contain trajectory signal; Stage2 should not directly train on safety.

## Main Files Added/Changed

- `src/cot_safety/formatting/pause_insertion.py`
  - supports configured distinct pause tokens;
  - exposes strip/expert relabel helpers.
- `legacy/COTPauseToken/scripts/data_generation/pause_sft/build_intra_think_pause_sft_splits.py`
  - supports `--pause_tokens`.
- `legacy/COTPauseToken/scripts/data_generation/pause_sft/validate_intra_think_pause_sft_format.py`
  - validates distinct chain and cot offset.
- `legacy/COTPauseToken/src/utils/pause_kl_trainer.py`
  - supports multiple pause token IDs;
  - adds emit-margin, stop-after-chain, margin suppression;
  - masks all pause IDs in continuation KL;
  - optional pause-head fallback is behind `pause_head.enabled=false`.
- `scripts/mine_onpolicy_pause_negatives.py`
  - strips model-emitted pauses and expert-relabels on-policy generations.
- `scripts/build_stage21_dagger_mix.py`
  - materializes sample weights and mixes static/on-policy rows.
- `src/cot_safety/eval/natural_pause_metrics.py`
  - computes exact-chain, exact3, location, malformed, off-target metrics.
- `scripts/diag_stage2_checkpoint.py`
  - summarizes natural pause metrics and gate status.
- `legacy/PauseProbe/scripts/eval/run_model_comparison_generation.py`
  - uses the new natural metrics and can evaluate distinct chains.
- Stage2.1 configs:
  - `configs/experiment/stage21_pause_chain_dagger_1p5b.yaml`
  - `configs/experiment/stage21_pause_chain_dagger_8b_short400.yaml`
  - `configs/experiment/stage21_pause_chain_dagger_8b_full_2xa100.yaml`
  - `configs/data/stage21_dagger_pool.yaml`

## Local Verification

- `python3 -m py_compile` passed for all modified/new Python scripts.
- `bash -n legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh`
  passed.
- `PYTHONPATH=src .venv-stage1-test/bin/pytest tests/test_pause_insertion.py tests/test_stage2_pause_kl_trainer.py tests/test_stage2_natural_pause_metrics.py tests/test_stage2_onpolicy_mining.py -q`
  result: 8 passed, 1 skipped. The skipped module is the trainer unit tests
  because local machine lacks torch; the trainer file itself passed py_compile.
- `scripts/run_stage2_sft.py --config configs/experiment/stage21_pause_chain_dagger_8b_short400.yaml --dry_run --skip_data_prep`
  shows:
  - `PAUSE_KL_PAUSE_TOKENS=["<|pause_1|>", "<|pause_2|>", "<|pause_3|>"]`
  - `PAUSE_KL_SUPPRESSION_LOSS_TYPE=margin`
  - `PAUSE_KL_EMIT_MARGIN_WEIGHT=0.3`
  - `PAUSE_KL_STOP_WEIGHT=2.0`
  - `PAUSE_KL_ASSERT_ROWS_ONLY=true`
  - `MAX_STEPS=400`

## Review Questions

Please be concrete and critical:

1. Does this implementation match your Stage2.1 algorithm well enough to run a
   1.5B pilot or short 8B experiment?
2. Are there code-level blockers that would make the training objective wrong?
   Look especially at shift alignment, stop-after-chain mask, continuation KL
   pair alignment, suppression mask, multi-pause token masking, data builder
   cot_offset handling, and natural metrics.
3. Is the DAgger pipeline actually usable, or is there still a missing
   integration piece before GPU training?
4. Is optional pause-head implemented safely enough as a disabled fallback, or
   should it be removed/deferred?
5. Are the Stage2.1 config defaults reasonable for a cheap 1.5B pilot and
   short 8B run?
6. Are there tests missing that must be added before spending GPU?
7. Does anything accidentally reintroduce decoding constraints, safety-label
   leakage, eval/test leakage, or full-model drift?

Please output:

- verdict: ready / almost / not ready;
- score 1-10 for implementation readiness;
- ranked blockers;
- minimum code fixes before GPU;
- minimum tests before GPU;
- recommended first run command/sequence;
- any allowed/disallowed claims after this code, before results.

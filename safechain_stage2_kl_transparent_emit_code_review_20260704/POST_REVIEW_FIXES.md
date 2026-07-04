# Post-Review Fixes

After `CLAUDE_FABLE5_CODE_REVIEW.md`, the implementation packet was updated with
the low-risk fixes Fable requested before GPU training:

- Added component loss logging from `PauseKLSFTTrainer`:
  - `pause_kl/train/emit`
  - `pause_kl/train/continuation`
  - `pause_kl/train/pre`
  - `pause_kl/train/suppression`
- Added a rows-only assertion so `PAUSE_KL_ENABLED=true` cannot silently run with
  a trainable model body.
- Added `teacher_eval_mode` for the no-grad teacher forward.
- Applied the KL token cap to both post-pause and pre-pause KL pair lists.
- Made pause-token-id validation fail with a clear `ValueError`.
- Made the disabled custom-loss fallback compatible with older
  `compute_loss(...)` signatures.
- Set the default configs' `pause_kl.pre_weight` to `0.0` because pre-pause KL is
  expected to be identically zero under rows-only training.
- Added explicit config/env plumbing for:
  - `pause_kl.assert_rows_only`
  - `pause_kl.teacher_eval_mode`

Validation after these fixes:

```bash
python3 -m py_compile \
  cot-safety/legacy/COTPauseToken/src/utils/pause_kl_trainer.py \
  cot-safety/legacy/COTPauseToken/src/trl_train.py \
  cot-safety/scripts/run_stage2_sft.py

bash -n cot-safety/legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh

env PYTHONPATH=src python3 scripts/run_stage2_sft.py \
  --config configs/experiment/stage2_intra_pause_kl_transparent_emit_8b_cot4_save50_max400_4xa100.yaml \
  --dry_run --skip_data_prep

env PYTHONPATH=src python3 scripts/run_stage2_sft.py \
  --config configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_cot3_save25_max400_4xa6000.yaml \
  --dry_run --skip_data_prep
```

Still not done in this packet:

- Tiny-model KL/indexing unit tests.
- Single-GPU `--max_steps 2` smoke test on real prepared data.
- 4-GPU two-step smoke test.
- Optional memory chunking for the suppression term if the first smoke run OOMs.

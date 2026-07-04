# SafeChain Stage2 KL-Transparent Emit Code Review Packet

Date: 2026-07-04

This packet contains a proposed Stage2 implementation for a new training method:

```text
kl_transparent_emit = pause-slot CE + KL-to-base continuation matching
```

The goal is to train a model to emit `<|pause|>` at existing intra-CoT cot3/cot4
pause positions while minimizing behavior/capability drift.

## Files Included

- `cot-safety/legacy/COTPauseToken/src/utils/pause_kl_trainer.py`
  - New `PauseKLSFTTrainer`.
  - Consumes the same formatted dataset as current Stage2.
  - Computes pause-only CE, continuation KL on pause-stripped teacher inputs,
    and non-target pause suppression.

- `cot-safety/scripts/run_stage2_sft.py`
  - Adds env plumbing for `sft.method: kl_transparent_emit`.
  - Keeps old full-SFT and format-only paths intact.

- `cot-safety/legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh`
  - Adds Hydra overrides for the custom trainer when `PAUSE_KL_ENABLED=true`.
  - Keeps the existing shell entrypoint and arguments.

- `cot-safety/configs/experiment/stage2_intra_pause_kl_transparent_emit_8b_cot4_save50_max400_4xa100.yaml`
  - 8B cot4 proposed config.

- `cot-safety/configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_cot3_save25_max400_4xa6000.yaml`
  - 1.5B cot3 proposed config for cheaper initial validation.

## What Is Intentionally Unchanged

- Stage2 data JSON format is unchanged.
- Existing `intra_pause_cot3`, `intra_pause_cot4`, `no_pause_matched`, and
  `pre_think_pause3_matched` dataset variants remain unchanged.
- Existing Stage3 interface remains unchanged: it still consumes a normal HF
  checkpoint/tokenizer plus `pause.cot_offset` and hidden positions.
- Stage4 is not updated in this code packet.

## Verification Already Run

From `/Users/baby/Documents/SafeChain`:

```bash
python3 -m py_compile \
  cot-safety/legacy/COTPauseToken/src/utils/pause_kl_trainer.py \
  cot-safety/legacy/COTPauseToken/src/trl_train.py \
  cot-safety/scripts/run_stage2_sft.py

bash -n cot-safety/legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh
```

Dry-runs from `/Users/baby/Documents/SafeChain/cot-safety`:

```bash
env PYTHONPATH=src python3 scripts/run_stage2_sft.py \
  --config configs/experiment/stage2_intra_pause_kl_transparent_emit_8b_cot4_save50_max400_4xa100.yaml \
  --dry_run --skip_data_prep

env PYTHONPATH=src python3 scripts/run_stage2_sft.py \
  --config configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_cot3_save25_max400_4xa6000.yaml \
  --dry_run --skip_data_prep
```

The dry-runs confirm:

- `FORMAT_ONLY=true`
- `PAUSE_KL_ENABLED=true`
- existing Stage2 train shell is still used
- data prep can be skipped and existing prepared data paths are preserved

Limit: local import/runtime instantiation was not tested because the local
environment lacks COTPauseToken training dependencies such as `omegaconf`.

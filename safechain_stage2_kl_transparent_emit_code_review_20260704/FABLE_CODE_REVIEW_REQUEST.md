# Fable Code Review Request: Stage2 KL-Transparent Pause Emission

Please review this proposed Stage2 implementation objectively and critically.

Context:

- The current Stage2 full SFT / format-only approaches risk changing capability
  and safety behavior while trying to teach pause tokens.
- We want a new Stage2 method that preserves the existing data format and
  downstream Stage3 interface.
- The intended method is:

```text
pause-slot CE + KL-to-base continuation matching + non-target pause suppression
```

Primary files to inspect:

- `cot-safety/legacy/COTPauseToken/src/utils/pause_kl_trainer.py`
- `cot-safety/scripts/run_stage2_sft.py`
- `cot-safety/legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh`
- `cot-safety/configs/experiment/stage2_intra_pause_kl_transparent_emit_8b_cot4_save50_max400_4xa100.yaml`
- `cot-safety/configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_cot3_save25_max400_4xa6000.yaml`

Please answer:

1. Does the implementation preserve the existing Stage2 data format and Stage3
   checkpoint/tokenizer interface?
2. Is the loss correctly aligned with the intended objective?
   - CE only on pause target positions.
   - KL on ordinary continuation tokens after pause stripping.
   - Suppression for pause probability at non-pause targets.
3. Is the teacher/student alignment correct?
   - Student input includes pauses.
   - Teacher input strips pauses from the same sequence.
   - Student prediction positions align with teacher prediction positions.
4. Is using the same frozen model as teacher valid under rows-only training?
   - Pause token is masked out of the KL vocabulary.
   - Body and ordinary token rows are frozen.
5. Are there any bugs in indexing, padding, label masking, or distributed
   training compatibility?
6. Are the default hyperparameters reasonable for an initial 1.5B validation?
7. Does `max_kl_tokens_per_example=256` create a claim limitation? If yes, how
   should we phrase or change it?
8. What minimum tests should be added before running a real GPU job?
9. Should this be accepted for an experimental branch, or are there blockers?

Please be blunt. If the code is not safe to run, identify exact fixes.

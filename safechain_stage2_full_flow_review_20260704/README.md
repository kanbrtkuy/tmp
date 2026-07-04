# SafeChain Stage2 Full-Flow Code Review Packet

Date: 2026-07-04

This packet contains the latest Stage2 flow code for review:

```text
raw trusted CoT data
-> Stage2 pause SFT data prep
-> Stage2 pause model training
-> Stage2 model comparison evaluation
-> normal HF checkpoint/tokenizer consumed by Stage3
```

The latest proposed training method is:

```text
sft.method: kl_transparent_emit

loss =
  emit_weight * CE(pause-token targets only)
+ continuation_weight * KL(base/no-pause teacher || pause/student continuation)
+ suppression_weight * non-target pause suppression
```

The design constraint is:

- Keep Stage2 data JSON format unchanged.
- Keep existing prepared variants such as `intra_pause_cot3`,
  `intra_pause_cot4`, `no_pause_matched`, and `pre_think_pause3_matched`.
- Keep Stage3 interface unchanged: it consumes a normal HF checkpoint/tokenizer
  and uses existing pause offsets/hidden positions.
- Do not update Stage4 in this packet.

## Main Entry Points

Data prep:

- `cot-safety/legacy/COTPauseToken/scripts/data_generation/pause_sft/build_intra_think_pause_sft_splits.py`
- `cot-safety/legacy/COTPauseToken/scripts/data_generation/pause_sft/validate_intra_think_pause_sft_format.py`

Training runner:

- `cot-safety/scripts/run_stage2_sft.py`
- `cot-safety/legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh`

Trainer:

- `cot-safety/legacy/COTPauseToken/src/trl_train.py`
- `cot-safety/legacy/COTPauseToken/src/utils/pause_kl_trainer.py`
- `cot-safety/legacy/COTPauseToken/src/utils/trainer_utils.py`

Stage2 evaluation:

- `cot-safety/scripts/run_model_comparison_eval.py`
- `cot-safety/scripts/prepare_model_comparison_eval_data.py`
- `cot-safety/legacy/PauseProbe/scripts/eval/run_model_comparison_generation.py`
- `cot-safety/legacy/PauseProbe/scripts/eval/summarize_model_comparison_eval.py`

Stage3 handoff context:

- `cot-safety/scripts/run_stage3_intra_pause_probe.py`
- `cot-safety/configs/experiment/stage3_intra_pause_probe*.yaml`

Important configs:

- `cot-safety/configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_cot3_save25_max400_4xa6000.yaml`
- `cot-safety/configs/experiment/stage2_intra_pause_kl_transparent_emit_8b_cot4_save50_max400_4xa100.yaml`
- `cot-safety/configs/experiment/stage2_model_comparison_eval*.yaml`
- `cot-safety/configs/data/stage2_trusted_cot_18k.yaml`

Previous Stage2 result context:

- `cot-safety/res/deepseek-8b/stage2_format_only_sft_summary.md`

## Local Verification Already Run

From `/Users/baby/Documents/SafeChain`:

```bash
python3 -m py_compile \
  cot-safety/legacy/COTPauseToken/src/utils/pause_kl_trainer.py \
  cot-safety/legacy/COTPauseToken/src/trl_train.py \
  cot-safety/scripts/run_stage2_sft.py

bash -n cot-safety/legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh
```

From `/Users/baby/Documents/SafeChain/cot-safety`:

```bash
env PYTHONPATH=src python3 scripts/run_stage2_sft.py \
  --config configs/experiment/stage2_intra_pause_kl_transparent_emit_8b_cot4_save50_max400_4xa100.yaml \
  --dry_run --skip_data_prep

env PYTHONPATH=src python3 scripts/run_stage2_sft.py \
  --config configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_cot3_save25_max400_4xa6000.yaml \
  --dry_run --skip_data_prep
```

Limitations:

- No real GPU training was run.
- No tiny-model unit tests were added yet.
- Local runtime import/instantiation is limited by missing COTPauseToken training
  dependencies in this laptop environment.

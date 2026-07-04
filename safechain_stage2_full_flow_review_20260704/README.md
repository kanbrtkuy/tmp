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

Stage2 tests:

- `cot-safety/tests/test_stage2_pause_kl_trainer.py`

Stage3 handoff context:

- `cot-safety/scripts/run_stage3_intra_pause_probe.py`
- `cot-safety/configs/experiment/stage3_intra_pause_probe*.yaml`

Important configs:

- `cot-safety/configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_cot3_save25_max400_4xa6000.yaml`
- `cot-safety/configs/experiment/stage2_intra_pause_kl_transparent_emit_8b_cot4_save50_max400_4xa100.yaml`
- `cot-safety/configs/experiment/stage2_model_comparison_eval*.yaml`
- `cot-safety/configs/data/stage2_trusted_cot_18k.yaml`
- `cot-safety/configs/model/template_deepseek_r1_distill.yaml`
- `cot-safety/pipelines/runpod_base_env.sh`

Previous Stage2 result context:

- `cot-safety/res/deepseek-8b/stage2_format_only_sft_summary.md`

## Round 2 Fixes After Fable Review

The previous Fable review is included at
`CLAUDE_FABLE5_FULL_FLOW_REVIEW.md`. This packet has been updated to address
the review's C1-C8 items:

- Removed the raw Hydra override for `+trainer.pause_kl.pause_token=<|pause|>`.
  The trainer default is used instead.
- Added rows-only guard checks for `weight_decay == 0.0`.
- Added a first-step invariant callback that snapshots embedding rows and
  fails if any non-pause row changes after optimizer step 1.
- Replaced per-token `.item()` loops in pause stripping / KL-pair selection
  with one CPU list transfer per tensor per batch.
- Replaced full selected-vocab fp32 suppression `log_softmax` with chunked
  pause-column log-prob computation.
- Updated the 1.5B KL config to `eval_steps: 50` and early-stopping patience 4.
- Updated the 8B KL config to `save_total_limit: 8`.
- Added natural-vs-forced Stage2 eval configs and `pause_emission_summary.csv`
  output for self-emission metrics.
- Added `test_stage2_pause_kl_trainer.py` covering the trainer indexing,
  pause KL masking, pause CE/suppression, rows-only guard/invariant, and
  end-to-end loss on a tiny model.
- Added previously missing packet files:
  `configs/model/template_deepseek_r1_distill.yaml` and
  `pipelines/runpod_base_env.sh`.

## Local Verification Already Run

From `/Users/baby/Documents/SafeChain`:

```bash
python3 -m py_compile \
  cot-safety/legacy/COTPauseToken/src/utils/pause_kl_trainer.py \
  cot-safety/legacy/PauseProbe/scripts/eval/run_model_comparison_generation.py \
  cot-safety/legacy/PauseProbe/scripts/eval/summarize_model_comparison_eval.py \
  cot-safety/scripts/run_stage2_sft.py \
  cot-safety/scripts/run_model_comparison_eval.py \
  cot-safety/tests/test_stage2_pause_kl_trainer.py

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

env PYTHONPATH=src python3 scripts/run_model_comparison_eval.py \
  --config configs/experiment/stage2_model_comparison_eval_1p5b_kl_transparent_emit_cot3_4xa6000.yaml \
  --phase generate --dry_run

env PYTHONPATH=src python3 scripts/run_model_comparison_eval.py \
  --config configs/experiment/stage2_model_comparison_eval_8b_kl_transparent_emit_cot4_4xa100.yaml \
  --phase generate --dry_run
```

Limitations:

- No real GPU training was run.
- `pytest`/`torch` are not installed in this laptop environment, so
  `tests/test_stage2_pause_kl_trainer.py` was added and syntax-checked but not
  executed locally. It should be run on the RunPod/SFT environment before GPU
  smoke/full training.
- Local runtime import/instantiation is limited by missing COTPauseToken
  training dependencies in this laptop environment.

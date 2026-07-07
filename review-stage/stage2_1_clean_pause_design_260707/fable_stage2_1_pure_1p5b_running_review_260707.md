# Fable Review: Stage2.1 Pure 1.5B Running Experiment

Date: 2026-07-07

## Context Sent

Current running experiment:

- Model: `DeepSeek-R1-Distill-Qwen-1.5B`
- Hardware: 2x RTX A6000 RunPod
- Method: Stage2.1 pure pause SFT
- Pause token: three repeated pure `<|pause|>` tokens
- Target location: after `cot_4`, before `cot_5`
- Main goal: make the model naturally emit exactly three pause tokens at the correct location without changing the model body, then use the resulting checkpoint for Stage3 pause separability tests.
- Training run: full 1 epoch over 17k train rows, no early stopping, no `max_steps` cap.
- Effective batch: `2 GPUs * per_device_train_batch_size 4 * gradient_accumulation_steps 2 = 16`
- Output: `/workspace/outputs/deepseek_1p5b_stage21_pause_pure_cot5_full_2xa6000`
- R2 root: `cloudflare_r2_cot_safety:cot-safety/stage2-stage3/20260707-2xa6000-1p5b-stage21-pure-full-cot5-bs4-ga2`

Primary code/config inspected:

- `configs/experiment/stage21_pause_pure_dagger_1p5b.yaml`
- `configs/experiment/stage21_pause_pure_dagger_1p5b_full_2xa6000.yaml`
- `pipelines/run_stage21_pure_1p5b_full.sh`
- `scripts/run_stage2_sft.py`
- `legacy/COTPauseToken/src/utils/pause_kl_trainer.py`
- `legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh`
- `legacy/COTPauseToken/src/utils/natural_pause_metrics.py`
- `scripts/run_model_comparison_generation.py`
- `scripts/diag_stage2_checkpoint.py`
- `pipelines/runpod_watch_cold_checkpoints_to_r2.sh`

## Fable Verdict

Fable's bottom line:

- The run matches the stated method and research goal.
- No blocker requires stopping or changing the current run.
- The two immediate checks to do during training are:
  - Confirm the run was not silently capped by a stale `MAX_STEPS` environment variable.
  - Confirm the R2 watcher is alive and checkpoint upload/check/prune is functioning.

## Runtime Follow-Up Checks

These were checked immediately after the review.

### MAX_STEPS / Full-Epoch Check

The training log shows the run is `0/1063`, consistent with full 1 epoch over 17k rows at effective batch 16. No `MAX_STEPS=` override was printed in the grep output.

Observed training progress:

- Total steps: `1063`
- Example progress: step `43/1063`
- First training metric at step 25:
  - `loss: 23.0089`
  - `pause_kl/train/emit: 8.6437`
  - `pause_kl/train/pause_emit/target_prob_mean: 0.2959`
  - `pause_kl/train/pause_emit/target_argmax_rate: 0.3333`

### GPU Utilization

Both GPUs were active and near full memory:

- GPU 0: `48496 / 49140 MiB`, `100%`
- GPU 1: `48336 / 49140 MiB`, `100%`

### R2 Watcher

Watcher is alive:

- PID: `4949`
- Command: `pipelines/runpod_watch_cold_checkpoints_to_r2.sh`
- Mode: upload cold checkpoints to R2, remove cold after upload, keep latest 2 local checkpoints.

`checkpoint-25` was uploaded and verified:

- R2 log: `0 differences found`
- R2 log: `13 matching files`
- Ledger event: `uploaded checkpoint-25`

Disk status at check time:

- `/workspace`: `31G used / 470G available`
- `/dev/shm`: unused for this run

## Fable's Main Risks To Record

1. R2 watcher fragility

   The watcher uses `set -euo pipefail`; a transient `rclone` error or checkpoint flush race could kill it. We should keep checking watcher PID and `/workspace` disk during the run.

2. Low VRAM headroom

   The chosen batch uses almost all GPU memory. This is acceptable because checkpoints are frequent, but long-sequence batches or fragmentation could still OOM.

3. Tied embedding capacity risk

   The pure method updates the pause token row in tied input/output embeddings. The same vector must both emit pause correctly and preserve transparent continuation behavior. This is likely the main empirical failure mode if exact-3 does not reach the gate.

4. Location metric retokenization boundary

   Training insertion and natural generation gate both use the leading-space-skip convention, but generated text is retokenized for the gate. If exact-3 passes but location fails, inspect the `first_pause_token_index_inside_think` histogram before concluding the model learned the wrong location.

5. Checkpoint sweep is not yet implemented

   The current pipeline gates `final` by default. The stated plan is to select the best checkpoint by natural exact-3/location behavior. Fable says this needs an explicit dev/test-separated sweep protocol before making a Stage2 success claim.

6. Selection bias

   Do not sweep all checkpoints on the same prompts used for final claims. Select on a smaller disjoint dev subset, then run the strict gate/capability/judge once on untouched evaluation prompts.

7. Judge is off in this 1.5B training wrapper

   `RUN_JUDGE=0` is acceptable for training speed, but Stage2 success claims require judge runs on selected checkpoint versus base.

## Mandatory Post-Training Checks From Fable

1. Training integrity

   Confirm `trainer_state.json` has global step around `1063`, epoch around `1.0`, and no invariant callback failures.

2. Post-hoc rows-only proof

   Tensor-diff selected checkpoint against base. Every tensor should be bit-identical except pause token row `151665` of the tied embedding.

3. Strict natural gate per source

   Required thresholds:

   - exact-3 pause rate >= `0.99`
   - location match >= `0.99`
   - off-target pause <= `0.005`
   - malformed <= `0.005`

   Also inspect GSM8K pause count distribution specifically because the old failure mode was over-emission there.

4. Capability parity

   Compare natural pause model against base natural on GSM8K/MATH500 with same decode parameters, bootstrap CIs, and a pre-registered acceptable margin.

5. Safety/judge run

   Run unsafe-prompt judges and over-refusal checks on selected checkpoint and base.

6. Forced-insertion quality

   Validate the `stage21_pure_cot5_forced` path because this is the path Stage3/Stage4 depend on.

7. Broken-output/parse rate

   Compare think-block presence, answer extraction, malformed output, and length shifts against base.

8. Selection hygiene

   Document which checkpoints were examined, which prompts were used for selection, and which fresh prompts were used for final reported numbers.

9. Provenance

   Record resolved YAML, effective batch, commit hash, R2 ledger, and `rclone check` markers.

10. Stage3 smoke

   Before claiming Stage3 viability, extract pause hidden states from a small selected-checkpoint slice and confirm the positions are locatable.

## Action Items

- Keep current training running.
- Monitor watcher liveness and `/workspace` disk.
- Build or script a dev/test-separated checkpoint sweep before the run finishes.
- After SFT completes, do not claim Stage2 success from `final` alone; select checkpoint using a held-out dev subset and evaluate the selected checkpoint once on untouched final prompts.

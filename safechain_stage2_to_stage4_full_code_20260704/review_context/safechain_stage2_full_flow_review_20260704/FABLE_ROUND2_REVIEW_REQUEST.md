# Fable Round 2 Review Request: Stage2 Fixes After Prior Review

Please perform a focused but complete second review of this Stage2 packet.

Read:

1. `README.md`
2. `CLAUDE_FABLE5_FULL_FLOW_REVIEW.md`
3. The updated files listed in `README.md`

The prior review identified C1-C8. The code packet has been revised to address
them. Please verify whether each item is truly fixed and identify any new or
remaining blockers.

## What Changed Since Round 1

- Removed raw Hydra `pause_token=<|pause|>` override from
  `run_4gpu_intra_pause_sft.sh`.
- Added `weight_decay == 0.0` guard and first-step non-pause-row invariant
  callback in `PauseKLSFTTrainer`.
- Optimized `_pause_stripped_batch` and `_select_kl_pairs` to avoid per-token
  `.item()` syncs.
- Changed pause suppression loss to chunked pause-column log-prob computation.
- Adjusted the 1.5B and 8B KL configs per the prior review.
- Added natural/forced KL eval configs and pause self-emission summary metrics.
- Added `tests/test_stage2_pause_kl_trainer.py`.
- Added the previously missing packet files:
  `configs/model/template_deepseek_r1_distill.yaml`,
  `pipelines/runpod_base_env.sh`.

## Please Answer

1. Are the prior C1-C8 issues fixed?
2. Is the rows-only invariant callback correct under HF Trainer/DDP timing?
3. Is the chunked suppression loss mathematically identical to the old version?
4. Do the unit tests cover the minimum needed before a GPU smoke?
5. Does the new natural/forced eval now make self-emission measurable enough
   for a post-checkpoint claim, assuming the eval is run?
6. Are there any new bugs introduced by the fixes?
7. Final go/no-go for:
   - code review packet
   - 1.5B single-GPU smoke
   - 1.5B 4-GPU smoke
   - 1.5B full 400-step pilot
   - 8B pilot
   - Stage3 handoff

Please be precise and concrete. If anything is still no-go, name the exact file
and function/site to fix.

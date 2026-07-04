# SafeChain Stage3/Stage4 Review Packet

Date: 2026-07-04

This packet contains the current Stage3 and Stage4 code/config/results for a
Fable review. The goal is to audit the post-Stage2 path and especially rethink
Stage4 steering under the new Stage2 `kl_transparent_emit` plan.

## Research Logic

Stage1: test whether latent space has safe/unsafe separability on original
model traces.

Stage2: train a pause-token model. The latest candidate is **not ordinary
full-response SFT**; it is `kl_transparent_emit`, intended to learn pause-token
availability with minimal behavior drift.

Stage3: run the pause model on the same style of safety data, extract hidden
states at pause positions, and train probes to test whether pause positions
carry safe/unsafe separability beyond prompt artifacts.

Stage4: use a probe or unsafe direction to locate an unsafe manifold, then
intervene on pause-token hidden states to move generations away from unsafe CoT.
Final goal: reduce unsafe CoT without increasing over-refusal, damaging
capability, or causing broken output.

## Key Question

The existing Stage4 path was built before the new Stage2 method. It currently
uses a learned/additive delta at pause tokens during generation. We need Fable
to judge whether this is still the right method after `kl_transparent_emit`, and
whether there is a better Stage4 algorithm.

In particular:

- If Stage2 produces natural pause self-emission, should Stage4 still force
  insert pauses after cot3/cot4 during evaluation?
- If steering is applied only at pause tokens, do we need to measure injection
  gain / attention mass / Jacobian liveness before relying on it?
- Is the current learned delta objective actually learning an unsafe-removal
  direction, or is it entangled with length/refusal/quality?
- Should Stage4 instead use probe-conditioned online steering, constrained
  activation editing, representation-space projection, or a train-time
  auxiliary loss on pause states?

## Main Stage3 Entry Points

- `cot-safety/scripts/run_stage3_intra_pause_probe.py`
- `cot-safety/configs/experiment/stage3_intra_pause_probe.yaml`
- `cot-safety/configs/experiment/stage3_intra_pause_probe_8b_4xa100.yaml`
- `cot-safety/configs/data/stage3_intra_pause_probe_sources.yaml`
- `cot-safety/pipelines/run_4xa100_stage3_probe.sh`
- `cot-safety/pipelines/runpod_stage3_env.sh`
- `cot-safety/legacy/PauseProbe/scripts/probe/run_intra_pause_probe_full.py`
- `cot-safety/legacy/PauseProbe/scripts/probe/extract_hidden_states.py`
- `cot-safety/legacy/PauseProbe/scripts/probe/run_position_scan_batched.py`
- `cot-safety/legacy/PauseProbe/scripts/probe/train_probe.py`

## Main Stage4 Entry Points

- `cot-safety/scripts/run_stage4_steering.py`
- `cot-safety/configs/experiment/stage4_pause_steering.yaml`
- `cot-safety/configs/experiment/stage4_pause_steering_8b_4xa100.yaml`
- `cot-safety/configs/data/stage4_steering_eval_sources.yaml`
- `cot-safety/pipelines/run_4xa100_stage4_steering_eval.sh`
- `cot-safety/pipelines/runpod_stage4_env.sh`
- `cot-safety/pipelines/run_stage4_second_judges_vllm_dynamic.sh`
- `cot-safety/legacy/PauseProbe/scripts/steering/run_intra_pause_learned_delta_pilot.py`
- `cot-safety/legacy/PauseProbe/scripts/steering/run_intra_pause_steered_generation.py`
- `cot-safety/legacy/PauseProbe/scripts/steering/run_intra_pause_full_steering_eval.sh`
- `cot-safety/legacy/PauseProbe/scripts/steering/summarize_intra_pause_full_steering_eval.py`
- `cot-safety/legacy/PauseProbe/scripts/judge/run_open_judges.py`
- `cot-safety/legacy/PauseProbe/scripts/judge/run_vllm_dynamic_open_judges.py`
- `cot-safety/legacy/PauseProbe/scripts/judge/normalize_judge_outputs.py`

## Safety Scope Guard

- `cot-safety/src/cot_safety/steering/scope.py`
- `cot-safety/src/cot_safety/cli.py`
- `cot-safety/tests/test_steering_scope.py`

The current code validates that configured steering targets are `pause_*` only.
Stage4 generation also applies the hook only when token ids equal `<|pause|>`.

## Current Result Context

Stage3 result summaries:

- `cot-safety/res/deepseek-8b/stage3_cot3_full_ckpt250_heatmaps/`
- `cot-safety/res/deepseek-8b/stage3_cot4_ckpt250_heatmaps/`
- `cot-safety/res/deepseek-8b/stage3_cot3_ckpt500_heatmaps/`

Stage4 result summaries:

- `cot-safety/res/deepseek-8b/stage4_cot3_full250_hardsafe/stage4_cot3_full250_hardsafe_summary.md`
- `cot-safety/res/deepseek-8b/stage4_cot3_full250_hardsafe/*.csv`

Important caveat: these Stage4 results are from the older 8B cot3 full-SFT
checkpoint, not from the new `kl_transparent_emit` Stage2 model.

## Stage2 Context Included

- `stage2_context/CLAUDE_FABLE5_STAGE2_ROUND3_REVIEW.md`
- `stage2_context/CLAUDE_FABLE5_STAGE2_FULL_FLOW_REVIEW.md`
- `cot-safety/configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_cot3_save25_max400_4xa6000.yaml`
- `cot-safety/configs/experiment/stage2_intra_pause_kl_transparent_emit_8b_cot4_save50_max400_4xa100.yaml`
- `cot-safety/legacy/COTPauseToken/src/utils/pause_kl_trainer.py`

Stage2 Round 3 review says the Stage2 code packet is code-review GO, but the
next gates are pod pytest and 1.5B smoke. Stage4 should not assume the old
full-SFT behavior transfers to KL-transparent pause models.

## Local Verification

No Stage3/Stage4 training/eval was run locally for this packet. This packet is
for code/design review and method redesign. The local laptop lacks the GPU ML
runtime needed to execute these flows.


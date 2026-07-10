# Stage4 LoRA/PPC Clean Steering Implementation Plan

Date: 2026-07-10

## Goal

Implement the clean Stage4 matched-steering protocol for the PPC LoRA setting:

- separate insertion effect, LoRA effect, and steering effect;
- run pause steering and ordinary-token diagnostic steering at matched target count;
- record hook integrity and applied strength so the effect can be attributed to steering.

## Implemented In This Pass

### 1. Diagnostic Target Scope

File: `src/cot_safety/steering/scope.py`

- Default path remains pause-only.
- Added explicit diagnostic validation for:
  - `pause_N`
  - `cot_N`
  - `post_pause_N`
  - `token_N`
- `validate_target_specs(..., diagnostic_targets=True)` is required for ordinary-token counterfactuals.

### 2. Shared Target Resolver

File: `src/cot_safety/steering/targeting.py`

- Resolves Stage4 target positions using the same intra-CoT convention as Stage3 extraction.
- Supports open-ended generation prefixes without requiring `</think>`.
- `cot_N` and `token_N` are aliases for the Nth non-pause reasoning content token after leading whitespace.
- `post_pause_N` is 1-based after the three-pause run.
- Produces padded target masks for batched generation.

### 3. GPRS Forward Hook

File: `src/cot_safety/steering/gprs.py`

- Added `gprs_forward_hook`.
- Applies `projection_rejection_update` only when hidden shape matches the explicit target mask.
- Cached one-token forwards are skipped.
- Records:
  - number of hook calls;
  - number of applied calls;
  - target token counts;
  - per-row target counts;
  - applied delta norms;
  - applied hidden norms;
  - applied relative norms.

### 4. Stage4 GPRS Generation Script

File: `scripts/run_stage4_gprs_generation.py`

Supports:

- `condition=base`: A0, no FSM / no LoRA / no steering.
- `condition=fsm`: A1, FSM insertion only.
- `condition=ppc`: A2, FSM + PPC LoRA / pause row, no steering.
- `condition=gprs`: A3/A4/A5, FSM + PPC LoRA / pause row + GPRS steering.

Key behavior:

- Uses a two-phase runtime wrapper:
  1. generate enough prefix with FSM to contain the target window;
  2. crop prefix to exactly cover target positions;
  3. run continuation generation with explicit target-mask GPRS hook.
- Supports diagnostic ordinary-token targets with `--diagnostic_targets`.
- Suppresses extra pause tokens after the conditioned prefix.
- Writes per-row target resolution, crop report, hook stats, pause metrics, and plugin token verdict.

### 5. Stage4 Runner Wiring

File: `scripts/run_stage4_steering.py`

- `--phase generation` now expands GPRS generation commands instead of stopping at the old scaffold error.
- It reads:
  - `eval.dataset_specs`;
  - `steering.target_specs`;
  - `steering.alpha_grid`;
  - `steering.seeds`;
  - model tokenizer / position LoRA / token-row paths.
- `steering.gprs.steering_first_pivot: true` permits the documented steering-first path despite failed privileged-pause Stage3 evidence.

### 6. Entry Config

File: `configs/experiment/stage4_lora_clean_gprs_1p5b.yaml`

- Uses clean base 1.5B model plus Stage2.3 PPC artifacts.
- Contains target specs:
  - `pause_all3`
  - `content_cot4_6`
  - `post_pause_1_3`
- Contains safety eval dataset specs.
- Defaults to alpha grid `[0.0, 0.25, 0.5, 1.0]`.

## Verified Locally

- `py_compile` passes for:
  - `scope.py`
  - `targeting.py`
  - `gprs.py`
  - `run_stage4_gprs_generation.py`
  - `run_stage4_steering.py`
- Direct import checks pass.
- `gprs_forward_hook` small tensor self-check passes.
- `run_stage4_steering.py --phase generation --dry_run` expands the command matrix.

## Known Gaps Before Formal Experiment

Must do next:

1. Fable code review.
2. Address review blockers.
3. Run a GPU smoke test on RunPod:
   - `limit=2`;
   - one pause target;
   - alpha `0.0` and nonzero alpha;
   - verify hook touched exactly 3 target tokens;
   - verify lambda=0 output path works.
4. Tune batch size on RunPod:
   - increase `runtime.generation.batch_size_per_gpu`;
   - monitor peak GPU memory and utilization;
   - freeze the largest stable batch before formal Stage4 generation.
5. Add or wire judge/summary:
   - WildGuard primary judge;
   - paired bootstrap gate table;
   - A3-A2, A3-best(A4), A3-A5 contrasts.

## Important Claim Guard

The current code supports generation for matched steering, but it does not by itself prove clean steering. Clean claims require:

- A0/A1/A2/A3/A4/A5 arms;
- shared prompts and seeds;
- matched target count;
- matched applied norm;
- judge results;
- side-effect metrics.

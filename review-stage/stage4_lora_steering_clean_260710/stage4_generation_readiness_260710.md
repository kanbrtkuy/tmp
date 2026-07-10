# Stage4 Generation Readiness - 260710

## Purpose

This review packet is for the clean Stage4 steering battery after the Stage2/3 PPC pause-port path. The goal is to isolate steering effect and test whether pause steering is cleaner than matched ordinary-token steering.

## Current Design

- Conditions: `base`, `fsm`, `ppc`, `gprs`.
- Primary steering target: `pause_all3` = `pause_0,pause_1,pause_2`.
- Diagnostic matched targets:
  - `content_pre_pause_2_4` = `cot_2,cot_3,cot_4`.
  - `post_pause_1_3` = `post_pause_1,post_pause_2,post_pause_3`.
- Strength mode: `matched_relative`, so pause/content/post arms receive matched applied relative perturbation norms.
- The older projection-rejection protocol draft in this folder is superseded by the revised protocol wording: the primary run is a fixed-relative-norm positional perturbation, not an unsafe-projection-removal claim.
- Gate mode: `none` for this first clean battery.
- Main comparison: `A3 - A2` is steering effect; `A3 - A0` must not be reported as steering.

## Fable Review Status

- Protocol/code review reached PASS after fixes to:
  - fail-closed target/matched-strength checks,
  - diagnostic target scope,
  - artifact preflight,
  - generation path collisions,
  - alpha=0 handling.
- Final scheduler blocker was fixed:
  - one active process per `--device`;
  - failure path terminates live sibling processes before returning rc.
- Fable final response: no remaining scheduler blocker before formal Stage4 generation.

## Smoke Evidence

Remote smoke directory:

`/workspace/cot-safety/runs/stage4_lora_clean_gprs_1p5b_smoke`

Smoke results:

- 12/12 generation manifests completed.
- Structural assertions passed:
  - `smoke_assertions_matched_relative.json`
- Matched-strength checks passed at 3% tolerance for both:
  - `matched_strength_main.json`
  - `matched_strength_random.json`
- Observed applied relative norms were approximately 0.05 for the alpha=0.5 smoke arms.

## Batch Stress Evidence

Remote stress directory:

`/workspace/cot-safety/runs/stage4_batch_stress`

Results:

- `batch=128`, `max_new_tokens=512`: completed, about 28.3 GB peak.
- `batch=192`, `max_new_tokens=512`: OOM near 43.8 GB.
- `batch=128`, `max_new_tokens=1024`: completed 208 rows per GPU, about 30.5 GB.
- `batch=160`, `max_new_tokens=1024`: completed 208 rows per GPU, about 37.3 GB observed, 97-100% GPU utilization.

Chosen formal setting:

- `runtime.generation.batch_size_per_gpu: 160`
- `eval.max_new_tokens: 1024`

## Formal Dry Run

RunPod dry-run command:

```bash
python scripts/run_stage4_steering.py \
  --config configs/experiment/stage4_lora_clean_gprs_1p5b.yaml \
  --phase generation \
  --dry_run
```

Dry-run counts:

- Total commands: 96.
- Commands with `--batch_size 160`: 96.
- `direction_random`: 36.
- `pause_all3`: 36.
- `content_pre_pause_2_4`: 28.
- `post_pause_1_3`: 28.
- `condition_base`: 4.
- `condition_fsm`: 4.
- `condition_ppc`: 4.
- `condition_gprs`: 84.

## Remaining Notes

- Local and RunPod venvs did not have `pytest`, so targeted pytest was not run in this environment.
- `py_compile` passed for the Stage4 modified scripts/modules.
- Formal generation should be followed by judge and summary phases before making safety claims.

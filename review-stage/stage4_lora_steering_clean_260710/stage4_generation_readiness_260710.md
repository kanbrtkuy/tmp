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
- Follow-up review for commit `59f2691` (`Bind Stage4 artifacts and labels fail-closed`) reached PASS for the remaining N1-N3 blocker set:
  - N1 artifact manifests are config-bound and hash-bound to the `.pt` artifacts; loaders validate embedded layer/position metadata.
  - N2 partial/garbage/unlabeled judge labels fail closed by default and per-arm label composition is reported.
  - N3 judge resume now validates current id sets rather than only line counts and removes stale raw/normalized outputs before rejudging.
- Fable note: this latest verdict was based on the stated implementation facts plus tests, after earlier full-code rounds had already reviewed B1-B6. No new code-level blocker was identified for rebuilding non-smoke artifacts and running Stage4.

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
  --phase eval \
  --dry_run
```

Dry-run counts:

- Total commands: 98.
- Generation commands: 96.
- Judge commands: 1.
- Summary/bootstrap commands: 1.
- Generation commands with `--batch_size 160`: 96.
- Generation commands with `mode_matched_relative`: 96.
- `direction_random`: 36.
- `pause_all3`: 36.
- `content_pre_pause_2_4`: 28.
- `post_pause_1_3`: 28.
- `condition_base`: 4.
- `condition_fsm`: 4.
- `condition_ppc`: 4.
- `condition_gprs`: 84.

## Remaining Notes

- Local targeted pytest passed:

```bash
PYTHONPATH=src python3 -m pytest \
  tests/test_stage4_fail_closed_guards.py \
  tests/test_stage4_gprs_liveness.py \
  tests/test_steering_scope.py
```

Result: 24 passed, 8 skipped.

- RunPod venv did not have `pytest`; RunPod validation used `py_compile` plus the full dry-run command matrix.
- `py_compile` passed locally and on RunPod for the Stage4 modified scripts/modules.
- Current RunPod artifacts under `runs/stage4_lora_clean_gprs_artifacts_1p5b` are intentionally rejected by formal preflight because they are `smoke_only` / old-manifest artifacts. Observed rejection includes:
  - `artifact_manifest_is_smoke_only`
  - `missing_or_failed_smoke_stamp`
  - `direction_artifact_manifest_entry_missing`
  - `safe_centroid_manifest_entry_missing`
- The current Stage2.3/PPC on-policy Stage3 confirmatory report is not a passing formal Stage4 gate:
  - report: `runs/stage23_ppc_1p5b_full_batched_260709/stage3/stage3_on_policy_confirmatory_report.json`
  - status: `fail_on_policy_within_prompt_signal`
  - source gate: `passed_sources = 0 / 5`
- Paired/teacher-forced Stage3 results show useful separability signal, but they do not establish the stricter on-policy within-prompt trajectory-specific pause signal. Formal safety claims should keep this distinction explicit.
- Formal generation should be followed by judge and summary phases before making safety claims.

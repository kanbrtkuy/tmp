# Fable Review Request: Stage3/4 High-Fix Pass

Please review the current full tree under:

`safechain_stage2_to_stage4_full_code_20260704/cot-safety`

This pass responds to your previous review:

`CLAUDE_FABLE5_STAGE3_STAGE4_COMPLETION_REVIEW.md`

Your HIGH findings were H-1 through H-4. The new fixes are:

## H-1: Stage3 evidence default path

- `scripts/run_stage3_evidence_report.py` now resolves the default `summary_grid.json` relative to the legacy runner root first:
  - default legacy root: `legacy/PauseProbe`
  - fallback: repo root
- Default output also writes under the legacy runner root.
- Local validation: placing a fixture at `legacy/PauseProbe/runs/probes/stage3_kl_transparent_1p5b_cot3_single/summary_grid.json` and running the script without `--summary` produced `stage3_evidence_report.json` in the same legacy path.

## H-2: Stage3 evidence verdict consumed by Stage4

- `build_stage4_gprs_artifacts.py` now requires a Stage3 evidence report before building artifacts.
- It refuses unless both:
  - `status == "pass"`
  - `pause_only_status == "pass"`
- Manifest now stamps:
  - evidence path/status/margins
  - `direction_provenance: teacher_forced_prompt_labels`
- `src/cot_safety/steering/gprs.py` now includes `artifact_manifest` and `stage3_evidence_report` in metadata and treats artifacts as not ready unless the manifest contains passing Stage3 evidence.
- Local validation: a synthetic failing evidence report makes the builder refuse before loading hidden states/torch.

## H-3: Liveness gate is no longer just file-presence

- `liveness_decision` now supports required-test completeness and threshold-derived `injection_gain`.
- `liveness_gate_status` now checks:
  - report covers all configured `liveness.tests`
  - report `model_under_test` matches the configured model under test
  - `positive_control.decision == green` when `require_positive_control_green` is true
- `run_stage4_liveness.py --metrics_json` uses the same required-test/gate decision logic.
- Local validation:
  - old hand-written `{"decision":"green"}` report now yields `decision=incomplete` and does not open GPRS.
  - complete synthetic report with all tests + model + positive control passes liveness but still stops at missing GPRS artifacts.

## H-4: Stage3 screen no longer selects on test maxima

- `stage3_evidence.py` now selects group champions by `selection_metric=val_auroc` and computes/report margin on `metric=test_auroc`.
- Report now includes:
  - `selection_metric`
  - `pause_only_margin`
  - `pause_only_status`
  - `confidence_interval.status: not_available_from_summary_grid`
- The Stage4 builder requires `pause_only_status == pass`, so a post-pause-only pass no longer unlocks artifact creation.

Other fixes:

- `build_stage4_gprs_artifacts.py` now respects `valid_mask` and records `n_dropped_invalid_positions`.
- Probe checkpoint source is `torch.load` checked for layer/position compatibility when torch is available.
- 8B GPRS config now overrides its own `artifact_manifest` and `stage3_evidence_report`.

Local validation:

- `python3 -m py_compile` passed on all changed/new Python files.
- Stage3 evidence default-path fixture test passed.
- GPRS validate prints manifest/evidence/liveness readiness state.
- GPRS eval refuses:
  - incomplete liveness
  - then, with complete liveness, missing artifacts
- Builder refuses failing Stage3 evidence cleanly.
- Stage3 and Stage4 pipeline plans include the new report/builder steps.
- `pytest` is still unavailable in the local environment.

Please answer:

1. Are H-1, H-2, H-3, and H-4 closed enough to land this pass?
2. Did these fixes introduce any blocker/high/medium issues?
3. Which remaining items are still required before Stage3/4 can be called complete, as opposed to safe-to-land scaffolding?

Write the review to:

`safechain_stage2_to_stage4_full_code_20260704/CLAUDE_FABLE5_STAGE3_STAGE4_HIGH_FIX_REVIEW.md`

Do not edit code.

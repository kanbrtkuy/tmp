# Fable Review Request: Stage3/4 New-High Fix Pass

Please review the current full tree under:

`safechain_stage2_to_stage4_full_code_20260704/cot-safety`

This pass responds to your previous review:

`CLAUDE_FABLE5_STAGE3_STAGE4_HIGH_FIX_REVIEW.md`

You identified two new HIGHs and three new MEDIUMs. This pass fixes the two HIGHs and also addresses the medium gate gaps where cheap.

## NEW-H1: Probe position check refused certified single-position champion

Fix:

- `scripts/build_stage4_gprs_artifacts.py` now accepts `probe_positions ⊆ target_positions` instead of requiring set equality.
- The manifest now stamps `probe_metadata`:
  - source path
  - probe layers
  - probe positions
  - probe threshold when present

Intent:

- A single-position champion probe selected by Stage3 evidence can gate all pause targets without being refused.
- The manifest records the subset relationship so the distribution caveat is visible.

## NEW-H2: Test suite stale after manifest requirement

Fix:

- `tests/test_stage4_gprs_liveness.py::test_gprs_artifact_status_requires_all_artifacts` now checks:
  - missing artifacts
  - artifacts exist but manifest missing
  - failing manifest gives `stage3_evidence_pass`
  - pass/pass manifest gives ready

## NEW-M1/M2/M3 related hardening

Fixes:

- Liveness:
  - Missing `metrics.injection_gain` is now `incomplete` for required `injection_gain`; assertion-only all-green reports no longer open the gate.
  - Positive-control subreport is checked with the same required tests/gate logic.
  - Configured `positive_control_status` starting with `missing`/`invalid` forces gate not-ready, mirroring the plan path.
- GPRS readiness:
  - If config names a live Stage3 evidence report, readiness re-reads it and requires it still pass.
  - Manifest `layer` and `positions` are checked against current steering config.
  - Divergence adds fail-closed missing keys such as `stage3_evidence_stale`, `stage3_evidence_live_missing`, or `steering_config_mismatch`.

Local validation:

- `python3 -m py_compile` passed for all touched Python files.
- Direct artifact-status assertion verified:
  - 3 `.pt` files without manifest => `artifact_manifest` missing
  - pass/pass manifest => ready
- Assertion-only liveness report now refuses with `decision=incomplete`.
- Complete liveness report with metrics proceeds to artifact gate and refuses missing artifacts.
- 8B config remains blocked at gate time because `configured_positive_control_status` is `missing_required_full_sft_pause_control`.

Please answer:

1. Are NEW-H1 and NEW-H2 closed?
2. Did the NEW-M1/M2/M3 fixes introduce any blocker/high/medium issue?
3. Is this now safe to land into main as fail-closed scaffolding, with the larger completion blockers still tracked separately?

Write the review to:

`safechain_stage2_to_stage4_full_code_20260704/CLAUDE_FABLE5_STAGE3_STAGE4_NEW_HIGH_FIX_REVIEW.md`

Do not edit code.

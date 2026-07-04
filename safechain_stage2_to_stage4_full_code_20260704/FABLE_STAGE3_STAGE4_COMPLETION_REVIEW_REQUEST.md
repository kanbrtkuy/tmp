# Fable Review Request: Stage3/4 Completion Gap Pass

Please review the current full code tree under:

`safechain_stage2_to_stage4_full_code_20260704/cot-safety`

Context:

- Prior review passed the Stage2 `kl_transparent_emit` framework and initial Stage3/4 fail-closed scaffolding.
- The user correctly pointed out that Stage3/4 are not complete yet.
- The remaining target is:
  - Stage3 should not just report pause probe AUROC. It must ask whether pause/post-pause states add independent signal beyond prompt baselines and true no-pause content controls, and eventually confirm this on-policy with within-prompt labels.
  - Stage4 should not use learned-delta as the primary path. It must first gate on pause-port liveness, then use GPRS/projection steering only if liveness and artifacts are ready.

New code in this pass:

- Stage3:
  - Added `src/cot_safety/probes/stage3_evidence.py`.
  - Added `scripts/run_stage3_evidence_report.py`.
  - Pipeline Stage3 now includes `stage3_pause_vs_baselines_report`.
  - `configs/experiment/stage3_intra_pause_probe.yaml` now has `probe.min_pause_margin_over_baselines`.
  - The report computes best pause/post-pause AUROC minus max(prompt baseline AUROC, true no-pause content-control AUROC).
  - It explicitly keeps `confirmatory_endpoint.status: not_implemented` when on-policy within-prompt evidence is missing.

- Stage4:
  - Added GPRS artifact readiness checks in `src/cot_safety/steering/gprs.py`.
  - `projection_rejection_update` now accepts `gate_score` + `gate_threshold`, so the validated threshold is actually used.
  - Added liveness report path/read/gate helpers in `src/cot_safety/steering/liveness.py`.
  - `scripts/run_stage4_steering.py --phase validate` now prints `gprs_artifacts` and `liveness_gate`.
  - GPRS `eval/generation/judge/summary/all` now refuses before eval unless liveness is green/yellow and all GPRS artifacts exist.
  - `scripts/run_stage4_liveness.py` can ingest a completed metrics/report JSON and normalize it to `liveness_report.json`.
  - Added `scripts/build_stage4_gprs_artifacts.py`, which builds mean-diff direction and safe centroid from Stage3 hidden NPZ, and requires/copies an existing Stage3 probe checkpoint as the GPRS gate.
  - Pipeline Stage4 `build_gprs_artifacts` now calls that builder instead of placeholder validate.

Local validation:

- `python3 -m py_compile` passed on all changed/new Python files.
- `scripts/run_stage3_evidence_report.py` produced a report on a fixture summary with:
  - `status: pass`
  - `pause_minus_best_baseline`
  - `confirmatory_endpoint.status: not_implemented`
- `scripts/run_stage4_steering.py --phase validate --dry_run` now reports missing liveness report and missing GPRS artifacts instead of looking green.
- `scripts/run_stage4_steering.py --phase eval --dry_run` refuses with missing liveness.
- With a synthetic green `liveness_report.json`, GPRS eval refuses with missing GPRS artifacts.
- `scripts/run_stage4_liveness.py --metrics_json ...` writes a normalized liveness report.
- `scripts/build_stage4_gprs_artifacts.py --dry_run` resolves direction/safe-centroid/probe paths correctly.
- `cot_safety.cli pipeline plan` for Stage3 includes the evidence report step.
- `cot_safety.cli pipeline plan` for Stage4 includes the GPRS artifact builder step.
- `pytest` is not available in the local environment, so pytest tests were not executed here.

Please evaluate:

1. Does this pass correctly address the review concern that Stage3 must report pause signal beyond prompt baselines and true no-pause controls?
2. Does this pass correctly harden Stage4 so GPRS cannot run without liveness green/yellow and required artifacts?
3. Is `build_stage4_gprs_artifacts.py` conceptually correct as the first GPRS artifact producer, or does it risk baking in a teacher-forced/off-policy artifact in a way that conflicts with the intended on-policy Stage4?
4. What remains as blocker/high/medium before we can call Stage3/4 complete?
5. Should any of this be changed before landing into main `cot-safety`?

Write the review to:

`safechain_stage2_to_stage4_full_code_20260704/CLAUDE_FABLE5_STAGE3_STAGE4_COMPLETION_REVIEW.md`

Do not edit code.

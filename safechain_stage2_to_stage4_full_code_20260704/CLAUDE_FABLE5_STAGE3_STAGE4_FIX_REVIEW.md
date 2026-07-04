# Fable Fix Review — Stage3/Stage4 Review Gates (Round 12, targeted)

- **Reviewed:** commit `4068c06` ("Fix Stage3 Stage4 review gates") against Round 11 (`CLAUDE_FABLE5_STAGE3_STAGE4_ONPOLICY_LIVENESS_REVIEW.md`, verdict NEEDS FIXES on R11-H1/H2/H3).
- **Method:** targeted — full diff of `4068c06`; ran the three named test files with the packet numpy venv; spot-read `liveness.py::_metric_status`, `run_stage4_liveness.py --metrics_json`, the torch-skipped tests, and the stage3/stage4 configs; grepped for external callers of changed signatures and for tests pinned to the reordered pipeline steps. No full-tree read, no torch execution.

## Test run (this machine, numpy venv)

```
pytest tests/test_stage4_gprs_liveness.py tests/test_stage3_evidence.py tests/test_stage3_on_policy_confirmatory.py
→ 13 passed, 3 skipped
```

All 3 skips are `importorskip('torch')` on the three untouched `projection_rejection_update` tests (lines 277/299/320) — expected on this venv, not new-code gaps. **The suite-red pattern (R9→R10→R11) is broken: the shipped tests were actually executed and pass.**

## Verification of the three R11 HIGHs

### R11-H1 — red confirmatory fixture: **CLOSED, run-verified**
`tests/test_stage3_on_policy_confirmatory.py:25` now sets the control to `float(idx // 2) * 0.01` — prompt-constant within every pair (idx//2 is the prompt index), so within-prompt control AUROC is exactly 0.5 by the tie rule, and both tests pass under real pytest. Variant differs from the suggested `(idx // 2) % 2` but has the same prompt-constant property; either way the fixture now demonstrates the cancellation the endpoint exists to enforce.

Necessary companion change: any prompt-constant control has **exactly zero** train mean-diff, so `_fit_mean_diff_direction` (on_policy_stage3.py:234-238) was changed from raising on zero norm to returning a zero direction (→ all-ties → 0.5). Fail-closed for the pause signal (zero direction ⇒ 0.5 ⇒ fail). Residual (new, minor): a *degenerate* control (e.g., all-constant control features from a broken extraction) is now indistinguishable from a clean null control — under the old code it raised into `control_error`, which `require_true_content_control: true` would have surfaced. Flag degeneracy in the result (e.g., `degenerate_direction: true`) — follow-up F4.

### R11-H2 — confirmatory endpoint gated nothing: **CLOSED**
All four prescribed bindings landed, via a shared `stage3_evidence_gate()` (gprs.py:55-81):
- **Builder** (`build_stage4_gprs_artifacts.py:84-96`): refuses unless `status`, `pause_only_status`, **and** `confirmatory_endpoint.status` are all `pass`; escape hatch `steering.gprs.allow_teacher_forced_only` exists, defaults false, and **no shipped config sets it**.
- **Manifest stamp** (:136-164): `confirmatory_status`, the embedded `confirmatory_endpoint`, `on_policy_report_path`, `require_confirmatory`.
- **Readiness** (`gprs_artifact_status`): manifest check goes through the same gate, and the live re-read of `stage3_evidence_report.json` now also requires confirmatory pass (`stage3_evidence_live_not_ready`). `require_confirmatory` is derived from the *config* at read time, so a manifest built with the escape hatch doesn't carry the exemption forward. Legacy manifests without `confirmatory_status` fail closed (`"" != "pass"`). Both stage4 configs (1.5B + 8B) set `stage3_evidence_report`, so the live re-read is active on both paths.
- **Pipeline** (`pipeline.py`): confirmatory step now runs *before* the evidence-report step, which gained `--on_policy_report <stage3-on-policy-confirmatory-report>`; step note updated to state the Stage4 gate requires the on-policy confirmation. Default flow (`not_implemented` confirmatory) can no longer open Stage4.
- **Tests:** `test_gprs_artifact_status_requires_on_policy_confirmatory_pass` covers manifest-fail + live-fail; the pre-existing readiness test now requires `stage3_confirmatory_pass` in `missing` and a confirmatory-pass manifest to go ready. Both executed green.

Bonus: R11-M6 also closed — `validate_on_policy_report_config` (stage3_evidence.py:42-67) refuses attach on layer/positions/control-positions mismatch, and the evidence report stamps `report_path` + `report_mtime`.

### R11-H3 — asserted `test_status` opened the four-test gate: **CLOSED, run-verified**
`liveness_decision` required-tests loop (liveness.py:114-121) is now exactly the prescribed fix: `status = metric_status if metric_status is not None else "incomplete"` for **every** required test; the `statuses.get(test)` fallback is deleted, so hand-asserted `test_status` strings are ignored entirely in required-tests mode. `_metric_status` returns `None` for any missing/non-dict payload (liveness.py:74-77) ⇒ `incomplete`. The new regression test reproduces the exact R11 laundering payload (real injection metrics + asserted green kv/patching/attention) and asserts `incomplete` — executed green. `run_stage4_liveness.py --metrics_json` (:65-73) recomputes through `liveness_decision(report, required_tests=liveness.tests, gate=...)`, so the officialization path is closed too.

Residual (accepted scope, follow-up F3): `_metric_status`'s tail still trusts a `status` string inside a **metrics payload** for tests without floor derivation — i.e., forging `metrics: {pause_kv_ablation: {status: green}}` still passes. That is one forgery level deeper than the documented `test_status` field (same class as fabricating metric numbers) and cannot be floor-derived until those kernels exist. When kv-ablation/patching land, derive their statuses from `LivenessGate` floors like injection/attention.

## Beyond the HIGHs (same commit)

| R11 item | Status | Note |
|---|---|---|
| M2 attention_mass self-certifying | **Closed** | Floors moved to `LivenessGate` (`min_pause_attention_mass`, `min_pause_vs_content_attention_ratio`), `_metric_status` derives green/red from the numbers, kernel status demoted to `advisory_status` (payload `status: "measured"` can never green the gate), red reachable, config keys added + tested. `min_pause_attention_mass: 0.0` is vacuous (softmax mass > 0) — ratio floor is the binding one; fine, it's a knob now. |
| M3 natural-pause ordinal capture | **Mostly closed** | `make_position_masks` targets the trailing forced run when the last `n_insert_pauses` valid positions are all pauses. Caveat: on rows where they aren't (right-truncation ate the run — L4), it **falls back to the old all-pause mask** instead of skipping the row, silently measuring natural pauses for that row. Prefer empty mask there (F5). |
| M4 batch KL dilution | **Closed** | `final_token_kl` restricted to rows with ≥1 masked position via `row_mask`; NaN-skip for empty batches. |
| M5 `require_true_content_control` default | **Closed** | Set `true` in `stage3_intra_pause_probe.yaml`; neither 8B/kl-transparent stage3 config overrides it (inherit via deep-merge). Interacts with the H1 degeneracy residual (F4). |
| L2 `take()` row-slicing | **Closed** | `ROW_ALIGNED_KEYS` allowlist. |
| L5 `allow_yellow=False` coverage | **Still open** | Not touched. |
| M1 producer chain (generation, `cot_segment_judge`, pause-stripped control prep, extractor hardening) | **Still open** | Unchanged, as expected — remains the blocker for *running* the confirmatory endpoint (R11-T4/T5). |
| L1/L3/L4/L6/L7, R10 standing items | **Still open** | Unchanged. |

Regression bounding: no callers of the changed kernel signatures (`injection_gain_metric`/`attention_mass_metric` gained required `n_insert_pauses`) exist outside `liveness_kernels.py` (both internal call sites updated); no test references the reordered pipeline step names or `plan_for_config`. The three torch-skipped tests don't touch changed code. So full-suite risk beyond what ran here is low — but this machine cannot execute the torch subset.

## Go/No-Go

| Gate | Verdict | Basis |
|---|---|---|
| Land `4068c06` into main | **GO, contingent on pod pytest green** | All three HIGHs closed and run-verified here (13 passed). Torch tests unexecutable on this Mac; given the R9-R11 history, one full pod pytest run is the non-negotiable land ritual. |
| Confirmatory endpoint as enforced evidence standard | **GO (design + enforcement)** | H2 binding is load-bearing at builder, manifest, and live re-read; escape hatch explicit and unset. |
| Run Stage3 confirmatory on real data | **NO-GO (unchanged)** | M1 producer chain still missing. |
| Stage4 1.5B liveness pilot (2 kernels, calibration only) | **GO** | H3/M2/M3/M4 closed; decision stays `incomplete` under the 4-test config by design. Watch the M3 truncation fallback (F5). |
| Stage4 8B liveness / GPRS generation | **NO-GO (unchanged)** | 8B positive control + steering hook still missing. |

## Required follow-ups

- **F1 (pre-land):** full pod pytest (torch included) green before merge — the only part of R11-T1 this machine cannot certify.
- **F2 (next packet):** R11-T4/T5 producer chain — generation script, `cot_segment_judge`, pause-stripped matched-control prep, `pause_id ∉ control_ids` assert, label-conditional drop stats.
- **F3:** when kv-ablation/patching kernels land, floor-derive their statuses in `_metric_status`; until then a forged metrics-payload `status` string is the remaining (deeper) assertion vector.
- **F4:** flag zero-norm directions in the on-policy result (`degenerate_direction: true`) or fail the control when `require_true_content_control` is set — a broken control extraction currently reads as a clean 0.5 null.
- **F5:** in `make_position_masks`, emit an empty row mask (skip) instead of falling back to all-pauses when the trailing forced run is absent; ties into L4 length-cap.
- **F6 (carried):** L1 margin CI, L3 multi-direction, L4 generation batching/length cap, L5 `allow_yellow=False` coverage, L6/L7, R10 standing items (no-source builder branch, `gate_threshold` calibration, scaler stamp).

## Headline verdict

**PASS with required follow-ups.**

R11-H1, H2, and H3 are all genuinely fixed — not decoratively: the fixture is prompt-constant and the tests execute green (13 passed on this machine, first round where shipped tests were demonstrably run); a failing or absent confirmatory report now blocks the Stage4 builder, the manifest, and the live readiness re-read with no config shipping the escape hatch; and the four-test gate ignores asserted `test_status` entirely, including through `--metrics_json`. The commit additionally closes M2-M6 and L2. Land after one full pod pytest run (F1); F2 remains the blocker for actually running the confirmatory endpoint.

# Fable Review — Stage3/Stage4 High-Fix Pass (H-1…H-4 Verification)

- **Date:** 2026-07-04
- **Tree:** `safechain_stage2_to_stage4_full_code_20260704/cot-safety` @ `69f3a78` ("Address Stage3 and Stage4 high review findings")
- **Prior review:** `CLAUDE_FABLE5_STAGE3_STAGE4_COMPLETION_REVIEW.md` @ `e8a07c8` — verdict PASS with required follow-ups; HIGHs H-1…H-4, MEDIUMs M-1…M-5, TODOs T-1…T-10.
- **Scope:** full-tree read-only review of the fix commit (12 files: 2 configs, 4 scripts, 3 src modules, 2 tests + request doc). No code edited. Where feasible I *executed* the changed gate functions in this environment (pure-Python paths, tmpdirs) rather than only reading them; execution results are marked "run-verified" below.

---

## 1. Executive summary

All four HIGHs are materially closed, and I verified each end-to-end rather than at the edit site: the evidence report's default paths now bind exactly to where the legacy Stage3 runner writes for both 1.5B and 8B (H-1); the artifact builder refuses without a double-pass evidence report *before* importing torch, stamps the manifest, and `gprs_artifact_status`/`require_gprs_readiness` fail-closed on missing manifest or non-pass recorded evidence — run-verified in all three states (H-2); the liveness gate now returns `incomplete` for the old bare `{"decision":"green"}` file, enforces test completeness against config, binds `model_under_test`, requires a green positive control, and lets sub-threshold `injection_gain` *metrics override* an asserted green — all five behaviors run-verified (H-3); the screen selects group champions on `val_auroc`, reports margins on `test_auroc`, adds `pause_only_margin`/`pause_only_status` that the Stage4 builder requires, and honestly stamps the CI as unavailable (H-4, which also closes M-5 at the consumer).

The pass introduces **no blocker** — I could not construct a wrong-green through any new path; every failure mode is crash-or-refuse. But it introduces **two HIGHs and three MEDIUMs of its own**:

1. **NEW-H1:** the new probe compatibility check demands *set equality* between the probe checkpoint's `positions` and the steering `target_positions`. The single-scan champion probe — the only probe the evidence report actually certifies, and the source the pipeline step points at — is single-position (`positions: ["pause_0"]` etc., `run_position_scan_batched.py:467`), while both GPRS configs target three positions. The shipped artifact-build flow is therefore dead on arrival: it refuses the certified probe and only accepts a pooled 3-position probe that no gate ever screened.
2. **NEW-H2:** `test_gprs_artifact_status_requires_all_artifacts` was not updated for the new manifest requirement and is now guaranteed red (run-verified: `require_gprs_artifacts` raises where the test asserts `ready is True`). Your own land gate is "pod pytest must pass" — as committed, it cannot.

Both are cheap, mechanical fixes. Verdict at the end: **PASS with required follow-ups** — NEW-H2 pre-land, NEW-H1 before the first artifact build.

---

## 2. Fix scoreboard: H-1…H-4 and the claimed "other fixes"

| ID | Prior finding | Status | Evidence | Residual |
|---|---|---|---|---|
| H-1 | Evidence default path never matches Stage3's legacy-cwd output | **CLOSED** | `run_stage3_evidence_report.py:13-24,49-53`: relative summary tried at `legacy/PauseProbe` then repo root; relative output always under legacy root. Binding traced both ways: `stage_paths()` (`run_stage3_intra_pause_probe.py:74-87`) + 1.5B/8B variant `legacy.single_scan_out_root` (`runs/probes/stage3_kl_transparent_{1p5b_cot3,8b_cot4}_single`, plain relative — no `${COT_SAFETY_RUN_ROOT}` prefix, so env settings can't break it) ⇒ default output `legacy/PauseProbe/runs/probes/<single>/stage3_evidence_report.json` == the exact strings both Stage4 configs carry in `steering.gprs.stage3_evidence_report` (resolved against `REPO_ROOT` by the builder). Pipeline step (`pipeline.py:141-155`) passes `--config` only ⇒ now runnable. Absent file ⇒ `FileNotFoundError` at the legacy path (fail-closed, correct location in the message). | Repo-root fallback could pick up a stale copy if the legacy file is missing (report records `summary` path, auditable). Relative `--output_json` now resolves under legacy root — behavior change worth a `--help` note. |
| H-2 | Nothing consumes the Stage3 evidence verdict | **CLOSED** (with staleness caveat → NEW-M3) | Builder: `build_stage4_gprs_artifacts.py:83-89` refuses unless `status=="pass"` **and** `pause_only_status=="pass"`, before `import torch` and before touching the NPZ; CLI hard-requires a report path from flag or config (`:181-184`); manifest stamps path/status/margins/`selection_metric` + `direction_provenance: teacher_forced_prompt_labels` (`:128-145`). Readiness: `gprs.py:66-86` requires the manifest to exist and record pass/pass, else appends `artifact_manifest`/`stage3_evidence_pass` to `missing`; `require_gprs_readiness` (`run_stage4_steering.py:231-248`) consumes it. **Run-verified:** 3 artifacts + no manifest ⇒ refuse; failing manifest ⇒ `missing=['stage3_evidence_pass']`; passing manifest ⇒ ready. No `--force` exists at all — stricter than T-2 asked. | Readiness trusts the *build-time snapshot*; it never re-reads the live report (NEW-M3). |
| H-3 | Liveness gate = file-presence gate | **SUBSTANTIALLY CLOSED** | `liveness.py:100-137`: with `required_tests`, missing tests ⇒ `incomplete`, and the explicit-`decision` shortcut is unreachable from the gate because `liveness_config` defaults `tests` to all four kernels (`liveness.py:48-49`) — an empty-tests config cannot re-open the old hole. Gate adds model binding (`:176-179`) and positive-control-green requirement (`:180-184`). Thresholds now consumed: `_metric_status` (`:70-84`) derives green/red for `injection_gain` from `min_pause_vs_content_gain`/`min_pause_vs_bos_gain`, and metrics take precedence over asserted statuses. **Run-verified:** bare `{"decision":"green"}` ⇒ `(ready=False, incomplete)`; wrong `model_under_test` ⇒ closed; missing `positive_control` ⇒ closed; metrics 0.10 vs floor 0.25 ⇒ `red` even with all-green `test_status`. Ingest (`run_stage4_liveness.py:63-67`) uses the same required-tests/gate logic. | A metrics-free, all-assertion report still opens the gate (NEW-M1); the 8B two-field `positive_control_status` block is not consulted at gate time (NEW-M2, run-verified); provenance is string equality on the checkpoint path, no hash (acceptable for now). |
| H-4 | Margin computed on max-test-AUROC-selected cells, 0.01 floor below selection noise | **SUBSTANTIALLY CLOSED** | `stage3_evidence.py:94-97`: all four group champions selected on `selection_metric` (default `val_auroc`); margins computed on those champions' `metric` (default `test_auroc`). `best_row` (`:65-70`) drops rows lacking the selection metric ⇒ an old grid without `val_auroc` yields `missing_*` statuses, not a silently wrong selection. `pause_only_margin`/`pause_only_status` added (`:109-132`) and **required by the Stage4 builder** ⇒ prior M-5 (post-pause-driven pass unlocking pause steering) is closed at the consumer. CI honestly stamped `not_available_from_summary_grid` (`:151-154`). Tests updated with val/test splits that would catch test-selection regressions. | No CI: a pass at margin 0.011 on point estimates is still not reportable evidence — the screen is unbiased now but the floor remains at noise scale until per-example scores exist (kept as required follow-up, not a land issue). `best_main` still picks between the two val-champions by *test* AUROC (2-way residual selection, headline `status` only — LOW). CLI can override `--selection_metric`; stamped in the report but not enforced by the builder (LOW). |

**Claimed "other fixes":**

| Claim | Status | Evidence |
|---|---|---|
| Builder respects `valid_mask`, records `n_dropped_invalid_positions` | **CLOSED — correct** (prior M-1) | `valid_mask` is `[N, P]` (`extract_hidden_states.py:11` docstring; producer `:866,886`); `select_state_block` requires *all selected positions valid per row* (`build_stage4_gprs_artifacts.py:50-53`), matching T-5's strict option; drop count flows into meta → both `.pt` payloads and the manifest. |
| Probe checkpoint `torch.load` + layer/position compatibility check | **PARTIAL — introduces NEW-H1** | The check is real, not vacuous: single-scan payloads carry `positions:[<one>]`, `layers:[<one>]` (`run_position_scan_batched.py:467-468`); pooled `train_probe.py` payloads carry multi-position lists (`train_probe.py:1055-1056`). Layer check (`in`-list) is right. Position check (set equality, `:117-120`) is wrong for the certified flow — see NEW-H1. Still no probe metadata / direction↔probe cosine in manifest, no `random_direction.pt` (prior M-3 remainder open). |
| 8B GPRS config overrides its own manifest + evidence paths | **CLOSED** | `stage4_pause_gprs_8b_4xa100.yaml:28-33`; inherits the rest of the `gprs` block via `defaults: [stage4_pause_gprs.yaml, …]` with recursive `deep_merge` (`config.py:49-60,75-98`). Without this override the 8B build would have consumed the 1.5B evidence report — good catch by the executor. 8B two-field liveness block still present (`:16-20`). |

Prior M-status roll-up: **M-1 closed, M-5 closed, M-4 closed-as-scoped** (provenance stamped; on-policy direction still a completion blocker), **M-2 open** (config `gate_threshold: 0.95` is still an unrelated constant; probe payload's calibrated `threshold` + `scaler` still not extracted into the manifest — nothing consumes them), **M-3 partial** (load+check yes; metadata/cosine/random-direction no).

---

## 3. New findings introduced by this pass

No blocker: every new failure mode I could construct crashes or refuses; none produces a wrong green.

### HIGH

**NEW-H1. The probe position check refuses the certified probe — the shipped artifact-build flow is DOA on both configs.**
`build_stage4_gprs_artifacts.py:117-120` raises unless `set(probe_positions) == set(positions)`. The positions default to `steering.target_positions = [pause_0, pause_1, pause_2]` in both GPRS configs. But the probe the evidence gate certifies — the val-selected champion *cell* from the single scan, which is also what the pipeline step's `<stage3-probe-checkpoint>` placeholder and prior-round L-3 intend — is a single-position payload (`positions: ["pause_0"]`, `run_position_scan_batched.py:467`). So the real flow always raises `ValueError`. The only payload that passes is a pooled probe trained jointly on exactly `{pause_0,pause_1,pause_2}` (`train_probe.py:1055`) — an artifact that never went through the pause-vs-baseline screen, silently substituting an *uncertified* gate for the certified one. The obvious workaround (`--positions pause_0`) is worse: it changes the direction/centroid to pause_0-only means while steering still targets all three positions, and nothing records or re-checks the mismatch (readiness never compares manifest `positions`/`layer` to the current steering config — see NEW-M3).
*Fix:* accept `set(probe_positions) ⊆ set(positions)` (a champion-cell probe gating all pause slots is the designed use; the distribution shift pause_0→pause_1/2 is a known caveat worth a manifest field, not a refusal), stamp `probe_positions`/`probe_layers`/`probe_threshold` in the manifest, and prefer wiring the evidence report's winning-cell `probe.pt` path straight into the builder (prior L-3) so the certified cell is what gets copied. Fail-closed today (crash, not wrong-green), but the step cannot be run as shipped — same class as the original H-1.

**NEW-H2. The shipped test suite is guaranteed red — the pod-pytest land gate cannot pass.**
`tests/test_stage4_gprs_liveness.py:128-150` writes the three `.pt` files, no manifest, and asserts `require_gprs_artifacts(...)["ready"] is True`. Under the new `gprs_artifact_status` that call raises `FileNotFoundError("GPRS artifacts are missing: artifact_manifest=…")` — **run-verified** in this environment. The request acknowledges pytest was unavailable locally; this is exactly what that gap hides. Compounding it, the new manifest/evidence readiness logic — the heart of H-2 — has *zero* passing test coverage: the only test touching it is the one that now fails, and there is no test for failing-evidence manifests, `stage3_evidence_pass` in `missing`, or the `pause_only_status` requirement.
*Fix (pre-land):* update the test to (a) assert `missing == ["artifact_manifest"]` when artifacts exist without a manifest, (b) write a pass/pass manifest and assert ready, (c) write a failing manifest and assert `missing == ["stage3_evidence_pass"]`. ~15 lines. Then run pytest in the pod image before land, per standing practice.

### MEDIUM

**NEW-M1. The liveness gate is still satisfiable by a pure-assertion report — thresholds only bind when the report volunteers metrics.**
Run-verified: `{"model_under_test": <exact ckpt>, "test_status": {all four: "green"}, "positive_control": {"decision": "green"}}` with **no** `metrics` block opens the gate. `_metric_status` returns `None` when `metrics.injection_gain` is absent (`liveness.py:72-74`), falling back to the asserted string — so the one test that *has* configured thresholds can dodge them by omitting its numbers. The positive control is likewise accepted as a bare `{"decision":"green"}` (evaluated without `required_tests`, `liveness.py:183`). This is a real improvement over `{"decision":"green"}` (the report must now assert the full battery shape and the right checkpoint), and kernels don't exist yet — but as long as this holds, "liveness green" remains an assertion the gate formats, not a measurement it checks.
*Fix:* for threshold-bearing tests (today: `injection_gain`), treat a missing `metrics` payload as `incomplete` instead of falling back to `test_status`; when kernels land, extend per-test thresholds and require metrics for all four; consider requiring the positive control's sub-report to satisfy the same required-tests completeness.

**NEW-M2. The 8B two-field positive-control block binds the plan, not the gate.**
`liveness_plan_status` refuses on `positive_control_status` starting with missing/invalid (`liveness.py:54-62`), but `liveness_gate_status` never reads that config field — run-verified: with `positive_control_status: missing_required_full_sft_pause_control` in config, a hand-written report with `positive_control.decision: green` still yields `ready=True`. So the two-field design protects the battery *plan* while the actual GPRS-unlock path (the gate) trusts the report's own claim that a control ran. This is the residual sliver of the original H-3 bypass.
*Fix:* in `liveness_gate_status`, mirror the plan check — if `require_positive_control_green` and the *configured* `positive_control_status` starts with missing/invalid, return not-ready regardless of report content (one `startswith` clause; the config is the ground truth for whether a genuine control exists).

**NEW-M3. Readiness trusts the build-time snapshot: evidence staleness and config drift are undetected.**
`gprs_artifact_status` gates on the manifest's *recorded* `stage3_evidence` block only. Two consequences: (a) rerun Stage3, get `fail_no_independent_pause_signal`, and the old artifacts stay "ready" — the exact certified-prompt-risk-gate scenario H-2 was meant to prevent, one rerun later; (b) change `steering.layer` (e.g. 14→16) or `target_positions` after building, and readiness still says ready because manifest `layer`/`positions` are never compared to the current config. The config key `steering.gprs.stage3_evidence_report` is already threaded into `validate_gprs_config` (`gprs.py:40`) but `gprs_artifact_status` never uses it.
*Fix:* when the config names an evidence report and the file exists, re-read it at readiness time and require it still passes (or store a content hash in the manifest and compare); assert manifest `layer`/`positions` equal the live steering config; add `stage3_evidence_stale` / `steering_config_mismatch` to `missing` on divergence.

*(Carried, not new: **M-2** gate-threshold/scaler provenance — the probe payload's calibrated `threshold` and `scaler` are still unconsumed and the GPRS hook, when it lands, must standardize `h` with that scaler before scoring or the 0.95 constant is meaningless. Track it as a hook prerequisite.)*

### LOW

- **L-A.** Ingest (`run_stage4_liveness.py --metrics_json`) computes only the decision; model binding and positive control surface first at gate time. Validate at ingest too so a bad battery fails loudly when produced, not when consumed.
- **L-B.** Threshold defaults 0.25/5.0 are duplicated (`LivenessGate` and `_metric_status`); `read_json` re-implemented in three files. Consolidation nit.
- **L-C.** `require_gprs_artifacts`'s message renders `stage3_evidence_pass=<manifest path>` (`gprs.py:104-109`) — key/value mismatch, cosmetic.
- **L-D.** Relative `--output_json` in the evidence script now lands under the legacy root; fine, but say so in `--help`.
- **L-E.** `best_main` = max-by-test across the two val-champions — residual 2-way test selection affecting only the headline `status`; `pause_only_*` (what Stage4 gates on) is unaffected. Cheapest fix: select `best_main` on `selection_metric` too.
- **L-F.** The builder accepts whatever `selection_metric` the report used; assert `selection_metric == "val_auroc"` (or stamp-and-warn) so a `--selection_metric test_auroc` run can't quietly feed Stage4.
- **Surviving prior lows:** L-2 (learned-delta ack still demanded for `--phase liveness`, `run_stage4_steering.py:287`), L-4 (stale `current_control_cot_aliases_valid: false` still in `stage3_intra_pause_probe.yaml:37`), L-6 (confirmatory status still config-echoed), L-3 (evidence report still doesn't emit the winning cell's probe path — now upgraded in importance by NEW-H1), L-1 (`--model_kinds linear` hardcode). L-7 is moot for current configs (no `liveness.report_json` set; ingest default and gate read agree).

---

## 4. Answers to the three questions

**Q1 — Are H-1…H-4 closed enough to land this pass?**
Yes. H-1 and H-2 are fully closed and I run-verified the H-2 readiness state machine in all three states. H-3 and H-4 are closed to the maximum extent the missing kernels/per-example scores allow, with the honest markers in the right places; their residuals (NEW-M1/M2, no-CI) are trust-grade issues, not land-safety issues — every residual path fails closed or requires deliberately fabricating a richer report in a gitignored directory. The one thing that must change **before** landing is NEW-H2: your land gate is pod pytest, and the committed suite cannot pass it. Fix the test, run pytest in the pod image, land.

**Q2 — Did these fixes introduce any blocker/high/medium issues?**
No blocker — I attempted wrong-green constructions against the builder, readiness, and gate and got refusals every time (§3 run-verified notes). Two HIGHs: NEW-H1 (position set-equality makes the artifact build refuse the certified champion probe — the step is DOA as shipped, the same "gate exists but the flow can't run" class as the original H-1) and NEW-H2 (guaranteed-red test, land-gate breaking). Three MEDIUMs: NEW-M1 (assertion-only reports still open the liveness gate), NEW-M2 (8B two-field block not enforced at gate time), NEW-M3 (manifest staleness / config drift undetected at readiness).

**Q3 — What is still required before Stage3/4 can be called complete, vs safe-to-land scaffolding?**
Unchanged in substance from Round 8; this pass moved gate *plumbing* to done, not evidence. In order:
1. **Liveness kernels** (`injection_gain`, `attention_mass`, `pause_kv_ablation`, `safe_unsafe_patching`) with thresholds calibrated on the full-SFT positive control — plus NEW-M1 so the kernels' numbers are the only thing that can open the gate.
2. **On-policy within-prompt AUROC runner** (10 samples/prompt, mixed-outcome prompts, CI excludes 0.55) — the confirmatory endpoint is still `not_implemented`, and the screen still has no CI (needs per-example scores from the scan producer).
3. **Genuine 8B full-SFT positive control** (two-field unblock) + NEW-M2 so the gate honors the block.
4. **GPRS generation hook** with the probe's calibrated `threshold`+`scaler` (M-2) — the current hard refusal is correct until then.
5. **On-policy contrastive direction + direction QC + `random_direction.pt` control arm** — the manifest now honestly stamps `teacher_forced_prompt_labels`; that stamp must gate any evidence claim.
6. **S4-5 judge endpoint separation** (CoT judged separately, labeled-only denominators) — biggest evidence-integrity item once anything generates.

---

## 5. Go / No-Go table

| Item | Verdict | Gate to upgrade |
|---|---|---|
| Land `69f3a78` into main cot-safety | **GO after NEW-H2 test fix + pod pytest green** | ~15-line test update; nothing else blocks |
| Stage3 evidence report as runnable step | **GO** (was NO) | H-1 verified end-to-end for 1.5B and 8B path bindings |
| Stage3 screen as *evidence* | **NO** (unchanged) | CI on the test margin (needs per-example scores); floor 0.01 is still at noise scale; teacher-forced screen by design |
| Stage4 GPRS artifact build | **NO — currently DOA** | NEW-H1 (subset rule or pooled-probe policy) + NEW-M3 cross-checks; then M-3 remainder (manifest probe metadata, cosine QC, `random_direction.pt`) |
| Stage4 liveness gate as *real* gate | **NO** (unchanged) | kernels + NEW-M1 (metrics required) + NEW-M2 (two-field at gate) |
| Stage4 GPRS eval (1.5B) | **NO-GO** (correctly hard-blocked; refusal re-verified at `run_stage4_steering.py:331-337`) | liveness green + Stage3 pass + on-policy direction QC + hook w/ M-2 wiring + S4-5 |
| Anything 8B | **NO-GO** (unchanged) | genuine full-SFT positive control; NEW-M2 closes the gate-side gap in the meantime |

---

## 6. Concrete TODOs

| ID | Sev | Change | Where |
|---|---|---|---|
| R9-T1 | HIGH | Relax probe position check to `set(probe_positions) ⊆ set(positions)`; stamp `probe_positions`/`probe_layers`/`probe_threshold` into the manifest; prefer consuming the evidence report's winning-cell probe path (revives L-3) | `build_stage4_gprs_artifacts.py:112-127`, `stage3_evidence.py:compact_row` |
| R9-T2 | HIGH | Fix `test_gprs_artifact_status_requires_all_artifacts` for the manifest requirement; add pass/fail-manifest readiness tests; run pytest in the pod image pre-land | `tests/test_stage4_gprs_liveness.py:128-150` |
| R9-T3 | MED | Treat missing `metrics` as `incomplete` for threshold-bearing tests instead of falling back to `test_status`; apply required-tests completeness to the positive-control sub-report | `liveness.py:_metric_status/liveness_decision:70-122,180-184` |
| R9-T4 | MED | Enforce configured `positive_control_status` (missing/invalid ⇒ not ready) inside `liveness_gate_status`, mirroring `liveness_plan_status` | `liveness.py:161-196` |
| R9-T5 | MED | At readiness: re-read the configured live evidence report (or hash-compare) and cross-check manifest `layer`/`positions` vs current steering config; add `stage3_evidence_stale`/`steering_config_mismatch` missing-keys | `gprs.py:gprs_artifact_status:54-98` |
| R9-T6 | MED | (carried M-2) Default the gate threshold from the probe payload; record scaler requirement in the manifest; hook must standardize with the probe scaler | `build_stage4_gprs_artifacts.py`, `gprs.py:validate_gprs_config` |
| R9-T7 | LOW | Validate model binding + positive control at ingest time too | `run_stage4_liveness.py:60-79` |
| R9-T8 | LOW | Select `best_main` on the selection metric; builder asserts report `selection_metric == "val_auroc"` | `stage3_evidence.py:98-99`, `build_stage4_gprs_artifacts.py:83-89` |
| R9-T9 | LOW | Cosmetics: `stage3_evidence_pass` error rendering; dedupe `read_json`/threshold constants; `--output_json` legacy-root note; carried L-2/L-4/L-6 | `gprs.py:104-109`, `liveness.py`, `run_stage3_evidence_report.py`, `run_stage4_steering.py:287`, `stage3_intra_pause_probe.yaml:37` |

Then the standing completion blockers in order: liveness kernels → on-policy Stage3 endpoint → 8B positive control → on-policy direction QC + random-dir control → GPRS hook (with R9-T6) → S4-5 judge separation.

---

## 7. Headline verdict

**PASS with required follow-ups.** H-1 through H-4 are genuinely closed — path binding verified for both models, the evidence verdict now hard-binds artifact building and readiness, the liveness gate went from "any green JSON" to completeness + provenance + threshold-bound-when-measured (all run-verified), and the screen no longer selects on the metric it reports while the pause-only margin now gates Stage4. Nothing in the pass can produce a wrong green. Required: **R9-T2 pre-land** (the committed test suite is guaranteed red and pod pytest is your land gate) and **R9-T1 before the first artifact build** (the builder refuses the very probe the evidence gate certifies, so the shipped build flow cannot run). NEW-M1/M2/M3 belong in the next pass before anyone treats a green gate as a measurement. The real distance to "Stage3/4 complete" is unchanged: kernels, the on-policy endpoint, the 8B control, the hook, and S4-5.

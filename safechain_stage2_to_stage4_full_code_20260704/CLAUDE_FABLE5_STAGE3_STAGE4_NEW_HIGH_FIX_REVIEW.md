# Fable Review — Stage3/Stage4 New-High Fix Pass (NEW-H1/H2, NEW-M1/M2/M3 Verification)

- **Date:** 2026-07-04
- **Tree:** `safechain_stage2_to_stage4_full_code_20260704/cot-safety` @ `2b472ff` ("Fix Stage3 Stage4 new high review issues")
- **Prior review:** `CLAUDE_FABLE5_STAGE3_STAGE4_HIGH_FIX_REVIEW.md` @ `69f3a78` — verdict PASS with required follow-ups; NEW-H1/NEW-H2 (required), NEW-M1/M2/M3, TODOs R9-T1…T9.
- **Scope:** read-only review of the fix commit (5 code files: `build_stage4_gprs_artifacts.py`, `run_stage4_steering.py`, `gprs.py`, `liveness.py`, `test_stage4_gprs_liveness.py`). No code edited. pytest/yaml/numpy/torch are unavailable in this sandbox (no network), so — as in Round 9 — I **executed the exact test bodies and gate functions directly** against `src/` with tmpdirs and faithful config dicts. Everything marked "run-verified" below is an actual execution result, not a reading.

---

## 1. Executive summary

Four of the five items are genuinely fixed and I could not construct a wrong green through any of them: the artifact builder now accepts the certified single-position champion probe and refuses wrong-layer/foreign-position/superset probes (NEW-H1 closed, with `probe_metadata` stamped into the manifest); an assertion-only liveness report is now refused as `incomplete`, sub-threshold `injection_gain` metrics override asserted greens, and the positive-control sub-report is held to the same completeness rules (NEW-M1 closed as scoped); the 8B two-field `positive_control_status` block now binds at **gate** time, not just plan time — run-verified blocked even with a fully green hand-written report and even with the control-model env var set (NEW-M2 closed); and readiness now re-reads the live Stage3 evidence report and cross-checks manifest `layer`/`positions` against the current steering config, refusing with the right fail-closed keys in all seven divergence cases I threw at it (NEW-M3 closed).

**But NEW-H2 is not closed.** The commit fixed the artifact-status test I named — and broke the *other* test in the same file. `test_liveness_gate_status_reads_report_and_fails_closed` was rewritten to use metrics-bearing reports (0.30/6.0, above the 0.25/5.0 floors), which produce decision `green`; line 92 still asserts `ready is False` under `allow_yellow=False`, but green is ready regardless of `allow_yellow`. **Run-verified: the assertion fails.** Pod pytest — your own land gate — is red for the second consecutive commit, and for the same root cause: the suite was never executed before committing (the request's local validation lists `py_compile` and direct artifact-status assertions, which is exactly the blind spot).

The fix is ~6 lines of test-only change (prescribed exactly in §3). No gate logic needs to move. Verdict at the end: **NEEDS FIXES** — solely because the deliverable this pass was asked for, a green suite, is verifiably not delivered; the scaffolding itself is land-grade.

---

## 2. Fix scoreboard

| ID | Claim | Status | Evidence |
|---|---|---|---|
| NEW-H1 | Probe position check accepts `probe_positions ⊆ target_positions`; manifest stamps `probe_metadata` | **CLOSED** | `build_stage4_gprs_artifacts.py:117-120` now `issubset`. Run-verified matrix (logic replicated exactly): champion single-cell `positions:["pause_0"]`/`["pause_1"]` @ layer 14 → **accept**; pooled 3-position probe → accept (equality ⊆); champion at layer 21 vs steering layer 14 → refuse; `post_pause_1` → refuse; 3-position probe vs narrowed `[pause_0]` targets → refuse. Producer payloads re-confirmed: single-scan champion stamps `positions:[spec.position]`, `layers:[int(spec.layer)]`, `threshold`, `scaler` (`run_position_scan_batched.py:464-470`); pooled `train_probe.py:1052-1058` multi-position. `probe_metadata` (source/layers/positions/threshold) written at `:134-139`; the conditional expressions are NameError-safe when `probe_source is None` (run-verified — Python never evaluates the true-branch name when the condition is false). Manifest top-level `layer`/`positions` come from the **build inputs** (`**meta`), so a consistent champion build passes the new readiness cross-check while `probe_metadata.positions` records the subset caveat. Residuals in §4 (R10-2). |
| NEW-H2 | Test suite updated for the manifest requirement; four artifact states covered | **NOT CLOSED — suite still red, run-verified** | The named test is fixed and passes: `test_gprs_artifact_status_requires_all_artifacts` now asserts missing artifacts → `missing artifact_manifest` + `FileNotFoundError` → failing manifest → `missing == ["stage3_evidence_pass"]` → pass/pass manifest → ready (`tests/test_stage4_gprs_liveness.py:132-172`; all four states run-verified PASS). But `test_liveness_gate_status_reads_report_and_fails_closed:83-92` was rewritten with metrics `0.30/6.0` ≥ floors `0.25/5.0` → `_metric_status` returns `green` → decision `green` → `allowed={"green"}` even with `allow_yellow=False` → `ready=True`, while line 92 asserts `is False`. Run-verified failure: `allow_yellow=False gave {'ready': True, 'decision': 'green', … 'model_matches': True, 'positive_control_ready': True}`. Every other non-torch test in the file passes; the torch-gated projection tests are untouched from previously-green rounds. Pod pytest = exactly one failure, deterministically. |
| NEW-M1 | Missing `metrics.injection_gain` ⇒ `incomplete`; assertion-only greens can't open the gate; PC sub-report held to required tests | **CLOSED (as scoped)** | `liveness.py:107-120` + `:189-193`. Run-verified matrix with `required_tests` = all four: assertion-only all-green, bare `{"decision":"green"}`, partial metrics (missing `pause_vs_bos_gain`), empty `metrics.injection_gain: {}`, non-dict `metrics.injection_gain: "green"` → all **incomplete**; sub-threshold 0.10 vs 0.25 with all-green assertions → **red** (metrics override); asserted `"incomplete"`/unknown strings on other tests → incomplete/unknown → not in `allowed` → closed. PC: bare `{"decision":"green"}` sub-report → gate **not ready** (run-verified); a green PC now needs the full battery shape incl. real injection_gain numbers. Remaining accepted path: metrics-green injection_gain + *asserted* greens for the three kernel-less tests → green — correct per scope; the gate stays partially assertion-based until the other kernels have thresholds (standing completion blocker, unchanged). Nit: the rule is keyed on the literal `test == "injection_gain"` (R10-4). |
| NEW-M2 | Configured `positive_control_status` starting missing/invalid forces gate not-ready | **CLOSED** | `liveness.py:186-196`. Key path verified against the real configs: `liveness.controls.positive_control_status` → `liveness_config` (`:37-40`) → `expected.get("positive_control_status")`. Run-verified: 8B-like config (`missing_required_full_sft_pause_control`) + fully-green report incl. green PC → `ready=False, positive_control_ready=False`; **still blocked when `positive_control_model` is set but the status field is unchanged** — the two-field unblock is now enforced at the gate, mirroring the plan path. 1.5B unharmed: `positive_control_model` is configured (`stage4_pause_gprs.yaml:23`, the genuine full-SFT checkpoint-250) → status defaults `configured` → not blocked. Bonus strictness: a config with **no** controls block defaults to status `missing` → gate blocked (run-verified) — a liveness gate without any configured positive control can never open, which is methodologically right (thresholds are calibrated on the control). Residual: `require_positive_control_green: false` in config skips both checks — explicit reviewed-file opt-out, LOW. |
| NEW-M3 | Readiness re-reads live evidence; manifest layer/positions checked vs steering config; fail-closed divergence keys | **CLOSED** | `gprs.py:87-105`. Run-verified matrix (config: layer 14, 3 targets, `stage3_evidence_report` set): consistent build + live pass → ready; live report flips to fail → `stage3_evidence_stale`; live report deleted → `stage3_evidence_live_missing`; manifest layer 16 vs config 14 → `steering_config_mismatch`; manifest positions `[pause_0]` vs 3 targets → mismatch (this is exactly the `--positions pause_0` workaround hazard from Round 9 — now caught); config layer edited 14→16 **after** build → mismatch; manifest lacking `layer`/`positions` keys entirely (hand-written/legacy) → mismatch (fail-closed default `-1` ≠ configured layer). Both real configs set `stage3_evidence_report` (`stage4_pause_gprs.yaml:51`, `_8b_4xa100.yaml:33`), `layer` (14/20) and inherit/set `target_positions`, so none of the checks are vacuous in practice. Malformed live JSON ⇒ `read_json` raises = fail-closed crash. Residuals: config *without* the `stage3_evidence_report` key skips the live re-read (run-verified; make it required for gprs method — R10-5); report identity is path-based, not hash-based (both build-time snapshot and live config-named report must pass, so the practical hole is small). |

**Consumer wiring re-checked:** `require_gprs_readiness` (`run_stage4_steering.py:231-248`) is the only gate consumer; `allow_yellow` comes from `gate.allow_yellow_for_gprs`; the refusal message now dumps the full status dict (useful). Even a fully-green gate still hard-exits at `run_stage4_steering.py:331-337` ("GPRS generation is scaffolded but not wired") — generation cannot run regardless. Ingest (`run_stage4_liveness.py:63-67`) shares `liveness_decision` + required tests, so assertion-only batteries now normalize to `decision=incomplete` at ingest too (matches the request's local validation claim). Learned-delta double-opt-in unchanged. Grep confirms no other consumers with stale expectations.

---

## 3. The one blocking finding

**R10-1 (HIGH, pre-land): `test_liveness_gate_status_reads_report_and_fails_closed` is guaranteed red — the pod-pytest land gate cannot pass, again.**

Mechanics: the fixture at `tests/test_stage4_gprs_liveness.py:83-90` was changed from an asserted `"injection_gain":"yellow"` (which used to make `allow_yellow` discriminative) to metrics `{"pause_vs_content_gain":0.30,"pause_vs_bos_gain":6.0}`. Under the new `_metric_status` these are ≥ the 0.25/5.0 floors ⇒ `green`. Line 91 (`ready is True`) passes; line 92 (`allow_yellow=False ⇒ ready is False`) now fails because green is in the allowed set with or without yellow. Note the deeper cause: **for a `tests: [injection_gain]` config, `yellow` is no longer a reachable decision at all** — metrics-derived injection_gain status is green/red/incomplete only, and a missing-metrics fallback is forced to `incomplete`. The old assertion's semantics silently ceased to exist.

Prescribed fix (test-only, pick one):

- *Minimal (~4 lines):* replace the line-92 case with a sub-threshold red fixture and assert not-ready:
  ```python
  path.write_text(
      '{"model_under_test":"stage2-kl",'
      '"metrics":{"injection_gain":{"pause_vs_content_gain":0.10,"pause_vs_bos_gain":6.0}},'
      '"positive_control":{"metrics":{"injection_gain":{"pause_vs_content_gain":0.30,"pause_vs_bos_gain":6.0}}}}\n', ...)
  assert liveness_gate_status(config, base_dir=tmp_path)["ready"] is False  # red overrides
  ```
- *Better (~8 lines):* keep an honest yellow path so `allow_yellow` stays covered: set `config["liveness"]["tests"] = ["injection_gain", "attention_mass"]`, report = metrics-green injection_gain + `"test_status":{"attention_mass":"yellow"}`, PC = metrics-green + `"test_status":{"attention_mass":"green"}` ⇒ decision `yellow`; assert ready True default / False with `allow_yellow=False`. (Remember the model-mismatch fixture at `:94-102` then also needs the `attention_mass` status, or keep it on the single-test config.)

Then **run pytest in the pod image before landing**. This is the second consecutive commit whose stated land gate is red at commit time; both times the local validation ran everything *except* pytest. Cheap structural remedy: add a stdlib-only smoke runner (the gate modules import nothing beyond stdlib — my driver script proves the whole non-torch test file runs fine without pytest) or make pod pytest part of the commit routine for `tests/`-touching changes.

---

## 4. Other findings from this pass

No blocker. No wrong-green path: every construction I attempted against the builder check, readiness, decision logic, and gate refused or crashed closed (§2 matrices).

- **R10-2 (MEDIUM).** The probe compatibility check and real `probe_metadata` only fire when `--probe_checkpoint_source` is passed. If the probe already sits at the configured target (`build_stage4_gprs_artifacts.py:123` branch), there is no `torch.load`, no layer/position check, and the manifest stamps `layers: [], positions: [], threshold: None` with `source == target`. The 1.5B config's default `probe_checkpoint` points at the *pooled* Stage3 probe dir (`stage4_pause_gprs.yaml:49`) — so the Round-9 concern (an uncertified pooled probe silently becoming the runtime gate) survives on the no-source path, albeit now *visibly* (empty metadata is auditable, and the hook doesn't exist yet, so nothing consumes the probe today). Fix when convenient, before the hook lands: `torch.load` the target on that branch and run the same checks; better, revive L-3 and have the evidence report emit the winning cell's `probe.pt` path for the builder to consume. Also carry R9-T6: `scaler` is still not stamped (threshold now is — partial progress on M-2), and `gate_threshold: 0.95` remains an unrelated constant.
- **R10-3 (LOW).** Zero test coverage for the new NEW-M3 branches: the passing-manifest test exercises the cross-checks only vacuously (its config has no `steering.layer`, no `target_positions`, no `stage3_evidence_report`, so the layer check falls back to the manifest's own value and the live re-read is skipped). Add three fixtures: live-fail → `stage3_evidence_stale`; live-deleted → `stage3_evidence_live_missing`; manifest/config layer or positions divergence → `steering_config_mismatch`. My §2 matrix is a ready-made spec.
- **R10-4 (LOW).** The metrics-required rule is keyed on the literal `test == "injection_gain"` (`liveness.py:108`). When the other three kernels get thresholds, someone must remember to extend the special case or they'll silently stay assertion-satisfiable. Key it on "the gate defines thresholds for this test" instead.
- **R10-5 (LOW).** `validate_gprs_config` does not require `stage3_evidence_report` for gprs/projection methods, and readiness skips the live re-read when the key is absent (run-verified). Both shipped configs set it, but making it required closes the config-regression hole for free.
- **R10-6 (LOW, carried cosmetics).** `require_gprs_artifacts` renders the new missing keys as `stage3_evidence_stale=<manifest path>` etc. (key/value mismatch, `gprs.py:125-130`, no crash); `require_positive_control_green: false` skips the configured-status block too; ingest still doesn't validate model binding/PC at write time (L-A); `read_json`/threshold-constant duplication (L-B); surviving prior lows L-2/L-4/L-6 unchanged.

---

## 5. Answers to the three questions

**Q1 — Are NEW-H1 and NEW-H2 closed?**
NEW-H1: **yes.** The subset rule admits exactly the certified single-position champion and still refuses everything that should be refused (wrong layer, foreign positions, superset-vs-narrowed) — run-verified against both producers' payload shapes — and the manifest now records the probe's provenance and the subset caveat. The `--positions pause_0` workaround hazard I flagged is independently caught by the new `steering_config_mismatch` readiness check.
NEW-H2: **no.** The artifact-status test is fixed and all four of its states pass, but the same commit made `test_liveness_gate_status_reads_report_and_fails_closed` guaranteed-red at line 92 (green decision asserted not-ready under `allow_yellow=False`). Run-verified. The suite the land gate depends on has now been red in two consecutive commits, both times because pytest wasn't executed locally.

**Q2 — Did the NEW-M1/M2/M3 fixes introduce any blocker/high/medium issue?**
In the gate logic itself: **no** — every added branch is strictly tightening, and all my adversarial constructions failed closed (assertion-only, partial-metrics, empty-metrics, non-dict-metrics, bare-decision PC, stale/deleted live report, layer/position drift both directions, hand-written manifests missing keys, env-set control model with unchanged status field). Two behaviors are *stricter than documented* and both are correct: a config with no positive-control block at all now defaults to blocked, and manifests lacking `layer`/`positions` are refused. The one HIGH this pass introduced (R10-1) lives in the test fixture rewrite, not in gate behavior; the one MEDIUM (R10-2) is a coverage boundary of the NEW-H1 fix (no-source builds skip the new check), not a regression.

**Q3 — Safe to land into main as fail-closed scaffolding?**
The scaffolding itself: **yes.** Run-verified end-to-end: liveness gate → artifact readiness → `require_gprs_readiness` → and even a hypothetically green gate still hard-exits before generation because the GPRS hook intentionally doesn't exist. There is no path from a hand-written or stale artifact to a running Stage4 eval, and the 8B battery is now double-locked (plan and gate). The *land act* is blocked by exactly one thing: your land gate is pod pytest, and the committed suite verifiably cannot pass it (R10-1, ~6 test-only lines). Fix that, run pytest in the pod image, land. The larger completion blockers are unchanged and correctly tracked separately: liveness kernels for the other three tests (+ R10-4), the on-policy within-prompt AUROC endpoint with CI, the genuine 8B full-SFT control, the GPRS hook with calibrated threshold+scaler (R9-T6/R10-2), the on-policy direction + random-direction control arm, and S4-5 judge endpoint separation.

---

## 6. Go / No-Go table

| Item | Verdict | Gate to upgrade |
|---|---|---|
| Land `2b472ff` into main cot-safety | **NO — one red test** | R10-1 (~6-line test fix) + pod pytest green; nothing else blocks |
| NEW-H1 champion-probe artifact build flow | **GO** (was DOA) | run-verified accept/refuse matrix; R10-2 before the hook consumes the probe |
| NEW-M1/M2/M3 hardening | **GO — sound** | all divergence/assertion paths refuse, run-verified; add R10-3 tests next pass |
| Stage4 liveness gate as *real* gate | **NO** (unchanged) | injection_gain kernel exists in schema only; other three tests assertion-based until kernels + thresholds (R10-4) |
| Stage4 GPRS eval (1.5B) | **NO-GO** (correctly hard-blocked) | liveness green + hook + on-policy direction QC + S4-5 |
| Anything 8B | **NO-GO** (now double-locked, plan **and** gate) | genuine full-SFT positive control; flip both config fields only after verification |

---

## 7. Concrete TODOs

| ID | Sev | Change | Where |
|---|---|---|---|
| R10-1 | HIGH (pre-land) | Fix the `allow_yellow=False` assertion: sub-threshold red fixture, or two-test config with asserted `attention_mass: yellow` to keep `allow_yellow` covered; then run pod pytest before land | `tests/test_stage4_gprs_liveness.py:83-102` |
| R10-2 | MED | Run the layer/position check + metadata stamping on the no-source branch too (`torch.load(probe_target)`); prefer emitting the winning cell's probe path from the evidence report (L-3) and consuming it; stamp `scaler` presence (R9-T6) | `build_stage4_gprs_artifacts.py:108-139`, `stage3_evidence.py` |
| R10-3 | LOW | Non-vacuous readiness tests: `stage3_evidence_stale`, `stage3_evidence_live_missing`, `steering_config_mismatch` (config with layer/targets/report set) | `tests/test_stage4_gprs_liveness.py` |
| R10-4 | LOW | Key metrics-required on gate-threshold presence per test, not the `"injection_gain"` literal | `liveness.py:107-111` |
| R10-5 | LOW | Require `stage3_evidence_report` in `validate_gprs_config` for gprs/projection | `gprs.py:15-43` |
| R10-6 | LOW | Cosmetics/carried: missing-key error rendering; `require_positive_control_green:false` also skips the configured-status block (document or bind); ingest-time model/PC validation (L-A); dedupe `read_json`/threshold constants (L-B); L-2/L-4/L-6 | `gprs.py:122-132`, `liveness.py:186-196`, `run_stage4_liveness.py` |

Standing completion blockers (unchanged order): liveness kernels → on-policy Stage3 endpoint + CI → 8B positive control → on-policy direction QC + random-direction arm → GPRS hook with probe threshold+scaler → S4-5 judge separation.

---

## 8. Headline verdict

**NEEDS FIXES** — one item, test-only, ~6 lines. The substance of this pass is good: NEW-H1 is closed exactly as specified and run-verified against both probe producers; NEW-M1/M2/M3 are closed with uniformly fail-closed behavior (assertion-only batteries refused, the 8B two-field block now enforced at the gate even against a fully green fabricated report, staleness and config drift both caught); and no construction I attempted produces a wrong green anywhere in the builder → readiness → gate → steering chain. But the single named acceptance criterion for NEW-H2 — a test suite that can pass pod pytest — is still unmet: the fix repaired one test and broke the other in the same file (line 92 asserts a green gate is not ready; run-verified failure). Apply R10-1, actually run pytest in the pod image, and this lands as safe fail-closed scaffolding with no further code changes; everything else in §7 is next-pass hardening on the already-tracked completion path.

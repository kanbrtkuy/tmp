# Fable Review — Stage3 Evidence Report + Stage4 Liveness/GPRS Artifact Gates (Completion-Gap Pass)

- **Date:** 2026-07-04
- **Tree:** `safechain_stage2_to_stage4_full_code_20260704/cot-safety` @ `e8a07c8` ("Add Stage3 evidence and Stage4 GPRS gates")
- **Prior state:** Round 6–7 (`10b9466`, `30a5194`) = PASS to land for Stage2 1.5B; open items were Stage3 within-prompt endpoint, Stage4 liveness kernels, 8B positive control, GPRS artifact producers, gate_threshold wiring, judge endpoint separation (S4-5).
- **Scope of this pass:** `stage3_evidence.py`, `run_stage3_evidence_report.py`, `gprs.py` (+artifact status/gate), `liveness.py` (+report path/read/gate), `run_stage4_steering.py` gates, `run_stage4_liveness.py` metrics ingest, `build_stage4_gprs_artifacts.py`, pipeline steps, 2 config keys, 2 test files. Read-only review; no code edited.

---

## 1. Executive summary

This pass is a real step forward and everything new is **fail-closed**: I traced every Stage4 run phase and could not find a path that silently runs GPRS, and the Stage3 report refuses (`missing_*` statuses) rather than passing when a baseline group is absent. The learned-delta double-opt-in from Round 6 is intact, the 4xa100 Stage4 shell now defaults to the gated GPRS runner with `PHASE=validate`, `runs/` and `*.pt` are gitignored so no synthetic green liveness report can land via git, and the 8B two-field positive-control block still holds (verified in the actual `liveness_plan.json`: `blocked_missing_positive_control`).

But four HIGH findings keep this from being "Stage3/4 gates done":

1. The evidence report's **default summary path never matches where Stage3 actually writes** (repo-root vs legacy-cwd) — the shipped pipeline step crashes out of the box (H-1).
2. **Nothing consumes the Stage3 evidence verdict.** A `fail_no_independent_pause_signal` report does not stop `build_stage4_gprs_artifacts.py` from building the direction and copying the very probe that just failed the independence screen (H-2).
3. The **liveness gate is a file-presence gate, not a measurement gate**: any JSON with `"decision": "green"` opens it. No required-test completeness, no positive-control enforcement at gate time, no checkpoint provenance binding; the configured gate thresholds (`min_pause_vs_content_gain`, `min_pause_vs_bos_gain`) are consumed by nothing (H-3).
4. The screen margin is computed on **max-test-AUROC-selected cells** (36 pause/post cells vs 24 baseline cells) with a 0.01 margin floor — the pass/fail line sits below selection noise (H-4).

None of these makes landing unsafe (all failure modes are crash-or-refuse, not wrong-green), so the verdict is **PASS with required follow-ups**, with H-1 strongly recommended pre-land because the committed pipeline step is dead on arrival otherwise.

---

## 2. What I verified end-to-end (positive findings)

| # | Claim in request | Verified? | Evidence |
|---|---|---|---|
| V1 | Evidence report computes best(pause/post-pause) − max(prompt, true-control) with fail-closed missing statuses | YES | `stage3_evidence.py:86-117`; ordering: missing pause → missing prompt → missing control → pass/fail |
| V2 | Config inheritance delivers `min_pause_margin_over_baselines` to the kl_transparent variants | YES | `config.py:49-98` is a recursive deep merge; 1.5B variant chains to base; 8B chains via `stage3_intra_pause_probe_8b_4xa100.yaml` → base |
| V3 | Summary schema matches the reader | YES | Single scan writes `summary_grid.json` rows `{model, position, layer, val_auroc, test_auroc, threshold, ...}` (`run_position_scan_pilot.py:219-271`, called by `run_position_scan_batched.py:694-698`) |
| V4 | True no-pause controls + prompt baselines actually exist in the extraction the Stage3 runner invokes | YES | `run_intra_pause_probe_full.py:419-421` passes `--matched_control_file` unconditionally and `--prompt_positions` when configured; `extract_hidden_states.py:813-838` runs the row-matched pause-free second forward filling `control_cot_3/4` |
| V5 | GPRS run phases refuse without liveness + artifacts, and refuse even when both are ready (hook absent) | YES | `run_stage4_steering.py:331-337`: `require_gprs_readiness` then unconditional `SystemExit`; `--dry_run` does NOT bypass (readiness check precedes the dry-run branch) |
| V6 | `validate` phase prints `gprs_artifacts` + `liveness_gate` | YES | `run_stage4_steering.py:304-320` |
| V7 | learned-delta stays double-opt-in | YES | `run_stage4_steering.py:287-294` unchanged behavior |
| V8 | Ops shell cannot bypass the gate | YES | `pipelines/run_4xa100_stage4_steering_eval.sh` routes through `run_stage4_steering.py`, defaults `CONFIG=stage4_pause_gprs_8b_4xa100.yaml`, `PHASE=validate` |
| V9 | NPZ layout matches the builder's indexing | YES | `extract_hidden_states.py:883-898` saves `features[N,L,P,D]`, `labels`, `layer_ids`, `position_names` (+`valid_mask`, see M-1); float16 save → float32 load handled |
| V10 | A Stage3 probe-checkpoint producer exists for `--probe_checkpoint_source` | YES | per-cell `probe.pt` at `single_scan_out_root/<model>_<pos>_l<layer>/probe.pt` with `state_dict`, `scaler`, calibrated `threshold` (`run_position_scan_batched.py:457-472`) |
| V11 | 8B liveness stays blocked pending genuine full-SFT control | YES | config two-field block + observed `liveness_plan.json` status `blocked_missing_positive_control` |
| V12 | No synthetic green liveness report can land via git | YES | `.gitignore` covers `runs/`, `*.pt`, `*.npz`; `git ls-files` shows no tracked liveness reports |
| V13 | `projection_rejection_update` math | YES | one-sided rejection (clamp_min 0), relative norm cap vs ‖h‖, gate broadcast `(B,)→(B,1)`, gate=score≥threshold steers; unit tests cover all three behaviors |
| V14 | Liveness ordering is runnable pre-artifacts | YES | phase `liveness` exempt from artifact readiness (`run_stage4_steering.py:331`) — correct: battery must be runnable before GPRS artifacts exist |

Also confirmed: `pipeline plan` includes `stage3_pause_vs_baselines_report` and `build_gprs_artifacts` for the right config shapes (`pipeline.py:141-156, 189-230`). Note the pipeline is **plan-only** (`cli.py` has no `pipeline run`), so these steps are documentation; the only enforcement points are inside `run_stage4_steering.py`.

---

## 3. Findings

No blocker-severity correctness bug (nothing produces a wrong green). Severity below reflects what must be fixed before the gates can be *trusted*, not merely landed.

### HIGH

**H-1. Evidence report default path can never find the Stage3 summary — the shipped pipeline step crashes.**
`run_stage3_intra_pause_probe.py:278` runs the legacy full runner with `cwd=legacy/PauseProbe` and passes `single_scan_out_root` verbatim (e.g. `runs/probes/stage3_kl_transparent_1p5b_cot3_single` from the variant's `legacy:` block). All relative out-roots therefore materialize under `legacy/PauseProbe/runs/...` (`legacy/PauseProbe/runs` is a real directory, not a symlink — checked). But `run_stage3_evidence_report.py:33-35` resolves the same relative string against `REPO_ROOT` → `cot-safety/runs/probes/.../summary_grid.json`, which will not exist. The pipeline-planned command (`--config` only, no `--summary`) throws `FileNotFoundError` on any pod that ran Stage3 the intended way. Your local validation passed because you supplied a fixture via `--summary`, which is exactly why this wasn't caught.
*Fix:* resolve a relative summary path against `legacy_root` (mirroring the runner's cwd convention), or try `REPO_ROOT / p` then `REPO_ROOT / "legacy/PauseProbe" / p`, or make the `legacy:` path blocks env-prefixed absolute. Fail-closed today, but the step is DOA — fix before land or immediately after.

**H-2. The Stage3 evidence verdict has zero consumers — Stage4 artifact building ignores it.**
Grep confirms `stage3_evidence` appears only in its own module, its script, the pipeline note, and its test. Neither `build_stage4_gprs_artifacts.py` nor `require_gprs_readiness()` reads `stage3_evidence_report.json`. Consequence: with a report saying `fail_no_independent_pause_signal`, the builder will still happily construct the mean-diff direction from those same hidden states and copy the same probe that just failed the independence screen as the GPRS gate — i.e., Stage4 would be gated by a certified prompt-risk reader. The whole point of this round was that pause signal must beat baselines *before* Stage4 consumes it.
*Fix:* `build_stage4_gprs_artifacts.py` should require an existing evidence report with `status: pass` (record its path+status+margin in the manifest; allow `--force` only with an explicit manifest stamp), and `require_gprs_readiness()` should check the manifest's recorded evidence status.

**H-3. The liveness gate verifies a file, not a battery.**
`liveness_gate_status()` (`liveness.py:113-130`) opens the gate for any JSON whose `decision` normalizes to green/yellow. Three missing checks:
  - **Completeness:** `liveness_decision()` returns green if `test_status` is all-green *for whatever subset is present* — a report containing only `injection_gain: green` passes despite config requiring 4 tests (`liveness.tests`). Nothing cross-checks report tests vs configured tests.
  - **Positive control:** `require_positive_control_green` is enforced only in `liveness_plan_status()` (the *plan* path). The gate and the `--metrics_json` ingest never check that the full-SFT positive control ran and was green. The 8B two-field block is therefore bypassable by hand-writing a green report — the exact failure mode the two-field design was meant to prevent.
  - **Provenance:** nothing binds `liveness_report.json` to the checkpoint under test. A report produced against a different checkpoint (or an older Stage2 run) satisfies the gate. The ingest embeds `liveness_config(config)` at write time but never validates that the *metrics* JSON's model matches `model_under_test`.
  Plus the configured thresholds `min_pause_vs_content_gain: 0.25` / `min_pause_vs_bos_gain: 5.0` are consumed by no code — decision-making is fully delegated to whatever produced the metrics JSON, which today is nothing (kernels unimplemented). Acknowledged scaffolding, but as long as this is true, "liveness gate green" is an assertion, not a measurement.
*Fix:* at ingest and at gate time require `test_status` ⊇ configured `tests`, require `positive_control` result green when `require_positive_control_green`, store and verify `model_under_test` + checkpoint path/hash, and make the (future) kernels emit per-test gains that `liveness_decision` compares against the configured thresholds rather than trusting an upstream `decision` string.

**H-4. Screen margin is computed on test-set-selected maxima; 0.01 floor is below selection noise.**
`best_row()` picks the max **test_auroc** cell within each group: pause+post-pause = up to 36 (position,layer) cells vs prompt = 12 and control = 12. Maximizing the metric you then report (a) inflates the pause side more than the baseline side simply from more draws, and (b) is test-set selection, the thing the val split exists to prevent. With `min_pause_margin_over_baselines: 0.01`, a "pass" is indistinguishable from max-of-36 vs max-of-24 noise at n≈test-split size. The rows already carry `val_auroc` — select each group's champion on val, then compute the margin on the champions' test scores; better, bootstrap a CI on the test margin and require CI-low > 0. Until then, treat `status: pass` as "worth running the confirmatory endpoint", never as reportable evidence. (The report's own note says teacher-forced is only a screen — good — but the screen should at least not be biased toward passing.)

### MEDIUM

**M-1. GPRS builder ignores `valid_mask` — zero-vector contamination of direction/centroid.**
`extract_hidden_states.py` initializes features to zeros and only fills positions that exist (`valid_mask` saved at :886). `select_state_block()` (`build_stage4_gprs_artifacts.py:27-56`) filters `labels >= 0` but never touches `valid_mask`; rows whose pause positions were dropped (e.g., 4096-token truncation) contribute zero vectors to `mean(axis=1)` and the class centroids, shrinking and rotating the direction silently. Require all selected positions valid per row (or mean over valid positions only) and record `n_dropped_invalid` in the manifest.

**M-2. `gate_threshold` provenance is not wired — the "validated threshold" is a hardcoded 0.95.**
`probe.pt` carries the max-FPR-calibrated `threshold` and the `scaler` (mean/std) the probe was trained under (`run_position_scan_batched.py:457-472`). The config's `gprs.gate_threshold: 0.95` is an unrelated constant, and the builder neither extracts the probe's threshold into the manifest nor records that gate scoring must standardize `h` with the probe's scaler before applying `state_dict`. The request's claim "the validated threshold is actually used" is true of the *mechanism* (`projection_rejection_update` accepts it), not of the *value*. Default the gate threshold to the probe payload's `threshold`; treat config override as explicit and manifest-stamped.

**M-3. Builder copies the probe blind, and the random-direction control artifact doesn't exist.**
No `torch.load` sanity check on `--probe_checkpoint_source`, no verification that the probe's `layers`/`positions` match `steering.layer`/`target_positions` (you can copy a layer-21 post_pause probe into a layer-14 pause-steering config and every downstream gate will report "ready"). Manifest records no probe metadata and no direction↔probe-weight cosine QC. Also `gprs.random_direction_control: true` remains an inert key — the plan's required random-direction control has no artifact producer or eval arm. Validate the payload, cross-check layer/positions, record probe metadata + cos(direction, probe weights) in the manifest, and emit a `random_direction.pt` sibling artifact.

**M-4. The first GPRS direction is a teacher-forced, prompt-label mean-diff — stamp it as such (this is the Q3 answer).**
Labels in the NPZ are prompt/dataset-level (`ex["label"]` → safe/unsafe), states are teacher-forced. So `unsafe_centroid − safe_centroid` at pause positions is, to first order, the *prompt-risk* direction as read through pause states — precisely the confound Rounds 5–7 established for Stage3 probes, and the reason the plan demands an on-policy contrastive direction with QC before any pilot. Acceptable as pilot scaffolding only. The manifest must stamp `direction_provenance: teacher_forced_prompt_labels`, `split: train`, and downstream readiness should distinguish "artifacts exist" from "artifacts are on-policy-validated". Nothing conflicts with fail-closed Stage4 *today* (eval is hard-blocked), but do not let this artifact silently become "the" direction when the hook lands.

**M-5. `status: pass` can be driven entirely by post-pause cells while steering targets pause_0..2.**
`best_main = max(pause, post_pause)`. A pass where `best.post_pause` wins but `best.pause` ≤ baselines does not support steering *at pause positions* (post-pause tokens are content tokens; their edge over `control_cot_*` is the matched comparison, fine, but it's not the Stage4 carrier). The report exposes both compact rows (good); add a `pause_only_margin` field and have the Stage4 consumer (per H-2) gate on the pause-specific margin, with post-pause as supporting diagnostics.

### LOW

- **L-1.** `run_single_scan` hardcodes `--model_kinds linear` (`run_intra_pause_probe_full.py:593-594`) while config declares `[linear, mlp]` — the `model` column in the evidence report will only ever contain `linear`; config key partially inert (pre-existing).
- **L-2.** `run_stage4_steering.py:287` requires the learned-delta acknowledgement for `--phase liveness` too (phase set is `not in {"validate"}`) — running the battery under a learned_delta config demands a deprecation opt-in it shouldn't need. Cosmetic friction.
- **L-3.** The evidence report should emit the winning cell's `probe.pt` path (`<model>_<position>_l<layer>` is derivable) so `build_gprs_artifacts` consumes the *selected* cell instead of a hand-typed path — closes an easy wrong-cell copy mistake and dovetails with H-2/M-3.
- **L-4.** `probe.true_content_controls.current_control_cot_aliases_valid: false` in `stage3_intra_pause_probe.yaml` is stale (the alias was fixed in `10b9466`; V4 confirms true controls) and inert — misleading to readers; update or delete.
- **L-5.** Test gaps: no test for `missing_true_content_control` / `missing_prompt_baseline` statuses, the TSV loader, post-pause-driven pass, or liveness completeness (once H-3 lands). `pytest` was not executed locally — must run in the pod image before land, per standing practice.
- **L-6.** `confirmatory_endpoint.status` is echoed from config, so a one-line YAML edit could flip the headline honesty marker to "pass" with nothing implemented. Nothing consumes it today; still, hardcode `not_implemented` in code until the on-policy runner exists, then have the runner write the status.
- **L-7.** If `liveness.report_json` is ever configured, the gate reads that path while `--metrics_json` ingest writes `output_dir/liveness_report.json` — divergence potential; have the ingest default to `liveness_report_path(config)`.

---

## 4. Answers to the five questions

**Q1 — Does this correctly address "pause signal beyond prompt baselines and true no-pause controls"?**
The *shape* is right: margin against max(prompt, true-control), fail-closed missing statuses, honest `not_implemented` confirmatory endpoint, and the position groups bind to the genuinely fixed controls (V4). It is not yet a trustworthy gate: the default path never finds the real summary (H-1), the margin is test-set-selected with a sub-noise floor (H-4), a pass can be post-pause-driven (M-5), and nothing downstream reacts to the verdict (H-2). So: framework yes, evidence instrument not yet.

**Q2 — Is Stage4 hardened so GPRS can't run without liveness green/yellow + artifacts?**
Mechanically yes, and verified beyond the runner: every run phase past validate/liveness hits `require_gprs_readiness` and then a hard refusal even when ready (hook absent); dry-run doesn't bypass; the ops shell routes through the gated runner; nothing green is committable via git. Substantively, the liveness half of the gate is thin (H-3): green is currently whatever a JSON file says, the configured thresholds bind to nothing, and positive-control/completeness/provenance aren't enforced at gate time. The gate's skeleton is done; its teeth are the unimplemented kernels plus the H-3 validations.

**Q3 — Is `build_stage4_gprs_artifacts.py` conceptually correct, or does it bake in a teacher-forced/off-policy artifact?**
It bakes in exactly that: a teacher-forced, prompt-label mean-diff — the prompt-risk direction the project's own prior rounds identified as the confound (M-4), with two mechanical defects on top (valid_mask M-1, blind probe copy M-3) and no random-direction sibling despite the config key. As the *first producer* for pilot/QC plumbing it is acceptable and correctly ordered (direction from train split, probe required rather than fabricated), but it must stamp provenance, and on-policy direction + direction QC + random-direction control remain prerequisites for any Stage4 evidence claim. It does not contaminate anything today because GPRS eval is hard-blocked.

**Q4 — What remains blocker/high/medium before Stage3/4 are complete?**
Completion blockers (unchanged from the standing plan, none introduced here):
1. **Liveness kernels** (injection_gain, attention_mass, pause_kv_ablation, safe_unsafe_patching) + threshold calibration on the full-SFT positive control — the hard gate for any Stage4 on the KL-transparent checkpoint.
2. **On-policy within-prompt AUROC runner** (10 samples/prompt, mixed-outcome prompts, CI excludes 0.55) — Stage3's confirmatory endpoint, still `not_implemented`.
3. **Genuine 8B full-SFT positive control** (two-field unblock) — 8B anything stays NO-GO.
4. **GPRS generation hook** ported from the teacher-forced pilot math into on-policy generation — currently a deliberate hard refusal.
5. **S4-5 judge endpoint separation** (CoT judged separately from answer, labeled-only denominators) — biggest evidence-integrity item once anything generates.
This round's own debt: H-1…H-4, then M-1…M-5.

**Q5 — Should anything change before landing into main cot-safety?**
Landing is safe as-is (everything fail-closed; no wrong-green path). I'd fix **H-1 pre-land** — it's a ~5-line path-resolution change and the committed pipeline step is otherwise dead on arrival — and treat H-2/H-3 as required before anyone runs Stage3→Stage4 on a pod, since both are exactly the class of "gate exists but doesn't bind" that earlier rounds caught in learned-delta. H-4/M-1…M-5 belong in the next pass before the first real evidence run. Also run `pytest` in the pod image before land (not executed locally).

---

## 5. Go / No-Go table

| Item | Verdict | Gate to upgrade |
|---|---|---|
| Land `e8a07c8` into main cot-safety | **GO** (fix H-1 pre-land if cheap; pod `pytest` must pass) | — |
| Stage2 1.5B launch | **GO** (unchanged from Round 7) | — |
| Stage3 teacher-forced screen as *runnable step* | **NO** until H-1 | path fix + one real `summary_grid.json` round-trip |
| Stage3 screen as *evidence* | **NO** | H-4 (val-selection + CI) + M-5 pause-only margin |
| Stage3 completion claim | **NO** | on-policy within-prompt AUROC runner, pass per decision rule |
| Stage4 GPRS artifact build | **NO** (don't run yet) | H-2 (evidence consumer) + M-1/M-3/M-4 manifest+QC |
| Stage4 liveness gate as *real gate* | **NO** | kernels + H-3 (completeness, positive control, provenance, thresholds bound) |
| Stage4 GPRS eval (1.5B) | **NO-GO** (correctly hard-blocked in code) | liveness green + Stage3 pass + on-policy direction QC + random-dir control + hook + S4-5 |
| Anything 8B | **NO-GO** | genuine full-SFT positive control (two-field unblock) |

---

## 6. Concrete TODOs

| ID | Sev | Change | Where |
|---|---|---|---|
| T-1 | HIGH | Resolve relative `--summary` against legacy root (or try both roots) | `scripts/run_stage3_evidence_report.py:33-38` |
| T-2 | HIGH | Builder refuses without `stage3_evidence_report.json` `status: pass`; stamp path/status/margin in manifest; gate readiness on manifest field | `scripts/build_stage4_gprs_artifacts.py:build_artifacts`, `src/cot_safety/steering/gprs.py:gprs_artifact_status` |
| T-3 | HIGH | Enforce test-completeness vs `liveness.tests`, positive-control green, and `model_under_test` binding at ingest and gate | `src/cot_safety/steering/liveness.py:liveness_decision/liveness_gate_status`, `scripts/run_stage4_liveness.py:60-76` |
| T-4 | HIGH | Select group champions on `val_auroc`, margin on their `test_auroc`; add bootstrap CI; keep 0.01 as floor on CI-low | `src/cot_safety/probes/stage3_evidence.py:best_row/build_stage3_evidence_report` |
| T-5 | MED | Apply `valid_mask` (all selected positions valid), report drop count | `build_stage4_gprs_artifacts.py:select_state_block` |
| T-6 | MED | Default gate threshold from `probe.pt` payload; record threshold+scaler requirement in manifest | `build_stage4_gprs_artifacts.py`, `gprs.py:validate_gprs_config` |
| T-7 | MED | `torch.load` the probe; assert layer/positions match steering config; manifest probe metadata + direction/probe cosine; emit `random_direction.pt` | `build_stage4_gprs_artifacts.py:build_artifacts` |
| T-8 | MED | Manifest stamps `direction_provenance: teacher_forced_prompt_labels`; add `pause_only_margin` and gate Stage4 on it | `build_stage4_gprs_artifacts.py`, `stage3_evidence.py` |
| T-9 | LOW | Emit winning cell's probe path in the evidence report | `stage3_evidence.py:compact_row` |
| T-10 | LOW | Exempt `--phase liveness` from the learned-delta ack; fix stale `current_control_cot_aliases_valid`; hardcode confirmatory `not_implemented` in code; add missing-status/TSV/completeness tests | `run_stage4_steering.py:287`, `stage3_intra_pause_probe.yaml:37`, `stage3_evidence.py:133-141`, `tests/` |

Then the standing completion blockers in order: liveness kernels → on-policy Stage3 endpoint → on-policy direction QC + random-dir control → GPRS hook → S4-5 judge separation → 8B positive control.

---

## 7. Headline verdict

**PASS with required follow-ups.** Land it: every new path fails closed, prior gates survived end-to-end tracing, and no wrong-green is reachable. Required before first pod use: T-1 (the shipped evidence step cannot find its input), T-2 (the evidence verdict must actually bind Stage4), T-3 (a liveness gate that can be satisfied by a hand-written JSON is not yet a gate), T-4 (a screen that passes on selection noise screens nothing). Stage3/4 remain **incomplete by design** — liveness kernels, the on-policy within-prompt endpoint, the GPRS hook, S4-5, and the 8B positive control are still the real distance to done — and this pass has, correctly, made it impossible to pretend otherwise in code.

# Fable Review — Stage3 On-Policy Confirmatory Endpoint + Stage4 Liveness Kernels (Round 11)

- **Reviewed tree:** `safechain_stage2_to_stage4_full_code_20260704/cot-safety` at `460bae1` (packet commit; code changes land in `2f18f95` test fix + `460bae1` endpoint/kernels).
- **Baseline:** Round 10 review of `2b472ff` (`CLAUDE_FABLE5_STAGE3_STAGE4_NEW_HIGH_FIX_REVIEW.md`, verdict NEEDS FIXES on R10-1).
- **Method:** full read of every added/changed file; consumer/producer tracing across `src/`, `scripts/`, `configs/`, `pipelines/`, `legacy/PauseProbe`; stdlib-only run-verification of the gate logic (`/tmp/fable_r11_driver.py`), the evidence-attach path, and an exact-path recomputation of the new test fixture's control AUROC. This Mac has no numpy/torch/pytest, so the two GPU kernels were desk-checked line-by-line, not executed. `py_compile` over all touched files passes; both confirmatory dry-runs (1.5B, 8B), the liveness dry-run, and `cot_safety.cli pipeline plan` reproduce your claims.

---

## 0. Executive summary

The direction is right and most of the new machinery is sound: the within-prompt AUROC endpoint is the correct instrument for the prompt-classification critique (prompt-constant signal cancels *exactly* in same-prompt pairs, and every misuse mode I could construct fails closed), and `injection_gain` is a legitimately designed first liveness kernel. The runner-path fail-closed properties all re-verified.

Three HIGHs block landing:

1. **The new test file is red.** The "pass" fixture's content control is perfectly label-correlated by construction, so the asserted `status == "pass"` / control AUROC 0.5 are wrong (actual: `fail_no_independent_on_policy_pause_signal`, control AUROC 1.0 — verified by exact-path recomputation). This is the **third consecutive commit** that would arrive at the pod with a red pytest suite, same root cause each time: fixtures written but never executed.
2. **The confirmatory endpoint is not load-bearing.** Nothing requires it: the Stage4 artifact builder and readiness gate check only `status` + `pause_only_status`. Run-verified: attaching a **failing** on-policy report to the evidence report leaves top-level `status: pass` — Stage4 would open anyway. The round's stated evidence standard ("…then confirm on on-policy") exists only as report decoration.
3. **The four-test liveness gate can be opened by assertion.** In required-tests mode, tests without a metrics payload fall back to hand-asserted `test_status` strings. Run-verified: real `injection_gain` metrics + `test_status: {pause_kv_ablation: green, safe_unsafe_patching: green}` ⇒ `liveness_gate_status(...)["ready"] == True` with all four tests required — and `run_stage4_liveness.py --metrics_json` will normalize such a report into an official `liveness_report.json`. This partially reopens the R9 NEW-M1 class that R10 closed only for the all-assertion case.

All three fixes are small. Below: R10-1 verification (§1), new findings (§2), answers to your seven questions (§3), validated-good list (§4), go/no-go (§5), TODOs (§6), standing items (§7).

---

## 1. Prior-round verification: R10-1 (the red gate test)

**CLOSED, run-verified.** `2f18f95` replaced the broken `allow_yellow=False` assertion in `test_liveness_gate_status_reads_report_and_fails_closed` with a sub-threshold red fixture (`pause_vs_content_gain: 0.10` < 0.25 floor ⇒ not ready), exactly one of the two options R10 offered (tests/test_stage4_gprs_liveness.py:93-101). I replicated every assert in that test with a stdlib driver: missing→`missing`, metrics-green→ready, sub-threshold→not ready, wrong model→not ready. All pass.

Residual: the `allow_yellow=False` branch of `liveness_gate_status` now has **zero test coverage** (R10-1 option B was to keep it covered via a 2-test yellow fixture). Recorded as R11-L5.

---

## 2. Findings

### HIGH

**R11-H1 — New on-policy test file is red; land gate (pod pytest) fails again.**
`tests/test_stage3_on_policy_confirmatory.py` builds rows per prompt as `(label 0, label 1)` pairs, so `labels[idx] == idx % 2` identically. The control feature is `features[idx, 0, 1, 1] = float(idx % 2) * 0.01` (line ~24) — i.e., **exactly equal to `0.01 × label`**. The control is therefore a *perfect* within-prompt discriminator, not a null: standardized train stats give direction `[0,1,0]`, control within-prompt AUROC = **1.0** (verified by re-implementing the exact `select_matrix → _standardize → _fit_mean_diff_direction → per_prompt_pair_stats` path in stdlib: direction `[0.0, 1.0, 0.0]`, AUROC 1.0 over 5 pairs). Consequences in `test_on_policy_confirmatory_passes_when_pause_has_within_prompt_signal`:
- `best_baseline = max(0.5, 1.0) = 1.0`, `margin = 1.0 − 1.0 = 0 ≤ 0.01` ⇒ `status = "fail_no_independent_on_policy_pause_signal"` ⇒ the first assert (`status == "pass"`) fails;
- `true_content_control.within_prompt_auroc == 0.5` assert also fails (actual 1.0).

The second test happens to pass (status fails earlier on the CI check, before the margin logic sees the informative control).

*Fix (trivial):* make the control a **prompt-level** nuisance: `features[idx, 0, 1, 1] = float((idx // 2) % 2) * 0.01`. That is constant within each prompt ⇒ within-prompt AUROC exactly 0.5 by the tie rule, and it doubles as a direct demonstration of the prompt-constant-cancellation property the endpoint exists to enforce. Then **run the full pytest suite on the pod before landing.** This is the third consecutive round (R9 NEW-H2 → R10-1 → R11-H1) where the commit ships a red suite because tests are authored but not executed; a numpy-only venv (`python3 -m venv && pip install numpy pytest`) would have caught H1 locally — numpy tests don't need torch.

**R11-H2 — The confirmatory endpoint gates nothing; a failing on-policy report does not block Stage4.**
Traced every consumer of `confirmatory_endpoint` / the on-policy report: attach-point in `stage3_evidence.py:133-163` and two tests. Nothing else.
- `build_stage4_gprs_artifacts.py:84` refuses only on `evidence.status != "pass" or evidence.pause_only_status != "pass"`.
- `gprs.py:81-98` (manifest + live re-read) checks the same two fields; the manifest stamp (`build_stage4_gprs_artifacts.py:140-148`) records `status`, `pause_only_status`, margins — **not** the confirmatory status.
- Run-verified with the real `build_stage3_evidence_report`: attaching `{"status": "fail_on_policy_within_prompt_signal"}` yields top-level `status: pass`, `pause_only_status: pass`, `confirmatory_endpoint.status: fail_on_policy_within_prompt_signal` — the two gated fields are untouched, so the builder proceeds.
- Additionally, `--on_policy_report` is opt-in and the pipeline's `stage3_pause_vs_baselines_report` step (`pipeline.py`) does not pass it, so the default flow produces an evidence report whose confirmatory block says `not_implemented` forever, and Stage4 opens on the teacher-forced screen alone.

This is exactly the pattern this project keeps re-litigating (R5 "nothing consumes stage3_evidence_report" → R8 H-2 → closed; now recreated one level up). The evidence *standard* changed in prose; the *gate* did not. Fix in §6 (R11-T3): builder requires `confirmatory_endpoint.status == "pass"` unless an explicit `steering.gprs.allow_teacher_forced_only: true` opt-in is set; stamp the confirmatory block into the manifest; extend the readiness live re-read; add `--on_policy_report` to the pipeline step command.

**R11-H3 — Asserted `test_status` strings still open the full four-test gate when combined with real injection metrics.**
`liveness_decision` required-tests mode (`liveness.py:104-127`): for each required test, `_metric_status` derives green/red from gate floors **only for `injection_gain`**; for any other test it returns the payload's own `status` (line 83-84), and if the metrics payload is absent entirely it returns `None` and the code falls back to `statuses.get(test)` — a hand-asserted string (line 111). Only `injection_gain` has the missing-payload→`incomplete` special case (108-109). Run-verified:

```
report = { model_under_test: stage2-kl,
           metrics: { injection_gain: {pause_vs_content_gain: 0.90, pause_vs_bos_gain: 9.0},
                      attention_mass: {status: green} },
           test_status: { pause_kv_ablation: green, safe_unsafe_patching: green },
           positive_control: <same> }
liveness_gate_status(config with tests=[all four])  ->  ready=True, decision=green
```

So "the full four-test liveness gate does not open prematurely" holds on the honest runner path (run-verified: stub payloads ⇒ `incomplete`) but **not** against a hand-assembled report — and `run_stage4_liveness.py --metrics_json` (lines 65-85) will launder exactly such a JSON into the official `liveness_report.json` with `status: green`. R10 closed NEW-M1 for the assertion-*only* case (injection's special case anchors that); this mixed case was left open. *Fix (one-line class):* in required-tests mode, treat a missing metrics payload as `incomplete` for **every** required test — i.e., generalize the `injection_gain` special case and delete the `statuses.get(test)` fallback from that branch. Legitimate runner output always writes a payload per configured test (`liveness_kernels.py:510-512`), so nothing honest breaks.

### MEDIUM

**R11-M1 — The on-policy NPZ has no producer; and the existing extractor's control branch is unsafe for generated text.**
The endpoint consumes NPZs with `features/labels/valid_mask/prompt_keys/layer_ids/position_names`. The only producer of that schema is `legacy/PauseProbe/scripts/probe/extract_hidden_states.py`, which *can* serve: it accepts a `generated` field (`row_output`, line 146), hashes prompt text into `prompt_keys` (line 872) so 10 samples/prompt share a group, and 8B configs do include layer 20 in `hidden.layers`. But the chain upstream is missing and one link is broken:
- No generation script exists (nothing samples `samples_per_prompt: 10` and writes JSONL), and `cot_segment_judge` (`probe.on_policy.label_source`) matches **nothing** in `scripts/`/`src/` — it is a config string with zero implementations. The pipeline step ships placeholder args (`<on-policy-train-hidden-npz>`). As shipped, the confirmatory endpoint cannot be run end-to-end. This repeats the R5 "config-only keys" pattern unless the producers are next on the queue.
- **Control contamination hazard:** for a matched-control row that only has `generated` text, `row_output(matched, pause_token, 0)` (extract_hidden_states.py:696) returns the raw text **with pause tokens still in it** — the pause-stripping only happens on the structured `reasoning`/`final_answer` branch. The control forward is then parsed with `pause_layout="none"` and `control_cot_3/4` land at reasoning offsets 3-4, which for on-policy text can sit **on or after pause tokens**. Nothing asserts the control token stream is pause-free. So the "true no-pause content control" property — the core R6-7 fix — silently does not hold on the on-policy route unless a (nonexistent) prep step writes a pause-stripped matched file. *Fix:* assert `pause_id ∉ control_ids` in the control branch (cheap, catches every misuse), and have the future on-policy prep write the pause-stripped matched JSONL.
- **Survivorship:** rows without a contiguous `n_pause_tokens` run are dropped entirely (lines 676-680), and on-policy generations are exactly where layout deviations happen; if deviation correlates with the judge label, the surviving per-prompt label mix is biased. Drop counts exist but not per label. *Fix:* stamp label-conditional drop/parse stats into extraction meta and surface them in the confirmatory report.

**R11-M2 — `attention_mass` is self-certifying and can never be red.**
The green criterion (`ratio ≥ 0.25`) is hardcoded in the kernel (`liveness_kernels.py:417`), not in `liveness.gate`, and the gate accepts the payload's asserted status verbatim (`liveness.py:83-84`). Softmax attention gives every position strictly positive mass, so `pause_mean > 0` always ⇒ `red` is unreachable; the test only emits green/yellow. Today this can't wrongly green the 4-test gate (the two stubs hold it at `incomplete`), but the moment the last two kernels land, a required test's pass criterion lives in kernel code instead of gate config — the exact anti-pattern R9 H-3 removed for `injection_gain`. *Fix:* add `min_pause_attention_mass` / `min_pause_vs_content_attention_ratio` to `LivenessGate`, derive the status in `_metric_status` like injection, keep the kernel's status as advisory.

**R11-M3 — Pause ordinal masks can bind to *natural* pauses instead of the forced run.**
`make_position_masks` selects target ordinals by per-row cumsum over all pause tokens (`liveness_kernels.py:152-159`). `build_liveness_prefixes` samples `insert_pause_after_cot_tokens` (=3/4) CoT tokens — precisely the depth at which the Stage2 model was *trained to emit pauses* — then appends the forced run. Any naturally sampled pause in the prefix takes ordinal 0..k and displaces the appended pauses out of `pause_0..2`, so the battery measures a mixture of natural and forced ports. Consistent with the known steering-ordinal caveat, but this battery exists specifically to certify the forced port. Under left padding the appended run is always the **last `n_insert_pauses` positions of the row** — a mask over those positions is exact and free. *Fix:* target the trailing forced run (or at minimum count and report natural-pause incidence per batch so contamination is visible).

**R11-M4 — `final_token_kl` dilutes pause/content slopes at `batch_size > 1`.**
The KL is averaged over **all rows** in the batch (`liveness_kernels.py:184-193`), including rows where the mask selected nothing (those rows are unperturbed ⇒ KL≈0). BOS exists in every row, pause/content don't (truncation, layout deviation) ⇒ `pause_vs_bos_gain` biased downward. Fail-closed in direction, but it distorts calibration against the positive control whenever batch compositions differ. Default `--batch_size 1` avoids it (rows without targets are skipped via `mask.any()`). *Fix:* compute KL only over rows where the mask has ≥1 position, or hard-require `batch_size == 1` for `injection_gain`.

**R11-M5 — `require_true_content_control` defaults to `false`, so the shipped standard silently degrades to prompt-baseline-only.**
If control positions are missing/invalid in the NPZ (e.g., extraction ran without a matched-control file — the likely first-run state given M1), `evaluate_signal` raises, `control=None`, `control_error` is recorded, and status can still be `pass` against the 0.5 constant alone (`on_policy_stage3.py:366-401`). Your own framing this round says pause must add signal beyond prompt baselines **and** true no-pause content controls. Either set `require_true_content_control: true` in the experiment configs, or emit a qualified status (e.g., `pass_no_content_control`) so the evidence report can't display an unqualified `pass` without the control. Note the design tension to resolve deliberately: the earlier agreed decision rule (follow-up review) was "pause > prompt required; pause > content NOT required." The implemented margin (`pause − max(0.5, control) > 0.01`) is *stricter* when control is present and *weaker* (M5) when it's absent. Since `control_cot_3/4` are depth-matched early-content positions (not late content), the stricter bar is defensible — but pick one standard and encode it; don't let data availability choose it.

**R11-M6 — No attach-time consistency check on the on-policy report.**
`run_stage3_evidence_report.py --on_policy_report` attaches any JSON object; nothing verifies the attached report's `layer`/`positions`/`config` against the evidence config (stage3_evidence.py:137-138). A 1.5B confirmatory report attaches to an 8B evidence report without complaint; a stale pre-rerun report attaches after configs change. Same class as the blind-probe-copy (M-3) and manifest layer/positions checks R10 closed. *Fix:* refuse attach when `on_policy_report["layer"] != probe.on_policy.layer` or positions mismatch; stamp the on-policy report path + mtime for the staleness re-read once H2's binding exists.

### LOW

- **R11-L1 — Margin over control is a point-estimate with a 0.01 floor** (`on_policy_stage3.py:388-399`): pause CI is computed but the pause−control margin has none; same class as the R8 H-4 residual. Bootstrap the per-prompt paired margin and gate on its CI low.
- **R11-L2 — `split_by_prompt.take()` slices any array with `shape[0] == n_rows`** (`on_policy_stage3.py:198-205`): if `len(position_names)` or `len(layer_ids)` happens to equal the row count (tiny pilots), metadata gets row-sliced silently. Slice an explicit key allowlist (`features/labels/valid_mask/prompt_keys` + any per-row extras).
- **R11-L3 — Single shared random direction** (`liveness_kernels.py:312`): one draw in ~1.5-4k dims is direction-luck; average slopes over k≈4-8 directions. Also `liveness.directions: [random, probe_weight, mean_diff]` and `negative_control_model` are inert — plan advertises capability the kernels don't have.
- **R11-L4 — Prefix re-tokenization truncates from the right** (`liveness_kernels.py:317-321`): near-`max_input_length` prompts lose the appended forced pauses (then bs=1 skips the row; bs>1 dilutes per M4). Also `build_liveness_prefixes` generates **all** prompts in a single `generate` call — 200 left-padded rows at 8B is an OOM risk; batch it. Cap prompt length at `max_input_length − insert_tokens − n_insert_pauses` at generation time.
- **R11-L5 — `allow_yellow=False` has no test coverage** post-R10-1-fix (see §1).
- **R11-L6 — Control depth hardcoded `[3,4]`** in the extractor's control branch (extract_hidden_states.py:713) while 8B is cot4 — controls are one token shallower than the 8B pause site; make it config-driven when the on-policy prep lands. Also `epsilon_base` isn't surfaced in `liveness_config`/plan output (provenance).
- **R11-L7 — Battery runs one layer** (`steering.layer`) while the plan lists `liveness.layers: [7,14,17,21]`; the yellow-path instruction "proceed on live layers only" has no per-layer results to act on. Fine for v1; note in the plan text.

---

## 3. Answers to your review questions

**Q1 — Does the on-policy endpoint address the prompt-classification critique?**
**Yes — this is the strongest part of the change.** Any prompt-constant component of the score cancels exactly in same-prompt pairs (a constant shifts both members of every pair), so within-prompt AUROC is provably immune to prompt-only leakage; the fixed 0.5 prompt-constant baseline with the explanatory note (`on_policy_stage3.py:417-420`) is correct. Direction and standardization are fit on train prompts only; the split is prompt-level with outcome-profile stratification (uses labels only, not features — legitimate). Misuse fails closed in every mode I constructed: teacher-forced NPZ ⇒ singleton prompt groups ⇒ `split_by_prompt` raises or `insufficient_mixed_prompts`; prompt-level labels copied across samples ⇒ zero mixed prompts ⇒ fail; missing `prompt_keys` ⇒ per-row groups ⇒ fail. Residual vulnerabilities are *upstream*, not in the metric: extraction survivorship (M1) and control contamination (M1) can bias what reaches the metric, and the endpoint is unrunnable until producers exist (M1).

**Q2 — Split / within-prompt AUROC / bootstrap / content-control: correct?**
Mechanically **yes** with caveats. `_rank_auroc` is a correct tie-averaged Mann-Whitney implementation (used only for the reference `global_auroc`); the pair statistic (`wins + 0.5·ties`) is the exact within-prompt AUROC; pooled aggregation weights prompts by pair count (documented behavior — mixed-balance prompts dominate; acceptable, and `per_prompt` rows expose everything for re-weighting); the cluster bootstrap resamples prompts, which is the right unit. Caveats: margin vs control is a point estimate (L1); `require_true_content_control` default weakens the shipped standard (M5); nondeterministic `set` iteration in `split_by_prompt` affects only row order, and everything downstream is order-invariant — fine. The CI-low > 0.55 pass rule matches the agreed endpoint.

**Q3 — Does the evidence report attach/represent the confirmatory endpoint honestly?**
Representationally yes: the report is embedded verbatim, the status string propagates, `not_implemented` remains when absent, and the "teacher-forced is only a screen" note stays. **Gate-honestly no (H2):** top-level `status` is unaffected by a failing confirmatory report (run-verified), nothing downstream reads `confirmatory_endpoint.status`, the manifest doesn't stamp it, and the pipeline never passes `--on_policy_report`. Plus no attach-time consistency check (M6). As shipped, a reader of `stage3_evidence_report.json` sees the truth; the *machine* gates ignore it.

**Q4 — Are `injection_gain` and `attention_mass` valid first battery pieces?**
`injection_gain`: **yes, conceptually valid** — norm-controlled (relative to the mean hidden norm of the targeted position class, which is the right matched-relative design), same direction across position classes and epsilons (paired comparison), KL at the final token, slopes compared pause/content/BOS, thresholds enforced *at the gate* rather than trusted from the payload. Known approximations to accept knowingly: KL is ~quadratic in ε so `KL/ε` grows ~linearly and the mean over `[1,2,4]` is dominated by ε=4 (comparison across classes at the same ε set remains apples-to-apples); slope scaling inherits pause-state norms (if KL-transparency shrank pause norms, relative-ε under-perturbs them — that's arguably the honest operating-point question, but note it when reading results). Real issues: M3 (ordinal contamination), M4 (batch dilution), L3 (single direction), L4 (truncation/OOM).
`attention_mass`: **weakest piece** — last layer only, head-averaged, self-asserted hardcoded threshold, red unreachable (M2). Fine as an advisory diagnostic; not yet fit to be one of four *required* gate tests. Also note `output_attentions=True` under sdpa/flash either falls back to eager (recent transformers) or yields no attentions — the `not attentions ⇒ incomplete` guard fails closed, but consider loading with `attn_implementation="eager"` for this pass so the test can actually run.

**Q5 — Masks under left padding and forced pause insertion?**
Correct on the mechanics I could check: `padding_side="left"` is set before tokenization; pause mask is `input_ids == pause_id ∧ attention_mask` so pads never match; BOS anchor is the first *valid* token per row (correct under left padding); the content window is a pause-count-matched span immediately before the first target pause, clipped at row start; final-token indices via the flip/argmax idiom are correct (and with pure left padding every row ends at `seq_len−1` anyway). Two real gaps: natural-pause ordinal capture (M3) and right-side truncation deleting the appended run for near-max-length prompts (L4). One-token pause-id assertion (`liveness_kernels.py:455-457`) correctly hard-stops models without the pause token.

**Q6 — Does the liveness runner remain fail-closed with two kernels incomplete?**
On the honest path, **yes — run-verified**: the runner stubs every configured-but-unimplemented test with `status: incomplete` ⇒ 4-test decision `incomplete` ⇒ gate not ready; `--tests` subsets can't shrink the required list (decision always uses config `liveness.tests`, which lists all four in both stage4 configs); restricting the *config* to the two implemented tests greens the gate, but that's a deliberate config edit, consistent with prior rounds' config-scoping; PC missing/8B two-field block hold at plan **and** gate (run-verified earlier rounds; config unchanged); sub-threshold injection metrics override asserted green ⇒ red (re-verified). **Against report-level assertion, no (H3)**: metrics-plus-asserted-`test_status` opens the full gate, and `--metrics_json` officializes it. Close H3 and the answer becomes an unqualified yes.

**Q7 — What must be fixed before landing?** See §5/§6 — H1 (+ pod pytest) is the land blocker; H2 and H3 are small and should ride the same fix commit; M1 is the blocker for ever *running* the confirmatory endpoint; M2-M4 before the first real battery result is interpreted.

---

## 4. Verified good (run-verified unless noted)

- R10-1 fixture fix correct (§1); 4-test stub report ⇒ `incomplete`; 2-test config ⇒ `green` (your synthetic check reproduces exactly).
- Failing/absent confirmatory report attach behavior characterized end-to-end (basis of H2).
- Both confirmatory dry-runs resolve output under `legacy/PauseProbe/<single_scan_out_root>/stage3_on_policy_confirmatory_report.json` — adjacent to `stage3_evidence_report.json`, reusing the H-1 path pattern correctly; 1.5B/8B configs deliver `on_policy` keys via deep-merge (layer 14/20, positions, thresholds).
- Layer provenance now aligned: 1.5B steering 14 == on_policy 14; 8B steering 20 == on_policy 20, and 8B `hidden.layers` includes 20; 8B `insert_pause_after_cot_tokens: 4` matches cot4. The old layer16-vs-20 break does not recur here.
- `run_stage4_steering.py --phase liveness` shells to the real runner (dry-run only when asked); GPRS phases beyond validate/liveness still hard-exit pre-generation (hook absent — unchanged, correct).
- Endpoint math: tie-correct rank AUROC; exact pair-win statistic; prompt-level cluster bootstrap; train-only standardization and direction; `global_auroc` reported but never gated on; full `per_prompt` audit trail embedded; atomic `.tmp`-rename JSON writes throughout.
- Kernel plumbing: hidden-state layer convention (`hidden_states[L]` ⇔ block `L−1` hook) matches the probe extraction convention; multi-device `device_map` handled in the hook (direction/step moved to `hidden.device`); base forward computed once per batch; `use_cache=False` on scored forwards.

---

## 5. Go/No-Go

| Gate | Verdict | Basis |
|---|---|---|
| Land `460bae1` into main | **NO-GO** | R11-H1: new test file red (control fixture label-correlated; verified). Same land-gate breach class as R9/R10. Fix fixture + pod pytest green first; take H2/H3 in the same commit. |
| Stage3 on-policy endpoint as the confirmatory evidence standard | **GO on design, NO-GO on enforcement** | Within-prompt design correct and fail-closed (Q1/Q2); but not bound to any gate (H2) and no producer chain (M1). |
| Run Stage3 confirmatory on real data | **NO-GO (blocked)** | M1: no generation script, no `cot_segment_judge`, control branch unsafe for generated text. |
| Stage4 1.5B liveness pilot (2 implemented kernels) | **GO after H3+M2, as a pilot** | Runner path fail-closed; decision stays `incomplete` by design under the 4-test config — run it for calibration data (PC vs model-under-test), not for a gate decision. Keep `--batch_size 1` (M4); watch natural-pause incidence (M3). |
| Stage4 8B liveness | **NO-GO (unchanged)** | Two-field positive-control block intact at plan and gate; genuine 8B full-SFT control still missing. |
| GPRS generation/eval | **NO-GO (unchanged)** | Steering hook not implemented; runner hard-exits. Plus H2: don't wire the hook before the confirmatory binding exists, or Stage4 opens on the screen alone. |

---

## 6. TODOs

Pre-land (blockers):
- **R11-T1 (H1):** fix the fixture: `features[idx, 0, 1, 1] = float((idx // 2) % 2) * 0.01` (prompt-constant nuisance ⇒ control AUROC exactly 0.5); keep a second control variant if you also want a within-prompt-noise-but-uninformative case. Then run full pod pytest; land only on green. Consider adding a numpy+pytest venv to the local loop — every red-suite incident so far was numpy-level, not torch-level.
- **R11-T2 (H3):** `liveness.py::liveness_decision` required-tests loop: replace the `injection_gain`-only special case with `if metric_status is None: status = "incomplete"` for **all** required tests (delete the `statuses.get(test)` fallback in that branch). Re-run the R10/R11 drivers: stub⇒incomplete, asserted kv/patching⇒incomplete (not green), metrics-green 2-test⇒green.
- **R11-T3 (H2):** bind the confirmatory endpoint: (a) `build_stage4_gprs_artifacts.py::build_artifacts` — require `evidence["confirmatory_endpoint"]["status"] == "pass"` unless `steering.gprs.allow_teacher_forced_only: true`; stamp `{confirmatory_status, on_policy_report_path}` into `manifest["stage3_evidence"]`; (b) `gprs.py` live re-read — same check against the live evidence report; (c) `pipeline.py` evidence step — add `--on_policy_report <stage3-on-policy-confirmatory-report>`; (d) test: manifest green + confirmatory fail ⇒ not ready.

Before the confirmatory endpoint can run (next packet):
- **R11-T4 (M1):** on-policy producer chain: generation script (N samples/prompt at fixed temp/top_p, writes `generated` + `prompt_id`), `cot_segment_judge` labeling (per-generation labels into a field `label_from_row` reads), and prep that emits the pause-stripped matched-control JSONL.
- **R11-T5 (M1):** extractor hardening: assert control token stream contains no `pause_id`; stamp label-conditional drop/parse counts into NPZ meta; surface them in the confirmatory report.
- **R11-T6 (M5/M6):** set `require_true_content_control: true` in experiment configs (or emit `pass_no_content_control`); attach-time layer/positions consistency check + staleness stamp.

Before interpreting the first battery run:
- **R11-T7 (M2):** move attention floors into `LivenessGate` and derive in `_metric_status`; make red reachable (e.g., ratio < yellow floor).
- **R11-T8 (M3):** mask the trailing forced pause run (last `n_insert_pauses` valid positions) or report natural-pause incidence per batch.
- **R11-T9 (M4):** restrict `final_token_kl` to masked rows, or assert `batch_size == 1` for injection_gain.
- **R11-T10 (L1-L4, L6-L7):** margin CI; `take()` key allowlist; k random directions; generation batching + length cap; config-driven control offsets for 8B; surface `epsilon_base` in the plan; note single-layer scope in plan text.

---

## 7. Standing items from prior rounds (unchanged this round)

R10-2 (no-source builder branch skips probe checks / stamps empty metadata — target-exists branch at `build_stage4_gprs_artifacts.py:123`), R10-3 (NEW-M3 branch coverage), R10-4 (injection_gain literal keying — partially superseded by R11-T2's generalization), R10-5 (`validate_gprs_config` doesn't require `stage3_evidence_report` — fold into R11-T3), R10-6 cosmetics; M-2 (`gate_threshold: 0.95` hardcoded vs probe.pt calibrated threshold; scaler still not stamped), `random_direction_control` key inert; S4-5 judge endpoint separation; GPRS hook + on-policy direction QC before any steering eval.

---

## Headline verdict

**NEEDS FIXES.**

The endpoint design and the first kernel are the right instruments, and every honest-path gate property re-verified. But: (1) the commit ships a red test suite for the third round running — `test_on_policy_confirmatory_passes_when_pause_has_within_prompt_signal` fails because its "null" control is a perfect label decoder (verified by exact-path recomputation: control AUROC 1.0, status `fail_no_independent_on_policy_pause_signal`); (2) the on-policy confirmatory endpoint — the stated point of this round — gates nothing: a failing confirmatory report leaves Stage4 fully openable (run-verified); (3) the four-test liveness gate is openable today by hand-asserted `test_status` strings riding on real injection metrics, through official `--metrics_json` tooling (run-verified `ready=True`). All three fixes are small (one fixture line, one decision-loop change, one builder/readiness binding). Fix, run pod pytest green, and this lands.

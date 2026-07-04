# Fable Four-Stage Objective-Alignment Review (Round 13)

- **Scope:** full objective-alignment review of the SafeChain Stage1–4 research workflow against the Stage2–4 code packet at `safechain_stage2_to_stage4_full_code_20260704/cot-safety` (tree at HEAD `2738453`, code identical to R12-reviewed `4068c06`; HEAD only adds the R12 review doc). This is not a diff check: every gate, kernel, trainer, and config on the Stage2→4 path was read against the staged research plan and the professor's five concerns.
- **Method:** full reads of the Stage2 trainer/format-only stack, Stage3 endpoint + evidence modules, all Stage4 steering/liveness modules and scripts, all stage configs; targeted reads of the legacy producers (generation, judge runner, hidden-state extractor); the **entire test suite executed on this machine** (see Test Run — including the torch tests R9–R12 could never run); consistency greps for escape hatches, dead knobs, and signature drift. No code edited.

## Test run (this machine)

No pytest exists in any local env and pip has no network, so the suite was run with a minimal pytest shim (implements `raises`/`approx`/`importorskip`/`mark.parametrize`/`tmp_path`/`monkeypatch` with pytest semantics; shim + runner live outside the repo in `/tmp/fable_r13_pytest/`) under `~/miniforge3/envs/pytorchenv` — **numpy + torch 2.3.1**, no transformers (the trainer test self-stubs it, tests/test_stage2_pause_kl_trainer.py:27-34).

```
all 13 tests/test_*.py files → 56 passed, 1 failed, 0 skipped
```

- **First round where the torch-gated tests actually executed:** the three `projection_rejection_update` tests (test_stage4_gprs_liveness.py:277/299/320) and all 7 KL-trainer tests ran on real torch and **pass**. The R12 residual "torch subset unexecutable on this Mac" is now largely discharged; only transformers-dependent integration paths remain pod-only.
- **1 genuine red test in main:** `test_safe_rewrite.py::test_merge_generated_pair_and_long_rows` (tests/test_safe_rewrite.py:124-159) feeds a 3-word `safe_reasoning` and expects a soft `length_match_pass: False`, but `update_pair_record_with_generated_safe` now hard-raises below `MIN_GENERATED_SAFE_REASONING_WORDS = 20` (src/cot_safety/data/safe_rewrite.py:30, :726-729). Not a shim artifact — plain function, real `ValueError`. Any pod pytest run fails on it, so **R12-F1 ("full pod pytest green before land") cannot pass on this tree**. Off the Stage2→4 gate path (safe-rewrite is Stage2.5-A corpus prep), but it must be reconciled (fix the test to feed ≥20 words, or make the floor config-driven) before anyone claims a green suite. → **R13-H1**.

---

## Q1 — Does Stage2 preserve the invariant (insert pause without changing data format or model behavior more than necessary)?

**Yes — and the shipped mechanism is materially stronger than the plan text.** The primary method is no longer plain full SFT; it is `kl_transparent_emit` (configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_cot3_save25_max400_4xa6000.yaml):

- **Rows-only training:** all parameters frozen except the `<|pause|>` embedding row in input *and* output embeddings, enforced by gradient row-masks (legacy/COTPauseToken/src/trl_train.py `configure_format_only_training` → `mask_embedding_gradients`), a trainable-param assert (`_assert_rows_only_training`), a `weight_decay == 0.0` assert (wd would decay frozen rows), and `_RowsOnlyInvariantCallback` re-checking non-pause rows are **bit-identical** every 50 steps and at train end (legacy/COTPauseToken/src/utils/pause_kl_trainer.py). Base-model behavior on pause-free text is preserved *by construction*, not by hope.
- **KL transparency:** loss = 0.3·CE(pause emission at forced slots) + 1.0·KL(student-with-pause ‖ pause-stripped same-model teacher, post-pause continuation) + 1.0·suppression(−log1p(−p_pause) off-slot) + 0.0·pre-pause KL. The pause logit is masked to −inf on **both** sides of the KL so transparency isn't gamed through the new token; teacher runs `no_grad`/eval on the pause-stripped batch with an explicit src→teacher index mapping. Pre-pause behavior needs no loss term: pauses insert at cot_3/cot_4 and causality freezes everything before them.
- **Format:** `<|pause|>` added as a special token; insertion is tokenizer-offset-exact (src/cot_safety/formatting/pause_insertion.py — splits the think block, tokenizes reasoning with offsets, skips leading whitespace tokens, inserts at the char offset of token `first_idx + cot_offset`). Chat template unchanged. `run_stage2_sft.py` hard-errors if the configured pause token ≠ `<|pause|>`.
- **Matched triplets:** `intra_pause_cot3` / `no_pause_matched` / `pre_think_pause3_matched` built from identical rows and identical shuffles with per-variant manifests (legacy builder + 3 format validators) — the comparisons Stage3 needs are baked into data prep.
- **Validation harness exists:** `run_model_comparison_eval.py` + configs score capability (gsm8k 500 / math500 300), safety (strongreject, harmbench_standard, jailbreakbench), and over-refusal (xstest, or_bench_hard) across `base_natural` / `kl_emit_cot3_natural` / `kl_emit_cot3_forced`, with wildguard/llamaguard/harmbench and conservative aggregation. Crucially, judges score `generated_for_judge = strip_pause_tokens(generated)` including the CoT (legacy/PauseProbe run_model_comparison_generation.py:456; judge `response_from_row` prefers that key), so "pause model unsafe-CoT rate ≈ original" is measured on pause-free text with the full trajectory visible — pause tokens can't confuse the judges, and CoT unsafety isn't hidden behind answer-only judging.
- All 7 shipped trainer tests (stripped-batch mapping, KL pair token alignment, pause-slot masking, CE/suppression math vs manual, wd/body-param rejection, row-mutation detection, finite loss with grads only on embeddings) **executed green on real torch here**.

**Divergences from the plan text (documentation debt, not code defects):**

| Plan says | Code does | Assessment |
|---|---|---|
| Full SFT, no LoRA, UltraFeedback chosen responses | Primary = rows-only `kl_transparent_emit` on **trusted CoT 18k** (Sky-T1 6k + Bespoke-Stratos 6k + OpenThoughts 6k, configs/data/stage2_trusted_cot_18k.yaml); `stage2_intra_pause_sft.yaml` full-SFT retained as control | Better for the professor's SFT-confound concern (nothing but one embedding row can change). Full-SFT checkpoint is repurposed as the Stage4 liveness **positive control** — coherent. But the plan text is stale and must be updated before running, or reviewers will score the run against the wrong protocol. |
| "model learns to emit pause tokens" | Emission trained at forced slots; **suppressed elsewhere** (suppression_weight 1.0); eval covers natural + forced insertion | Fine — but claims should say "emits at trained intra-CoT slots," not free emission. |

**Q1 verdict: invariant preserved; Stage2 is the strongest link in the chain.**

## Q2 — Does Stage3 actually answer "pause adds trajectory/sampling signal beyond prompt classification"?

**The measurement design now does; the code cannot yet produce the inputs the measurement needs.**

Design (correct, and enforced):
- **Teacher-forced screen:** `build_stage3_evidence_report` (src/cot_safety/probes/stage3_evidence.py:113-207) requires best pause/post-pause AUROC to beat max(prompt-only baseline, true no-pause content control) by `min_pause_margin_over_baselines`, fails closed on any missing baseline (`missing_prompt_baseline` / `missing_true_content_control`), and reports `pause_only_status` separately so post-pause can't carry pause. Positions/layers configured in stage3_intra_pause_probe.yaml (layers [7,14,17,21,22,28]; prompt baselines `last_prompt_token`,`pre_think`; controls `control_cot_3/4`); heldout source (`reasoningshield_test`) supports the generalization check.
- **Confirmatory endpoint (the real answer to the professor's #1):** within-prompt AUROC on on-policy samples (src/cot_safety/probes/on_policy_stage3.py) — prompt-level split with outcome-profile stratification, train-only standardization, mean-diff direction, cluster bootstrap over prompts, status ladder `insufficient_mixed_prompts` → CI-low ≤ 0.55 fail → missing control fail (with `require_true_content_control: true` set in config, R11-M5 closed) → margin ≤ 0.01 over max(0.5, control) fail → pass. Prompt-constant signal cancels **exactly** within prompt pairs; the red-fixture test proving cancellation and the green-fixture test both executed green here. Zero-norm (degenerate) directions return all-ties → 0.5 → fail-closed, with the R12-F4 residual that a *broken* control extraction is indistinguishable from a clean null.
- **Attach-time integrity:** `validate_on_policy_report_config` (stage3_evidence.py:42-67) refuses layer/positions/control-positions mismatches; the evidence report stamps `report_path` + `report_mtime`; omitting `--on_policy_report` leaves confirmatory at `not_implemented` → downstream fail.

**The gap (unchanged blocker, R11-M1 / R12-F2):** the producer chain for the confirmatory endpoint does not exist —
1. **No on-policy generation for the intra-pause layout.** The only multi-sample generator, legacy/PauseProbe/scripts/generation/generate_target_trajectories.py, is vLLM n-samples/prompt (defaults 50 @ temp 0.6 vs plan's 10 @ 0.7 — knobs exist) but forces the **pre-think** prefix `<|pause|>×3<think>\n` (:31), not pauses after cot_3/cot_4; its `generated_for_judge` strips only **leading** pauses (:58-59, :217), so intra-CoT pauses would leak to any judge; and it explicitly "does not judge safety."
2. **`cot_segment_judge` is a config string with no implementation** (grep: only `configs/experiment/stage3_intra_pause_probe*.yaml`). No per-generation CoT-segment labels ⇒ no labels for the endpoint.
3. **No pause-stripped matched-control prep for generated text.** The extractor control branch (legacy/PauseProbe/scripts/probe/extract_hidden_states.py:660-730) hardcodes `cot_offsets=[3, 4]`, assumes curated matched rows, and never asserts the control token stream is pause-free (`pause_id ∉ control_ids`) — unsafe for self-generated text and wrong for the 8B cot4 path.
4. No wiring from generation → judge → NPZ with the `features/labels/prompt_ids` row-aligned contract `run_stage3_on_policy_confirmatory.py` consumes.

**Second-order divergence:** Stage3 probe data is the curated stack (reasoningshield train/dpo, star41k, star1, aidsafe×2, unsafechain, harmthoughts — configs/data/stage1_positionscan_sources.yaml), not the plan's StrongReject+WildJailbreak SafeChain data. Defensible (labels and CoT quality), but the "Stage3 holds on SafeChain distribution" claim in the plan is not what this code will test; say so explicitly or add the SafeChain sources.

**Q2 verdict: design PASS / execution NO-GO.** The screen can run today; the confirmatory endpoint — the thing that actually answers the question — is blocked on the producer chain (blocker B1 below).

## Q3 — Does Stage4 wait for Stage3 evidence + liveness, and is GPRS aligned with the goal?

**Yes. Verified independently in code, the gating is real, layered, and fail-closed — four gates deep:**

1. **Scope:** steering targets must be `pause_*` and must not match `pre_pause_/post_pause_/cot_/control_cot_` (src/cot_safety/steering/scope.py; groupwise for `target_specs`); wired as the first pipeline step and as `cot-safety steer validate-scope`.
2. **Liveness:** `liveness_gate_status` (src/cot_safety/steering/liveness.py:169-212) recomputes the decision from **metrics** through `LivenessGate` floors — asserted `test_status` strings are ignored in required-tests mode (:114-121), missing/non-dict metric payloads ⇒ `incomplete` (:74-77), and the `--metrics_json` officialization path recomputes through the same function (scripts/run_stage4_liveness.py:65-85). It additionally requires the report's `model_under_test` to match the config's checkpoint and requires a **green positive control recomputed by the same battery** (seed+17), not an asserted status. The 1.5B positive control is the full-SFT pause checkpoint (stage4_pause_gprs.yaml:23) — the right "ports definitely live" model; the 8B config deliberately stamps `positive_control_status: missing_required_full_sft_pause_control` (stage4_pause_gprs_8b_4xa100.yaml:16-21) ⇒ plan `blocked_missing_positive_control`, gate not ready. Under the shipped 4-test config, `pause_kv_ablation`/`safe_unsafe_patching` are `kernel_not_implemented` stubs (liveness_kernels.py:560-562) ⇒ decision `incomplete` ⇒ **liveness cannot go green at all yet** — going green requires either implementing the kernels or a visible config edit narrowing `liveness.tests`.
3. **Stage3 evidence:** `stage3_evidence_gate` (src/cot_safety/steering/gprs.py:55-81) requires `status`, `pause_only_status`, **and** `confirmatory_endpoint.status` all `pass`. Enforced at the artifact **builder** (build_stage4_gprs_artifacts.py:84-97 — refuses before even importing torch), stamped into the **manifest** (:136-167 incl. `confirmatory_status`, embedded endpoint, report path, `require_confirmatory`), re-checked at **readiness** (`gprs_artifact_status` gprs.py:83-154) including a **live re-read** of `stage3_evidence_report.json` (`stage3_evidence_live_not_ready`) with `require_confirmatory` derived from *config at read time* — a manifest built under the escape hatch doesn't carry its exemption forward; legacy manifests without the field fail closed. Escape hatch `allow_teacher_forced_only` defaults false and **no shipped config sets it** (grepped). Layer/positions mismatch between config and manifest ⇒ `steering_config_mismatch`.
4. **Terminal stop:** even with every gate green, `run_stage4_steering.py --phase eval` for gprs/projection raises `SystemExit("GPRS generation is scaffolded but not wired…")` after `require_gprs_readiness` (:331-337). The deprecated learned-delta path demands an explicit `--allow_learned_delta` or `acknowledge_deprecated: true` (only set `false` in configs) and is labeled archival.

**Is GPRS logically aligned?** Yes: `projection_rejection_update` (gprs.py:170-207) computes `h ← h − λ·max((h−μ_safe)·û, 0)·û` — one-sided rejection along the normalized unsafe−safe mean-diff, active only when the state sits beyond the safe centroid on the unsafe side, gated by probe score ≥ `gate_threshold` (0.95), with the delta norm-capped at 10% of ‖h‖. That is precisely "identify the unsafe direction at pause states and pull back toward the safe manifold without wrecking the representation," restricted to pause positions by scope. The three geometry tests (positive-side-only movement + norm cap; probe-gate threshold; uncapped reach of safe halfspace) executed green on real torch here. Controls are designed in: `alpha_grid` includes 0.0, eval conditions include `sft_random_direction`, `random_direction_control: true` in config.

**Aligned residuals (not blockers, keep visible):**
- **Direction + probe are teacher-forced-fit** (`direction_provenance: "teacher_forced_prompt_labels"` stamped honestly in the manifest) while steering acts during self-generation. The Stage3 confirmatory gate ensures the *signal* survives on-policy before steering is allowed, and liveness measures on prefixes sampled from the model's own CoT (`build_liveness_prefixes`, kernels :273-319), but the direction itself is never refit on-policy — first GPRS results must carry this caveat.
- **Dead config knobs:** `liveness.directions: [random, probe_weight, mean_diff]` is never consumed — `injection_gain_metric` draws a single random direction (:347). Random-only is the conservative (anti-inflationary) choice for liveness, but the knob promises coverage that doesn't exist. `steering.gprs.random_direction_control: true` is likewise consumed nowhere yet (generation hook absent). → R13-M2.
- `min_pause_attention_mass: 0.0` is vacuous (softmax mass > 0); the ratio floor (0.25) is the binding one. `gate_threshold: 0.95` and `norm_cap: 0.10` are uncalibrated defaults (R10 standing).
- `make_position_masks` truncation fallback: rows whose trailing forced run was right-truncated fall back to the **all-pause mask**, silently including natural pauses (kernels :161-171; R12-F5).
- `allow_yellow_for_gprs: true` default; `allow_yellow=False` path untested (R11-L5).

**Q3 verdict: PASS on gating and steering-math alignment. Execution NO-GO by the code's own design** (unimplemented kernels ⇒ liveness incomplete; unimplemented generation hook ⇒ terminal SystemExit; 8B additionally blocked on positive control).

## Q4 — Code changes still required before running Stage2/3/4 for real

**Stage2 (1.5B kl_transparent_emit train + model-comparison eval): nothing on the training path.** Data prep, trainer invariants, and eval harness are implemented and test-validated (now on real torch). Required around it: fix **R13-H1** (red safe_rewrite test) so the pod-pytest land ritual can be green, and update the plan/README for the data+method divergence (D1/D2).

**Stage3 confirmatory (B1 — the big one, R11-T4/T5 / R12-F2):**
1. On-policy generation for the **intra-pause** layout: sample `insert_pause_after_cot_tokens` CoT tokens then force `n_insert_pauses` pauses (the pattern already exists in `build_liveness_prefixes`) — or natural-emission sampling — at plan knobs (10 samples/prompt, temp 0.7); extend, don't reuse, `generate_target_trajectories.py` (pre-think-only, leading-strip-only).
2. Implement `cot_segment_judge`: pause-stripped **full** generation → per-generation CoT-segment labels, with label-conditional drop stats.
3. Matched pause-stripped control prep + extractor hardening: control depth from config (cot3 vs cot4, not hardcoded `[3,4]`), assert `pause_id ∉ control_ids`, valid-row masks for generated text.
4. Wire producer output to the NPZ row-aligned contract of `run_stage3_on_policy_confirmatory.py`; stamp sampling params into the report.
5. Small: flag `degenerate_direction: true` in the result or fail the control under `require_true_content_control` (F4).

**Stage4 liveness as a real gate:** implement `pause_kv_ablation` + `safe_unsafe_patching` kernels with **floor-derived** statuses in `_metric_status` (F3 — until then a forged `metrics.<test>.status` string is the remaining assertion vector, one level deeper than the closed `test_status` one); F5 empty-mask-on-truncation; consume or delete `liveness.directions`; 8B needs a genuine full-SFT pause control checkpoint + both config fields flipped.

**Stage4 GPRS generation:** the forward-hook that applies `projection_rejection_update` during generation, incl. per-step probe gate scoring at pause positions; wire `random_direction_control`; calibrate `gate_threshold` on held-out probe scores; R10 standing items (scaler stamp, no-source builder branch note).

## Q5 — Blockers vs follow-ups

| ID | Item | Class | Blocks |
|---|---|---|---|
| R13-H1 | Red test in main: `test_merge_generated_pair_and_long_rows` vs `MIN_GENERATED_SAFE_REASONING_WORDS=20` (safe_rewrite.py:30,:726) | **Blocker for the land ritual / green-suite claim** (not for Stage2 execution itself) | R12-F1 |
| D1/D2 | Plan text stale: UltraFeedback→trusted-CoT-18k; full-SFT→kl_transparent_emit; StrongReject/WildJailbreak→curated Stage3 sources | **Blocker for claims, not code** — update plan/README before runs | honest reporting |
| B1 | Stage3 on-policy producer chain (generation @ intra layout, `cot_segment_judge`, pause-stripped control prep, extractor hardening, NPZ wiring) | **Blocker** | Stage3 confirmatory ⇒ Stage4 anything |
| B2 | kv-ablation + patching kernels with floor-derived statuses (F3) | **Blocker** for liveness green (by design: decision stays `incomplete`) | Stage4 GPRS |
| B3 | GPRS generation hook + per-step probe gating | **Blocker** | Stage4 eval |
| B4 | 8B full-SFT positive control checkpoint + config flip | **Blocker** (8B only) | 8B liveness |
| F4 | `degenerate_direction` flag on zero-norm control | Follow-up (medium) | control-integrity reporting |
| F5 | Empty row mask on truncated forced run in `make_position_masks` | Follow-up (medium) | liveness measurement purity |
| R13-M2 | Dead knobs: `liveness.directions`, `random_direction_control` | Follow-up (low) | config honesty |
| — | `gate_threshold`/`norm_cap` calibration, vacuous mass floor, `allow_yellow=False` coverage, L1 margin CI, L3 multi-direction, L4 length cap, R10 scaler stamp | Follow-ups (carried) | polish |

## Q6 — Contradictions with the professor's concerns

| Concern | Where the workflow answers it | Verdict |
|---|---|---|
| 1. Probe may be prompt classification | Within-prompt AUROC confirmatory endpoint (prompt-constant signal cancels exactly), hard-wired into the Stage4 builder/manifest/live-re-read gate; prompt-only baselines + margin in the screen | **Addressed in design; unproven until B1 lands** — no contradiction, but no run either |
| 2. Source/format artifacts | Matched triplets from identical rows; true no-pause content controls at matched depth; heldout source; attach-time layer/position validation | Addressed; note curated-source divergence (D2) when claiming generality |
| 3. SFT confounds safety/capability | Rows-only training (only the pause row can change, bit-identity checked), KL-transparency to the model's own pause-stripped distribution, capability + over-refusal + broken-output eval, full-SFT demoted to liveness control | **Addressed more strongly than the plan itself** |
| 4. Teacher-forced ≠ on-policy distribution | Confirmatory endpoint requires on-policy generations + per-generation judge labels; liveness prefixes sampled from the model's own CoT | Addressed at the gate level; residual: steering direction/probe still teacher-forced-fit (honestly stamped) — carry as a caveat on any GPRS result |
| 5. Residual unsafe rates must stay visible | `primary_endpoint: unsafe_cot_rate`; conservative multi-judge aggregation; `unlabeled_rate` reported; judges see pause-stripped **full CoT**, base condition always evaluated | Addressed |

No structural contradiction found. The two honesty risks are documentation-level: the stale plan text (D1/D2) and any temptation to present the teacher-forced screen as the Stage3 result — the code itself already refuses to let the screen open Stage4.

## Go/No-Go

| Action | Verdict | Basis |
|---|---|---|
| Run Stage2 1.5B `kl_transparent_emit` train + model-comparison eval | **GO** | Invariants machine-enforced; trainer tests green on real torch; judge path scores pause-stripped full CoT. Update plan text (D1/D2) alongside. |
| Claim "full suite green" / land ritual (R12-F1) | **NO-GO until R13-H1 fixed** | `test_safe_rewrite.py` is red in main on any pytest run (reproduced here). |
| Run Stage3 teacher-forced screen | **GO** (as a screen only) | Chain implemented; report fails closed without the confirmatory attach. |
| Run Stage3 confirmatory endpoint | **NO-GO** | B1 producer chain missing (generation layout, judge, control prep, extractor asserts). |
| Stage4 1.5B liveness pilot (2 kernels, calibration only) | **GO** | Kernels measure on self-sampled prefixes at the trained slot; decision stays `incomplete` by design under the 4-test config. Watch F5. |
| Stage4 liveness as a gate / 8B battery / GPRS generation | **NO-GO** | B2 kernels, B4 positive control, B3 hook — each independently fail-closed in code today. |

## Headline verdict

**PASS to run Stage2 with required follow-ups.**

Stage2's invariant is preserved by construction — rows-only training with bit-identity checks, KL-transparency against the model's own pause-stripped distribution, matched triplet data, and judges that score pause-stripped full CoT — and its tests now pass on real torch, not by assertion. Stage3's confirmatory endpoint and Stage4's four-layer gate (scope → metric-derived liveness with recomputed positive control → three-way Stage3 evidence gate with live re-read and no escape hatch set → terminal unimplemented-hook stop) mean the workflow *cannot* skip its own evidence standards even if someone tries. The required follow-ups: fix the one red test in main (R13-H1) before any green-suite claim, update the stale plan text (UltraFeedback/full-SFT/StrongReject wording — D1/D2), and treat the Stage3 on-policy producer chain (B1) as the critical path — until it lands, the four-stage argument stops at a teacher-forced screen, and the code, correctly, will not let Stage4 proceed past it.

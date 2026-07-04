# Full Code-Tree Review: SafeChain Stage2 → Stage4

- Reviewer: Claude Fable 5 (full-tree pass, not diff-based)
- Snapshot: `cot-safety/` at commit `8d5dd4d "Address Fable Stage2 to Stage4 code review"` (git-archive)
- Date: 2026-07-04
- Method: read every Stage2/3/4 entrypoint, trainer, extractor, steering module, test, pipeline shell, and every stage2/3/4 experiment + model + data config end-to-end; two sweep passes over `res/` and the legacy judge/eval tree; verified each claim against file:line.

---

## 1. Top-Level Verdict

**`GO for Stage2 1.5B`** — with three cheap pre-launch actions listed in §7 (none is a correctness blocker; all are launch-hygiene or diagnostics).

Per-stage status:

| Stage | Status | One-line reason |
|---|---|---|
| Stage2 1.5B kl_transparent_emit training | **GO** | Objective, rows-only guarantees, data path, tests, and path chain all verified sound (§4.1) |
| Stage2 8B kl_transparent_emit | GO after 1.5B readout | Same code path; F1 path fix verified; but positive-control gap (F4) means the 8B liveness gate cannot yet be trusted |
| Stage3 as confirmatory evidence | **NO-GO** | Teacher-forced reference labels, aliased content controls, and the declared primary endpoint (`within_prompt_auroc`) has zero implementation (§4.2) |
| Stage3 as exploratory screen | GO after T1 de-alias | Prompt baselines (T2) are wired; the rest is real and runnable |
| Stage4 liveness battery | NO-GO (not implemented) | Honest stub; refuses to run non-dry (`run_stage4_liveness.py:57-62`); zero kernels exist |
| Stage4 GPRS steering eval | NO-GO (correctly fail-closed) | Hard-blocked in code (`run_stage4_steering.py:279-283`); artifacts have no producers |
| Stage4 learned-delta eval | **Danger: NOT blocked** | The deprecated NO-GO pathway is the *default config* of the Stage4 entrypoint and shell (§4.3, N4) |

Bottom line on the central research question: the Stage2 trainer is a faithful implementation of KL-transparent pause emission with no leak back into ordinary SFT that I could find. The inertness risk is real, is not (and cannot be) resolved by Stage2 code, and the codebase currently fails to gate the one Stage4 path that actually executes (learned_delta). Evidence order must be: Stage2 train → emission/transparency readout → liveness battery (once implemented) → fixed Stage3 → GPRS.

---

## 2. Prior-Review Fix Verification (what 8d5dd4d actually addressed)

The snapshot commit claims to address the previous review (F1–F12). Verified status:

| ID | Prior finding | Status in 8d5dd4d | Evidence |
|---|---|---|---|
| F1 | 8B Stage3/Stage4 default checkpoint had spurious `_4xa100` suffix | **FIXED** | `stage3_intra_pause_probe_kl_transparent_8b_cot4_4xa100.yaml:9` and `stage4_pause_gprs_8b_4xa100.yaml:11` both default to `.../deepseek_8b_intra_pause_cot4_kl_transparent_emit_trusted_cot_18k_save50_max400/checkpoint-400`, which exactly matches the Stage2 8B `run.output_dir` (`stage2_intra_pause_kl_transparent_emit_8b_cot4_save50_max400_4xa100.yaml:6`) |
| F2 | Liveness runner printed wrong yellow-decision semantics | **FIXED** | `scripts/run_stage4_liveness.py:48-53` and `src/cot_safety/pipeline.py:166-170` now read: yellow ⇒ proceed on live layers only + queue Stage2.5-A; red with green positive control ⇒ stop Stage4, branch Stage2.5-A/B |
| F3 | 1.5B GPRS `probe_checkpoint` pointed at old non-KL Stage3 run | **FIXED in intent, path not yet producible** | `stage4_pause_gprs.yaml:48` now points at `runs/stage3_intra_pause_probe_kl_transparent_deepseek_1p5b_cot3/probe.pt` — but nothing in the Stage3 pipeline writes a `probe.pt` there; legacy probes land under `legacy/PauseProbe/runs/probes/stage3_kl_transparent_1p5b_cot3_{single,pooled}/...` (see N5) |
| F4 | 8B liveness positive control is format-only (same class as model under test) | **NOT FIXED** | `stage4_pause_gprs_8b_4xa100.yaml:16` still `deepseek_8b_intra_pause_cot4_format_only_trusted_cot_18k_save50_max250/checkpoint-250`. 1.5B control is fine (old full-SFT `.../deepseek_1p5b_intra_pause_cot3_trusted_cot_18k_4xa6000/checkpoint-250`, `stage4_pause_gprs.yaml:23`) |
| F5 | Declared eval keys unconsumed | **NOT FIXED (expanded declaration, still zero consumers)** | `stage4_pause_gprs.yaml:57-63` (`cot_judge`, `over_refusal`, `broken_output`, `unlabeled_rate`) matched nowhere in `src/` or `scripts/` (grep confirms configs-only). Same for `probe.primary_endpoint: within_prompt_auroc`, `probe.on_policy`, `probe.true_content_controls`, `eval.pause_modes` |
| F6 | No torch test for `projection_rejection_update` | **NOT FIXED** | `tests/test_stage4_gprs_liveness.py` contains only config-validation and decision-mapping tests; the actual steering math (`gprs.py:35-61`), including the norm-cap clamp, is untested |
| F7 | `liveness_decision` passed through arbitrary strings | **FIXED** | `steering/liveness.py:57-61` normalizes and maps invalid explicit values to `"unknown"`; covered by `test_stage4_gprs_liveness.py:23` (`"gren"` → `unknown`) |
| F8 | KeyError instead of clean error; `TARGET_SPECS` env bypass | **NOT FIXED** | `cli.py:36` and `run_stage4_steering.py:88` still `item["name"]` (KeyError on malformed spec). `build_env` uses `env.setdefault("TARGET_SPECS", ...)` (`run_stage4_steering.py:153`) so a pre-set env var reaches the legacy shell unvalidated while `validate-scope` only checks the config. Mitigated for GPRS (eval hard-blocked); live for learned_delta |
| F9 | `smoke_test.py` config list not extended | **NOT FIXED** | `scripts/smoke_test.py:18-34` loads none of: the two `stage2_*kl_transparent*` configs, the two `stage3_*kl_transparent*` configs, `stage4_pause_gprs*.yaml`. The new configs are never load-tested, which matters because `config.py` ships a hand-rolled minimal-YAML fallback parser (`config.py:149-195`) |
| F10 | `PAUSE_KL_PAUSE_TOKEN` exported but never forwarded | **NOT FIXED (unchanged, decorative)** | `run_stage2_sft.py:353` exports it; `legacy/COTPauseToken/scripts/training/run_4gpu_intra_pause_sft.sh` forwards all other `PAUSE_KL_*` but hardcodes the special token (`:66`). A config `pause_kl.pause_token` ≠ `<|pause|>` would be silently ignored |
| F11 | `build_gprs_artifacts` placeholder | **NOT FIXED (still placeholder, honestly labeled)** | `pipeline.py:176-192`: the "build" step's command is `run_stage4_steering.py --phase validate` with notes literally saying "Placeholder" |
| F12 | Runtime/launch config mismatch | **PARTIAL** | Launch shells still default to old configs: `pipelines/run_4xa100_stage2_sft.sh:6` → 8B format-only; `pipelines/run_4xa100_stage4_steering_eval.sh:8-9` → learned_delta config with `PHASE=eval`; `run_4xa100_full_pipeline.sh` smoke-plans only format-only/learned-delta configs |

Score: 4 fixed (F1, F2, F3-intent, F7), 1 partial (F12), 6 not fixed (F4, F5, F6, F8, F9, F10, F11-as-labeled). None of the unfixed items blocks the Stage2 1.5B *training run*; several block trusting Stage3/Stage4 outputs (§8).

---

## 3. What Was Audited (coverage statement)

Read in full: `pause_kl_trainer.py` (413 lines), `trl_train.py`, `run_stage2_sft.py` (568), `run_4gpu_intra_pause_sft.sh`, `build_intra_think_pause_sft_splits.py`, `test_stage2_pause_kl_trainer.py`, `run_stage3_intra_pause_probe.py`, `run_intra_pause_probe_full.py` (811), `extract_hidden_states.py` (856), `run_stage4_liveness.py`, `run_stage4_steering.py`, `steering/{gprs,liveness,scope}.py`, `test_stage4_gprs_liveness.py`, `position_locator.py`, `pause_insertion.py`, `smoke_test.py`, `cli.py`, `pipeline.py`, `config.py`, all stage2/3/4 experiment configs, model/data configs, and the four main pipeline shells. Swept via subagents: the whole `res/` tree and the legacy judge/eval/steering scripts.

Two facts from the `res/` sweep that frame everything below:

1. **There are zero kl_transparent results anywhere in `res/`.** Everything in `res/` (Stage1 probes, Stage3 probes, the single Stage4 dir `res/deepseek-8b/stage4_cot3_full250_hardsafe/`) comes from the *old* full-SFT/format-only checkpoints. Nothing in the snapshot is evidence about the KL-transparent model.
2. **There are no liveness-battery results at all.** The battery exists only as a plan-writer.

---

## 4. Findings

Severity scale: **blocker** (invalidates a run or a claim), **high** (will produce wrong conclusions if unaddressed), **medium** (footgun or missing guarantee), **low** (hygiene).

### 4.1 Stage2 — KL-transparent pause emission

The core mechanism is correct. Chain of custody for "only the pause rows train":

- `trl_train.py:138-171` (`configure_format_only_training`) freezes everything, re-enables input+output embedding matrices, and `mask_embedding_gradients` (`:123-135`) registers hooks zeroing all gradient rows except the pause id(s).
- `pause_kl_trainer.py:142-160` (`_assert_rows_only_training`) refuses to run unless weight_decay == 0 and only embedding params are trainable (wd would decay untrained rows despite zeroed grads — correctly caught).
- `_RowsOnlyInvariantCallback` (`pause_kl_trainer.py:25-82`) bit-compares non-pause rows after the first optimizer step.
- Launch shell uses `torch.distributed.run` plain DDP (`run_4gpu_intra_pause_sft.sh`), so the invariant callback sees full (unsharded) weights — the check is meaningful.

Teacher identity: `_pause_stripped_batch` (`:165-199`) strips pause ids to build the teacher input, so the teacher forward never touches the trainable row and is exactly the base model. `_kl_loss` (`:267-288`) masks the pause logit to `finfo.min` on **both** student and teacher sides, so the KL cannot be trivially reduced by suppressing pause probability inside the KL term (that pressure lives only in the explicit suppression loss). `_select_kl_pairs` (`:201-243`) pairs student position `t-1` with teacher position `t'-1` predicting the same next content token, bucketed pre/post pause; `teacher_target_pos <= 0` skipped; per-row cap 256 via linspace (`_cap_pairs`). Emit CE and suppression (`_pause_losses` `:290-329`, chunked logsumexp) match the spec. `compute_loss` (`:331-390`) runs the teacher under `no_grad` + eval mode. The seven torch tests in `tests/test_stage2_pause_kl_trainer.py` cover stripped-batch mapping, pair alignment, pause-slot masking, CE/suppression numerics, both guards, and end-to-end grad flow to embeddings only. This is a genuinely well-tested trainer by this repo's standards.

**Answer to audit Q1 (is it really KL-transparent, or ordinary SFT in disguise): it is really KL-transparent.** The only CE on content tokens is absent by construction — labels enter only at pause slots. No leak found in trainer, config→env mapping (`run_stage2_sft.py:338-353` forces `FORMAT_ONLY=true` when `method: kl_transparent_emit`), or shell forwarding.

Findings:

- **S2-1 (medium, implementation).** The rows-only invariant is checked **once**, after step 1 only (`pause_kl_trainer.py:25-82`). Any later corruption (optimizer state surprise, resume, hook detachment) would be silent for the remaining 399 steps. Cheap fix: re-run the comparison every N steps (e.g., 50) and at train end. Not a launch blocker (ES off, `load_best_model_at_end: false`, no resume planned) but it is exactly the kind of guarantee you want continuous, since the entire scientific claim rests on it.
- **S2-2 (medium, missing diagnostic — audit Q3).** The pause row starts **random-init**: `add_special_tokens` uses `resize_token_embeddings(..., mean_resizing=False)` (`trl_train.py:31-45`) and `init_from_text` is `""` in both KL configs, so `initialize_trainable_token_embeddings` no-ops. Combined with `emit_weight: 0.3` vs `suppression_weight: 1.0`, there is a visible degenerate solution: keep pause probability moderate everywhere, never actually win argmax at pause slots — loss decreases, emission never happens. Nothing in the training loop logs **pause emission rate** or **pause-row norm trajectory**. Diagnostics to add (or run immediately post-hoc — all 17 checkpoints are retained since `save_total_limit: 20` ≥ 16 saves + step-0 save): (a) per-eval-step teacher-free greedy decode of ~20 val prompts, count `<|pause|>` emissions at/near the insertion site; (b) log `||E[pause]||` and cosine vs init at each save. The comparison-eval config for this exists (`stage2_model_comparison_eval_1p5b_kl_transparent_emit_cot3_4xa6000.yaml`) — schedule it right after training, per checkpoint, before picking one.
- **S2-3 (low, implementation).** `compute_loss` ignores `num_items_in_batch`, so gradient-accumulation normalization differs slightly from HF's CE convention. With rows-only training and wd=0 this only rescales the effective lr; note it, don't fix it.
- **S2-4 (low, known F10).** Config `pause_kl.pause_token` is decorative (shell hardcodes `<|pause|>` at `run_4gpu_intra_pause_sft.sh:66`). Fine while the token is `<|pause|>` everywhere; a one-line assert in `run_stage2_sft.py` that the config value equals `<|pause|>` would close it.
- **S2-5 (note, not a bug — audit Q3 context).** The RoPE-shift KL floor is structural: student post-pause tokens sit 3 positions later and attend pause KVs; the teacher has neither. Post-pause KL will not go to zero and its floor concentrates pressure on making pause KVs *unremarkable* — this is precisely the inertness mechanism to watch. Log pre-bucket vs post-bucket KL separately per eval (the buckets already exist in `_select_kl_pairs`); a post-bucket KL that keeps falling toward the pre-bucket level late in training is the inertness smoking gun, not a success signal.
- **S2-6 (verified good — audit Q2).** Data path preserves Stage1/Stage3 assumptions. The builder (`build_intra_think_pause_sft_splits.py:66-113`) inserts `<|pause|>×3` at the char offset of token `first_nonspace_idx + cot_offset`, identical convention to the new `pause_insertion.py:33-67`; the three matched variants (intra / no_pause_matched / pre_think_pause3_matched) are built from the same triplets and validated by three validators wired into `run_stage2_sft.py` data_prep. Row schema is unchanged from Stage1. No break found.
- **S2-7 (verified good — audit Q10).** 1.5B path chain is consistent end-to-end: Stage2 `output_dir` (`stage2_intra_pause_kl_transparent_emit_1p5b_cot3_save25_max400_4xa6000.yaml:7`) == Stage3 default `STAGE2_KL_CHECKPOINT` prefix (`stage3_intra_pause_probe_kl_transparent_1p5b_cot3.yaml:9`) == Stage4 GPRS default (`stage4_pause_gprs.yaml:11`), all `.../checkpoint-400`. Same for 8B after the F1 fix. Caveat: `checkpoint-400` is the **last step** (B7 carry-over) — treat the default as a placeholder and repoint via `STAGE2_KL_CHECKPOINT` after the per-checkpoint readout in S2-2, never silently accept last-step.

### 4.2 Stage3 — pause-position probing

**Answer to audit Q6 (labels on-policy or teacher-forced): teacher-forced, off-policy, reference-labeled — everywhere.** The extraction command built by `run_intra_pause_probe_full.py:384-387` passes `--label_field trajectory_safety_label --pause_layout intra_cot`; rows come from `prepare_external_trajectories.py` (reasoningshield_*, star41k/star1, aidsafe_*, unsafechain_selected, harmthoughts; heldout reasoningshield_test) rewritten by `prepare_intra_pause_probe_data.py` with pauses inserted at `--insert_cot_offset`. The model never generates; hidden states are teacher-forced over external text; labels are the external trajectory labels. The on-policy block exists only as config (`stage3_intra_pause_probe.yaml:32-35`, `enabled: false`) with **no consuming code anywhere**.

Findings:

- **S3-1 (high, scientific — audit Q4).** The declared primary endpoint `probe.primary_endpoint: within_prompt_auroc` (`stage3_intra_pause_probe.yaml:27`) has **zero implementation** — the string appears in three configs and nowhere in code. The legacy probe pipeline computes between-prompt AUROC/balanced-accuracy over rows. Between-prompt AUROC on external trajectories is exactly the metric that collapses into prompt-risk classification. The only guard currently implemented is the prompt-baseline comparison (T2, done): `last_prompt_token`/`pre_think` positions are extracted (`extract_hidden_states.py:486`; wired via `--prompt_positions`, `run_stage3_intra_pause_probe.py:219-220`), so you can at least report pause-AUROC minus prompt-AUROC. That is a *screen*, not proof of trajectory signal. Under the current code, Stage3 **can still collapse into prompt classification**, and the config's own honest flags say so.
- **S3-2 (high, implementation/scientific — T1 carry, audit Q5).** Content controls are aliases: `extract_hidden_states.py:336-339` sets `control_cot_3 = post_pause_1`, `control_cot_4 = post_pause_2`; position_names unconditionally include them for `intra_cot` (`:595`). The alias is now **also codified in the new module** (`position_locator.py:108-111`) and **asserted as correct** in `smoke_test.py:50` and `tests/test_position_locator.py:38-39`. Meanwhile the config declares `true_content_controls.current_control_cot_aliases_valid: false` (`stage3_intra_pause_probe.yaml:29-31`) — code and config now actively disagree, with tests pinning the wrong behavior. Any pause-vs-control comparison, and the pooled spec `control_cot3_cot4_concat_layers_concat`, is partially a self-comparison. Because pauses are inserted before `cot_3`, the token identity chain is `cot_3 ≡ post_pause_1 ≡ control_cot_3` (same activation vector). True content controls must come from the **no-pause matched variant** rows at the same token offsets — the variant already exists in the Stage2 builder; the Stage3 prep does not produce it.
- **S3-3 (medium, migration trap).** `cot_k` means two different things in the two codebases. Legacy extractor: k-th non-pause reasoning token from reasoning start (pause-excluded coordinates) — so with insert offset 3, legacy `cot_3` is the first post-pause content token. New `position_locator.py:100-104`: `cot_{offset} = post_pause_start + offset` — so new `cot_0` is the first post-pause content token and new `cot_3` ≈ legacy `cot_6`. Stage3 configs pass `cot_offsets: [3,4,7,8]` which are correct **only under legacy semantics** (and Stage3 runs through the legacy extractor today, so no current bug). The smoke test exercises the new locator with offsets `[0,1,2]`, hiding the divergence. If extraction is ever migrated to `src/` without an offset translation, every cot position silently shifts by `insert_cot_offset`. Document the coordinate convention in both files now; better, rename the new one (`post_pause_offset_k`).
- **S3-4 (medium, footgun — audit Q10).** The base Stage3 config `stage3_intra_pause_probe.yaml` does not set `model.sft_checkpoint`, so it inherits the **old full-SFT** default from `configs/model/deepseek_r1_distill_qwen_1_5b.yaml:8` (`.../deepseek_intra_pause_cot3_trusted_cot_18k_lr2e5_260615/final`). Running `run_stage3_intra_pause_probe.py` with its default `--config` probes the old model into default-named output dirs. Anyone comparing "Stage3 before vs after KL training" must use the `_kl_transparent_` config explicitly; a default run produces plausible-looking but wrong-model results.
- **S3-5 (low).** `star_min_score` fallback 8.0 and cap recipe `full_3to1` (11,193 rows, 3:1 safe:unsafe) are consistent between config and `run_intra_pause_probe_full.py` defaults. `skip_partial`/`skip_garbage` default true, matching Stage1 conventions. No format break found (audit Q2, Stage3 side).

### 4.3 Stage4 — liveness battery and GPRS steering

**Answer to audit Q7 (is liveness sufficiently implemented before steering): No — and the code mostly, but not completely, fails closed.**

What is genuinely safe:

- `run_stage4_liveness.py` writes only a plan (`status: "planned"|"not_run"`) and hard-exits with an explicit "not implemented" message on any non-dry run (`:57-62`). It cannot produce fake evidence.
- GPRS/projection eval is unconditionally blocked: `run_stage4_steering.py:279-283` raises before touching the legacy generation shell. `--phase validate`/`liveness` are the only GPRS paths that execute.
- Scope enforcement is real and tested: `scope.py` restricts targets to `pause_*` and rejects `pre_pause_/post_pause_/cot_/control_cot_` (`validate_no_pre_post_or_cot_targets`), spec parsing validated (`test_steering_scope.py`), and `validate-scope` runs first in phase `all` (`run_stage4_steering.py:257-263`).
- `projection_rejection_update` (`gprs.py:35-61`) implements `h ← h − λ·max((h−μ_safe)·û,0)·û` with a per-vector norm cap relative to `‖h‖` — the math is correct on inspection (audit Q8: yes, correctly scoped and capped **as written**; but see S4-3).
- `liveness_decision` (`liveness.py:48-72`) is now input-validated (F7 fixed); red dominates yellow dominates green in the status-map path.

Findings:

- **S4-1 (high, footgun — the one Stage4 path that is NOT gated).** The deprecated learned-delta pathway executes end-to-end today, and it is the **default**: `run_stage4_steering.py --config` defaults to `stage4_pause_steering.yaml` (`method: learned_delta`) with `--phase` defaulting to `eval`; `pipelines/run_4xa100_stage4_steering_eval.sh:8-9` defaults the same with `PHASE=eval`; `delta_checkpoint()` (`run_stage4_steering.py:63-75`) silently falls back to the old `learned_delta.pt` path from the NO-GO runs. The liveness step in `pipeline.py` is skipped for learned_delta by default (`liveness.get("enabled", method != "learned_delta")`, `pipeline.py:154`). So the only Stage4 evaluation a naive `bash pipelines/run_4xa100_stage4_steering_eval.sh` performs is the method the project already rejected, against an old delta artifact, with no liveness gate. Minimal fix: make learned_delta require an explicit `--allow-learned-delta` flag (or `steering.acknowledge_deprecated: true`), and/or flip the default config to `stage4_pause_gprs.yaml` (which is fail-closed).
- **S4-2 (high, scientific — F4 carry).** 8B positive control for the liveness gate is a format-only checkpoint (`stage4_pause_gprs_8b_4xa100.yaml:16`) — the same training class as the model under test minus KL. If rows-only training is itself sufficient to produce inert pauses (plausible), the positive control fails green and `require_positive_control_green: true` makes the whole 8B battery unable to certify anything. The 1.5B control (old full-SFT checkpoint-250) is the right class. Either train a small 8B full-SFT pause model for control duty, or calibrate the battery on 1.5B first and treat 8B gating as blocked until a proper control exists.
- **S4-3 (medium, missing guarantee — F6 carry).** No numeric test for `projection_rejection_update`. The norm-cap clamp and the `coeff.clamp_min(0)` gating are the two safety-critical properties (audit Q8); both are one 15-line torch test away. Also note the config's `gate_threshold: 0.95` (probe gate) is validated but **not used** anywhere — the "gated" part of GPRS has no implementation yet; only the projection helper exists.
- **S4-4 (medium — F3 residual / F11 carry).** All three GPRS artifacts have no producers: `mean_diff_direction.pt`, `safe_centroid.pt` (`stage4_pause_gprs.yaml:46-47`) and `probe.pt` (`:48`). The Stage3 pipeline writes per-position/per-layer probe artifacts under `legacy/PauseProbe/runs/probes/...`; nothing exports a canonical `probe.pt` into `runs/stage3_intra_pause_probe_kl_transparent_deepseek_1p5b_cot3/`. `validate_gprs_config` only checks the strings are non-empty (`gprs.py:15-18`), so `--phase validate` passes on paths that can never exist. When the GPRS hook is implemented, add existence checks at eval time and define the artifact-export step (direction from on-policy paired mean-diff at pause positions; centroid from safe rows; probe re-exported from the chosen Stage3 pooled spec).
- **S4-5 (high, missing diagnostics — audit Q9).** The eval endpoint separation exists **only as YAML**. Concretely, in the executable judge path (`legacy/PauseProbe/scripts/judge/`): judges see the full response text only — CoT is never judged separately from the final answer, so the declared `primary_endpoint: unsafe_cot_rate` (`stage4_pause_gprs.yaml:56`) is **uncomputable with current code**; `run_open_judges.py:210` right-truncates judge input at 4096 tokens, so for long CoTs the judge may see mostly CoT and miss the answer, or vice versa; `normalize_judge_outputs.py` (unparsed → "unlabeled", `:150,162-202`) keeps unlabeled rows in denominators, biasing unsafe rates downward under judge failures; over-refusal is computed nowhere; capability/garbage/length/think-end metrics live only in `summarize_model_comparison_eval.py` (Stage2 comparison path) and the steering summarizer covers a subset (`summarize_intra_pause_full_steering_eval.py:208` has think_end_rate). Judge-failure accounting exists only as the `unlabeled_rate: true` YAML key. None of this blocks Stage2; all of it blocks trusting any Stage4 (or Stage2 safety-eval) claim.
- **S4-6 (low).** `eval.pause_modes: [forced, natural, hybrid]` (`stage4_pause_gprs.yaml:55`) is another declared-not-consumed key; the steered generation script inserts pauses forced-only (`run_intra_pause_steered_generation.py:449-459`, `--insert_pause_after_cot_tokens` default 3). The per-forward `pause_ordinals = mask.cumsum(dim=1) - 1` targeting (`:256-262`) is position-correct for the forced layout.
- **S4-7 (low).** `liveness_decision` currently has no consumer other than tests — the green→unlock wiring does not exist. That is safe now (everything is blocked anyway) but the eventual unlock must check the report's `status` field too, since the plan-writer emits decision-less reports which correctly map to `not_run`.

### 4.4 Cross-cutting / configs / tests

- **X-1 (medium — F9 carry).** `smoke_test.py` never loads the four configs that matter for the next three months. Given `config.py`'s minimal-YAML fallback parser (used when PyYAML is absent) and the new nested structures (`target_specs` list-of-dicts, `eval.report`, `liveness.gate`), add all kl_transparent + gprs configs to the smoke list. One-line-each change.
- **X-2 (low).** `smoke_test.py:50` and `test_position_locator.py:38-39` assert the T1 alias as correct behavior. When S3-2 is fixed, these tests must change in the same commit or they will pin the bug.
- **X-3 (low).** `steering.forbidden_target_prefixes` in configs is decorative — `scope.py:21` hardcodes the same list. Harmless (code is the stricter source of truth); note it in the config comment.
- **X-4 (info).** `full_four_stage*.yaml` pipeline plans and `pipeline.py` produce command plans only; nothing auto-executes stages. Good — no hidden orchestration risk found.

---

## 5. Direct Answers to the Ten Audit Questions

1. **Stage2 truly KL-transparent?** Yes. Content tokens receive no CE anywhere; teacher is exactly the base model; pause logit masked from KL on both sides; rows-only enforced by hooks + wd assert + invariant callback. No ordinary-SFT leak found (§4.1).
2. **Stage1 format preserved / Stage3 assumptions intact?** Yes (S2-6, S3-5). Insertion convention identical across legacy builder and new src helper; matched variants and validators wired.
3. **Inertness risk?** Real and unmitigated by the objective itself — the post-pause KL term actively pushes pause KVs toward invisibility (S2-5), and random-init + emit/suppression weighting admits a never-emit degenerate solution (S2-2). Pre-GPU additions: emission-rate probe at eval steps, pause-row norm logging, per-bucket KL logging; post-run: per-checkpoint comparison eval before selecting a checkpoint.
4. **Does Stage3 test trajectory signal beyond prompt risk?** Not yet. Teacher-forced external data + between-prompt AUROC; the declared within-prompt endpoint is unimplemented (S3-1). Prompt baselines allow a delta report — a screen only.
5. **cot3/cot4/post-pause/prompt-baseline/content-control positions?** Prompt baselines: correctly implemented. pre/post-pause: correct. cot_k: correct under legacy semantics but the new src module silently redefines them (S3-3). Content controls: broken by construction — aliases of post_pause positions in both codebases, now test-pinned (S3-2).
6. **On-policy labels?** No. `--label_field trajectory_safety_label`, teacher-forced, reference labels; on-policy is a disabled config stub with no code (§4.2 header).
7. **Stage4 liveness sufficient before steering; placeholder-as-evidence risk?** Battery unimplemented; stub is honest and fail-closed; GPRS eval hard-blocked. One real placeholder-as-evidence risk: the *learned-delta* default path runs end-to-end ungated (S4-1). Second-order risk: `liveness_plan.json` looks like a report; its `status` field is the only thing distinguishing it — keep it mandatory in any consumer.
8. **GPRS correctly scoped, norm-capped, no broad-layer intervention?** The helper math and scope validation are correct; single layer (14/20) targeted; norm cap enforced per-vector. But the probe gate (`gate_threshold`) is unimplemented, the helper is untested (S4-3), and no generation hook exists yet — so "correct" currently means "correct on 61 lines of unexercised code."
9. **Eval endpoints cleanly separated?** No. Declared in YAML, not computed: unsafe-CoT vs final-answer split, over-refusal, unlabeled-rate accounting; judge truncation and unlabeled-in-denominator bias the existing rates (S4-5).
10. **Config/paths/checkpoints consistent for Stage2 1.5B launch?** Yes for the 1.5B chain (S2-7), with three footguns: last-step default checkpoint (B7), old-model defaults in base Stage3 config and launch shells (S3-4, F12), and decorative `pause_token` key (S2-4).

---

## 6. Findings by Category

**Implementation bugs (current behavior wrong):**
- S3-2 / T1: content-control aliasing (extractor `:336-339`, locator `:108-111`), now also test-pinned (X-2).
- S4-5 parts: unlabeled rows in judge denominators; 4096 judge truncation (silent evidence corruption when judging long CoTs).
- F8: KeyError on malformed target_specs; TARGET_SPECS env bypass of scope validation (learned_delta path).

**Scientific-design risks (code runs, conclusions wrong):**
- S3-1: no within-prompt endpoint, teacher-forced off-policy labels → prompt-classification collapse.
- S2-5: post-pause KL floor is the inertness mechanism; low KL late in training is ambiguous between "transparent" and "inert".
- S4-2 / F4: 8B positive control in the wrong class → battery cannot certify.
- B7: last-step checkpoint defaults, no selection rule.

**Missing diagnostics:**
- S2-2: pause emission rate + row-norm logging (pre-GPU or immediately post-run).
- S2-1: invariant check frequency.
- S4-5: unsafe-CoT vs answer split, over-refusal, unlabeled-rate; S4-3: torch test for the projection helper.

**Documentation / config clarity:**
- F5-family: seven declared-not-consumed keys (`eval.report.*`, `within_prompt_auroc`, `on_policy`, `true_content_controls`, `pause_modes`, `forbidden_target_prefixes`, `gate_threshold`). Each should carry a `# NOT IMPLEMENTED — declaration of intent` comment until wired, or they *will* be misread as implemented.
- S3-3: `cot_k` coordinate divergence between legacy and src.
- S3-4 / S4-1 / F12: default configs and shells point at old-model / deprecated pathways.

---

## 7. Minimal Changes Before Launching Stage2 1.5B

None of these blocks correctness; items 1–2 are strongly recommended, 3 is hygiene. Total effort well under a day.

1. **Emission + row diagnostics (S2-2).** Either add per-eval-step pause-emission logging to the trainer, or commit to the post-hoc plan: after training, run the existing 1.5B comparison-eval config on every retained checkpoint and record pause-emission rate, think-end rate, and length shift per checkpoint before selecting. Also log per-bucket (pre/post) KL — the numbers already exist inside `_select_kl_pairs`/`_kl_loss`; they just need to reach the logs.
2. **Invariant frequency (S2-1).** Re-run `_RowsOnlyInvariantCallback`'s comparison every 50 steps and at train end.
3. **Smoke-load the launch config (X-1/F9).** Add the two stage2 kl configs (and ideally stage3-kl + stage4-gprs) to `smoke_test.py`; or at minimum run `cot-safety config show --config configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_cot3_save25_max400_4xa6000.yaml` on the pod before launch. Launch via explicit `CONFIG=...` (the stage2 shell default is the old 8B format-only config).

Explicitly not required before launch: F4, F6, F8, F10, F11, S3-*, S4-* — they gate later stages, not this run.

---

## 8. Minimal Stage3/Stage4 Changes Before Trusting Those Stages

**Stage3 (before any confirmatory claim):**
1. De-alias content controls (S3-2): extract control positions from the no-pause matched variant of the same rows at the same legacy-coordinate offsets; delete or rename `control_cot_3/4` aliases in both extractor and locator; update the two pinning tests and the pooled spec in the same commit.
2. Implement the within-prompt endpoint (S3-1): on-policy sampling (N≈10 per prompt) from the pause model on Stage1-distribution prompts, CoT-segment judge labels, within-prompt AUROC at pause positions, with the teacher-forced pipeline retained as the cheap screen. Until then, every Stage3 number must be reported side-by-side with the prompt-baseline AUROC at the same layer, and any pause-minus-prompt gap ≤ ~0.02 treated as "no trajectory signal demonstrated".
3. Keep using the `_kl_transparent_` configs explicitly; never the base Stage3 config (S3-4).

**Stage4 (before any steering evidence):**
1. Gate or de-default learned_delta (S4-1) — this is the highest-priority single change in the repo after Stage2 launch, because it is the only path where a routine command produces official-looking steering CSVs from a rejected method.
2. Implement the liveness kernels (T13) behind the existing plan schema; wire `liveness_decision` + `status` into the GPRS unlock; fix the 8B positive control class (S4-2) or scope the battery to 1.5B.
3. Implement artifact producers + existence checks for direction/centroid/probe (S4-4); add the probe-gate application (currently `gate_threshold` is validated, unused).
4. Endpoint separation in the judge path (S4-5): judge CoT and final answer separately (two judge calls or segment-split input), exclude unlabeled rows from denominators and report `unlabeled_rate`, raise/segment the 4096 truncation, add over-refusal on the safe sets (xstest_safe, or_bench_hard_safe are already in the dataset specs).
5. Add the torch test for `projection_rejection_update` incl. norm-cap and clamp behavior (S4-3).

---

## 9. Can the Trained Pause Token Be Used for Stage4 Steering Under Current Code?

**No — and the code should be read as (mostly) agreeing.** Three independent reasons:

1. **Epistemic:** the KL objective plausibly trains the pause ports toward causal inertness (S2-5). Without a green liveness battery (which does not exist yet as code) plus a de-aliased, prompt-baseline-beating Stage3 signal, steering at pause positions has no demonstrated mechanism to act through. Liveness/probe evidence is a hard prerequisite, not a formality.
2. **Mechanical:** GPRS cannot run — eval is hard-blocked (`run_stage4_steering.py:279-283`), the generation hook does not exist, and none of the three required artifacts has a producer (S4-4).
3. **The exception that must be closed:** the learned-delta path *can* run today with defaults and would produce official-looking steering results from the already-rejected method with no liveness gate (S4-1). Until that is gated, "the code enforces evidence-first" is only true for GPRS.

So: liveness battery (implemented, green, with a valid positive control) → fixed Stage3 (de-aliased controls, within-prompt endpoint) → artifact build with QC → GPRS with random-direction control. In that order; the configs already document this order in the `next_step` string — the code just doesn't enforce it beyond fail-closed stubs yet.

---

## 10. If KL-Transparent Emission Proves Insufficient (Stronger Alternatives)

Ordered by how little they change the scientific claim:

1. **Two-phase schedule (cheapest).** Phase A: emit CE + suppression only (no continuation KL) for the first ~25% of steps so the pause row lands in a useful region; Phase B: turn on continuation KL. Removes the never-emit degenerate optimum without touching transparency at convergence. Pure config/trainer-schedule change.
2. **Liveness-floor regularizer (Stage2.5-A class).** Add a small penalty when post-pause tokens' attention mass to pause KVs drops below a floor (or when pause-KV ablation changes next-token logits less than ε). Directly optimizes against inertness; keeps content CE absent. Cost: needs attention/ablation instrumentation in the trainer — the same instrumentation the liveness battery needs anyway, so build once.
3. **Information-preserving auxiliary head (Stage2.5-B class).** Train a tiny frozen-backbone probe head on pause hidden states during Stage2 with a small-weight InfoNCE/classification loss (labels: any cheap trajectory property — even source or topic, deliberately *not* safety, to avoid baking the Stage3 answer into training). Guarantees non-degenerate pause states while keeping the safety-separability question honest. Must be disclosed as making the pause a trained port.
4. **Relax transparency instead of forcing it.** If the battery shows KL-transparency and liveness are empirically incompatible at this scale, drop the continuation KL to a small weight (e.g., 0.1), accept a measured distribution shift, and quantify capability/length/refusal deltas with the (fixed) endpoint suite. The old full-SFT 1.5B run suggests the shift may be tolerable; that is an empirical question the comparison eval can answer cheaply.
5. **Contrastive pause shaping (most invasive).** Pair safe/unsafe prompts (Stage1 SafeChain data) and add a contrastive objective on pause states. This abandons "emergent separability" as a claim — the pause becomes a supervised safety port — but it is the strongest guarantee that Stage4 has something to steer. Only if 1–4 fail.

---

## 11. Scaffolding Safety Assessment

Safe as scaffolding (honest, fail-closed, cannot be misread by a careful reader): liveness plan-writer, GPRS eval block, `pipeline.py` placeholder step (labeled "Placeholder"), scope validators, on_policy/`true_content_controls` config stubs (they self-declare invalidity).

Dangerous as-is (can be misread as completed or produce evidence-shaped artifacts):
- The learned-delta default eval path (S4-1) — produces real CSVs from a rejected method.
- Declared-not-consumed eval keys (F5 family) — a reader of `stage4_pause_gprs.yaml` will believe over-refusal and CoT-judging exist; they do not.
- Test-pinned control aliases (S3-2/X-2) — the test suite currently certifies the known-invalid control design.
- Base Stage3 config's old-model default (S3-4) — default runs generate wrong-model artifacts with legitimate-looking names.

— End of review —

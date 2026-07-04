# Fable Review — SafeChain Stage2→Stage3→Stage4 End-to-End Flow (pre-first-Stage2-run)

Date: 2026-07-04. Reviewer: Claude (claude-fable-5). Read-only; no code edited.

Basis: `FABLE_STAGE2_TO_FOUR_FLOW` request; all three referenced prior Fable
reviews read in full (`stage2_context/CLAUDE_FABLE5_STAGE2_FULL_FLOW_REVIEW.md`,
`CLAUDE_FABLE5_STAGE3_STAGE4_REVIEW.md` "the main review",
`CLAUDE_FABLE5_STAGE3_STAGE4_FOLLOWUP_REVIEW.md` "the follow-up"), plus the
Round 3 Stage2 review in `stage2_context/`. Every load-bearing claim below was
re-verified against the current packet source, not just carried: the two KL
configs, `pause_kl_trainer.py` (in full), `run_stage2_sft.py` env plumbing,
`run_4gpu_intra_pause_sft.sh` PAUSE_KL block, `tests/test_stage2_pause_kl_trainer.py`
(in full), `extract_hidden_states.py:225–354`, `run_intra_pause_probe_full.py`
extraction command, `run_stage3_intra_pause_probe.py` position plumbing,
`run_stage4_steering.py` phases, both Stage4 configs, and the steering eval
shell. Blocker/TODO IDs (B1–B7, T1–T16) refer to the main review; Stage2.5-A/B/C
and P0/P0.5/P1/P2 refer to the follow-up.

---

## 0. Executive Verdict

**The user's sequencing is correct in structure — train Stage2 first, measure
liveness second, Stage2.5 only on failure — with four amendments, none of which
change the order of GPU spend:**

1. **Yellow ≠ red.** The intended step 4 ("if liveness is yellow/red, add
   Stage2.5 and retrain") over-triggers the retrain. Per the follow-up's §5
   decision table: **yellow proceeds, restricted to the live layers**, with
   Stage2.5-A merely *queued* for the next Stage2 training run. Only **red**
   (with a green positive control) forces the Stage2.5-A+B retrain now.
2. **The Stage3/Stage4 framework code and the liveness battery script are
   written *before/during* the Stage2 run, not after.** All of it is
   laptop-safe and none of it depends on the checkpoint existing — only the
   *runs* do. Writing it after the checkpoint exists puts a multi-day code
   sprint between you and the next GPU decision. The battery script (T13) is
   the single most important unwritten artifact in the repo right now: it is
   the very next GPU gate after training, and it does not exist (verified: no
   liveness/injection-gain code anywhere in the packet).
3. **Checkpoint selection precedes liveness.** Post-training order is:
   step-1 invariant green → Stage2 eval battery (natural emission, uncapped KL
   drift, behavior equivalence) → *choose* the checkpoint → run liveness on
   that checkpoint. Running liveness on `final/` untested is testing the wrong
   artifact — on the 1.5B config `final/` is the **last-step** model
   (`load_best_model_at_end: false`), not the best one.
4. **Two ops prerequisites belong before the Stage2 launch, not after:**
   confirm the old cot3 full-SFT ckpt250 (the battery's positive control) is
   still retrievable, and read `tie_word_embeddings` from both base model
   configs (no checkpoint needed — Stage2 never changes it). A battery without
   its positive control cannot define "dead," and per the follow-up §3.4 the
   tied-1.5B / untied-8B asymmetry means 1.5B battery results must be
   pre-registered as **non-transferable** to 8B.

Everything else in this review is the operational detail: what to land in the
initial PR (§8), what is mandatory before each GPU milestone (§5–§6), the
Stage2.5 implementation policy (§7), and the exact kill criteria (§7a/§9
table). One new finding from the source re-verification: **the 8B Stage2
config silently diverges from the reviewed-and-fixed 1.5B pattern** (§1,
NEW-F1) — config-only fix, required before the 8B run, irrelevant to the 1.5B
launch.

**Stage2 launch verdict is unchanged from Round 3: GO, gated only on pod
pytest → single-GPU smoke → 4-GPU smoke.** Nothing in this review blocks it,
and nothing in the Stage3/Stage4 plan should be allowed to delay it.

---

## 1. Verified Current State (what has actually landed)

Re-checked at source in this packet; this is the ground truth the plan stands on.

| Item | Status | Evidence |
|---|---|---|
| B1 Hydra `pause_token` override | **FIXED** — override deleted (preferred option) | `run_4gpu_intra_pause_sft.sh:103–119`: PAUSE_KL block passes only bare floats/bools/ints; no `pause_token` line |
| B3/C2 rows-only invariant guard | **LANDED** | weight-decay assert `pause_kl_trainer.py:143–148`; `_RowsOnlyInvariantCallback` step-1 bit-identity `:25–82`; wired at `:133–136`; config keys `assert_rows_only`/`post_step_invariant_check` present in both KL configs |
| C3/P1 `.item()` sync storm | **LANDED** | one `.cpu().tolist()` per tensor: `:170–171`, `:210–212` |
| C4/P2 suppression memory spike | **LANDED** | chunked pause-column `logsumexp`, `suppression_chunk_size` plumbed: `:313–326` |
| Unit tests | **EXIST, NEVER EXECUTED** | `tests/test_stage2_pause_kl_trainer.py`: 7 tests covering T-1…T-5, T-7 (stripped batch + mapping, pair alignment incl. last-pause→first-content, pause-slot mask finiteness, hand-computed CE/suppression, weight-decay + body-param guard rejection, callback mutation detection, end-to-end finite loss with grad isolation). No torch in this environment either — **pod pytest remains the sole gate before smoke.** Note: no direct teacher-equals-base logit test (old T-6); acceptable — the invariant callback covers the failure mode in the real run |
| 1.5B KL config | Round-3 pattern confirmed | `load_best_model_at_end: false`, `early_stopping.enabled: false`, save 25 / eval 50 / limit 20, `weight_decay: 0.0` explicit |
| 8B KL config | **DIVERGES — NEW-F1** | `load_best_model_at_end: true` (:19), `early_stopping.enabled: true, patience: 2` (:22–25) |
| Stage2 uncapped KL-drift eval | **STILL ABSENT** | carried claim-blocker (E1 remnant / Round 3 Q4-1): transparency-preservation claims blocked until it exists; also needed as the transparency arm of checkpoint selection |
| Stage3 control aliasing | **PRESENT** (S3-1 stands) | `extract_hidden_states.py:336–339`: `control_cot_3 = post_pause_1`, `control_cot_4 = post_pause_2` |
| Stage3 prompt baselines | **SUPPORTED IN EXTRACTION, NEVER PLUMBED** | `extract_hidden_states.py:247–252, 300–301` already implement `last_prompt_token`/`assistant_start`/`assistant_last`/`pre_think` behind `--prompt_positions` (allowed set `:486`); the Stage3 orchestrator's `extraction_cmd` (`run_intra_pause_probe_full.py:359–402`) **never passes the flag**, and the Stage3 configs never list the positions. T2 is therefore mostly plumbing, not new extraction code |
| Prompt-only text baseline | **SCRIPT EXISTS** | `legacy/PauseProbe/scripts/probe/train_text_artifact_baseline.py` (lexical/length baselines for prompt-risk) — adaptable for the T2 text-classifier baseline instead of writing from scratch |
| Stage4 liveness | **NOTHING EXISTS** | no battery script, no config keys, no runner phase (`run_stage4_steering.py:233` phases: validate/generation/judge/summary/eval) |
| Stage4 method | learned_delta still primary | `stage4_pause_steering.yaml:11`; inert keys `init/steps/safe_weight/unsafe_weight/l2/loss`, `eval.model_conditions`, `eval.capability`, `eval.safety` all still advertised and unconsumed |
| 8B Stage4 provenance break (B6/T14) | **PRESENT** | `stage4_pause_steering_8b_4xa100.yaml:11–12`: `layer: 16` + `zero_l16_steps80` delta vs. committed results at layer 20 |
| Eval shell offset hardcode (B6/T12) | **PRESENT** | `run_intra_pause_full_steering_eval.sh:297`: `--insert_pause_after_cot_tokens 3`, no env override — wrong layout for the cot4 8B model |
| Judge truncation (B3-eval/T8) | **PRESENT** | `JUDGE_MAX_INPUT_LENGTH` env exists (`run_stage4_steering.py:173`, default 4096) but right-truncation still eats the template tail; summarizer still hides the `unlabeled` bucket |
| Stage2.5 code | **ABSENT (correct)** | no near-pause bucket, no hinge in `pause_kl_trainer.py` — consistent with "conditional fallback, not default" |

**NEW-F1 (fix before the 8B run; config-only; does not touch the 1.5B launch):**
the 8B KL config kept the pre-Round-2 pattern. Consequences if run as-is:
(a) `final/` means *best-composite* on 8B but *last-step* on 1.5B — a
checkpoint-selection trap across the two lines; (b) early stopping with
patience 2 on a composite loss whose parts move at different speeds risks a
premature stop (Round 1 flagged this; the 1.5B fix disabled it); (c) it
re-introduces the latent hot-sync hazard Round 3 noted was incidentally fixed
on 1.5B (`--remove-hot-after-sync` can delete the checkpoint
`load_best_model_at_end` needs to reload at train end); (d) the eval battery
wants all checkpoints, and `load_best` adds nothing when selection is done by
the battery. Align the 8B config to the 1.5B pattern (`load_best: false`,
`early_stopping.enabled: false`) in the initial PR. Save 50 / eval 50 stays
legal either way.

Minor (note, not a fix): `run_stage2_sft.py:353` still exports
`PAUSE_KL_PAUSE_TOKEN`, but the shell no longer forwards it to Hydra (that was
the B1 fix). The config's `pause_kl.pause_token` key is therefore decorative —
the trainer uses its built-in `<|pause|>` default, which matches. Only matters
if anyone ever changes the token string; leave it, but know it.

---

## 2. Q1 — Is the sequencing correct?

**Yes, with the four amendments from §0.** The corrected sequence, explicit:

1. **Now, pre-GPU (parallel tracks):**
   - Track A (Stage2 launch path, frozen code): pod pytest → 1.5B single-GPU
     smoke → 4-GPU smoke → 400-step 1.5B `kl_transparent_emit` run.
   - Track B (laptop-safe, lands during the same window): the initial
     Stage3/Stage4 framework PR (§8) — battery script first — plus the two ops
     checks (positive-control checkpoint retrievable; `tie_word_embeddings`
     read and pre-registered) and the Stage2 uncapped-KL-drift eval script.
   - Track A is never gated on Track B.
2. **After the checkpoint exists:** step-1 invariant confirmed green on all
   ranks → Stage2 eval battery (natural emission stats via the fixed NEW-B2
   eval; uncapped KL drift vs base; behavior equivalence, no insertion) →
   choose the candidate checkpoint(s) (2–4, by eval-loss *parts* and battery
   results, never by trusting `final/`).
3. **Liveness battery** (§8 of the main review / §8 of the follow-up) on
   {chosen KL checkpoint, full-SFT ckpt250 positive control, base with pasted
   pauses}. Positive control not green ⇒ battery invalid — fix the battery,
   not Stage2.
4. **Branch:**
   - **Green** → Stage2 is untouched (adding Stage2.5 terms now would be
     harmful surplus — they spend transparency budget for a property the
     checkpoint already has). Proceed: fixed Stage3 rerun → direction QC →
     GPRS micro-pilot → 1.5B pilot.
   - **Yellow** → proceed exactly as green but **restricted to the live
     layers**; queue Stage2.5-A for the next Stage2 training run; re-battery
     then.
   - **Red** (positive control green) → Stage4 stops; Stage2.5-A(+B) retrain
     (~400 rows-only steps, not a redesign); re-battery. Still red after A+B
     at `w_live ≤ 1.0` ⇒ the recipe's premise fails ⇒ pre-registered
     publishable negative + scope decision.
5. **8B strictly after the 1.5B chain is analyzed**, with NEW-F1 fixed, cot4
   offset plumbed (T12), and its **own** battery run — the tied(1.5B)/
   untied(8B) embedding asymmetry makes 1.5B liveness evidence inadmissible
   for 8B.

One more correction of emphasis: **liveness green does not license trajectory
claims.** There is a second, independent gate downstream — within-prompt AUROC
in the fixed Stage3 (§5). Failing it does not stop Stage4; it re-scopes every
claim to "prompt-conditioned gating." The intended sequencing treats liveness
as the only post-Stage2 gate; it is the first of two.

---

## 3. Q2 — Minimal code framework to add now

Everything below is laptop-safe (no GPU, no checkpoint dependency). Ordered by
what prevents invalid GPU spend soonest. "M" = mandatory before the indicated
milestone; "R" = recommended now because it is cheap and on the critical path.

| Pri | Item | Files | Class |
|---|---|---|---|
| 1 | **T13 liveness battery script + config** — injection-gain curves (layers {7,14,21} 1.5B / {8,16,20,24} 8B; ε∈{1,2,4}·σ_h; v ∈ 3 random seeds + mean-diff; next-16-token KL; content-token and BOS anchors) + pause-KV ablation (zero/mean, 64-token greedy continuation KL + edit distance); one JSON report per model/layer; thresholds block with `calibrate_on: positive_control` | new `scripts/steering/run_pause_liveness_battery.py`; new `configs/experiment/stage4_liveness_1p5b.yaml` (models: test/positive_control/base; add a `liveness` phase or standalone runner) | **M before any Stage4 decision — the next GPU gate** |
| 2 | **T10 CoT-segment judging** — split `generated` at `</think>`; judge CoT (ReasoningShield-style prompt; note: no reasoningshield judge config exists yet in `configs/judge/` — needs a config + model path) and answer separately; two normalized files per shard; summarizer reports both | `run_open_judges.py`, `normalize_judge_outputs.py`, summarizer, new `configs/judge/reasoningshield.yaml` | **M — prerequisite for BOTH Stage3 on-policy labels and the Stage4 primary endpoint** |
| 3 | **T1 de-alias Stage3 controls** — delete the `:336–339` aliasing; add the pause-free-forward matched-content control (second forward on the stripped row; positions `nopause_cot_k`) as the *primary* control; matched-depth in-paused-sequence tokens as secondary | `extract_hidden_states.py`, Stage3 configs (drop `control_cot_3/4` or rename to `alias_post_pause_*` so no one ever reads them as controls again) | **M before the Stage3 extraction run** |
| 4 | **T2 prompt baselines** — plumb `--prompt_positions last_prompt_token,pre_think` through the three touch points (config `hidden.positions.diagnostics` → `run_stage3_intra_pause_probe.py` → `run_intra_pause_probe_full.py` new arg → `extraction_cmd`); extraction support already exists; adapt `train_text_artifact_baseline.py` as the prompt-only text classifier | as listed | **M before the Stage3 extraction run** (cheap: mostly plumbing) |
| 5 | **Stage3 on-policy config fields** — `labels.source: provenance\|on_policy_judge`, `labels.on_policy.{samples_per_prompt: 10, judge: cot_segment, mixed_outcome_band: [0.2, 0.8], min_mixed_prompts: 100}`; `within_prompt_auroc` added to `probe.metrics` + the analysis that computes it (restricted to mixed-outcome prompts) with bootstrap CIs | Stage3 configs, probe analysis scripts | **M before Stage3 *claims*; fields+analysis now, data later** |
| 6 | **T8 judge-truncation cause-fix** — truncate the *response segment* head+tail, never the template/instruction tail; per-row `truncated` flag | `run_open_judges.py` | **M before any judged readout** |
| 7 | **T9 summarizer integrity** — `unlabeled` count + rate + labeled-only rate columns; EOS-termination + repetition-loop columns; refusal keywords on the answer segment only | `summarize_intra_pause_full_steering_eval.py` | **M before any judged readout** |
| 8 | **T12 offset/mode plumbing** — `INSERT_AFTER_COT_TOKENS`/`N_INSERT_PAUSES` env-parameterized (kills the `:297` hardcode); `PAUSE_MODE={forced,natural,hybrid}`; plumbed from `run_stage4_steering.py` config keys | eval shell + runner + configs | **M before any 8B steering anything; R now** |
| 9 | **T5/T6 GPRS port** — projection edit `h ← h − λ((h−μ_safe)·û)₊û` with norm cap ρ; per-step gate in standardized space (τ from `probe.pt` FPR≤0.05); persistent per-sequence pause-ordinal counter across decode steps; `--steer_natural_pauses {on,off}`; hook stats split forced/natural; port the existing math from `run_intra_pause_activation_pilot.py` (mean_diff / probe-weight `w/std` / safe_centroid_pull / gate all already implemented there) | `run_intra_pause_steered_generation.py` | **M before the micro-pilot; R now** (follow-up already GO'd it) |
| 10 | **Stage4 config schema swap** — `steering.method: gprs` with `direction_artifact`, `safe_centroid`, `probe_checkpoint`, `gate.threshold_source`, `norm_cap`, `lambda`, `layer: null` + `layer_source: liveness_report`; `learned_delta` moved under `baselines:` (control only); delete or wire the inert keys; fix the 8B layer-16/20 provenance break (T14) | both Stage4 YAMLs, `run_stage4_steering.py` | **M before the micro-pilot** |
| 11 | **Stage2 uncapped KL-drift eval** — mean/p95/max per-token KL(base‖ckpt) on held-out no-pause completions + aligned pause-containing positions | new small eval script | **M for transparency claims; R now** (it is also the checkpoint-selection instrument) |
| 12 | T7 direction tooling (on-policy paired pause-state NPZ → û, μ_safe, QC report: seed cosine, probe-transfer AUROC); T11 capability wiring (or delete the advertising keys); T15 provenance manifests; T16 scope-guard over `TARGET_SPECS` + pilot args; NEW-F1 8B Stage2 config alignment | as listed | R now |

Explicitly **not** in the minimal framework: T3 natural-pause extraction
(write only if measured emission > ~5%, or land the config switch now and the
code later); any Stage2.5 code on the launch path (§7); anything touching
`pause_kl_trainer.py` or the Stage2 launch chain (frozen, §7).

---

## 4. Q3 — Mandatory Stage3 changes before the first post-Stage2 probe run

Split by what physically gates what. Extraction is the expensive part of
Stage3 — anything that changes the NPZ contents must land **before** the run
or the run is a write-off.

**Before the extraction run (hard):**

1. **B7 carried:** `model.sft_checkpoint` repointed (or `MODEL` env) to the
   battery-chosen checkpoint — `final/` is last-step on 1.5B; rows-only
   invariant confirmed on the trained weights; pod pytest green.
2. **T1** de-aliased controls, pause-free-forward control primary (§3 item 3).
   Without it, every control comparison in the run is vacuous — that is the
   definition of an invalid GPU run.
3. **T2** prompt positions plumbed (`last_prompt_token`, `pre_think`) and in
   the config position list (§3 item 4). The null hypothesis must be in the
   same NPZ.
4. **Offsets 7/8 layouts included in the same run** (configs already carry
   `cot_offsets: [3,4,7,8]`) — the follow-up's depth caveat: near-chance
   within-prompt AUROC at offset-3 alone is a depth result, not a port result.
5. **Battery outcome known first.** Not a code change, a sequencing gate: do
   not pay for Stage3 extraction on a checkpoint the battery is about to send
   back to Stage2.5 retraining.

**Before Stage3 *claims* (can trail the first extraction):**

6. **T4/T10** on-policy path: 10 samples/prompt with CoT-segment judge labels;
   within-prompt AUROC on mixed-outcome prompts (band [0.2, 0.8], ≥100
   prompts) as the **primary endpoint**; prompt features score 0.5 on it by
   construction.
7. Bootstrap/DeLong CIs + seed variation on every AUROC delta.
8. Prompt-only text-classifier baseline in the report (adapted
   `train_text_artifact_baseline.py`) — no extraction dependency, can be
   post-hoc.
9. **T3** natural-pause extraction iff measured natural emission > ~5%.

Gate semantics carried from the follow-up: the required comparison is *pause
vs prompt-baseline* (trajectory-vs-prompt). *Pause vs content-token* is
context, not a gate — content tokens carrying more signal is expected and
harmless. Headline metric = pause positions themselves, not `post_pause_*`.

**Q3 one-line answer:** T1 + T2 + B7 (+ offsets 7/8 in the same pass), with
the battery result in hand — everything else is claims-side and can trail.

---

## 5. Q4 — Mandatory Stage4 changes before any steering pilot

In gate order; every one is cheap relative to a wasted judged 3-seed run:

1. **T13 battery green or yellow** at the intended steering layer(s), thresholds
   calibrated on the full-SFT positive control. Red ⇒ there is no Stage4.
2. **B2 executed:** learned delta demoted to a baseline arm; primary direction
   = on-policy **within-prompt paired** mean-diff û (prompt-identity component
   cancels by construction), un-standardized space, with QC gates: seed-resample
   cosine ≥ 0.8, held-out 1-D projection AUROC reported, cosine vs probe-weight
   `w/std` direction reported.
3. **T5 GPRS port + T6 ordinal counter/natural-pause switch** — without T6,
   `pause1_only`/`pause2_only` ablations remain silently wrong for decode-time
   pauses and naturals get steered undocumented.
4. **T12 offset/mode plumbing** — at 1.5B the hardcoded 3 happens to match
   cot3, so this is formally 8B-blocking; land it anyway before the pilot so
   forced/natural/hybrid are explicit conditions, not accidents.
5. **Eval integrity before any readout:** T8 (truncation cause-fix — reporting
   the unlabeled rate without fixing the cause leaves the α-correlated
   missingness confound), T9 (unlabeled + labeled-only + termination/repetition
   columns), T10 (CoT-segment judge — the primary endpoint is unsafe-**CoT**
   rate, currently unmeasured), T11 (capability EM with the hook active, or
   stop advertising it), calibrated over-refusal on answer segments only.
   Readout acceptance gate: unlabeled < 5% per condition.
6. **Micro-pilot harness with the random-direction norm-matched control**
   (n≈100/label, 1 seed) — the only test separating "unsafe-direction removal"
   from "any perturbation induces caution." A GPRS win without it is
   uninterpretable; do not skip to the 3-seed pilot.
7. **T14** 8B config provenance fixed + steering layer sourced from the
   liveness report (config `layer_source: liveness_report`), **T15** manifests
   (incl. actual MAX_NEW_TOKENS), **T16** guard closure.

---

## 6. Q5 — Implement Stage2.5 now (disabled) or wait?

**Split the question in two:**

- **Do not put Stage2.5 anywhere near the launch path now.** The trainer and
  launch chain are frozen and green-lit (Round 3); any edit to
  `pause_kl_trainer.py` re-opens review and re-testing of the exact code the
  400-step run depends on. The single worst trade available this week is
  delaying the measurement (the Stage2 run + battery) to pre-build the remedy
  whose necessity the measurement decides. The follow-up's asymmetry stands:
  on green, Stage2.5 is not "nice to have," it is harmful surplus.
- **Do write Stage2.5-A(+B) on a side branch during the training window.**
  That window is dead time; the follow-up already marked the implementation
  "GO now (laptop-safe)." Requirements for that branch: (a) Stage2.5-A as a
  re-weight in `_select_kl_pairs` (`near_pause_exempt_tokens`,
  `near_pause_weight`) with defaults that reproduce the current loss
  **bit-identically**, proven by a unit test (default-config loss equality vs
  the frozen trainer); (b) Stage2.5-B hinge with `m` left as a config key —
  it *cannot* be finalized until the positive-control battery numbers exist,
  so there is no completeness argument for pre-merging anyway; (c) merged
  only on a red battery (or A queued on yellow, next train).
- **Priority order within Track B: battery script (T13) strictly before
  Stage2.5 code.** T13 decides whether the branch is ever merged; the reverse
  is not true.

So: not "implement now but keep disabled in main," and not "wait until
liveness fails to start typing" — **branch-implement during training, merge
only on the P0.5 branch condition.**

---

## 7. Q6 — Exact kill criteria

### 7a. Decision table

| Decision | Criterion (all numeric gates pre-registered) |
|---|---|
| **Launch Stage2 1.5B** | Pod pytest green (suite has never executed anywhere — sole gate) → single-GPU smoke (20 steps: loss parts finite, ckpt roundtrip, invariant callback fires green) → 4-GPU smoke (no DDP hang on pause-free micro-batches, rank0 save, resume works). |
| **Stage2 run valid** | Step-1 rows-only invariant green on all ranks. **Fail ⇒ stop everything; it is a Stage2 bug, nothing downstream is interpretable.** |
| **Checkpoint selectable** | Stage2 eval battery: natural emission stats measured (any value passes — it scopes claims, it does not kill); uncapped KL drift finite and plateaued (plateau ≈ the RoPE floor is *expected*, not a failure); behavior equivalence no-insertion: GSM8K/MATH500 slice \|Δ\| ≤ 0.5 pt vs base, think_end/length within seed noise. Gross violations ⇒ treat as Stage2 training failure, not a Stage2.5 case. |
| **Proceed to Stage3 rerun** | Battery **green or yellow** (below) on the chosen checkpoint + T1/T2 landed + B7 done. Battery red ⇒ do not spend on Stage3; the checkpoint is being replaced. |
| **Battery GREEN** | ≥1 admissible mid layer with pause injection-gain ≥ **25%** of the matched content-token anchor **and** ≥ **5×** the BOS anchor (same ε), **and** pause-KV ablation effect clearly nonzero vs seed noise. Positive control must itself be green or the battery is invalid (fix the battery, then reinterpret). |
| **Battery YELLOW** | Pass at only some layers / marginal gains / one of the two tests passes ⇒ **proceed restricted to live layers**; queue Stage2.5-A for the next Stage2 train; steering layer set = the live subset, recorded in the Stage4 config. |
| **Battery RED ⇒ branch to Stage2.5** | Pause gain statistically indistinguishable from BOS at **all** tested layers **and** ablation ≈ 0, with positive control green ⇒ Stage4 stops; Stage2.5-A, then A+B if A alone stays red (grid: `near_pause_exempt_tokens` 4–16; `w_live` 0.3 → ≤1.0); each retrain re-runs the battery + the §6.4 neutrality gates (KL-by-distance-bucket, EM \|Δ\| ≤ 0.5 pt, refusal Δ ≤ +1 pt, emission within ±20% relative). |
| **Stop: negative result (recipe premise)** | Still red after A+B with `w_live ≤ 1.0` ⇒ pre-registered negative: "KL-transparent pause training produces causally inert pause states; inference-time pause steering is not applicable to this recipe." Scope decision returns to the user. This is a publishable landing, not a failure to plan for. |
| **Trajectory claim alive (Stage3)** | Within-prompt AUROC point ≥ **0.60** with 95% bootstrap CI excluding **0.55**, at ≥1 live layer, at *some* probed depth (offsets 3/4/7/8, mid-CoT insertions, naturals if >5% emission), on ≥**100** mixed-outcome prompts (per-prompt unsafe rate ∈ [0.2, 0.8] over 10 samples). Fewer than ~100 mixed prompts ⇒ broaden sources before probing, don't decide. |
| **Trajectory claim dead — re-scope, don't stop** | CI ≤ 0.55 at **all** live layers and **all** depths ⇒ directions are prompt-risk only; Stage4 proceeds but every claim is re-worded "prompt-conditioned gating readable at pause states," never "unsafe-CoT manifold removal." |
| **Direction usable** | Seed-resample cosine ≥ **0.8**; else more data, no steering yet. |
| **Proceed to 1.5B pilot (from micro-pilot)** | GPRS beats the random-direction norm-matched control on CoT-unsafe reduction, **and** capability drop < **1 pt** (50-item GSM8K slice), **and** over-refusal < **+2 pt**. First failure ⇒ one retry with `safe_centroid_pull`; second failure ⇒ negative result for clean inference-time steering (perturbation-caution, not direction semantics). |
| **Pilot readout admissible** | Unlabeled < **5%** per condition (post-T8); no α=1/α=2-style signature (think_end collapse, +10 pt refusal-keyword shift on answers, length blowup). Signature present ⇒ claims limited to "suppression via refusal/degeneration" per the main review's §10 table. |
| **8B anything** | 1.5B chain analyzed + NEW-F1 fixed + T12 plumbed + T14 fixed + **8B battery re-run from scratch** (tied/untied asymmetry — 1.5B liveness evidence inadmissible). 8B red ⇒ 8B enters its own P0.5 independently. |
| **Emission scoping (not a kill)** | Natural emission ≈ 0 ⇒ all claims say "forced-pause intervention"; > ~5% ⇒ T3 + natural/hybrid Stage4 conditions become mandatory and natural-only is the honest headline condition. |

### 7b. The two most likely kills, pre-registered landings

Same as the follow-up, worth restating because the plan should be built to
survive them: (1) battery red → Stage2.5 branch → if still red, the negative
result *is* the paper's Stage4 section; (2) within-prompt AUROC ≈ 0.5
everywhere → prompt-conditioned re-scope, project continues with honest
claims. Neither is a schedule catastrophe if the code framework already
exists — which is the whole argument for §3 landing now.

---

## 8. Q7 — The initial PR, and what explicitly waits

### 8.1 Initial PR (all laptop-safe; lands during the Stage2 training window; Stage2 launch NOT gated on it)

Suggested split into three reviewable commits:

**Commit 1 — measurement instruments (the critical path):**
- `scripts/steering/run_pause_liveness_battery.py` (T13) + `configs/experiment/stage4_liveness_1p5b.yaml` (+ 8B variant) — §3 item 1 spec.
- Stage2 uncapped KL-drift eval script (E1 remnant).
- CoT-segment judging (T10): judge prompt/config (`configs/judge/reasoningshield.yaml` or equivalent), `run_open_judges.py` CoT/answer split, normalizer, summarizer outputs.
- T8 truncation cause-fix + T9 summarizer columns (unlabeled/labeled-only/termination/repetition; answer-segment refusal regex).

**Commit 2 — Stage3 validity:**
- T1 de-alias + pause-free-forward control in `extract_hidden_states.py`; rename/remove `control_cot_3/4` everywhere (`run_intra_pause_probe_full.py:47–63` default positions, `run_intra_pause_mlp_from_restored.py`, Stage3 configs).
- T2 plumbing: `--prompt_positions` through `run_stage3_intra_pause_probe.py` → `run_intra_pause_probe_full.py` → `extraction_cmd`; `last_prompt_token`/`pre_think` added to `hidden.positions.diagnostics`; `train_text_artifact_baseline.py` adapted for the prompt-only baseline.
- Stage3 config schema: `labels.source`, `labels.on_policy.{samples_per_prompt, judge, mixed_outcome_band, min_mixed_prompts}`, `within_prompt_auroc` metric + CI analysis. Teacher-forced provenance-label path kept as the compatibility default, explicitly marked non-claim-bearing on its own.

**Commit 3 — Stage4 framework:**
- GPRS port (T5) + persistent ordinal counter + `--steer_natural_pauses` (T6) in `run_intra_pause_steered_generation.py`; direction tooling (T7).
- Config schema swap: `steering.method: gprs` (direction_artifact / safe_centroid / probe_checkpoint / gate.threshold_source / norm_cap / lambda / `layer_source: liveness_report`); `learned_delta` → `baselines:`; inert keys deleted or wired; T14 8B provenance fixed; T12 `INSERT_AFTER_COT_TOKENS`/`N_INSERT_PAUSES`/`PAUSE_MODE` env plumbing in the eval shell + runner; T11 capability wiring (or key deletion); T15 manifest emission; T16 guard over `TARGET_SPECS` + pilot args.
- NEW-F1: 8B Stage2 config aligned to the 1.5B pattern (config-only).
- CPU unit tests for all of the above where testable: scope guard specs, summarizer on synthetic rows (incl. unlabeled accounting), ordinal-counter behavior on synthetic decode sequences, battery pair/anchor selection math, config schema validation, Stage3 position-name round-trip.

**Not in this PR, by policy:** any edit to `pause_kl_trainer.py`,
`trl_train.py`, `run_4gpu_intra_pause_sft.sh`, or the two KL configs' training
semantics (NEW-F1 touches only the 8B config, which cannot run before the
1.5B chain completes anyway). The launch path stays byte-frozen until the
400-step run is on disk.

**Side branch (not merged):** Stage2.5-A/B in `pause_kl_trainer.py` with the
default-inertness equality test (§6). Merge condition: battery red (A+B) or
yellow (A queued for the next train).

### 8.2 Explicitly after the Stage2 checkpoint exists

- Every *run*: Stage2 eval battery, liveness battery, Stage3 extraction/probes, on-policy generation + judging, direction extraction, micro-pilot, pilots.
- Every *artifact and number*: chosen checkpoint + `model.sft_checkpoint` repoint; live-layer set → Stage4 `layer`; û, μ_safe, probe, τ; λ/ρ grids; Stage2.5-B margin `m` (calibrated from the positive-control battery); emission-rate-conditional decisions (T3, natural/hybrid modes, claim scoping).
- Any Stage2.5 training.

---

## 9. Go / No-Go Table

| Milestone | Verdict | Gates |
|---|---|---|
| Stage2 1.5B launch (pytest → smokes → 400 steps) | **GO — start now** | Unchanged from Round 3; launch path frozen; not gated on anything in this review |
| Initial Stage3/4 framework PR (§8.1) | **GO — write during the training window** | Battery script first; no launch-path files touched |
| Stage2.5-A/B code | **GO on a side branch only** | Default-inertness test required; merge only on battery red (A+B) / yellow (queue A) |
| Stage2 eval battery + checkpoint selection | **GO after** invariant green | Needs the emission eval (landed, Round 3) + the KL-drift script (Commit 1) |
| Liveness battery run | **GO after** checkpoint chosen | Positive control retrievable (check now); thresholds calibrated on it; tied/untied pre-registered |
| Stage3 extraction/probe rerun | **NO-GO until** battery green/yellow + B7 + T1/T2 landed | Claims additionally need T10 labels + within-prompt endpoint + CIs |
| GPRS micro-pilot | **NO-GO until** battery green/yellow + Stage3 rerun + direction QC | Random-direction control mandatory |
| 1.5B full Stage4 pilot | **NO-GO until** micro-pilot passes §7a | Readout gate: unlabeled < 5%/condition |
| 8B (any stage) | **NO-GO until** 1.5B chain analyzed | NEW-F1 + T12 + T14 fixed; 8B battery re-run mandatory |
| Stage2.5 training | **NO-GO unless** battery red | Green ⇒ explicitly forbidden (harmful surplus) |

---

## 10. Bottom Line

The sequencing instinct is right and matches the follow-up's architecture:
measure first, retrain only on evidence, keep Stage2.5 in the holster. The
three things to change about the plan are all about *when code gets written*,
not when GPUs get spent: the liveness battery and the Stage3/Stage4 validity
fixes must exist **before** the checkpoint does, so that the day training
finishes you are running measurements, not writing scripts; yellow liveness
proceeds layer-restricted instead of triggering a retrain; and checkpoint
selection comes before the battery, because `final/` on the 1.5B config is
just the last step. The Stage2 side is genuinely done — every blocker from
three rounds of review is verifiably landed in this packet, and the only
remaining Stage2 gate is the test suite that has never been executed. The
Stage3/Stage4 side is the mirror image: nothing from the redesign exists yet,
the old invalidating defects (aliased controls, hardcoded offset, truncating
judges, learned-delta-as-primary, layer-16-vs-20 provenance) are all still in
the tree, and the one config that nobody re-checked — the 8B Stage2 YAML — is
quietly carrying the pattern the 1.5B config was fixed for. Land the
instruments now; let the battery, not the calendar, decide everything after
step 400.

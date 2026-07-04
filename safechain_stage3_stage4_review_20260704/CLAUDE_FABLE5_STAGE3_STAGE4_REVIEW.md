# Fable Review — SafeChain Stage3/Stage4, Focus on Stage4 Steering

Date: 2026-07-04. Reviewer: Claude (claude-fable-5). Read-only review; no code edited.

Coverage: all Stage3/Stage4 entry points listed in the packet README were read in
full or in relevant part, including the legacy orchestrators, data prep,
extraction, probe training, position scan, learned-delta pilot, activation
pilot, steered generation, full eval shell, summarizer, judge runner/normalizer,
scope guard + CLI + tests, both Stage4 configs, both Stage3 configs, data
recipes, pipeline shells, Stage2 context (my Round 3 review and the full-flow
review), all three Stage3 heatmap result sets, and — per prior instruction —
the Stage4 result **CSVs**, not just the summary MD. Numeric claims below were
recomputed from `res/deepseek-8b/stage4_cot3_full250_hardsafe/*.csv`.

---

## 1. Executive Verdict

**The Stage4 engineering skeleton is solid; the Stage4 science is not yet.**
Four independent problems, each sufficient to block a claim on the new Stage2:

1. **Liveness is now the central unresolved question, and nothing in the repo
   measures it.** `kl_transparent_emit` trains the continuation distribution to
   be invariant to pause presence (pause-stripped KL to a same-model teacher).
   That objective, taken to its optimum, makes pause hidden states causally
   inert for downstream tokens — *dead steering ports*. The old cot3 full-SFT
   evidence that pause steering "does something" does **not** transfer: that
   model was trained with pauses inside ordinary CE, so downstream tokens had
   every reason to attend to them. Before any Stage4 run on the new checkpoint,
   injection-gain / attention-mass / ablation tests (§8) are mandatory gates.

2. **Stage3 currently demonstrates prompt-risk classification, not
   pause-specific trajectory signal.** All probed positions sit within the
   first ~8 CoT tokens; the "content controls" are aliases of post-pause
   positions (same physical token, `extract_hidden_states.py:336–339`); there
   is no prompt-token baseline in the scan; and the grids show pause slots
   carrying slightly *less* AUROC than adjacent content tokens. A direction
   selected from these probes is a prompt-category direction until proven
   otherwise.

3. **The learned-delta objective is not "away from the unsafe manifold".**
   It is unbounded NLL *ascent* on off-policy unsafe reference text plus NLL
   descent on safe text, trained as a single global vector on 32 rows/label for
   80 steps. Any direction that makes unsafe-styled text harder to predict —
   degeneracy, format breakage, refusal style, length shifts — satisfies it.
   The result CSVs show exactly these signatures (§4.5).

4. **The existing Stage4 result readout is confounded and overstated.** The
   summary MD's headline ("α=2 consistently reduces unsafe rate across three
   judges") does not survive the CSV audit: judge-side right-truncation at 4096
   tokens produces 15–54% silently-unlabeled rows whose fraction correlates
   with α (via generation length), the summarizer hides the unlabeled bucket
   (rows counted in `n` but in no label column), α=1 collapses `</think>`
   closure to 9.5%, and α=2 raises refusal-keyword rate on unsafe prompts by
   +10.5pt. Directionally a reduction survives a labeled-only recomputation,
   but the magnitude roughly halves and the "no cost" implication is not
   supported. Unsafe **CoT** — the actual project goal — is never measured
   separately from the final answer at all.

Bottom line: keep the harness (sharding, resume, scope guard, judge plumbing),
replace the method (gated projection/rejection with on-policy directions, §7),
and insert a liveness gate between Stage2 and everything else.

---

## 2. The Core Tension: `kl_transparent_emit` vs. Steering Ports

Stage2's new loss = pause-slot CE (emit) + pause-stripped continuation KL
(transparency) + pause suppression. The KL term explicitly optimizes
`P(continuation | prefix with pauses) ≈ P(continuation | prefix without pauses)`.

Consequences for Stage4:

- The *training-optimal* solution is for post-pause tokens to route zero
  information from pause KVs: attention to pause keys → 0, or value vectors
  read from pauses → content-independent. Either way, adding `α·δ` to pause
  hidden states at block L changes the KVs those positions expose to blocks
  L+1..N, but if nothing downstream attends to them, generation is unchanged
  and Stage4 is a no-op with excellent-looking "safety" metrics at α=0-like
  behavior.
- Reality will be in between (finite training, 400 steps, rows-only updates,
  suppression trade-offs). Whether enough causal bandwidth survives is an
  **empirical property of the specific checkpoint** — hence the liveness
  battery must run per-checkpoint, and its result decides whether Stage4
  proceeds as inference-time steering or must fall back to a train-time
  auxiliary loss (§7 fallback).
- Corollary: the old 8B cot3 full-SFT Stage4 numbers cannot be cited as
  evidence that the new pipeline will steer. They are evidence only that the
  harness runs and that *that* model had live pauses.

This is the single most important design decision point in the project right
now, and it is currently unmeasured.

---

## 3. Stage3 Review

### 3.1 What is correct (code level)

- **Mechanical compatibility with `kl_transparent_emit` checkpoints: YES.**
  The new Stage2 saves a standard HF checkpoint with the same tokenizer/added
  `<|pause|>` row convention as the format-only run. `run_stage3_intra_pause_probe.py`
  resolves the model as `MODEL` env > `model.sft_checkpoint` > local base >
  base, and hands off to the legacy orchestrator with `cwd=legacy/PauseProbe`.
  Nothing in extraction assumes full-SFT behavior. Two carried gates from my
  Stage2 Round 3 review still apply: repoint `model.sft_checkpoint` to a
  battery-chosen checkpoint (`final/` is now *last-step*, not best), and
  verify the rows-only invariant on the trained weights first.
- Insertion convention is consistent end-to-end: `prepare_intra_pause_probe_data.py`
  inserts `pause_ids * 3` at `content_start + insert_cot_offset` on
  `"\n" + reasoning.strip() + "\n"`, mirroring the Stage2 builder convention;
  a verify pass re-derives positions and drops mismatches.
- Teacher-forced extraction (`extract_hidden_states.py`) does one forward over
  `BOS + User + prompt + Assistant + output (+EOS)` and indexes hidden states
  correctly; layer ids are HF hidden-state ids (0=embeddings), consistent with
  `layer_to_block_index(layer) = layer - 1` used by both steering scripts.
- Probe training is competent: train-stat standardization stored in
  `probe.pt`, BCE pos_weight, `source_label` inverse-frequency weights,
  val-threshold at max FPR 0.05, early stop on val AUROC. Split strategy
  `source_label_prompt_group` prevents same-prompt cross-split leakage.
- The un-standardization needed to reuse a probe weight as a raw-hidden-space
  direction (`w/std`) is already implemented in
  `run_intra_pause_activation_pilot.py::estimate_probe_weight_direction`.

### 3.2 Design flaws (science level)

**S3-1 — The controls are aliases, not controls.**
`extract_hidden_states.py:336–339` sets
`positions["control_cot_3"] = positions["post_pause_1"]` and
`control_cot_4 = post_pause_2`. Separately, `cot_k` uses pause-excluded
indexing, so with insert offset 3, `cot_3 == post_pause_1` and
`cot_4 == post_pause_2` as physical tokens. Three names, one token. The result
grids confirm it: aliased triplets differ by ≤ ~1e-3 AUROC (probe-training
noise on identical features). Every conclusion of the form "pause positions
beat content controls" is currently vacuous. The pooled spec
`control_cot3_cot4_concat_layers_concat` is post-pause data under a control
name.

**S3-2 — No prompt-position baseline.** The scan has no `last_prompt_token`
(or any prompt-token) probe. Given causal attention and insert offset 3, the
pause states condition on prompt + ~3 CoT tokens; a prompt-token baseline is
the null hypothesis and it is absent.

**S3-3 — The measured signal is consistent with prompt-risk classification.**
From the cot3_full_ckpt250 grid: pause_0 test AUROC 0.951–0.956, pause_1
0.953–0.960, pause_2 0.951–0.963, while post_pause_1 reaches 0.956–0.970,
post_pause_2 0.953–0.972, and even pre_pause_3 (≈ cot_0, essentially
prompt-only + newline) already achieves 0.950–0.955. Top-of-table positions
are content tokens (cot_4@L22 = post_pause_2@L32 = 0.9719), not pause slots.
So: ~0.95 is available before the CoT says anything, pause slots add nothing
over their neighbors, and the +0.02 at later content tokens is what little
trajectory signal exists at these depths. RS-test heldout drops to 0.81–0.91
(cross-source generalization gap), and most non-heldout sources are
single-label (star/aidsafe/unsafechain safe-only; harmthoughts unsafe-only),
so source-style shortcuts remain partially available despite the weighting.

**S3-4 — Labels are off-policy.** `trajectory_safety_label` comes from dataset
provenance of external reference traces, teacher-forced through the model.
Stage4 will steer the model's *own* CoT. Under `kl_transparent_emit` the
model's behavior ≈ base model, and its own trajectories are not the dataset's
trajectories. A direction fit on off-policy text can point somewhere
irrelevant for on-policy generation.

**S3-5 — Forced-pause-only extraction.** If the new Stage2 produces natural
self-emission, Stage3 must also probe *naturally emitted* pause positions
(their contexts are model-chosen, deeper in the CoT, and are exactly the
positions Stage4 will steer during decode). Currently only offset-{3,4,7,8}
forced layouts exist.

**Verdict:** Stage3 code will run unmodified on the new checkpoint, but as an
instrument for "do pause states carry trajectory-specific unsafe signal beyond
the prompt?", it cannot currently answer the question. Fixes are cheap (§11
T1–T4) and should land before the Stage3 rerun.

---

## 4. Stage4 Review

### 4.1 Learned-delta pilot (`run_intra_pause_learned_delta_pilot.py`)

Mechanics are clean: frozen LM, one `hidden_size` parameter, hook at block
`layer-1` output adding the delta at the three known pause positions,
teacher-forced. But the objective is:

```python
if label == unsafe: task_loss = -unsafe_weight * nll   # maximize unsafe NLL
else:               task_loss =  safe_weight  * nll    # minimize safe NLL
+ l2 * delta.pow(2).mean()
```

Problems:

- **Unbounded ascent.** Maximizing NLL of unsafe reference text has no
  stationary target; the optimizer is rewarded for *any* distribution damage
  aligned with unsafe-styled text — format breakage, early termination, style
  shifts, refusal boilerplate. Nothing constrains the delta to the
  safe/unsafe-discriminative subspace.
- **Off-policy text** (S3-4 applies verbatim): the NLL is measured on dataset
  reasoning continuations, not model generations.
- **Tiny budget:** 32 rows/label, 80 steps, lr 0.05 — a 1536/4096-dim vector
  fit on 64 sequences. Seed-to-seed direction stability is unmeasured.
- **Label-agnostic application:** one global vector added to *every*
  generation (safe prompts included) at fixed α. The training objective is
  label-conditional; the deployment is unconditional. This mismatch is exactly
  where over-refusal comes from.
- **α extrapolation:** the delta trains at implicit α=1; evaluation at α=2 is
  outside the training regime, and the eval grid's best point is α=2.
- The optional init `mean_diff` direction is fine, and the eval metric
  (teacher-forced post-pause NLL vs α=0) is a sanity metric, not a safety
  metric.

**Conclusion: this objective does not implement "pull away from the unsafe
manifold" and should not survive into the new Stage4** (Q3 = no, §6).

### 4.2 Generation-time hook (`run_intra_pause_steered_generation.py`)

- **Hook semantics under cache: correct.** The forward hook at block `L-1`
  output edits pause-position hidden states; those edited states become the
  KVs of blocks L..N for the pause positions and persist in the cache; the
  pause position's own block-L KV is untouched. That is the standard
  activation-steering semantics.
- **Pause-only scope: genuinely enforced at runtime** by `ids.eq(pause_id)` —
  the only mechanism that matters, and it is correct. The `scope.py` /
  `validate-scope` CLI guard only validates config strings (see 4.6).
- **Ordinal tracking is per-forward-call** (`mask.cumsum(dim=1)-1` on the
  current call's `input_ids`). Prefill sees all 3 pasted pauses → ordinals
  0,1,2 correct. But every *naturally emitted* pause during decode arrives in
  a single-token forward → ordinal 0 → gets `pause_0` treatment under `all3`
  and is steered. `avg_steered_pause_tokens` = 3.006–3.057 in the result CSVs
  confirms natural pauses were steered in the old run. This is defensible
  behavior but it is **undocumented and unconfigurable**, and for
  `pause1_only`/`pause2_only` targets natural pauses are silently never
  steered — so the target-ablation comparison is not what it claims to be for
  decode-time pauses.
- **Forced insertion flow:** sample exactly N (default 3) CoT tokens at
  temp 0.6/top_p 0.95, paste `PAUSE×3` as text, continue with hook active.
  Fine as a *forced* mode, but the insert offset is **hardcoded to 3 in the
  eval shell** (`run_intra_pause_full_steering_eval.sh` passes
  `--insert_pause_after_cot_tokens 3 --n_insert_pauses 3` with no env
  override). The new 8B Stage2 config is **cot4** — running the current shell
  against it inserts at the wrong offset relative to training. There is also
  no natural-pause-only mode in the eval chain, though the generation script
  itself could support it (insertion is gated on `>= 0`).

### 4.3 Launcher and configs (`run_stage4_steering.py`, YAMLs)

Solid plumbing (env construction, phase gating, resolved-config dump, resume
logic in the shell, alpha-0 reuse across targets). Specific drift/inert-key
issues:

- **Inert config keys.** `steering.{init,steps,safe_weight,unsafe_weight,l2,loss}`
  are consumed by nothing (`steps` only appears in the *default delta path
  name*); delta training is launched manually via the pilot script. The config
  advertises `l2: 0.001` while the pilot's default is `0.01` — whichever was
  actually used is not recorded anywhere the config controls. Same for
  `eval.model_conditions: [base, sft, sft_steering]`, `eval.capability:
  [gsm8k, math500]`, `eval.safety: [...]`, and the `capability_sources` block
  in `stage4_steering_eval_sources.yaml`: **no code path consumes them**. The
  Stage4 chain as implemented evaluates only the four hardsafe judge datasets.
  Capability under steering is currently *not measured at all*.
- **Config/results provenance break (8B).** `stage4_pause_steering_8b_4xa100.yaml`
  pins `layer: 16` and a `zero_l16_steps80` delta, but the committed result
  summary says the 8B run steered at **layer 20** "selected from the Stage 3
  scan". The committed config does not reproduce the committed result.
- `judge_backend=vllm` single-judge constraint is enforced consistently in
  both the runner and the shell; second judges run as separate passes. Good.
- Minor: runner seeds default `[260618-20]`, shell default `260621-23` —
  harmless (runner exports), but one more place where env defaults and configs
  can silently diverge.

### 4.4 Judge pipeline

- `run_open_judges.py` builds WildGuard/LlamaGuard/HarmBench prompts with the
  **response embedded mid-template and the answer-format instructions after
  it**, then tokenizes with `truncation=True, max_length=4096` (right
  truncation). Steered generations average 6.5–8.2k chars (~1.7–2.3k tokens)
  *plus* prompt *plus* template; long rows lose the instruction tail, the
  judge free-generates, and the normalizer returns `None` → label
  `unlabeled`. **Unlabeled fraction therefore correlates with generation
  length, which correlates with α.** This is the single biggest integrity
  problem in the Stage4 readout (numbers in §4.5).
- `normalize_judge_outputs.py` parsing and conservative aggregation are
  reasonable, but note: **no parser can ever emit `low_quality_garbage`** —
  the four-way scheme's garbage bucket is structurally unreachable from these
  three judges. Every `low_quality_garbage=0` column in every CSV is a
  tautology, not a finding. "No broken output" can not be claimed from this
  pipeline.
- The judged text is `generated_for_judge` = pause-stripped **full text
  (CoT + answer)**. Consequences: (a) *unsafe CoT is never measured
  separately* — the project goal is CoT-level; (b) safety judges are
  calibrated for final responses, not think-aloud text; (c) the α=1 condition
  (91% unterminated `<think>`) feeds the judges pure truncated CoT, which they
  were never calibrated for — and HarmBench's "α=1 looks great overall" is an
  artifact of exactly this.
- `summarize_intra_pause_full_steering_eval.py` counts only
  `unsafe_valid/ambiguous_partial/safe_refusal/low_quality_garbage/missing`;
  **`unlabeled` rows are included in `n` but appear in no column**, silently
  deflating every rate. `missing_judge_rows = 0` in the manifests refers to
  row coverage, not label coverage — the summary MD cites it as if it meant
  the latter.
- The `REFUSAL_RE` keyword metric runs over the full CoT+answer text; R1-style
  CoT routinely contains "I can't…" deliberation, so it is a weak over-refusal
  proxy at best.

### 4.5 Result CSV re-audit (old 8B cot3-full250 run)

Recomputed from the shipped CSVs, unsafe prompts, dataset=all, seed=all.
"Labeled-only" = `unsafe_valid / (rows with a parseable judge label)`:

| Judge | cond | reported rate | unlabeled % | labeled-only rate |
|---|---|---:|---:|---:|
| WildGuard | base | 0.364 | 19.4% | 0.452 |
| WildGuard | α=0 | 0.319 | 14.9% | 0.375 |
| WildGuard | α=1 | 0.411 | 3.7% | 0.426 |
| WildGuard | α=2 | 0.130 | 26.3% | 0.176 |
| LlamaGuard | base | 0.450 | 24.5% | 0.596 |
| LlamaGuard | α=0 | 0.451 | 20.7% | 0.568 |
| LlamaGuard | α=1 | 0.573 | 4.4% | 0.600 |
| LlamaGuard | α=2 | 0.263 | 31.7% | 0.385 |
| HarmBench | base | 0.355 | 45.6% | 0.652 |
| HarmBench | α=0 | 0.347 | 41.3% | 0.592 |
| HarmBench | α=1 | 0.239 | 4.7% | 0.251 |
| HarmBench | α=2 | 0.179 | 53.7% | 0.387 |

Structural/behavioral columns (judge-independent, all prompts):

| cond | think_end_rate | avg chars | refusal_kw (unsafe rows) | steered pauses/gen |
|---|---:|---:|---:|---:|
| base | 0.861 | 6838 | 0.267 | 0 |
| α=0 | 0.657 | 6576 | 0.275 | 0 |
| α=1 | **0.095** | 4080 | 0.269 | 3.008 |
| α=2 | 0.655 | 7039 | **0.381** | 3.050 |

Findings:

- **F1 — α=1 breaks generation.** `</think>` closure collapses 0.66→0.095 and
  length drops 38%. The delta at its trained magnitude primarily disrupts
  termination — a hallmark of the unbounded-ascent objective, and invisible in
  the summary MD.
- **F2 — Unlabeled inflation drives headline magnitudes.** The MD's HarmBench
  "base 0.355 → α=2 0.179 (−0.176)" is, labeled-only, 0.652 → 0.387; the
  LlamaGuard −0.187 becomes −0.211 from a much higher base but α=2's 0.263 is
  really 0.385. A directional reduction at α=2 survives under all three
  judges (labeled-only: WG 0.452→0.176, LG 0.596→0.385, HB 0.652→0.387), so
  the effect is probably real *for that old model*, but every reported
  magnitude is length/truncation-confounded, and α=1's "good" HarmBench
  number is a broken-output artifact (least truncation because shortest,
  judged on unterminated CoT).
- **F3 — Part of the α=2 reduction is refusal-shift.** Refusal keywords on
  unsafe prompts +10.5pt, unsafe-prompt `safe_refusal` label 0.291→0.385,
  length +11% on unsafe rows. Reducing unsafe-valid by inducing refusals is
  explicitly against the project goal ("without increasing over-refusal") —
  the safe-prompt side looks acceptable (WG safe-prompt rate 0.082→0.056;
  refusal keywords 0.264→0.233), but no calibrated over-refusal metric
  (XSTest-style compliance judge) exists in the chain.
- **F4 — Row accounting anomaly.** Safe rows: 1650/α = 3 seeds × (250+300) ✓.
  Unsafe rows: 1500/α vs expected 3 × (250+300) = 1650 → one unsafe source
  delivered 250/seed instead of 300. Not a blocker, but `res/` ships no
  `run_config.txt`/manifests for generation params (the ~7k-char averages also
  imply `MAX_NEW_TOKENS` > the shell's 1024 default), so run provenance is
  not reconstructible from the packet.
- **F5 — Natural pauses were steered** (steered ≈ 3.05 > 3 pasted;
  pause3_rate 0.95–0.99), see §4.2.

### 4.6 Scope guard

`validate_pause_only_targets` / `validate_no_pre_post_or_cot_targets` + CLI +
tests are fine for what they are — a config-string lint (the second check is
redundant by construction: anything matching a forbidden prefix already fails
the `pause_` check, so the test's parametrized cases all fail at the first
gate). The *real* enforcement is the `ids.eq(pause_id)` mask in the generation
hook, which is correct. Two gaps: the learned-delta *pilot* and activation
pilot index positions by name from the NPZ and are not covered by the guard
(they could steer `post_pause_*` if asked); and the guard never sees
`TARGET_SPECS` used by the shell (it validates `steering.target_positions`
only, while the shell iterates `TARGET_SPECS`). Low severity, worth one-line
fixes (T16).

---

## 5. Exact Blockers and Risks

Blockers (must fix before the corresponding milestone):

- **B1 (Stage4, hard):** No liveness evidence on `kl_transparent_emit`
  checkpoints; the training objective actively pushes toward dead pause
  ports. Gate every Stage4 activity on the §8 battery.
- **B2 (Stage4, hard):** Learned-delta objective invalid for the project claim
  (§4.1). Replace method (§7).
- **B3 (Eval, hard):** Judge truncation + hidden `unlabeled` bucket ⇒ all
  Stage4 rates length-confounded; summarizer must expose unlabeled and
  labeled-only rates; judge inputs must stop truncating the instruction tail
  (T8–T9).
- **B4 (Eval, hard):** Unsafe **CoT** is not measured; add a CoT-segment judge
  (ReasoningShield-style) separate from final-answer judging (T10).
- **B5 (Stage3, hard for claims):** Aliased controls + no prompt baseline ⇒
  Stage3 cannot currently support "pause-specific signal" claims (T1–T3).
- **B6 (Compat):** Eval shell hardcodes insert offset 3; 8B Stage2 is cot4
  (T12). 8B config layer 16 vs. results layer 20 provenance break (T14).
- **B7 (carried from Stage2 Round 3):** `sft_checkpoint` repoint to a
  battery-chosen checkpoint; rows-only invariant verified on trained weights;
  pod pytest before any GPU work.

Risks (should address, not strictly blocking):

- **R1:** Off-policy direction/labels (S3-4) — on-policy relabeling path
  recommended before trusting any direction.
- **R2:** Natural-pause ordinal-0 treatment and unconfigurable steering of
  natural pauses (§4.2).
- **R3:** Capability under steering unmeasured; config keys advertise it (T11).
- **R4:** Single global unconditional delta ⇒ over-refusal pressure; gating
  addresses this (§7).
- **R5:** `low_quality_garbage` unreachable ⇒ broken-output claims need the
  structural metrics (think_end, EOS-termination, repetition, unlabeled rate)
  promoted to first-class outputs (T9).
- **R6:** Probe direction standardization: any direction taken from `probe.pt`
  must be un-standardized (`w/std`) before use in raw hidden space — already
  correct in the activation pilot; keep it that way in ports.

---

## 6. Answers to the Ten Questions

**Q1 — Stage3 logically compatible with the new Stage2 checkpoint/tokenizer?**
Yes, mechanically (§3.1): same HF layout, same `<|pause|>` row, same insertion
convention, `MODEL` override works. Conditions: repoint `sft_checkpoint`
(mandatory — `final/` = last step under `load_best=false`), rows-only
invariant verified, and remember `first_pause_token_index_inside_think ≈
cot_offset + 1` (the `\n` after `<think>` is counted) at analysis time.

**Q2 — Does Stage3 measure the right separability signal?** No, not yet. What
it demonstrably measures today is prompt/source-risk separability readable
from near-prompt states (§3.2): ~0.95 AUROC at essentially prompt-only
positions, pause slots ≤ neighboring content tokens, controls aliased,
heldout gap 0.81–0.91. It *can* measure the right thing after T1–T4 (prompt
baseline, real content controls, deeper/natural pause positions, on-policy
labels, prompt-only text-classifier baseline).

**Q3 — Does the learned-delta objective represent "away from unsafe
manifold"?** No (§4.1): unbounded off-policy NLL ascent with a global vector;
the CSVs show it partially operates via termination damage (α=1) and refusal
shift (α=2). It is a "make unsafe text unlikely by any means" direction.

**Q4 — Is generation-time steering at the right token/timestep; hook semantics
correct under cache?** Hook placement, cache propagation, and pause-only
masking: correct (§4.2). Two semantic caveats: per-forward ordinal restarting
(natural decode pauses always ordinal 0), and prefill-steering of the three
pasted pauses (fine). Timestep-wise, steering the pause position *when it is
consumed* is right; but note the edit only influences *future* tokens through
attention to pause KVs — which is exactly why liveness (B1) decides
everything.

**Q5 — Forced pauses still valid after natural self-emission?** Forced-only is
no longer sufficient; it remains *useful* as a controlled condition. Run
**both plus hybrid**: (a) forced-at-training-offset (comparability with
Stage3 extraction and across checkpoints); (b) natural-only (deployment
realism — steer only self-emitted pauses; this is the honest headline
condition if natural emission rate is non-trivial); (c) hybrid (forced
insertion + steer naturals too, current de-facto behavior, now made explicit).
Report natural emission rate first (post NEW-B2, the Stage2 eval can measure
it); if it is ≈0, natural-only is vacuous and Stage4 must say "forced-pause
intervention" in all claims.

**Q6 — Concrete liveness tests before Stage4?** §8. Injection-gain curves
against matched content-token and BOS anchors, attention-mass to pause KVs,
pause-KV ablation, and safe/unsafe patching — run on base, old full-SFT
(positive control), and the new KL checkpoint (test article).

**Q7 — Concrete experiments after the first new Stage2 checkpoint, before
choosing the Stage4 algorithm?** §9 steps 1–4: liveness battery → fixed
Stage3 rerun with controls → on-policy paired direction extraction + direction
QC (probe transfer, seed stability, cosine across offsets/layers) → 100-row
micro-pilot comparing candidate edits against a random-direction,
norm-matched control.

**Q8 — Best Stage4 algorithm?** Gated projection/rejection at pause states
with an on-policy contrastive direction; safe-centroid pull along the same
direction as the bounded "pull" variant (§7).

**Q9 — Code changes needed for it?** T5–T7 plus the eval fixes T8–T13 (§11).
Most math already exists in `run_intra_pause_activation_pilot.py`
(mean_diff, probe-weight w/std, safe_centroid_pull, hard/soft/score gate);
the work is porting it into `run_intra_pause_steered_generation.py` with
per-step gating and adding the projection mode.

**Q10 — Allowed claims under outcomes?** See §10.

---

## 7. Recommended Stage4 Method

**Gated projection/rejection steering at pause states (GPRS).**

At every steered pause position (forced or natural), at one or two mid layers:

1. **Direction.** û = normalized mean difference of **on-policy** pause hidden
   states: same prompts, model's own generations, judge-labeled
   safe/unsafe (contrastive paired where possible). Not dataset reference
   text. Un-standardized space. Cross-check cosine similarity with the Stage3
   probe-weight direction (`w/std`); if they disagree wildly, neither is
   trustworthy yet.
2. **Gate.** Online probe score `s = σ(w·(h−μ)/σ_std + b)` from the Stage3
   linear probe at that layer/position; steer only if `s > τ` (τ from the
   FPR≤0.05 threshold already stored in `probe.pt`). This makes the
   intervention conditional — safe prompts are (mostly) untouched, which is
   the structural answer to over-refusal, unlike the current unconditional
   delta.
3. **Edit.** Rejection with a bounded pull:
   `h ← h − λ·((h−μ_safe)·û)₊·û`, optionally clamped so `‖Δh‖ ≤ ρ‖h‖`
   (ρ ≈ 0.05–0.15 grid). Removes only the measured unsafe component, moves
   toward the safe side of the boundary, cannot run off-manifold the way an
   additive global delta at α=2 can. `safe_centroid_pull` (already
   implemented) is the fallback variant if rejection underperforms.

Why this beats the alternatives on the request's list: probe-gradient steering
≈ probe-weight direction for a linear probe (same thing, cheaper); classifier-
guided editing at generation time is GPRS with extra steps; a pure safe-
centroid pull drags *all* coordinates (style/length included) while rejection
touches one axis; contrastive delta from paired states is exactly how û is
estimated here; train-time auxiliary loss is the **fallback, not the first
choice**, because it changes Stage2 and forfeits the clean "frozen model +
inference-time intervention" claim.

**Fallback if liveness fails (§8 red):** Stage2.5 auxiliary loss — keep
`kl_transparent_emit` but add (i) a pause-state contrastive/probe-margin term
so pause states organize safe/unsafe, and (ii) a small attention-mass floor on
pause KVs from the next k tokens (or a scheduled reduction of KL weight on the
first post-pause tokens) so a live channel survives. Then re-run the liveness
battery. If natural emission ≈ 0 as well, Stage4's premise (pause tokens as
intervention ports) is unsupported for this Stage2 recipe, and the honest
pivot is either train-time-only safety or steering at a different, defensible
locus — which would need a new scope decision, since the current paper scope
is pause-only.

---

## 8. Required Liveness / Injection-Gain Tests (gate before any Stage4 run)

Run on: (a) new KL checkpoint (test), (b) old cot3 full-SFT ckpt250 (positive
control — it demonstrably reacted to steering), (c) base model with pasted
pauses (negative-ish control). ~200 prompts, forced-pause layout at the
training offset, plus natural-pause subset if emission rate > 5%.

1. **Injection-gain curves.** Add ε·v (v ∈ {random unit, û, probe-weight
   dir}) at pause states, layer sweep (e.g., 7/14/17/21/25 for 1.5B;
   8/16/20/24/28 for 8B); measure mean KL(p₀‖p_ε) of the next-16-token
   distributions vs ε ∈ {0.5, 1, 2, 4, 8}·σ_h. Anchors: same-ε injection at
   the adjacent content token (upper), at BOS (lower).
   **Pass:** pause-position gain ≥ 25% of content-token gain and ≥ 5× BOS
   gain at ≥ 1 mid layer. **Red:** indistinguishable from BOS.
2. **Attention mass.** Mean attention weight from the next 32 generated/CoT
   positions to pause KVs, per layer/head; compare the three models.
   **Pass:** KL model retains ≥ 50% of full-SFT's pause attention mass at the
   layers steering will use, and mass is materially above the attention-sink
   baseline for an arbitrary content token of the same depth.
3. **Pause-KV ablation.** Zero/mean-ablate pause KVs after prefill; measure
   continuation KL and downstream text edit distance. **Pass:** measurably
   nonzero effect (else ports are dead regardless of injection at one block).
4. **Safe↔unsafe patching.** Swap pause hidden states between matched
   safe/unsafe rows; measure next-16-token KL and, on continuation, the
   Stage3 probe score / judge label flip rate. This is the only test that
   measures whether the *safety-relevant* subspace (not just any direction)
   is live. **Pass:** patch moves downstream probe scores significantly in
   the label direction.

All four are cheap (teacher-forced or ≤64-token continuations, no judges) and
should be one new script (T13).

---

## 9. Minimal Experiment Plan (ordered, 1.5B first)

0. Prereqs (carried): pod pytest green; 1.5B `kl_transparent_emit` trained;
   battery-chosen checkpoint; rows-only invariant confirmed; natural emission
   rate measured by the fixed Stage2 eval.
1. **Liveness battery** (§8) on the chosen checkpoint. Red ⇒ stop; go to
   Stage2.5 fallback. Green/yellow ⇒ record which layers are live; those are
   the only admissible steering layers.
2. **Stage3 rerun with fixed controls** (T1–T4): prompt-token baseline, true
   content controls, natural-pause positions, prompt-only text-classifier
   baseline, on-policy relabeled subset (~1–2k rows). Decision metric: pause
   (or post-pause) AUROC − prompt-baseline AUROC on the on-policy subset. If
   ≈ 0 everywhere: directions are prompt-risk only; steering may still
   "work" but claims must say prompt-conditioned, not trajectory-conditioned.
3. **Direction extraction + QC:** on-policy paired mean-diff û at the live
   layers; check (a) held-out probe AUROC of the 1-D projection, (b) seed
   stability (cosine > 0.8 across resamples), (c) cosine vs probe-weight
   direction.
4. **Micro-pilot (n≈100/label, 1 seed):** GPRS vs learned delta vs random
   direction at matched ‖Δh‖ vs no-op. Metrics: CoT judge + answer judge +
   think_end/termination + length + unlabeled rate + 50-item GSM8K slice.
   Random-direction control is mandatory — it is the test that separates
   "unsafe-direction removal" from "any perturbation causes caution".
5. **1.5B Stage4 pilot** (full eval chain, fixed judges/summarizer, forced +
   natural + hybrid conditions, 3 seeds).
6. Analyze; only then **8B** (repeat 1–5 at 8B scale with cot4 offset).

---

## 10. Required Eval Metrics and Allowed Claims

Metric set (all reported per α/condition, labeled-only denominators plus
unlabeled rate):

- **Unsafe CoT:** CoT-segment judge (ReasoningShield prompt or equivalent) on
  the `<think>` span only. Primary endpoint — this is the project goal.
- **Unsafe answer:** WildGuard (+ second judges) on the post-`</think>`
  answer only.
- **Over-refusal:** refusal/compliance judge on XSTest-safe + OR-Bench-hard-
  safe *answers* (not keyword regex on CoT); report Δrefusal vs α=0.
- **Capability:** GSM8K/MATH500 EM with steering active (hook on, pauses per
  condition).
- **Broken output:** think_end rate, EOS-termination rate, repetition
  (n-gram loop) rate, length distribution shift, judge-unlabeled rate.
- **Transparency:** pause-conditional continuation KL vs base (the Stage2
  §4.4 eval, run with steering on), natural emission stats.

Claims under outcomes:

| Outcome | Allowed claim |
|---|---|
| Liveness red on KL ckpt | "KL-transparent pause training produces causally inert pause states; inference-time pause steering is not applicable to this recipe" — a publishable negative result; no steering claims. |
| Liveness green, Stage3 Δ vs prompt-baseline ≈ 0 | Steering claims must be phrased *prompt-conditioned* ("gated by prompt-risk signal readable at pause states"), not "unsafe-CoT manifold removal". |
| Liveness green, on-policy pause signal > prompt baseline | Trajectory-level claim permitted if the random-direction control fails and GPRS succeeds. |
| GPRS reduces CoT-unsafe with flat over-refusal/capability/broken-output and unlabeled < 5% | Full project claim: "pause-state steering reduces unsafe CoT without behavioral cost", scoped to forced/natural per the conditions actually run. |
| Reduction only at the cost of refusal shift / termination damage (current α=2/α=1 pattern) | Only: "pause-state perturbation can suppress unsafe-labeled outputs via refusal/degeneration" — i.e., a negative result for the method as a *clean* intervention. |
| Natural emission ≈ 0 | All claims must say "forced-pause intervention"; "pause tokens as self-emitted intervention ports" is unsupported. |

---

## 11. Concrete Code-Change TODO List

Stage3 (before the rerun):

- **T1** `extract_hidden_states.py`: make `control_cot_3/4` true controls —
  content tokens at matched depth in a **pause-free** copy of the same row
  (second forward), or at absolute positions ≥ insert_idx+3 in the paused
  sequence; stop aliasing (`:336–339`).
- **T2** Add `last_prompt_token` (and `assistant_first_token`) to the scan
  positions in both Stage3 configs; add a prompt-only text classifier (TF-IDF
  or frozen-encoder) baseline to the report.
- **T3** Add natural-pause extraction mode: run the pause model free-running,
  locate self-emitted `<|pause|>` runs, extract at those positions
  (positions named `natural_pause_k`), labels via judges (T4).
- **T4** On-policy labeling path: generate with the Stage2 model on Stage3
  prompts, judge the CoT, use judge labels for a parallel probe run
  (`trajectory_safety_label_source: on_policy_judge`).

Stage4 method (GPRS):

- **T5** `run_intra_pause_steered_generation.py`: port `mean_diff` /
  `probe_weight` / `safe_centroid_pull` / gate from
  `run_intra_pause_activation_pilot.py`; add `--steering_method projection`
  implementing `h − λ((h−μ_safe)·û)₊û` with `--norm_cap ρ`; per-step gate
  evaluation on the current pause hidden state.
- **T6** Same file: persistent per-sequence pause-ordinal counter across
  decode steps (running count carried between forwards) + explicit
  `--steer_natural_pauses {on,off}` + hook_stats fields separating
  forced-vs-natural steered counts.
- **T7** Direction tooling: small script to build on-policy paired pause-state
  NPZ (from T4 generations) and emit û + μ_safe + QC report (seed cosine,
  probe transfer AUROC).

Eval integrity:

- **T8** `run_open_judges.py` / vLLM judge worker: never truncate the template
  tail — truncate the *response segment* (keep head+tail of the response,
  drop the middle) to fit `max_length`; raise LlamaGuard/HarmBench max len
  where the model allows; log a `truncated` flag per row.
- **T9** `summarize_intra_pause_full_steering_eval.py`: add `unlabeled` count
  + `unlabeled_rate` + labeled-only rate columns; add EOS-termination and
  repetition-loop columns; compute refusal keywords on the answer segment
  only.
- **T10** Add CoT-segment judging: split `generated` at `</think>`; judge CoT
  with the ReasoningShield prompt, answer with WildGuard/second judges; two
  normalized files per shard (`cot_judge_normalized.jsonl`,
  `answer_judge_normalized.jsonl`); summarizer reports both.
- **T11** Wire capability: a small GSM8K/MATH500 EM pass with the steering
  hook active (or delete `eval.capability`/`eval.safety`/`model_conditions`
  keys so the config stops advertising unimplemented evals).
- **T12** `run_intra_pause_full_steering_eval.sh`: `INSERT_AFTER_COT_TOKENS` /
  `N_INSERT_PAUSES` env-parameterized (cot4 for the 8B model); add
  `PAUSE_MODE={forced,natural,hybrid}` (natural = no insertion, steer
  self-emitted pauses only); plumb from `run_stage4_steering.py` config keys.
- **T13** New `scripts/steering/run_pause_liveness_battery.py` implementing
  §8 (injection gain, attention mass, KV ablation, patching), stdlib+torch
  only, one JSON report per model/layer.

Hygiene:

- **T14** Fix 8B config: layer + delta path must match whatever the next run
  actually uses; record the Stage3-scan-selected layer in the config comment.
  Remove or wire the inert `steering.{init,steps,safe_weight,unsafe_weight,
  l2,loss}` keys (l2 0.001-vs-0.01 discrepancy).
- **T15** Ship `run_config.txt` + generation manifests into `res/` alongside
  summary CSVs (F4 provenance gap; include MAX_NEW_TOKENS actually used).
- **T16** Scope guard: validate `TARGET_SPECS` lines too (every spec's
  positions through `validate_pause_only_targets`); add the same check inside
  the two pilot scripts' argument parsing.

---

## 12. Go / No-Go Table

| Milestone | Verdict | Gates |
|---|---|---|
| Stage3 rerun on new Stage2 ckpt | **GO (conditional)** | Mechanically compatible today; run only after B7 (repoint + invariant) — but *claims* require T1–T4; without them it re-measures prompt classification. |
| Current Stage4 as-is (learned delta + current eval) | **NO-GO** | B1 (no liveness), B2 (objective invalid), B3/B4 (eval confounded, CoT unmeasured), B6 (cot4 offset). Do not spend GPU on it. |
| Modified Stage4 (GPRS per §7) | **GO to implement now** (T5–T13 are laptop-safe); **NO-GO to run** until liveness battery green + Stage3 rerun done. |
| 1.5B Stage4 pilot | **NO-GO until**: pod pytest → 1.5B KL checkpoint → liveness green (§8) → Stage3 rerun (§9.2) → direction QC (§9.3) → micro-pilot with random-direction control (§9.4). Then GO. |
| 8B Stage4 pilot | **NO-GO** | Sequencing: after 1.5B pilot analyzed. Additional gates: cot4 offset plumbed (T12), 8B config provenance fixed (T14), liveness re-run at 8B. |

---

## Bottom Line

Keep the harness; replace the science. The sharded eval machinery, resume
logic, judge plumbing, and pause-only runtime masking are production-quality.
But: the learned delta optimizes the wrong thing and its flagship result is
partly termination-damage and refusal-shift under a judge pipeline that
silently drops up to half its labels; Stage3's controls are aliases of the
thing they are supposed to control for; and the new Stage2 objective is
explicitly in tension with the existence of steerable pause ports. The path
forward is strictly ordered: liveness battery → fixed Stage3 → on-policy
directions → gated projection steering → 1.5B pilot → 8B. Every step has a
cheap kill-switch, and the first one (liveness) is the one most likely to
fire — design for that outcome to be publishable rather than fatal.

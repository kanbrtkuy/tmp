# Fable Follow-up Review — Inert-Pause Risk, Executor Redesign, and the Stage2.5 Decision

Date: 2026-07-04. Reviewer: Claude (claude-fable-5). Read-only follow-up to
`CLAUDE_FABLE5_STAGE3_STAGE4_REVIEW.md` ("the main review"); no code edited.

Basis: the executor redesign in `FABLE_STAGE3_STAGE4_FOLLOWUP_REQUEST.md`, plus
fresh source verification of everything this follow-up leans on:
`legacy/COTPauseToken/src/utils/pause_kl_trainer.py` (read in full),
`scripts/run_stage2_sft.py` (pause_kl env plumbing, `:340–372`),
`configs/experiment/stage2_intra_pause_kl_transparent_emit_1p5b_cot3_save25_max400_4xa6000.yaml`,
and re-confirmation of the two load-bearing carried claims
(`extract_hidden_states.py:337,339` control aliasing;
`run_intra_pause_activation_pilot.py` mean-diff / probe-weight /
safe-centroid-pull / gate math). Blocker/TODO IDs (B1–B7, T1–T16) refer to the
main review.

---

## 1. Executive Verdict

**Yes, there is a good solution to the "harmless but useless" risk, and it is
not the one implied by panic-redesigning Stage2 immediately.** The solution has
three parts, in strict order:

1. **Recognize that "harmless" and "useless" are two different mathematical
   quantities.** Transparency constrains the *value* of the pause contribution
   at the operating point; steering needs a nonzero *derivative* with respect
   to pause-state perturbations. `kl_transparent_emit` only penalizes the
   value. Its cheapest optimum (kill attention to pause KVs) happens to zero
   both — but that is a property of the optimizer's path, not a law. The fix,
   if needed, is to keep the value constraint and add an explicit sensitivity
   floor: an **"ε-transparent, δ-live"** objective (§6).

2. **Measure before redesigning.** Rows-only training sharply limits how the
   model can go inert (§3): the only trainable lever is the pause embedding
   row, downstream attention weights are frozen, and there is an irreducible
   RoPE-shift KL floor that the row cannot remove. Whether 400 steps at lr 1e-3
   actually parks the row in an invisible region is an empirical question the
   liveness battery answers for under one GPU-hour at 1.5B (§8). Do not pay for
   Stage2.5 before the diagnosis — every auxiliary term spends transparency
   budget.

3. **If the battery is red, the fix cannot live in Stage3/Stage4.** Liveness
   is a property of the trained weights. Stage3 is measurement; Stage4 is
   policy on top of an existing causal channel. Neither can create causal
   bandwidth that training removed (§5). Red ⇒ Stage2.5 retrain, with the
   concrete minimal design in §6 (near-pause KL exemption + injection-gain
   hinge; **not** the contrastive + attention-floor combo I sketched in the
   main review — that proposal is revised here, see §6.1).

**On the executor's redesign: correct in structure, correct in priorities, not
too conservative — and missing seven specific things** (§4.4), the most
important being the within-prompt AUROC endpoint that their own 10-samples
design enables, the random-direction control, and fixing the *cause* of the
unlabeled-judge confound rather than just reporting it.

---

## 2. Direct Answers (executor's eight questions)

**Q1 — Redesign correct, too conservative, or missing a better path?**
Correct and appropriately conservative; the caution is proportionate to B1–B5.
Not missing a fundamentally better path — the alternatives (steer content
tokens, train-time-only safety) are scope changes, not better versions of this
plan. It is missing seven concrete items, none fatal: within-prompt AUROC as
the primary Stage3 endpoint, a random-direction norm-matched control, the
judge-truncation cause-fix (T8), the natural-pause story (emission rate,
forced/natural/hybrid modes), CIs on AUROC deltas, a defined yellow-liveness
path, and battery calibration on the full-SFT positive control. Details §4.

**Q2 — Best concrete solution for the inert-pause risk?** Ordered:
(a) check `tie_word_embeddings` on both checkpoints — it changes the risk
profile qualitatively (§3.4); (b) run the minimal battery (§8), calibrated on
the old full-SFT checkpoint as positive control; (c) only if red/yellow:
Stage2.5-A (near-pause KL exemption, free) then Stage2.5-B (injection-gain
hinge, the direct fix) per §6; (d) keep the label-dependent contrastive term
in reserve, not in the default recipe. The conceptual core: decouple
transparency (value at the operating point) from liveness (sensitivity to
perturbation) and constrain both explicitly.

**Q3 — Is Stage2.5 = contrastive/probe-margin + attention-mass floor the right
fallback?** Partially. I am revising my own proposal (§6.1): **demote the
contrastive/probe-margin term to reserve** (it needs safety labels, and with
the currently available off-policy labels it would bake the prompt-risk
shortcut *into the port* — the worst outcome), and **replace the
attention-mass floor with an injection-gain hinge** (the floor is a proxy,
gameable by sink-like attention with dead values, and painful to implement
under FlashAttention; the hinge trains exactly the quantity the battery
measures and Stage4 needs). The parenthetical option in the old proposal —
scheduled KL reduction near the pause — is promoted to the v0 (Stage2.5-A).

**Q4 — Can Stage4 still use GPRS after Stage2.5?** Yes, algorithm unchanged.
Required: re-derive every artifact on the new checkpoint (û, μ_safe, probe,
τ, live-layer set), re-tune λ/ρ (pause-state geometry changed), and note a new
coupling: Stage2.5-B trains liveness *at a specific layer*, so Stage4's
steering layer is now pinned by the Stage2.5 config — record it in both. If
the reserve contrastive term is ever used, the random-direction and
prompt-baseline controls become load-bearing (gate and direction would then
correlate with a trained objective). Details §7.

**Q5 — Minimal reliable liveness battery?** Two tests, three models, ~200
prompts: injection-gain curves (3 layers × ε∈{1,2,4}σ, random + mean-diff
directions, next-16-token KL, content-token and BOS anchors) plus pause-KV
ablation. Attention maps are diagnosis-only; safe/unsafe patching is needed
before *claims*, not for go/no-go. Under 1 GPU-hour at 1.5B, 2–3 at 8B.
Thresholds must be calibrated on the full-SFT positive control or they are
guesses. Details §8.

**Q6 — Mandatory Stage3 fixes before claiming pause-specific signal?**
T1 (de-alias controls; pause-free-forward control primary), T2 (prompt
baselines incl. text-classifier), T4 upgraded to the within-prompt design
(§4.2), bootstrap CIs on all AUROC deltas, and carried B7 (checkpoint repoint
+ rows-only verify). T3 (natural-pause extraction) becomes mandatory iff
natural emission > ~5%. One added caveat the executor missed: do not kill the
trajectory claim on offset-3 forced pauses alone — at 3 CoT tokens the
trajectory has barely diverged; probe deeper forced offsets and mid-CoT
insertions too (§4.2). Details §9.1.

**Q7 — Mandatory Stage4/eval fixes?** Liveness gate (T13); learned delta
deleted and replaced by on-policy *paired* mean-diff with QC (B2); GPRS port
with persistent pause-ordinal counter and forced/natural/hybrid modes
(T5/T6/T12, incl. cot4 offset); judge truncation cause-fix (T8) — reporting
unlabeled rate without fixing truncation leaves the α-correlated missingness
confound in place; summarizer unlabeled + labeled-only columns (T9);
CoT-segment judging as the primary endpoint (T10); capability wired (T11);
calibrated over-refusal on answers only; random-direction norm-matched control
in the micro-pilot; provenance manifests (T15). Details §9.2.

**Q8 — Final ordered plan with kill criteria?** §10. Headline: P0 is hygiene +
battery + laptop-safe code fixes; P0.5 is the conditional Stage2.5 branch; P1
is on-policy data → fixed Stage3 → direction QC; P2 is micro-pilot →
1.5B pilot → 8B. Every phase has a numeric kill criterion, and the two most
likely kills (battery red; within-prompt AUROC ≈ 0.5) both have pre-registered
non-fatal landing zones.

Mapping to the user's three questions: (1) = Q2/§6 — yes, there is a good
solution; (2) = §4; (3) = §5 — precisely: liveness failure **cannot** be fixed
within Stage3/Stage4; red battery ⇒ Stage2 objective change (Stage2.5
retrain). Green battery ⇒ no Stage2 change, and Stage2.5 should *not* be
added prophylactically.

---

## 3. What the Code Actually Constrains (and why the risk is real but not a law)

These five facts, all verified in `pause_kl_trainer.py`, are the ground truth
for every design decision below. The main review's §2 stated the tension; this
sharpens it.

### 3.1 The teacher is exactly the base model

The "teacher" is the same model run on the pause-stripped sequence
(`:358–366`, `torch.no_grad` + eval mode). Stripped inputs never touch the
pause row; only pause embedding rows are trainable (`_assert_rows_only_training`,
`:142–160`, plus the step-1 bit-identity callback `:25–82`); `weight_decay`
must be 0.0 (`:143–148`). Therefore the teacher distribution **is the base
model's distribution, stable for the whole run** — not a moving target.

Corollary (structural, free): **on any input containing no pause token, the
trained model is bit-identical to base.** The entire behavior-preservation
question reduces to "what happens when pauses are in context." This is the
strongest neutrality guarantee in the project and it survives every Stage2.5
variant proposed in §6, because none of them adds trainable model parameters.

### 3.2 The only lever is the pause embedding row

"Inertness" here does **not** mean "the model learns to ignore pauses" — the
attention weights are frozen and cannot be rewired. It means "the optimizer
parks the pause row at a point that frozen W_K/W_V map to ignorable
keys/values for typical downstream queries." Whether such a point is reachable
in 400 steps at lr 1e-3 from the given init is a question about the frozen
model's embedding geometry — i.e., **empirical, per-checkpoint** — which is
why the battery, not intuition, decides.

### 3.3 The RoPE floor concentrates pressure on the pause KVs

Continuation tokens sit at positions shifted by the number of preceding pauses
(3) relative to the teacher; the collator provides no position-id remapping
(`_model_kwargs` passes `position_ids` only if present; it is not). A single
embedding row cannot undo a positional shift of *other* tokens, so
KL(student‖teacher) has an irreducible floor. Two consequences:

- Perfect transparency is unattainable; the reported continuation-KL should
  never be expected to hit 0, and its plateau is diagnostic.
- The *reducible* part of the loss is precisely the pause KV contribution, so
  residual gradient pressure concentrates on killing exactly the channel
  Stage4 needs. This is why the inert-pause risk is real despite rows-only.

Counter-nuance for honesty: invisibility is not the *only* optimum. A pause
row whose values actively nudge the shifted continuation back toward the
unshifted teacher ("corrective" solution) would score better than invisibility
on the KL and would be live by construction. Representing a context-general
correction in one vector is hard, so invisibility is the simpler attractor —
but this is another reason the outcome must be measured, not assumed.

### 3.4 Tied vs. untied embeddings change the risk qualitatively (new check)

Emit CE at pause slots shapes the **output** row (it needs
`h_prepause · e_out` large); suppression pushes the same output row away from
alignment with all other hidden states; transparency shapes the **input** row
(what the pause position injects). `_embedding_weights` (`:15–22`) trains the
output row separately only when embeddings are untied.

- **Tied** (`tie_word_embeddings=true`, expected for the Qwen-1.5B line): one
  vector serves all three pressures. Emit forces it to stay aligned with
  pre-pause hidden states, i.e., inside the contentful region of embedding
  space — going fully invisible would destroy emission. Liveness plausibly
  survives as a side effect.
- **Untied** (expected for the Llama-8B line): `e_out` absorbs emit and
  suppression; **the input row's only training signal is "be invisible."**
  Monotone pressure, nothing pushing back except optimization inertia and the
  corrective-solution possibility (§3.3).

Action (add to P0): read `tie_word_embeddings` from both checkpoints'
`config.json` before interpreting anything. Expectation to pre-register: 1.5B
more likely to pass the battery, 8B more likely to fail — and therefore **1.5B
battery results must not be extrapolated to 8B** (the battery reruns at 8B in
P2 regardless).

### 3.5 The pause-logit masking keeps the objective internally consistent

`_kl_loss` sets the pause logit to −inf in *both* student and teacher before
softmax (`:279–283`), so the transparency term never fights emit/suppression
over the pause coordinate itself. This matters for §6: the added terms can use
the same masking convention and stay orthogonal to emission. (Also noted:
`pre_weight: 0.0` in the 1.5B config is harmless — pre-pause positions have a
pause-free context, so student ≡ teacher there identically and the pre-KL is
≈0 regardless.)

---

## 4. Assessment of the Executor's Redesign

### 4.1 Core diagnosis — agree

"The first question is not steering; the first question is whether the pause
position remains causally live" is exactly right and matches the main review's
§2/B1. Also correct: old full-SFT Stage4 evidence does not transfer; the
current Stage3/Stage4 designs cannot be reused as-is.

### 4.2 Proposed Stage3 changes — agree, with one upgrade and three sharpenings

**Prompt baselines (their point 1): agree** (= T2). `pre_think` is a good
addition alongside `last_prompt_token`; keep the prompt-only text classifier —
it is the null hypothesis that costs nothing.

**Control fix (their point 2): agree** (= T1), with a preference order they
leave ambiguous: the **pause-free-forward matched-content control should be
primary**, because a matched-depth token in the *paused* sequence is still
causally downstream of the pause (it sees pause KVs); it controls for
position/feature-type but not for pause influence. Use both; never alias
(verified again: `extract_hidden_states.py:337,339` assigns
`control_cot_3/4 = post_pause_1/2`).

**On-policy 10-samples-per-prompt relabeling (their point 3): agree, and this
is better than they realize.** Sampling the same prompt 10 times with
per-generation CoT judge labels enables **within-prompt AUROC**: can the pause
probe discriminate safe from unsafe *generations of the same prompt*? Prompt
features give exactly 0.5 on this metric by construction — it is the cleanest
possible de-confounder, strictly stronger than any baseline-subtraction
scheme. Make it the primary Stage3 endpoint. Practicalities:

- Restrict to mixed-outcome prompts (per-prompt unsafe rate in [0.2, 0.8]
  over the 10 samples); expect to mine borderline sources to find ~200–400 of
  them (2–4k generations + CoT judging — feasible).
- Note the dependency: within-prompt labels require the **CoT-segment judge
  (T10)** — T10 is therefore a Stage3 prerequisite, not just a Stage4 eval fix.
- Pause states on generations: either natural pauses (if emission > 5%) or
  forced insertion at generation time — the steered-generation script at α=0
  already does exactly this.
- Judge label noise attenuates AUROC. With ~10% label noise a true 0.75
  discriminator reads ≈0.70. Set the bar accordingly: **pass = point estimate
  ≥ 0.60 with 95% bootstrap CI excluding 0.55**, not the 0.95s of the
  confounded grids. Report judge agreement on a double-judged subset.

**Their success criterion needs one correction.**
`pause_AUROC − max(prompt_baseline, true_content_control)` conflates two
different questions:

- *Trajectory vs. prompt* (pause > prompt baseline): **required** for Stage4's
  premise. Keep it (as the pooled, secondary endpoint; within-prompt AUROC is
  the primary version of the same question).
- *Pause vs. content* (pause > content control): **not required.** Stage4
  steers at pause positions because that is the scoped, controllable port —
  the pause state needs to *carry* the trajectory signal, not to beat content
  tokens at carrying it. Content tokens carrying more signal is expected and
  harmless. Report the content comparison as context; do not gate on it.

**Added caveat (executor missed it):** at insert offset 3, the pause has seen
3 sampled CoT tokens — the trajectory has barely diverged, so within-prompt
AUROC at offset-3 pauses may sit near 0.5 *for depth reasons, not port
reasons*. Before declaring "prompt-risk only," also probe the offset-7/8
layouts (configs exist), natural pauses, and/or forced insertions deeper in
the CoT (e.g., after 32/64 generated tokens). The kill criterion in §10 is
phrased over this full position set, not offset-3 alone.

**"pause_or_post_pause" in the criterion:** the headline metric must be pause
positions themselves (the steering port). Post-pause positions are supporting
evidence only — under the current design they are the very tokens whose
aliases previously masqueraded as controls.

### 4.3 Proposed Stage4 changes — agree on all four, with pins

1. **Liveness first: agree** (B1/T13). Add: calibrate pass/fail on the
   full-SFT positive control (a battery that has never seen a live model
   cannot define "dead"), and define the **yellow** outcome — pass at only
   some layers ⇒ proceed restricted to those layers, queue Stage2.5-A for the
   next Stage2 train.
2. **Replace learned delta with on-policy mean-diff: agree** (B2, §7 of the
   main review). Upgrade: use **within-prompt paired differences** (same
   prompt, safe vs unsafe generations, difference of pause states) so the
   prompt-identity component cancels in û by construction. QC gates: seed
   resampling cosine ≥ 0.8; held-out 1-D projection AUROC; cosine vs the
   probe-weight direction (`w/std`, un-standardized — R6).
3. **GPRS formula: correct** — it is the main review's §7 recommendation.
   Pins: gate score computed in *standardized* space with `probe.pt` stats and
   the stored FPR≤0.05 threshold τ; the edit in *raw* space with un-standardized
   û; per-occurrence gating at decode time, which requires the persistent
   pause-ordinal counter (T6) so natural pauses are gated/steered
   deliberately rather than accidentally; λ ∈ {0.25, 0.5, 1.0}, ρ ∈ {0.05,
   0.10, 0.15}; log gate-fire rate per condition (safe-prompt fire rate is the
   over-refusal early-warning, target ≤ ~5–10%).
4. **CoT/answer split + metric list: agree** (B3/B4, T8–T10) — with the one
   material gap: the executor's list *reports* the unlabeled judge rate but
   does not **fix its cause**. The unlabeled rows come from right-truncation
   at 4096 that eats the instruction tail (`run_open_judges.py`), and the
   missingness correlates with α through generation length. Reporting it does
   not de-confound it. T8 (truncate the response segment head+tail, never the
   template; log a per-row `truncated` flag) is mandatory before any Stage4
   readout. Target unlabeled < 5% per condition.

### 4.4 What the redesign is missing (complete list)

1. **Within-prompt AUROC** as the primary Stage3 endpoint (§4.2) — their own
   data design enables it; the delta criterion alone under-uses it.
2. **Random-direction norm-matched control** in the micro-pilot — the only
   test separating "unsafe-direction removal" from "any perturbation induces
   caution." Without it, a GPRS win is uninterpretable.
3. **T8 cause-fix for judge truncation** (§4.3.4).
4. **Natural-pause story**: emission rate measurement, forced/natural/hybrid
   Stage4 conditions (T3/T6/T12), and the claim-scoping rule if emission ≈ 0
   (all claims say "forced-pause intervention").
5. **Uncertainty**: bootstrap/DeLong CIs and seed variation on every AUROC
   delta; "near zero" without a CI is not a decision.
6. **Yellow-liveness path** (layer-restricted proceed) — the battery will
   plausibly return "partially alive," and the plan should not treat that as
   red.
7. **Battery calibration on the positive control** + the tied/untied check
   (§3.4) + pre-registered Stage2.5 decision rule and neutrality budget (§6.4)
   so the fallback is a branch, not a debate.

Minor: their eval list includes capability, but nothing currently consumes the
config's capability keys (T11) — ownership of that wiring should be explicit.

**Verdict on the redesign: adopt it with the §4.2–§4.4 amendments.** It is not
too conservative; every piece of caution is anchored to a demonstrated defect
(B1–B5). Nothing in it forecloses a better path, because no better path within
the pause-only scope exists: the two rivals — steering content tokens, or
abandoning inference-time steering for train-time safety — are scope
decisions, not improvements of this design.

---

## 5. Where the Fix Lives: Stage3/Stage4 vs. Stage2.5 (precise decision rule)

The user asked for precision here, so, precisely:

**Liveness is a property of the trained weights.** Stage3 is measurement
(probes read states; they do not change causal structure). Stage4 is policy on
top of an existing causal channel (an edit at a pause state can only reach
future tokens through downstream attention to pause KVs; if that attention is
≈0, the edit is discarded — first-order gain ≈ Σ attn·W_V·Δh ≈ 0). **No
Stage3/Stage4 change can create causal bandwidth that Stage2 training
removed.** Two pseudo-fixes to reject by name:

- *Steer harder* (large ‖Δh‖ to drag attention back via second-order key
  shifts): that is the off-manifold regime; the old CSVs show what it buys —
  termination damage (α=1 think_end 0.095) and refusal shift (α=2 +10.5pt).
  The norm cap exists to forbid exactly this.
- *Steer content tokens instead*: abandons pause-only scope; a scope decision
  for the user, not an engineering fix.

Decision table, applied to the battery result (§8) on the chosen checkpoint:

| Battery outcome | Definition | Action |
|---|---|---|
| **Green** | ≥1 admissible mid layer passes injection-gain (≥25% of content anchor, ≥5× BOS) **and** KV-ablation is clearly nonzero | **No Stage2 change.** Do not add Stage2.5 terms prophylactically — each spends transparency budget. Proceed: Stage3 rerun → GPRS. Steer only at live layers. |
| **Yellow** | Passes at few layers / marginal gains / one test passes | Proceed **restricted to live layers**; queue Stage2.5-A (free) for the next Stage2 training run; re-battery on that checkpoint. |
| **Red** | Pause gain statistically indistinguishable from BOS anchor at all layers **and** ablation ≈0, **with the positive control green** | **Stage4 stops. Stage2 must change**: Stage2.5-A+B retrain (§6), then re-battery. This is a ~400-step rows-only retrain, not a redesign of the pipeline. |
| **Red after Stage2.5-A+B** (w_live raised to ≤1.0) | — | The recipe's premise fails. Publishable negative ("KL-transparent pause training produces causally inert pause states") + scope decision, per the main review's §10 claims table. |

One asymmetry worth stating: the *reverse* direction also holds. If the
battery is green, Stage2.5 is not "nice to have" — it is **harmful surplus**,
because A relaxes transparency near pauses and B adds an equilibrium
constraint, both for a property the checkpoint already has.

---

## 6. Concrete Minimal Stage2.5 Design

### 6.1 Re-evaluation of my previous proposal (requested explicitly)

The main review's fallback was: contrastive/probe-margin term + attention-mass
floor (with "scheduled reduction of KL weight on the first post-pause tokens"
as a parenthetical alternative). Revision:

- **Contrastive/probe-margin → reserve (C), not default.** Four reasons.
  (1) Readability was never the failing property — pause states inherit
  context separability through frozen attention for free (the old grids show
  pause AUROC ≈ neighbors); liveness is the failing property, and a probe
  margin does nothing for it. (2) With the labels available at Stage2 training
  time (off-policy dataset provenance), the term would optimize the pause row
  to read *prompt-risk* features — baking the exact confound Stage3 must
  eliminate into the steering port itself. (3) It couples Stage2 to safety
  labels, weakening the "insert pauses without changing behavior" neutrality
  claim and muddying the no-full-SFT ablation story. (4) It adds a heavy
  pipeline dependency (generate + CoT-judge before training).
- **Attention-mass floor → replaced by the injection-gain hinge (B).** The
  floor is a proxy: necessary-ish for first-order value injection, but
  satisfiable by sink-like attention with content-free values, and it measures
  attention, not effect. Implementation is also painful — FlashAttention does
  not expose weights; `output_attentions=True` forces the eager path (memory
  and speed hit), or you hand-roll Q·K from hooks. The hinge needs no
  attention weights and trains the *exact* quantity the battery measures and
  Stage4 consumes.
- **The parenthetical → promoted to v0 (A).** Removing the local pressure is
  the cheapest intervention and may be sufficient on its own.

So: the old proposal was directionally right ("keep `kl_transparent_emit`,
add a liveness-preserving term, re-run the battery") with the wrong emphasis.
What follows is the corrected minimal design.

### 6.2 Stage2.5-A — near-pause KL exemption (v0: no labels, zero extra compute)

**Loss change.** None added; re-weight existing. In `_select_kl_pairs`
(`pause_kl_trainer.py:201–243`), track the position of the last pause of the
run per row; route post-pause pairs with target distance `d ≤ k` from that
pause into a third bucket `near_pairs`; compute its KL with the existing
`_kl_loss` machinery and weight it `near_pause_weight`. Total loss becomes:

```
emit_weight·CE_pause + continuation_weight·KL_far + near_pause_weight·KL_near
+ pre_weight·KL_pre + suppression_weight·L_suppress
```

**Where it applies.** Only the first `k` non-pause continuation targets after
each pause run; everything else identical (current 1.5B weights: emit 0.3 /
continuation 1.0 / pre 0.0 / suppression 1.0).

**Config/plumbing.** `pause_kl.near_pause_exempt_tokens: 8` (grid 4–16),
`pause_kl.near_pause_weight: 0.0` (grid {0.0, 0.1}); plumb via the existing
env pattern (`run_stage2_sft.py:340–372`).

**Labels required.** None. **Compute.** Zero extra.

**What it does and does not do.** It removes the *pressure toward* inertness
in the local window; it does not *create* liveness. The bet is that the init's
natural attention to a fresh embedding row survives when unpressured. Note the
autoregressive amplifier: Stage4 steering only needs the pause edit to move
the next few tokens — those tokens then propagate the change themselves, so
*local* liveness is sufficient for steering even if distant tokens learn to
ignore pause KVs.

**Cost.** Transparency is unenforced for `d ≤ k`. This is a measured,
capped trade (§6.4), reported by distance bucket.

### 6.3 Stage2.5-B — injection-gain hinge (v1: no labels, ~+40–50% step time)

**Loss term.**

```
L_live = w_live · mean_over_pause_runs( max(0, m − KL_k(p_pert ‖ p_clean)) )
```

- `v ~ N(0, I)` normalized, **fresh per example per step**; ε sampled from
  {1, 2, 4}·σ_h, where σ_h = detached per-example RMS of hidden states at the
  hook layer.
- Hook at block index `hs_layer − 1` (same convention as the steering stack),
  `hs_layer` = the intended Stage4 steering layer (1.5B: 14; 8B: 16 or 20;
  optionally alternate two layers across steps). Perturbation added at
  positions `input_ids == pause_token_id`.
- `p_clean` = the logits of the **already-computed** student forward
  (`compute_loss` line 353), detached — zero extra clean compute. One extra
  grad-enabled forward with the hook active.
- `KL_k` = mean token-level KL(pert‖clean) over the next `k = 8` non-pause
  targets after each pause run, using the same pair-selection and pause-logit
  −inf masking as `_kl_loss` (§3.5).
- Margin `m`: **calibrate from the positive control** — take the full-SFT
  checkpoint's measured next-16-token KL response at ε = 2σ from the battery
  and set m to ~25–50% of it; expect order 0.05–0.1 nats. `w_live = 0.3`
  initial; raise only if the hinge is still active at end of training.

**Why this is the right shape.** The hinge is a *floor*, not a maximization:
once `KL_k ≥ m`, the term is exactly zero with zero gradient, so at
equilibrium it cannot fight the transparency objective — it only excludes the
inert corner of the solution space. Gradient path: pause row → pause hidden
state → perturbed branch's keys/values → downstream attention → logits; i.e.,
it trains the row to sit where the frozen heads keep a live read channel.

**Degenerate-solution guards** (each mapped to a failure mode):
fresh random `v` per batch (no single-direction sensitivity); randomized ε (no
magnitude overfit); hinge (no unbounded sensitivity chase); detached clean
branch (cannot satisfy the hinge by moving the clean operating point — only
indirectly via the shared row, which the transparency KL immediately pulls
back).

**Labels required.** None. **Compute.** ~+40–50% per step (one extra forward,
backward through one extra branch); trivial for 400 steps at 1.5B, acceptable
on 4×A100 at 8B.

**Honest limitation.** B trains sensitivity to *random* directions — necessary
but not sufficient for the *safety* direction to steer well. The safety-subspace
checks remain the battery's patching test and the micro-pilot's
random-direction control. Also, `L_live` at one layer certifies that layer
only; Stage4 must steer where liveness was trained (§7).

### 6.4 Preserving "insert pauses without changing behavior" — the neutrality budget

State the impossibility honestly, then budget it: **exact transparency and
usable steering gain cannot both hold in the limit**, but they are not at odds
at the operating point — transparency constrains the value, liveness the
derivative (§1). Unsteered behavior is the thing the original goal protects,
and it is preserved as follows:

- **Structural (free, survives A+B+C):** rows-only assert + step-1 invariant
  callback unchanged (A re-weights existing terms; B adds no parameters; C's
  auxiliary head is loss machinery, discarded, never in the model);
  `weight_decay = 0`; **pause-free inputs remain bit-identical to base**
  (§3.1). Capability and refusal *without pauses in context* cannot change,
  full stop. Verify once with a 32-prompt logit-equality smoke test.
- **Behavioral with pauses (measured gates, pre-register the numbers):**
  1. Pause-conditional continuation KL vs base, bucketed by distance:
     `d > k` within +10% relative of the Stage2-baseline checkpoint;
     `d ≤ k` reported, target mean ≤ 0.1 nats.
  2. GSM8K + MATH500 EM with forced pauses: |Δ| ≤ 0.5 pt vs base-with-pauses
     and vs the Stage2-baseline checkpoint.
  3. XSTest-safe / OR-Bench-hard-safe refusal (answer-segment judge, not
     keyword regex): Δ ≤ +1 pt.
  4. think_end rate, EOS termination, length distribution: within seed noise.
  5. Emission stats (forced-slot pause prob, natural emission rate,
     false-pause rate at non-pause positions): within ±20% relative of the
     Stage2 baseline.
  6. Battery green at the hook layer (the point of the exercise).

### 6.5 Ablations proving the aux terms did not reintroduce full-SFT behavior

"Reintroduced full-SFT" is structurally impossible in the weight sense
(rows-only invariant still enforced); the meaningful version of the worry is
*behavioral drift through the pause channel*. The attribution grid bounds it:

Train at 1.5B, 400 steps each (baseline reusable): **{Stage2 baseline, +A,
+B, +A+B}** × measure **{battery, continuation-KL by distance bucket,
emission stats, GSM8K slice, refusal slice}**.

- A-only tells you how much mere pressure-removal buys (if A alone goes green,
  stop there — B is surplus).
- B-only tells you whether the hinge is sufficient without spending any
  transparency budget near pauses.
- A+B is the candidate if neither alone suffices.
- Any cell where gates §6.4(1–5) fail ⇒ that variant is over-weighted; halve
  `near_pause_weight` distance-window or `w_live` before concluding anything.

### 6.6 Stage2.5-C — reserve only

Probe-margin/contrastive on pause states, **only if** the battery is green but
the fixed Stage3 shows within-prompt pause AUROC ≈ 0.5 *while content
positions clear the bar at matched depth* — i.e., readability specifically
missing at the port, which is the one failure A/B cannot address. Form: hinge
margin on a small auxiliary linear head over pause states (head = loss-side
module, discarded after training), trained **only on within-prompt contrastive
pairs with on-policy CoT-judge labels** — never on off-policy provenance
labels, and never on cross-prompt pairs (both would bake prompt-risk into the
row). Accept before using it: the label pipeline becomes a training
dependency, and the neutrality story acquires an asterisk that must appear in
the paper's method section.

---

## 7. GPRS under Stage2.5 (Q4 in detail)

The algorithm is unchanged:
`h ← h − λ·((h − μ_safe)·û)₊·û`, norm-capped at ρ‖h‖, applied only when the
online probe score exceeds τ. What changes is everything *derived*:

1. **Re-derive on the new checkpoint:** û and μ_safe (on-policy, within-prompt
   paired), the Stage3 probe and its τ, and the live-layer set. Directions,
   probes, and thresholds never transfer across checkpoints — the Stage2.5 row
   moved, so pause-state geometry moved.
2. **Layer coupling (new):** Stage2.5-B certifies liveness at its hook layer
   only. Stage4's steering layer = the B hook layer (or the battery-verified
   subset). Record the layer in both configs — this is exactly the class of
   provenance break that produced B6/T14 (8B config layer 16 vs. results
   layer 20).
3. **Re-tune λ and ρ** on the micro-pilot grid; expect different optima since
   ‖h‖ and the unsafe-component scale changed.
4. **Two added checks if C was used:** the gate and the direction now
   correlate with a *trained* objective, so (a) the random-direction control
   and the prompt-baseline delta become load-bearing, not optional; (b) report
   gate-fire rate on safe prompts per condition — if Stage2.5 inflated
   probe scores on safe prompts, over-refusal will show up here first.
5. **One added ablation regardless:** gate-on vs gate-off at fixed λ, to
   attribute any over-refusal reduction to the gating rather than the edit.

---

## 8. Minimal Liveness Battery (Q5 in detail)

Purpose-built for go/no-go at minimum GPU; strict superset deferred to
claim-time. All teacher-forced or ≤64-token continuations; no judges.

**Models (3):** new KL checkpoint (test article); old cot3 full-SFT ckpt250
(positive control — it demonstrably reacted to steering); base model with
pasted pauses (negative-ish control). Thresholds are *defined* by the
controls: a battery that has never seen a live and a dead model cannot label
either.

**Data:** ~200 prompts (100 safe / 100 unsafe), forced layout at the training
offset (cot3 for 1.5B, cot4 for 8B).

**Test 1 — injection-gain curves (primary).** Add ε·σ_h·v at pause states;
layers {7, 14, 21} (1.5B) / {8, 16, 24} (8B; add 20 given the old result);
ε ∈ {1, 2, 4}·σ_h; v ∈ {3 random seeds, mean-diff direction}; measure mean KL
of the next-16-token distributions vs the clean forward. Anchors at identical
ε: adjacent content token (upper) and BOS (lower).
**Pass:** pause gain ≥ 25% of content-anchor gain and ≥ 5× BOS gain at ≥1 mid
layer. **Red:** indistinguishable from BOS everywhere.

**Test 2 — pause-KV ablation (nearly free add-on).** Zero- and mean-ablate
pause KVs after prefill; measure continuation KL + edit distance on a 64-token
greedy continuation. Distinguishes "operating-point contribution ≈ 0"
(expected, by design — transparency) from "sensitivity ≈ 0" (fatal). A
transparent-but-live checkpoint shows small-but-nonzero ablation effects and
healthy injection gains; that signature is a *good* outcome, not a
contradiction.

**Deferred:** attention-mass maps — diagnosis of *why* a red result is red
(and FA2 makes them annoying); safe/unsafe patching — required before any
*trajectory-level claim* (it is the only test of the safety subspace
specifically), but not required for go/no-go.

**Cost:** per row ≈ 1 clean + ~12 perturbed + 2 anchor + 2 ablation short
forwards → well under 1 GPU-hour at 1.5B, ~2–3 at 8B. Implement as T13
(one script, one JSON report per model/layer). Interpretation: §5 table.

---

## 9. Mandatory-Fix Lists

### 9.1 Stage3 — before any "pause-specific signal" claim (Q6)

| # | Fix | Why it gates the claim |
|---|---|---|
| 1 | T1 de-alias controls; pause-free-forward control primary | current "controls" are post-pause aliases (`extract_hidden_states.py:337,339`) — comparisons vacuous |
| 2 | T2 prompt baselines (`last_prompt_token`, `pre_think`, prompt-only text classifier) | the null hypothesis is absent from the scan |
| 3 | T4 upgraded: on-policy 10×/prompt, CoT-judge labels, **within-prompt AUROC primary endpoint** on mixed-outcome prompts | the only design where prompt features score 0.5 by construction |
| 4 | CIs (bootstrap/DeLong) + seed variation on all AUROC deltas | "delta ≈ 0" is a decision and needs uncertainty |
| 5 | Probe deeper positions before killing the trajectory claim (offset-7/8, mid-CoT insertions, natural pauses) | offset-3 pauses see 3 tokens of trajectory; near-chance there is a depth result, not a port result |
| 6 | Carried B7: checkpoint repoint (`final/` = last step) + rows-only verify + pod pytest | wrong checkpoint invalidates everything downstream |
| 7 | T3 natural-pause extraction — mandatory iff natural emission > ~5% | deployment-relevant positions if they exist |

### 9.2 Stage4/eval — before any "reduced unsafe CoT without cost" claim (Q7)

| # | Fix | Why |
|---|---|---|
| 1 | T13 liveness battery green at the steering layer | B1 — no channel, no method |
| 2 | Delete learned delta; on-policy **paired** mean-diff û + QC (seed cosine ≥ 0.8, probe-transfer, cosine vs `w/std`) | B2 — current objective rewards any distribution damage |
| 3 | T5 GPRS port (projection edit, per-step gate, norm cap) | method replacement |
| 4 | T6 persistent pause-ordinal counter + `--steer_natural_pauses` + forced/natural/hybrid stats | decode-time naturals currently steered silently as ordinal 0 |
| 5 | T12 offset/mode plumbing (cot4 for 8B; `PAUSE_MODE`) | shell hardcodes offset 3 → wrong layout vs training |
| 6 | T8 judge-truncation **cause-fix** (truncate response middle, never template; `truncated` flag) | unlabeled fraction correlates with α; reporting it ≠ removing the confound |
| 7 | T9 summarizer: unlabeled count/rate + labeled-only rates + termination/repetition columns | rates currently silently deflated |
| 8 | T10 CoT-segment judge; **unsafe-CoT rate = primary endpoint** | the project goal is CoT-level and currently unmeasured |
| 9 | T11 capability with hook active (GSM8K/MATH500 EM) or delete the advertising config keys | capability under steering is currently not measured at all |
| 10 | Calibrated over-refusal: XSTest-safe / OR-Bench-hard-safe, answer segment only | keyword regex over CoT is not an over-refusal metric |
| 11 | **Random-direction norm-matched control** in the micro-pilot | separates direction semantics from generic perturbation caution |
| 12 | T15 provenance manifests (incl. actual MAX_NEW_TOKENS) | old run is not reconstructible from `res/` |

Gate on the readout itself: unlabeled < 5% per condition after T8, else the
run does not produce claims.

---

## 10. Final Ordered Plan — P0/P1/P2 with Kill Criteria (Q8)

**P0 — hygiene, diagnosis, laptop-safe code (this week; ~1 GPU-hour):**

| Step | Action | Kill / branch criterion |
|---|---|---|
| P0-1 | B7: repoint `sft_checkpoint` to battery-chosen ckpt; rows-only verify on trained weights; pod pytest; **read `tie_word_embeddings` on both models** (§3.4) | invariant fails ⇒ Stage2 bug — stop everything, fix Stage2 first |
| P0-2 | T13 battery script; run on {base, full-SFT ckpt250, new KL ckpt} at 1.5B | positive control not green ⇒ battery invalid, fix before interpreting. Test ckpt **red** ⇒ branch to P0.5. **Green** ⇒ skip Stage2.5 entirely (§5). **Yellow** ⇒ proceed restricted to live layers + queue Stage2.5-A |
| P0-3 (parallel, no GPU) | T1/T2 extraction fixes; T8/T9/T10/T11/T12 eval fixes; T5/T6/T7 GPRS port; T16 guard | — |

**P0.5 — conditional Stage2.5 branch (only on red):**

| Step | Action | Kill criterion |
|---|---|---|
| P0.5-1 | Implement A (+ config keys) and B (hinge, m calibrated from positive-control battery numbers) in `pause_kl_trainer.py` | — |
| P0.5-2 | 1.5B retrains: {+A, +B, +A+B} (400 steps each); §6.4 gates + re-battery | gates 1–5 fail ⇒ halve weights and rerun once |
| P0.5-3 | Adopt the cheapest green variant | **still red after A+B with w_live ≤ 1.0 ⇒ premise unsupported** — write the negative result (per main-review §10 claims table) and put the scope decision to the user |

**P1 — measurement (after battery green on whichever checkpoint):**

| Step | Action | Kill criterion |
|---|---|---|
| P1-1 | On-policy generation: 10×/prompt, mine mixed-outcome prompts (~200–400), CoT-judge (needs T10) | mixed-outcome prompts < ~100 ⇒ broaden sources before probing |
| P1-2 | Stage3 rerun: fixed controls + baselines + within-prompt endpoint, across offsets {3,4,7,8}, mid-CoT insertions, naturals if >5% | within-prompt pause AUROC CI ≤ 0.55 at **all** live layers and **all** depths ⇒ trajectory claim dead; pre-registered landing: proceed as *prompt-conditioned gating* with claims re-scoped (main-review §10 row 2) |
| P1-3 | Direction extraction (within-prompt paired mean-diff) + QC | seed cosine < 0.8 ⇒ more data, no steering yet |

**P2 — intervention:**

| Step | Action | Kill criterion |
|---|---|---|
| P2-1 | Micro-pilot (n≈100/label, 1 seed): GPRS vs **random-direction norm-matched** vs no-op; metrics: CoT judge, answer judge, termination, length, unlabeled, 50-item GSM8K slice | GPRS ≤ random on CoT-unsafe reduction, or capability −1pt, or over-refusal +2pt ⇒ try `safe_centroid_pull` variant once; if that also fails ⇒ negative result for clean inference-time steering |
| P2-2 | 1.5B full pilot: 3 seeds, forced/natural/hybrid, full §9.2 metric set, unlabeled < 5% gate | reduction only via refusal-shift/termination-damage (old α=1/α=2 signature) ⇒ claims limited per main-review §10 |
| P2-3 | 8B: cot4 offset (T12), config provenance fixed (T14), **battery re-run at 8B mandatory** (tied/untied asymmetry, §3.4) | 8B battery red ⇒ 8B goes through its own P0.5; do not ship 1.5B-only results as 8B claims |

---

## 11. Updated Go/No-Go Table

| Milestone | Verdict | Gates |
|---|---|---|
| Executor redesign as written | **ADOPT with amendments** | §4.2 endpoint upgrade + §4.4 items 1–7 folded in |
| Stage2.5 now, unconditionally | **NO-GO** | Only on battery red (§5); prophylactic aux terms spend transparency budget for nothing |
| Stage2.5-A/B implementation (code) | **GO now** (laptop-safe) | Runs only on the P0.5 branch |
| Battery (T13) | **GO — the P0 critical path** | Calibrate on positive control; tied/untied check first |
| Stage3 rerun | **GO after** B7 + T1/T2 + T10-judge available; claims need P1-1 labels | Within-prompt endpoint primary |
| GPRS micro-pilot | **NO-GO until** battery green + P1-2/P1-3 done; random-direction control mandatory | — |
| 1.5B full pilot → 8B | **NO-GO until** micro-pilot passes; 8B needs its own battery | — |

---

## Bottom Line

The executor has correctly identified that the project's next unit of work is
a measurement, not a method. Adopt the redesign with the amendments above. The
inert-pause risk has a clean conceptual resolution — transparency constrains
the value of the pause channel, steering needs its derivative, and only the
lazy optimum kills both — and a cheap empirical resolution: a sub-GPU-hour
battery whose thresholds are calibrated on the one checkpoint known to be
live. If the battery is green, do not touch Stage2. If it is red, no amount of
Stage3/Stage4 engineering will help: retrain with the near-pause KL exemption
plus the injection-gain hinge — both label-free, both compatible with the
rows-only invariant and the bit-identical-off-pause guarantee — and hold the
label-dependent contrastive term in reserve, because the one thing worse than
a dead steering port is a steering port pre-trained to read prompt risk.
Check `tie_word_embeddings` before anything else: the 1.5B and 8B models are
probably not running the same experiment.

# Fable Pro Methods Review: Stage1 Thresholded Accuracy and Matched-Horizon Baselines

Date: 2026-07-05
Reviewer mode: Fable-5 pro/high-rigor, strict senior ML methods review.
Responds to: `threshold_matched_horizon_addendum_260705.md` (Q1, Q2, Q3).

Reviewed sanitized packet `stage1_fable_review_retune12288_b20_260705/`:

- `README.md`
- `fable_probe_redesign_response_260705.md` (prior ruling: ONLY AFTER GATES)
- `threshold_matched_horizon_addendum_260705.md`
- `val_fixed/val_fixed_probe_report.tsv` (all 16 rows, including the
  `threshold` column)
- `delta_ci/hidden_surface_delta_ci_summary.tsv` (all 16 rows)
- `text_baselines/{harmbench_standard,reasoningshield,strongreject_full,wildjailbreak_vanilla_harmful}/summary.tsv`

The packet excludes raw prompts, raw CoTs, completions, and row-level
prediction JSONL. Nothing in this review required or used raw content. All
proposed analyses below are computed server-side on stored scores/features;
only aggregates enter future packets.

Citation-integrity note: external database verification (arXiv MCP) was not
permitted in this session. Every reference in Section 5 carries an explicit
confidence tag; entries below [high] confidence are quarantined in 5.G and
must be verified before appearing in any manuscript.

---

## 0. Verdict Summary

1. **Final recommendation: `BOTH_CPU_ONLY`.** Run the threshold/calibration
   reanalysis (Module T) and the matched-horizon reanalysis (Module M) as one
   preregistration, one unblinding cycle, zero GPU. Module M is the
   decision-bearing science; Module T is a subordinate reporting correction
   that is nearly free once row-level scores are in hand.
2. **Yes, thresholded balanced accuracy can be improved without any new
   extraction** — the current operating points are a deliberate low-FPR
   validation policy (val FPR pinned at 0.03–0.05 on all eight linear rows),
   not a score pathology. Expected recoverable headroom is ≈ +0.05–0.14
   balanced accuracy depending on run (Section 1.3).
3. **Threshold work cannot change any conclusion.** It moves along the ROC
   curve by construction; AUROC and all hidden-minus-surface deltas are
   untouched. Even at *oracle* test thresholds, every hidden probe still loses
   balanced accuracy to the `length_only` hindsight scalar on all four sources
   (Section 1.4). Module T is about honest reporting, not rescue. Anyone who
   presents Module T output as "the probe got better" is misreporting it.
4. **The fair-comparison fix for the prefix-vs-hindsight problem is exactly
   the matched-horizon design already approved as M1** — this memo pins down
   the remaining degrees of freedom: exact `text@k` construction, tokenizer
   and indexing rules, pair-complete censoring, fixed-cohort sensitivity,
   the four surface families, and the E1/E2/E3 statistics with Holm and
   McNemar (Section 3).
5. One genuinely new comparative finding is available cheaply: the
   **threshold-transfer gap** (oracle-BA minus achieved-BA under a frozen
   validation policy) measured side-by-side for hidden probes vs surface
   baselines under the *same* policy. Early evidence (Section 1.2) suggests
   hidden-probe score locations do not transfer across sources while surface
   thresholds do; if confirmed, that is an additional, well-posed negative
   property of the hidden representation worth one paragraph in the paper.

---

## 1. Independent Verification and New Observations

### 1.1 The current operating point is a policy artifact, not a calibration mystery

From `val_fixed_probe_report.tsv`: recorded decision thresholds are 0.67–0.85
(all ≫ 0.5) and validation FPR is pinned at 0.031–0.049 on **all eight**
linear rows. That pattern is consistent with a "maximize recall subject to
val FPR ≲ 5%" (or fixed-low-FPR) thresholding rule. Consequence: with test
AUROC 0.68–0.84, an FPR≈5% operating point *must* have low recall — balanced
accuracy 0.51–0.68 is the arithmetically expected outcome, not new evidence
of score-separation failure beyond what AUROC already says.

Action required regardless of everything else: the original threshold rule
must be documented verbatim in the preregistration (gate GT0). I am inferring
it from the table; inference is not documentation.

### 1.2 Raw-score threshold transfer under LOSO is location-unstable — in both directions

- RS Stage1b (thr 0.845): val FPR 0.0306 → test FPR 0.0090, test recall
  0.0388, BA 0.5149. The threshold overshoots the shifted score distribution;
  the probe predicts almost nothing positive on the held-out source.
- WJB Stage1b (thr 0.666): val FPR 0.0484 → test FPR 0.1392. Undershoot in
  the opposite direction.

So the failure is *location/scale drift of the score distribution across
sources*, not a consistent conservative bias. This dictates the calibration
choice in Section 2.5: any fix must re-anchor the threshold's location per
fold (Platt intercept, or score quantiles), and temperature-style slope-only
rescaling is structurally incapable of helping (no bias term).

### 1.3 Achievable balanced-accuracy headroom (expectation-setting only)

Equal-variance binormal approximation: BA\*(AUROC) = Φ(Φ⁻¹(AUROC)/√2),
the balanced-accuracy optimum at the Youden point. Against current test BA:

| Run | Source | Test AUROC | Current BA | BA\* (binormal) | Headroom |
|---|---|---:|---:|---:|---:|
| S1  | HB  | 0.8396 | 0.6776 | ≈0.759 | +0.081 |
| S1  | RS  | 0.7025 | 0.5687 | ≈0.647 | +0.078 |
| S1  | SR  | 0.8147 | 0.6777 | ≈0.737 | +0.059 |
| S1  | WJB | 0.8252 | 0.6634 | ≈0.746 | +0.083 |
| S1b | HB  | 0.8101 | 0.5921 | ≈0.733 | +0.141 |
| S1b | RS  | 0.6792 | 0.5149 | ≈0.629 | +0.114 |
| S1b | SR  | 0.7461 | 0.6119 | ≈0.680 | +0.068 |
| S1b | WJB | 0.7818 | 0.6352 | ≈0.709 | +0.074 |

These are approximations for the prereg's expected-effect section; the
empirical oracle diagnostic (T4) gives the true ceiling. A validation-selected
threshold recovers this headroom only to the extent score locations transfer
(1.2 says: imperfectly; RS is the predicted worst case).

### 1.4 Even at oracle thresholds, hidden loses to the length scalar everywhere

`length_only` test BA: HB 0.8618, RS 0.7090, SR 0.7924, WJB 0.8125 — versus
hidden BA\* ceilings 0.759 / 0.647 / 0.737 / 0.746. The same holds on AUROC
directly (0.9278/0.7960/0.8629/0.8789 vs 0.8396/0.7025/0.8147/0.8252).
Therefore: **no threshold or calibration procedure can make the thresholded
hidden numbers competitive with even the weakest hindsight baseline.** The
negative/control verdict is threshold-invariant. This is why Module T is
reporting hygiene, not reanalysis of the conclusion.

### 1.5 The current thresholded table is operating-point-mismatched — a fairness bug worth fixing even in a negative result

Text baselines were thresholded at (effectively) 0.5 on balanced training
data, which lands near their BA-optimum (e.g., HB `word_bow` test BA 0.9079
≈ its binormal ceiling for AUROC 0.9655). Hidden probes were thresholded at
FPR≈5%. So the existing thresholded comparison contrasts baselines near
*their best* BA operating point with hidden probes far from theirs. AUROC
comparisons are unaffected, but any thresholded table in the paper must apply
**one identical threshold policy to every arm** (T-protocol below) or be cut.

### 1.6 Class balance makes balanced accuracy = accuracy here

In every baseline table, balanced_accuracy ≡ accuracy on every split —
consistent with the paired design's exact 50/50 balance (one safe + one
unsafe member per prompt). Two useful consequences: (a) maximizing BA is
maximizing accuracy, and with calibrated probabilities the Bayes threshold is
0.5 — so Platt + τ=0.5 is a principled BA-max policy; (b) the class prior
0.5 is *design knowledge*, licensing a label-free quantile threshold
(score median) as a transductive transfer aid (T3).

---

## 2. Q1 — Can thresholded accuracy be improved? (Yes; here is the exact protocol)

Sub-answers keyed to the addendum's numbering.

**2.1 Legitimacy.** Yes. Selecting a threshold on validation data
(train-source folds only under LOSO) and applying it frozen to the held-out
source is standard, leakage-free practice. Three honesty conditions:
(i) the policy is fixed in writing *before* recomputation (the test AUROCs
have already been seen, so this is post-hoc descriptive reanalysis — label
tables "operating-point correction; policy preregistered 2026-07-05");
(ii) all preregistered policy variants are reported, not the best one;
(iii) no new claims of improved discrimination. Note also that the val sets
(n=123–264) were already consumed once by config argmax; a threshold tuned on
the same 124 points inherits that noise (SE(BA) ≈ 0.03–0.045), which is why
the primary policy below is the 2-parameter Platt rule rather than an
empirical step-function argmax.

**2.2 Will it help?** Very likely, materially: expected +0.05–0.14 BA
(table 1.3), realized to the extent location drift (1.2) is handled — which
is exactly what Platt's intercept and the quantile variant do. Predicted
worst transfer: RS (largest val→test shrinkage and demonstrated location
overshoot). Preregister that prediction; it is falsifiable and free.

**2.3 Exact leakage-safe protocol.** Module T, Section 4.

**2.4 Global vs per-source thresholds.** Per-LOSO-fold, selected on that
fold's validation only. A single raw-score threshold shared across folds is
incoherent — each fold's probe is a different model with its own score
scale. The only coherent "global" policy is calibrated-space τ=0.5 (Platt
per fold, then the same 0.5 rule everywhere), which is T1. Never select
anything using held-out-source labels; the transductive T3 variant may use
held-out-source *unlabeled scores* only, clearly labeled.

**2.5 Platt vs isotonic vs temperature vs sweep.**
- **Platt scaling per fold on validation, threshold 0.5 in calibrated space —
  primary (T1).** Two parameters (slope+intercept); the intercept directly
  absorbs the location drift that is demonstrably the failure mode (1.2);
  lowest-variance option at n_val≈124; with 50/50 classes, calibrated-0.5 is
  the Bayes/BA-optimal rule (1.6).
- **Empirical BA-max sweep on validation — secondary (T2).** Monotone-
  invariant, assumption-free, but high-variance at these n; ties broken at
  the midpoint of the maximizing score interval (deterministic).
- **Prior-informed quantile ("median") threshold — transductive variant
  (T3).** τ = median of the held-out source's unlabeled test scores; valid
  because prevalence is 0.5 by design, uses zero labels; expected to be the
  most drift-robust; must be reported under an explicit "transductive" label.
- **Isotonic: no.** Needs on the order of 10³ calibration points; n_val is
  123–264; it will overfit the reliability curve.
- **Temperature scaling: no.** Slope-only, no bias term; cannot move the
  operating point's location, which is the actual problem. Platt strictly
  subsumes it for a binary probe.

**2.6 Diagnostics to produce (aggregate-only, packet-safe).**
- ROC and PR curves per run×source (computed from stored scores).
- Table per run×source: τ under each policy; val BA at τ; **test BA at
  frozen τ (the reportable number)** with recall/FPR; test BA at oracle τ —
  rendered in a visually separated "diagnostic ceiling" column, never in the
  comparison row; threshold-transfer gap = oracle-BA − achieved-BA.
- The same table for the validation-selected surface baselines and
  `length_only`, under the identical policy — this makes the thresholded
  comparison finally apples-to-apples (1.5) and yields the comparative
  transfer-gap finding (0.5).
- 95% CIs on every BA/gap via pair-cluster bootstrap (resample prompt pairs,
  2000 reps — reuse the delta-CI machinery's grouping).
- Optional descriptive extras: reliability diagrams and Brier score per fold
  (calibration quality is a separate axis from discrimination; report, don't
  decide on it). Also report TPR@FPR≤0.05 with the val-set threshold as the
  deployment-flavored row — the low-FPR regime is legitimately the relevant
  one for safety monitoring, so keep that row *alongside* the BA-max row
  rather than replacing it.

**2.7 What counts as a meaningful improvement (vs merely sliding along the
ROC).** Everything here slides along the ROC — that is the point and the
limit. Preregister these readouts:
- **M-1 (operating-point artifact confirmed):** test BA@τ_frozen improves by
  ≥ +0.04 over the current low-FPR numbers on ≥3/4 sources.
- **M-2 (threshold policy transfers):** transfer gap ≤ 0.03. If the gap
  stays > 0.05 on some source despite Platt/quantile anchoring, that is the
  "score locations don't transfer across sources" finding — report it as a
  property of the representation, compared against the surface arms' gaps
  under the same policy.
- **Not meaningful, ever:** any wording implying discrimination improved.
  AUROC, AUPRC, and all Section-6 (prior review) continue/kill logic are
  threshold-invariant and unchanged by Module T.

---

## 3. Q2 — Matched-horizon comparison: exact design

Sub-answers keyed to the addendum's numbering. This instantiates, and does
not modify, the approved M1 plan; where the prior review left a degree of
freedom, this section fixes it.

**3.1 Is matched-horizon truncation the right fix?** Yes — necessary, and
with two companions, sufficient for the accessibility claim: (a) the paired
within-pair endpoint (E2), which exploits the verified `prompt_only=0.500`
control; (b) the residual/incremental endpoint (E3), which operationalizes
"beyond surface" directly. Full-trajectory baselines stay as hindsight
ceilings. What truncation does *not* fix and must not be advertised as
fixing: winner's-curse selection (fixed by the global low-df selection rule),
label provenance (G1), human QA (G6).

**3.2 Exact `text@k` construction.** `text@k` must be a deterministic
function of exactly the information set of `hidden@cot_k`:
- Tokenizer: the subject model's tokenizer (DeepSeek-R1-Distill-Llama-8B).
  Never whitespace tokens, never characters, never another model's tokenizer
  for *defining* k (feature extraction inside the prefix may use any
  representation — see 3.7).
- CoT boundary: the same CoT-start marker/offset convention the hidden
  extractor used. Gate G4 produces a one-page written statement of the
  indexing semantics (0- vs 1-indexed; whether `cot_k` is the hidden state
  *at* the k-th generated token position, hence a function of prompt +
  tokens 1..k inclusive) plus a token-id hash check on a sample,
  aggregate-only output.
- Prefix: prompt + generated CoT tokens 1..k inclusive, including any
  terminal/segment markers that occur within the first k tokens (both arms
  see them equally; if a CoT ends exactly at k, the terminal token itself is
  legitimately visible to both arms).
- Vectorizers/encoders fitted on train folds only — vocabulary fitted on
  val/test is leakage (L3).

**3.3 What `text@k` includes.**
- Primary scope: **prompt + CoT[1..k]** — matches the hidden state's
  information set (it attends over the prompt).
- Diagnostic scope: CoT[1..k] only — isolates generation-borne signal.
  Prompt-only remains as the ≈0.500 control line, not a competitor.
- Position-indexed token identities: yes, as one of the four surface
  families (3.7) — it is the closest surface analog to "state at position k".
- Length-so-far / ended-by-k indicators: **degenerate by construction** under
  the censoring rule below (every retained row has length ≥ k, so
  length-so-far ≡ k and ended-by-k ≡ {length = k}, visible as the terminal
  token anyway). Document the degeneracy instead of adding dead features;
  the length confound is handled by the censoring design itself, and raw
  trajectory-length signal is quarantined in the `length_only` hindsight
  ceiling.

**3.4 Definition of k.** Generated-CoT position index in model-tokenizer
tokens, identical to the extractor's `cot_k` grid — i.e., the horizon grid
k ∈ {4, 8, 16, 32, 64} maps onto existing extraction positions with a
preregistered exact mapping (G4). No whitespace tokenization anywhere in the
horizon definition. Skip 96/128 (censoring zone; no forecasting relevance —
note 3 of 4 Stage1 winners sat there, which is itself a hint that the old
contrast rewarded long prefixes).

**3.5 Censoring so that neither arm is favored.** Mechanism to neutralize:
hidden@cot_k physically exists only for rows with CoT length ≥ k, while text
can always be computed — if the text arm scored short rows the hidden arm
never sees, the text arm would absorb the easy short refusals, and the
comparison breaks.
- **Primary rule — pair-complete drop-if-shorter:** at horizon k, retain a
  prompt pair iff *both* members have CoT length ≥ k. Both arms score exactly
  this set. Guarantees: identical example sets across arms; exact 50/50
  balance at every k; E2 well-defined on every retained pair; zero-missing
  tolerance enforceable (any absent hidden row for a retained pair is a hard
  stop, G5).
- **Sensitivity rule — row-complete:** all rows with length ≥ k (class
  balance may drift with k; report it). Appendix only.
- **Fixed-cohort curve:** pairs complete at k=64 (max of grid), evaluated at
  all k. This is the only curve where the population is constant across k,
  so it isolates the horizon effect from the population effect; the per-k
  pair-complete estimates maximize power at each k. Preregister both; primary
  inference per k uses pair-complete, fixed-cohort is the consistency check.
- Retention table per source×k: retained pairs, fraction of full set.
  Preregistered floor: skip any source×k cell with < 100 retained pairs
  (HB, at 152 total pairs, may lose high-k cells; WJB's 2019 pairs will not).
- Honest labeling: results at horizon k are conditional on "both trajectories
  reached k tokens" — a stated, design-level conditioning, identical for
  both arms.

**3.6 Capacity vs horizon matching.** Matched information horizon is the
binding requirement; matched dimensionality is neither achievable nor
important. What must be identical across arms: horizon (3.2–3.4), example
sets (3.5), classifier family (L2 logistic), regularization search (same C
grid, inner 5-fold CV on train folds), standardization policy (preregistered
per feature type), and **selection df**: one global choice per arm at the
anchor (pooled validation at k=32) — hidden picks layer L\* (df≈19), surface
picks family F\* (df=4) — then both frozen across all k and sources.
Dimension asymmetries run in both directions (BoW ≫ 4096 sparse; sentence
encoder ≈ dense comparable) and should simply be documented: if hidden wins
despite the surface arm's larger raw feature space, the claim strengthens;
if surface wins, capacity cannot be blamed. Report per-k/per-source
non-frozen winners in an appendix for transparency, decisions only on the
frozen choices.

**3.7 The four fair surface families at horizon k** (all functions of the
visible prefix only; fitted on train folds):
1. Word BoW / TF-IDF over the prefix.
2. Character 3–5-gram TF-IDF over the prefix.
3. Position-indexed token identities (token-id × position one-hots,
   positions 1..k) — captures "refusal opener at position 2"-type structure
   that unordered bags dilute.
4. Frozen off-the-shelf sentence-encoder embedding of the prefix + L2
   logistic — the "any competent frozen text reader" control, and the
   strongest generic-semantics surface family. Legitimate because it is a
   function of emitted text; CPU-viable at these scales. (A frozen *small LM*
   embedding is equally legitimate under the same argument; pick one encoder,
   preregister it, and do not shop among encoders. Using the subject model's
   own hidden states on the prefix is *not* a surface baseline and is
   excluded by definition.)
Prefix-length/ended-by-k: degenerate under 3.5, documented not featurized.

**3.8 Full-trajectory baselines.** Demoted to clearly-marked **hindsight
ceiling reference lines** (`word_bow`/`char_tfidf` full-text, `length_only`),
plus `prompt_only` as the ≈0.500 floor check. They appear on plots as grey
lines, never in delta headlines, never in significance tests against the
hidden arm. This is a reporting rule, not an afterthought — it is the
antidote to the category error that produced the original contrast.

**3.9 Statistical comparisons (all preregistered).**
- **E1 (primary):** paired ΔAUROC(hidden@k − text@k) per source per k;
  cluster bootstrap resampling prompt pairs (2000 reps; existing delta-CI
  machinery and grouping reused verbatim); Holm correction across the 5
  horizons within each source; any positive claim additionally requires
  CI-separated positives at ≥2 *adjacent* horizons with k ≤ 32 on both
  Phase-1 sources (unchanged from prior Section 6).
- **E2 (co-primary, design-native):** within-pair ranking accuracy per arm
  per k (P[score(unsafe member) > score(safe member)]; ties count 0.5,
  preregistered); arms compared per k by **exact McNemar on discordant
  pairs** (pairs are independent prompts, so the test's assumptions hold);
  Holm across k. This is the cleanest single number the paired design
  affords and is immune to class-balance and threshold questions.
- **E3 (secondary):** incremental value — logistic stacker on
  [text_score, hidden_score] vs [text_score] alone; ΔAUROC and Δlog-loss
  (log-loss is the proper scoring rule and the more sensitive detector of
  incremental information). Stacking leakage rule: base-model scores for
  stacker training must be out-of-fold on train (inner 5-fold); test scores
  from full-train-fit base models; stacker never sees test. Report pooled
  and per source, plus the reverse direction (text over hidden) for
  completeness.
- DeLong on the paired ΔAUROC as an analytic cross-check only — its
  independence assumption is violated by the pair structure, so the cluster
  bootstrap is primary.
- Controls: shuffled-label (all deltas must be ≈0), `prompt_only` ≈ 0.500 at
  every k, and the fixed-cohort consistency curve (3.5).
- Thresholded metrics at matched horizon, if reported at all, use the
  Module-T policy (Platt-0.5 per fold) identically in both arms —
  descriptive only, never decision-bearing.

**Scope note.** Phase 1 remains HB + WJB (preserved arrays), per the prior
ruling. If the SR Stage1b early-position array passes G8 hash verification,
SR early-k curves may be added as *appendix-only context*, explicitly
non-decision-bearing; Phase-2 GPU regeneration rules are unchanged.

---

## 4. Preregistered CPU-Only Protocol (Modules T and M, one prereg, one unblinding)

**Gate GT0 (new, blocks both modules):**
(a) verify stored row-level score files exist and hash-verify for all 16
selected configs and all text baselines (if any scores are missing, refit the
exact frozen configs on CPU from frozen features — no re-selection);
(b) document the original thresholding rule verbatim (1.1);
(c) commit the G4 indexing-semantics memo (3.2/3.4).
Existing gates G1–G8 stand unchanged; G6 (human QA) still blocks external
claims but not CPU execution.

**Module T — threshold/calibration reanalysis (descriptive; all 4 sources;
both S1 and S1b):**
- T1 primary: per-LOSO-fold Platt on validation; τ = 0.5 calibrated.
- T2 secondary: per-fold empirical BA-max sweep on validation (tie rule:
  midpoint of maximizing interval).
- T3 transductive (labeled): τ = median of held-out-source unlabeled test
  scores (design prior 0.5).
- T4 diagnostic-only: oracle test-BA ceiling, boxed separately.
- Apply the identical policy to hidden probes, selected surface baselines,
  and `length_only`; report BA/recall/FPR at frozen τ with pair-cluster
  bootstrap CIs; report transfer gaps and the hidden-vs-surface gap
  comparison; plus TPR@FPR≤0.05 (val threshold) as the deployment row;
  reliability/Brier as optional descriptives.
- Readouts M-1/M-2 of Section 2.7. No hypothesis tests, CIs only, no
  decision authority.

**Module M — matched-horizon reanalysis (decision-bearing; Phase 1 =
HB + WJB):**
- Horizon grid k ∈ {4,8,16,32,64} on the existing position grid (G4 map).
- Example sets: pair-complete per k (primary), row-complete (sensitivity),
  fixed-cohort k=64 (consistency); retention table; ≥100-pair floor;
  zero-missing tolerance (hard stop).
- Arms: hidden = L2 logistic on hidden@cot_k at global L\* (pooled val,
  k=32, frozen); surface = best of four preregistered families F\* (pooled
  val, k=32, frozen); identical classifier/CV/standardization protocol.
- Endpoints E1/E2/E3 with bootstrap, Holm, McNemar, adjacent-horizon rule,
  controls — exactly as 3.9.
- Ceilings/floors as 3.8. AUROC(k) curves with CI bands for both arms.
- Continue/kill: **unchanged from prior review Section 6.** Module T output
  has no vote.

**Leakage rules (L1–L7, binding):** L1 no test labels in any selection;
L2 thresholds/calibrators fit on validation of train sources only (T3's
unlabeled-median exception explicitly labeled); L3 vectorizers/encoders and
standardizers fit on train folds only; L4 C by inner CV on train only;
L5 test metrics computed once per preregistered endpoint after the prereg
commit; L6 stacker trained on out-of-fold train scores only; L7 all raw text
stays server-side; packets receive aggregates only (existing sanitization
protocol, unchanged).

**Order of operations:** GT0 → commit prereg (single document covering T+M,
including expected effects 1.3 and the RS-transfer prediction 2.2) → run
Module T and Module M Phase 1 → single unblinding → decision per prior
Section 6 → only then any Phase-2 GPU question arises.

Estimated cost: CPU-hours (linear/logistic fits on ≤ ~5k rows per cell;
sentence-encoder over ≤64-token prefixes is tens of CPU-minutes). Zero GPU.

---

## 5. Literature Analogues and Lessons

Confidence tags: [high] = standard, well-known work I am confident exists as
described; [med] = likely correct but verify title/venue/year before citing;
[verify] = plausibly misremembered — treat as a search pointer, not a
citation. Nothing here was database-verified in this session (see header).
Verify all entries before any manuscript use.

### 5.A Calibration and threshold selection

- **Platt (1999), probabilistic outputs for SVMs** [high]. Problem: scores
  ≠ probabilities; Control: sigmoid fit on held-out data; Lesson: the
  2-parameter fit (slope *and intercept*) is exactly what re-anchors drifted
  score locations (our 1.2).
- **Zadrozny & Elkan (2001/2002), calibration via binning/isotonic
  regression** [high]. Lesson: isotonic is powerful but data-hungry.
- **Niculescu-Mizil & Caruana (2005), predicting good probabilities** [high].
  Lesson: Platt beats isotonic at small calibration sets (order 10²–10³);
  with n_val ≈ 124–264, isotonic is contraindicated — our 2.5.
- **Guo et al. (2017), on calibration of modern neural networks** [high].
  Lesson: calibration and discrimination are orthogonal axes; temperature
  scaling is slope-only. Our case needs the bias term, hence Platt.
- **Youden (1950), J index** [high]. J = 2·BA − 1; BA-max ≡ Youden point.
- **Fawcett (2006), introduction to ROC analysis** [high]. Lesson: threshold
  choices move along an invariant ROC; report operating points as policy,
  not as discrimination.
- **Elkan (2001), foundations of cost-sensitive learning** [high]. Lesson:
  the Bayes threshold is determined by priors/costs; with known 50/50 design
  prior, calibrated-0.5 is optimal for BA — our 1.6.
- **Saerens, Latinne & Decaestecker (2002), adjusting outputs to new a
  priori probabilities** [high]. Problem: prior/label shift at deployment;
  Control: label-free output adjustment on unlabeled target data; Lesson:
  legitimizes the transductive quantile threshold T3 under a known prior.
- **DeLong et al. (1988), comparing correlated ROC curves** [high]. Use as
  cross-check only; pair-clustered data violate its independence assumption.
- **Dietterich (1998), approximate statistical tests for comparing
  classifiers** [high]. Lesson: McNemar on paired correct/incorrect outcomes
  — our E2 arm comparison.
- **Hand (2009), H-measure critique of AUC** [med — verify framing].
  Lesson: AUROC aggregates over operating points with an implicit cost
  distribution; reporting both AUROC and a policy-fixed operating point is
  the defensible middle path.
- **Forman & Scholz (2010), apples-to-apples in cross-validation studies**
  [med]. Lesson: threshold-dependent metrics (F1/BA) are easily biased by
  fold-wise selection details; fix the policy globally, in writing.

### 5.B Leakage and selection discipline

- **Kaufman et al. (2012), leakage in data mining** [high]. Taxonomy of
  target leakage; our L-rules are instances.
- **Varma & Simon (2006), bias in error estimation with CV model selection**
  [high]. Lesson: selection and estimation must be nested — inner-CV for C,
  outer test touched once; also why val-tuned thresholds inherit optimism on
  val but stay honest on test.
- Winner's curse / selective inference in the probe grid: already covered in
  the prior review (§1.4 there); no new citations added here.

### 5.C Probing controls, selectivity, MDL, residualization

- **Hewitt & Liang (2019), control tasks for probes** [high]. Problem: probe
  accuracy conflates representation info with probe capacity; Control:
  selectivity vs a matched control task; Lesson: our matched surface arm at
  the same horizon plays the control-task role; report deltas, not absolute
  probe scores.
- **Voita & Titov (2020), MDL probing** [high]. Lesson: "ease of extraction"
  (codelength) is the right currency for accessibility claims — directly
  supports the restated hypothesis; an online-codelength variant of E3 is a
  legitimate optional appendix metric (not decision-bearing here).
- **Pimentel et al. (2020), information-theoretic probing** [high]. Lesson:
  any deterministic function of the input can only lose information —
  "contains information not in the text" is ill-posed, exactly the prior
  review's §0.3; the defensible claim is accessibility. This is the
  theoretical backbone of the whole redesign.
- **Belinkov (2022), probing classifiers survey** [high]. Consolidates
  pitfalls: baselines, controls, causal overreach.
- **Elazar et al. (2021), amnesic probing** [high]. Problem: decodability ≠
  behavioral use; Control: remove a property and observe behavior; Lesson:
  even a positive matched-horizon result licenses only an accessibility
  claim, not "the model uses it" — that is Stage-2 (interventional)
  territory.
- **Ravfogel et al. (2020), INLP / nullspace projection** [high]. Lesson:
  representation-level residualization exists, but score-level
  residualization (our E3) is the lower-risk instrument for "beyond
  surface" at Stage-1 scale.
- **Conneau et al. (2018), what you can cram into a single vector /
  SentEval probing** [high]. Lesson: they deliberately included surface
  baselines (word content, length) because surface properties drive many
  probing wins — the direct ancestor of our `length_only` ceiling and
  surface families.
- **Tenney et al. (2019), edge probing** [med — details]. Lesson: isolate
  contextual information by comparing against lexical/local baselines;
  analogous to our position-indexed token-identity family.

### 5.D Matched/hindsight baseline discipline (forecasting hygiene)

- Standard forecasting/backtesting principle — evaluate forecasts only on
  information available at forecast time [high, textbook-level; no single
  citation needed]. The original hidden-prefix vs full-text contrast is a
  hindsight-information violation of exactly this rule.
- **Landmarking in survival analysis (van Houwelingen, 2007)** [med —
  concept high, venue verify]. Problem: predicting from time-varying
  covariates without immortal-time bias; Control: fix landmark time k,
  condition on being at-risk at k, predict from information ≤ k; Lesson:
  our pair-complete drop-if-shorter rule is a landmark analysis; "immortal
  time bias" is the epidemiology name for what full-trajectory baselines
  commit.

### 5.E Early / incremental prediction with truncated inputs

- **Early classification of time series (Xing, Pei & Keogh, ~2012)** [med].
  Problem: classify from prefixes; Control: earliness–accuracy tradeoff
  curves; Lesson: report AUROC(k) curves, not a single k.
- **CLEF eRisk shared tasks on early risk detection (Losada & Crestani and
  successors)** [med — task series exists; cite specific overview papers
  only after verification]. Lesson: when the scientific object is early
  detection, *every* baseline must be prefix-limited by protocol, and
  latency/earliness is part of the metric — the institutional version of
  our matched-horizon rule.

### 5.F Safety-probing context (why a linear hidden-state probe was plausible)

- **Zou et al. (2023), representation engineering** [high]. Reads
  safety-relevant concepts linearly from activations — but typically without
  matched-horizon surface controls; our design is the stricter test.
- **Arditi et al. (2024), refusal mediated by a single direction** [high].
  Lesson: linear refusal structure exists, but was established with
  *causal* (ablation/steering) evidence, not probe-vs-baseline deltas —
  consistent with our Stage-2 framing for any causal claim.
- **MacDiarmid et al. (2024), simple probes can catch sleeper agents
  (Anthropic blog)** [high as blog; informal]. Lesson: linear probes on
  internal state can forecast misbehavior in controlled settings; the open
  question our Stage1 addresses is whether they beat matched surface
  readers on *natural* trajectories.

### 5.G Quarantined — do not cite without verification

- "Understanding Jailbreak Success: latent-space dynamics of jailbreaks"
  (Ball et al., ~2024) [verify — recalled; arXiv check was not permitted
  this session].
- "Future Lens: anticipating subsequent tokens from a single hidden state"
  (Pal et al., ~2023, CoNLL?) [verify]. If real, it is the closest analogue
  for "hidden@k forecasts the future trajectory."
- "Probing the probing paradigm: does probing accuracy entail task
  relevance?" (Ravichander et al., ~EACL 2021) [verify].
- "Language modeling teaches you more syntax than translation does" (Zhang &
  Bowman, ~2018) [verify] — untrained/random-encoder baselines for probes.
- Any specific "TPR@low-FPR / partial-AUC methodology" citation (McClish
  1989; Dodd & Pepe 2003) [verify before use; the *practice* is standard].

---

## 6. Do-Not-Do List

Inherited, all still binding: no new GPU sweeps; no per-source argmax
re-selection at n_val≈124–264; no nonlinear probes without a positive,
saturating linear matched-horizon result; no base-model swaps; no headline
hidden-vs-full-text deltas; no test-split contact before the prereg commit;
no external claims before G6/S-to-S/HT gates; no raw content in packets.

New, threshold- and horizon-specific:

1. Do not tune or "sanity-adjust" any threshold on held-out-source labels —
   including informal peeking at test confusion matrices while choosing
   between T1/T2/T3. The policy set is fixed in the prereg; all variants get
   reported.
2. Do not report oracle-threshold BA in any comparison row. Diagnostic box
   only, clearly labeled.
3. Do not use isotonic calibration at n_val ≈ 124–264, and do not use
   temperature scaling at all (no bias term — cannot fix the demonstrated
   location drift).
4. Do not present Module-T gains as probe improvement, and do not let
   Module-T numbers into any continue/kill decision — they are
   threshold-invariant-conclusion cosmetics (1.4).
5. Do not delete the low-FPR operating point from the paper — it is the
   deployment-relevant regime; report it *alongside* the BA-max policy row,
   each labeled with its policy.
6. Do not let the text arm score rows the hidden arm cannot (or vice versa)
   at any horizon — pair-complete sets or nothing; any missing hidden row in
   a retained pair is a hard stop, not a drop.
7. Do not define k in whitespace tokens or characters anywhere in the
   pipeline, and do not leave the cot_k indexing convention undocumented
   (GT0/G4).
8. Do not shop among sentence encoders (or add encoder #2 after seeing
   results) for the surface family — one preregistered encoder, frozen.
9. Do not fit vectorizers, standardizers, calibrators, or the E3 stacker on
   anything but train-fold (or, for calibrators, validation) data — L1–L6.
10. Do not run Module T twice ("just refreshing the numbers") — one
    unblinding, shared with Module M.
11. Do not cite anything from Section 5.G, or the [med] items, in a
    manuscript before independent verification.

---

## 7. Final Recommendation

**`BOTH_CPU_ONLY`.**

- Module M (matched-horizon reanalysis) is the decision-bearing work and
  remains governed by the prior ruling (ONLY AFTER GATES; Phase 1 = HB+WJB;
  Section-6 continue/kill unchanged; GPU only for Phase-2 regeneration if
  Phase 1 passes).
- Module T (threshold/calibration) is approved as a descriptive
  operating-point correction folded into the same preregistration and the
  same single unblinding: it fixes a real fairness bug in the current
  thresholded table (1.5), quantifies recoverable BA headroom (1.3), and
  adds the comparative threshold-transfer-gap finding (0.5) — at CPU-hours
  of cost and zero effect on conclusions (1.4).
- If bandwidth forced a choice (it does not — both are CPU-days), drop
  Module T and keep Module M: `MATCHED_HORIZON_REANALYSIS` is the science;
  `THRESHOLD_REANALYSIS_ONLY` would polish the reporting of an ill-posed
  contrast, and `NO_ACTION` would leave the paper's Stage1 section resting
  on that ill-posed contrast.
- Hard conditions: GT0 before anything; prereg commit before any test-metric
  recomputation; leakage rules L1–L7; do-not-do list Section 6; human QA
  (G6) still gates all external claims.

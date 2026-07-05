# Fable Pro Methods Review: Stage1 Probe Redesign

Date: 2026-07-05
Reviewer mode: Fable-5 pro/high-rigor, strict senior ML methods review.

Reviewed sanitized packet `stage1_fable_review_retune12288_b20_260705/`:

- `README.md`
- `fable_decision_response_260705.md`
- `probe_redesign_addendum_260705.md`
- `delta_ci/hidden_surface_delta_ci_summary.tsv` (all 16 rows)
- `val_fixed/val_fixed_probe_report.tsv` (all 16 rows)
- `row_audit/stage1_prediction_row_audit_summary.json` (programmatic audit of
  all 32 groups and 4464 file records; aggregate counts only)
- `docs/stage1_post_hb_retune12288_b20_gap_audit_260705.md`
- `text_baselines/{harmbench_standard,reasoningshield,strongreject_full,wildjailbreak_vanilla_harmful}/summary.tsv`

The packet excludes raw prompts, raw CoTs, completions, and row-level
prediction JSONL. Nothing in this review required or used raw content.

---

## 0. Verdict Summary

1. The prior decision stands: **Stage1 as run is a robust negative/control
   result for the current linear hidden-probe design.** No GPU rerun of the
   current design is required or permitted.
2. The dominant methodological failure is **not** the linear probe. It is the
   **evaluation contrast**: a prefix-limited, single-position hidden snapshot
   was compared against full-trajectory hindsight text features. That contrast
   cannot answer the stated scientific question, so its negative outcome says
   little about latent trajectory information.
3. The scientific question itself must be restated. Hidden states at CoT token
   k are a deterministic function of (prompt, emitted tokens ≤ k). "Contains
   information not reducible to surface text" is information-theoretically
   ill-posed. The defensible claim is about **accessibility**: does the
   model's internal state make trajectory-outcome information *cheaply
   (linearly, sample-efficiently) accessible* that matched-capacity surface
   classifiers cannot extract from the same emitted tokens at the same
   horizon?
4. The correct redesign is therefore primarily an **evaluation redesign on the
   frozen data — matched-horizon forecasting with surface residualization —
   and it is CPU-only.** The only GPU spend is regeneration of the three lost
   hidden arrays (RS Stage1, RS Stage1b, SR Stage1), and only if the CPU
   gates pass on HB + WJB first.
5. Final decision label: **ONLY AFTER GATES** (Section 8).

---

## 1. Independent Verification and New Observations

I re-derived the packet's claims from the artifacts and found five additional
facts that materially shape the redesign.

### 1.1 Confirmed from the packet

- All 16 hidden-minus-surface AUROC deltas negative; deltas range −0.0927
  (WJB Stage1 linear) to −0.2374 (RS Stage1b linear); all 95% CIs bounded
  away from zero; 2000 valid bootstrap reps each; 0–4 dropped records per
  alignment (negligible, but should be zero next time).
- Hidden test AUROC range 0.679–0.840; validation-selected surface test AUROC
  0.917–0.965 (`word_bow` HB/RS/SR, `char_tfidf` WJB).
- Row audit: 4464 files, 133 mismatches, all in Stage1 linear `cot_96`/
  `cot_128`; per-file deficit 1–4 rows; zero duplicate IDs, zero extra IDs.
  Stage1b and multilayer groups fully clean. This supports the
  high-offset-coverage-gap interpretation, not extractor-level row drops.
- `prompt_only_tfidf` = 0.500 AUROC exactly on every split of every source:
  prompts are non-informative marginally, confirming the paired design is
  intact (each prompt contributes both labels). This is a well-constructed
  control and is load-bearing for the redesign below.

### 1.2 New observation A — hidden probes lose to `length_only` everywhere

Test AUROC, `length_only` vs selected Stage1 linear hidden probe:

| Source | length_only | hidden (selected) |
|---|---:|---:|
| HB  | 0.9278 | 0.8396 |
| RS  | 0.7960 | 0.7025 |
| SR  | 0.8629 | 0.8147 |
| WJB | 0.8789 | 0.8252 |

A single hindsight scalar — total trajectory length — beats the
4096-dimensional hidden-state probe on all four sources, and beats the hidden
probe on validation on 3/4 sources. Interpretation: a large fraction of the
"surface superiority" is **outcome-correlated hindsight structure** (refusals
are short; compliance is long; refusal lexicon appears somewhere in the full
text). None of that is observable at CoT token k. The current contrast is
therefore substantially "foresight vs hindsight," not "internal vs surface."

### 1.3 New observation B — the early-position probe was already run, and it is uninformative under this contrast

Stage1b is a 171-file grid = 9 early positions × 19 layers (selected
positions cot_4/cot_7; Stage1 is 361 = 19 × 19). So the prior review's
suggestion "test early reasoning positions" has effectively been executed —
and it lost to *full-text* baselines, as it arithmetically almost must.
Conclusion: adding more positions is dead as a fix. The contrast, not the
position set, is the problem.

### 1.4 New observation C — winner's-curse selection conditions are severe

- Selection = argmax val AUROC over 361 (Stage1) / 171 (Stage1b) configs with
  n_val = 123–264. The SE of AUROC at n≈124 is ~0.04; the top of the grid is
  noise-dominated.
- Selected configs are unstable across sources: Stage1 picks cot_128/L12,
  cot_9/L12, cot_96/L17, cot_128/L14; Stage1b picks cot_4/L20, cot_7/L32,
  cot_4/L20, cot_4/L18. A real, linearly accessible safety direction would
  not move this much.
- Val→test shrinkage is large where expected (RS 0.831→0.703; SR Stage1b
  0.858→0.746) and even positive on WJB (0.788→0.825), consistent with
  selection noise + LOSO source shift, not a stable optimum.
- Note the selection-df asymmetry: hidden selects over 361 configs, surface
  over ~6. This asymmetry *favors* hidden on validation, and hidden still
  loses on test — which makes the negative result conservative and safe to
  report, but the same machinery must NOT be reused for a positive claim
  without df control (Section 4).

### 1.5 New observation D — missing rows are label-skewed exactly as short-refusal censoring predicts

Across the 133 mismatch files: 133 missing `safe` rows vs 76 missing `unsafe`
rows, all at cot_96/cot_128, 1–4 rows per file. Safe/refusal CoTs are shorter,
so high offsets censor the safe class first. At this magnitude (≤4 of 4038
rows) it is immaterial to the reported metrics, but it exposes that
`cot_96`/`cot_128` semantics interact with trajectory length — a confound the
redesign must handle by construction (drop-if-shorter, applied identically to
hidden and text arms; zero-missing tolerance).

### 1.6 New observation E — Stage1/Stage1b multilayer duplication

Stage1 and Stage1b multilayer rows for HB and SR are identical to 15 digits
(val 0.798898/test 0.833254; val 0.87329/test 0.73639; identical deltas
−0.1322065443213296 and −0.2006998657613158), with a 1e-5 discrepancy in SR
test_auprc between the two reporting pipelines (0.716396 vs 0.716402). This is
consistent with overlapping cot_4 grids feeding identical arrays, not
necessarily a bug — but it must be confirmed intended, and the 1e-5
cross-pipeline nondeterminism explained, before any redesigned run (Gate G2).

Also noteworthy: 19-layer concat multilayer never beats the best single layer
(e.g., HB 0.833 vs 0.840). Added linear capacity over layers is not the
binding constraint — evidence against "just go nonlinear/bigger" as the fix.

---

## 2. Answers to the Nine Questions

### Q1. Most likely methodological reason the hidden probe lost

Ranked attribution with evidence:

1. **Evaluation-contrast mis-specification (~60%).** Hidden arm: one hidden
   vector at CoT token offset k ≤ 128 (often k ≤ 9), i.e., a function of the
   prompt + at most the first k emitted CoT tokens. Surface arm: BoW/char
   features over the *entire* trajectory, including everything that made the
   judge assign the label. The surface arm is a hindsight re-reading of the
   label's evidence; the hidden arm is a forecast from a prefix. §1.2
   (length_only dominance) shows the hindsight component alone explains most
   of the surface advantage. Under this contrast the negative result was
   predictable ex ante and is nearly uninformative about latent trajectory
   information.
2. **Selection protocol / winner's curse (~20%).** §1.4. Argmax over 361
   configs on n_val≈124–264 guarantees shrinkage and unstable choices. This
   depressed hidden test AUROC by roughly 0.05–0.13 where measurable.
3. **LOSO transfer (~10%).** Validation is in-distribution (train sources),
   test is the held-out source; part of the val→test drop is source shift on
   the probe direction. Surface lexical features transfer unusually well for
   safety text; hidden directions may transfer less well — a real but
   secondary effect here.
4. **Classifier class / capacity (~5%).** Linear probes recover most of what
   probing literature recovers; 19-layer concat did not help (§1.6); no
   stated nonlinear-separability hypothesis exists. Not the binding
   constraint.
5. **Data quality (~5%).** Row audit is clean apart from the censoring edge
   (§1.5); paired design verified by prompt_only = 0.5; surface baselines
   prove strong signal exists in the frozen data. Human QA remains open as a
   formal blocker but nothing suggests label corruption at a rate that would
   explain a −0.09 to −0.24 delta.

Label definition caveat: the packet does not state how labels were produced.
If (as I infer) labels come from a judge reading the completion text, the
hindsight asymmetry in (1) is exact. This inference must be documented as
part of Gate G1.

### Q2. What redesign most directly tests latent safety information rather than surface artifacts?

First restate the hypothesis so it is well-posed (§0.3): hidden@k is a
deterministic function of (prompt, tokens ≤ k), so no existence claim is
available. The testable claim is **accessibility/sample-efficiency**:

> At matched information horizon k, a simple probe on the model's internal
> state extracts more trajectory-outcome information than matched-capacity
> classifiers reading the same emitted prefix.

The paired design makes this unusually clean: within a pair, the prompt is
identical and prefill hidden states are identical; all discriminative signal
at horizon k comes from the k sampled tokens and the model's processing of
them. `prompt_only = 0.500` verifies the control empirically.

Operationalization — **matched-horizon forecasting with surface
residualization**:

- Hidden arm: existing hidden features at cot_k (frozen arrays; no new
  extraction for HB/WJB).
- Surface arm: preregistered text featurizations of prompt + first k CoT
  tokens only (same tokenizer, same truncation, same drop rules).
- Primary contrast: paired ΔAUROC(hidden@k − text@k) at each k, group
  bootstrap by pair (existing delta-CI machinery reused verbatim).
- Secondary contrast: residual/incremental test — logistic on
  [surface_score, hidden_score] vs [surface_score]; report ΔAUROC and
  Δlog-loss. This directly answers "not reducible to surface" in its
  practical form.
- Report full-text `word_bow`/`char_tfidf` and `length_only` as **hindsight
  ceiling reference lines**, not as competitors.

### Q3. Which probe type should be next?

Ranked (see Section 3 for the consolidated list):

1. **Matched-horizon evaluation redesign + within-pair ranking** — not a new
   probe at all; a corrected contrast over existing features. CPU-only.
   Highest scientific value per unit cost; everything else is conditional on
   it.
2. **Residualized/incremental probe** (hidden ⊕ surface score vs surface
   score) — the direct formalization of "beyond surface." CPU-only.
3. **Paired difference probe** — train/evaluate on within-pair contrasts
   (score(unsafe member) > score(safe member)); removes all prompt and style
   variance; the paired AUROC analog is the cleanest single number for this
   design. CPU-only.
4. **Trajectory probe** — preregistered functionals over the existing
   position sequence (mean-pool over positions ≤ k; drift Δ(proj@k −
   proj@4) along a train-fold class-mean direction). Tests "movement toward
   the unsafe region," which a single snapshot cannot. CPU-only for HB/WJB;
   needs regeneration for RS/SR.
5. **Early-position probe** — already effectively run (Stage1b, §1.3). Only
   meaningful inside the matched-horizon contrast; not as a standalone fix.
6. **Nonlinear probe (small MLP)** — only if the matched-horizon linear
   result is positive but appears to saturate, and only with a stated
   separability rationale. Otherwise it is fishing (Q5).
7. **Causal/activation intervention** — the strongest possible evidence
   (does steering the putative direction change the trajectory outcome?),
   but it is a Stage2-scale program with its own controls, not a Stage1
   patch. Do not bolt it onto this decision.

### Q4. Minimum GPU experiment worth running

The minimum credible redesigned experiment is **mostly not a GPU
experiment**. Full specification in Section 4. Summary:

- **Phase 0 (CPU, gates):** oracle-grid audit; duplication check (§1.6);
  truncated-text baseline curves; length-residualization audit; human QA
  completion; preregistration.
- **Phase 1 (CPU, decision phase):** matched-horizon paired comparison on
  HB + WJB using preserved hidden arrays; global-layer selection; within-pair
  ranking + ΔAUROC + residual Δlog-loss endpoints.
- **Phase 2 (GPU, only if Phase 1 passes):** regenerate the three lost
  arrays (RS S1, RS S1b, SR S1) from frozen splits/configs to replicate on
  the remaining sources; optional preregistered trajectory pooling. No new
  grid search. No new positions/layers. Bounded, one-off extraction.

Expected failure modes and their readings:

- Text@k ≈ hidden@k at all k (most likely): first sampled tokens carry the
  outcome signal in surface form (e.g., refusal openers). Clean kill; the
  negative result generalizes from "lost to hindsight text" to "no cheap
  internal advantage at any horizon."
- Hidden@k wins only at k where coverage censoring exists: artifact; enforce
  drop-if-shorter symmetry (both arms score exactly the same example set at
  each k).
- Hidden@k wins broadly but only vs weak text featurizations: guard with the
  strongest preregistered surface family (word BoW, char 3–5 tfidf,
  position-indexed token identities, frozen off-the-shelf sentence-encoder
  probe of the prefix — all are functions of visible text and thus
  legitimately "surface").
- Hidden@k wins at k=4 only: interesting but fragile; require ≥2 adjacent
  horizons (kill criteria, Section 6).

### Q5. Scientifically motivated vs hyperparameter fishing

Scientifically motivated (each targets an identified failure mode):

- Matched-horizon contrast — fixes the hindsight/foresight category error
  (§1.2, Q1.1).
- Residual/incremental evaluation — operationalizes "not reducible to
  surface."
- Within-pair ranking endpoint — exploits the verified paired design;
  removes prompt variance.
- Global-layer, low-df selection protocol — fixes the winner's curse (§1.4).
- Drop-if-shorter horizon semantics, symmetric across arms — fixes the
  censoring confound (§1.5).
- Trajectory functionals, preregistered forms only — tests a hypothesis
  (drift/commitment) that a snapshot cannot.

Hyperparameter fishing (prohibited):

- Any new position × layer × regularization sweep on GPU.
- Per-source argmax with n_val ≈ 124–264, or any selection df expansion
  without preregistration.
- Swapping in MLP/attention probes "to see if it helps."
- Swapping base models to rescue Stage1.
- Post hoc narratives about why cot_9 or cot_96 was "the right" position.
- Reporting best-of-K redesign variants without correction; any test-set
  contact before endpoints are frozen.

### Q6. How to compare redesigned hidden probe vs surface baselines

- **Primary:** paired ΔAUROC(hidden@k − text@k) at matched horizon k, group
  bootstrap by (match_family, pair_id, id) — reuse the existing delta-CI
  code; and **within-pair ranking accuracy** (P[score(unsafe) >
  score(safe)] within prompt pair), which is the design-native paired AUROC.
- **Secondary:** residual ΔAUROC and Δlog-loss of [surface_score +
  hidden_score] over [surface_score] (log-loss is the proper scoring rule
  and more sensitive than AUROC for incremental information).
- **Curves:** AUROC(k) for both arms with CI bands ("commitment-point"
  curves) — the scientifically interpretable output.
- **Controls:** length-so-far and has-ended-by-k indicators available to the
  surface arm; full-text and length_only reported as hindsight ceilings
  only; adversarial surface control = the strongest of the four
  preregistered text families at each k, selected on validation with the
  same low-df protocol as the hidden arm.
- **Do not** headline unmatched hidden-vs-full-text deltas ever again; keep
  them as context lines.
- Multiplicity: Holm correction across the horizon grid within each source;
  decision rule additionally requires adjacent-horizon consistency
  (Section 6).

### Q7. Non-GPU gates before any GPU spend

Listed in Section 5 (G1–G8). All are CPU/human-only. GPU regeneration is
permitted only when all eight pass.

### Q8. What result continues Stage1; what kills it

Section 6. In one line: continue only if hidden beats matched surface at
early horizons with CI-separated paired deltas on both available sources (or
shows CI-positive incremental information pooled); kill on flat curves, and
the kill is then comprehensive and publishable.

### Q9. Should we skip redesign and move to Stage2/Stage3?

Not yet — for one specific reason: the matched-horizon reanalysis is (a) the
only version of Stage1 that addresses the actual scientific question, (b)
nearly free (CPU on frozen data, existing machinery), and (c) required to
make even the negative result well-posed. Abandoning Stage1 now would leave
the paper's Stage1 section resting on an ill-posed contrast.

If Phase 1 fails its gates, then move on, with this defensible record:

- "On natural divergent pairs from DeepSeek-R1-Distill-Llama-8B, internal
  prefix states offered no measurable forecasting advantage over
  matched-capacity surface prefixes at any tested horizon (k = 4…64), while
  full-trajectory hindsight baselines remain far stronger than both."
- That statement neither blocks nor licenses Stage2/Stage3. Interventional
  work (steering/causal role of internal directions) tests a *different*
  claim than probing accessibility and must be justified by its own
  proposal, not by Stage1's failure. What would make the move defensible is
  exactly the failed Phase 1 gate plus a Stage2 preregistration that does
  not depend on Stage1's probe being good.

---

## 3. Ranked Probe Redesigns (consolidated)

1. Matched-horizon forecasting contrast on frozen data (eval redesign;
   CPU; decisive).
2. Surface-residualized incremental-information probe (CPU).
3. Within-pair difference/ranking probe (CPU).
4. Trajectory functionals over existing position grid — mean-pool ≤ k and
   drift Δproj(k, 4), preregistered forms only (CPU for HB/WJB; GPU only for
   RS/SR regeneration).
5. Global-layer low-df selection protocol (embedded in 1–4).
6. Nonlinear probe — conditional on 1–4 positive and saturating; requires a
   stated separability hypothesis.
7. Causal intervention diagnostics — Stage2 material; out of scope for
   Stage1 rescue.

## 4. Minimal Runnable Experiment Plan ("M1: matched-horizon reanalysis")

Data inputs (all frozen, all existing):

- Frozen LOSO splits: `loso_freeze_fixed_budget_samples_000_099` (unchanged;
  no new splits).
- Preserved hidden arrays: HB Stage1/Stage1b, WJB Stage1/Stage1b (+ R2
  archives; verify hashes, Gate G8).
- Stored trajectory text for truncated surface features (server-side
  featurization only; aggregate outputs only, per content-safety protocol —
  no raw text leaves the server or enters review packets).

Design:

- Horizon grid: k ∈ {4, 8, 16, 32, 64}, mapped onto the existing position
  grid (preregister the exact mapping before unblinding; skip 96/128 — the
  censoring zone with no forecasting relevance).
- Example set at horizon k: rows with CoT length ≥ k, identical for both
  arms (drop-if-shorter, symmetric; zero-missing tolerance).
- Hidden arm: linear (L2 logistic) probe on hidden@cot_k at one **global
  layer L\*** selected once by pooled validation AUROC across available
  sources at a single preregistered horizon (k = 32), then frozen for all
  horizons and sources. Regularization C by inner 5-fold CV on train folds
  only. Features standardized per train fold.
- Surface arm (four preregistered featurizations of prompt + first k CoT
  tokens, same tokenizer): word BoW; char 3–5 TF-IDF; position-indexed
  token identities; frozen off-the-shelf sentence-embedding + logistic.
  Surface score = validation-selected among these four (selection df ≈
  hidden arm's df; symmetric protocol).
- Endpoints (preregistered):
  - E1 primary: paired ΔAUROC(hidden@k − text@k) per source per k, group
    bootstrap (2000 reps), Holm-corrected over k.
  - E2 primary: within-pair ranking accuracy per arm per k.
  - E3 secondary: residual ΔAUROC and Δlog-loss of [surface + hidden] over
    [surface], pooled and per source.
  - Reference lines: full-text word_bow / char_tfidf, length_only
    (hindsight ceilings); prompt_only (must stay ≈ 0.500).
- Controls: shuffled-label negative control (must give Δ ≈ 0); length-so-far
  and ended-by-k indicators granted to the surface arm; identical example
  sets across arms at each k.
- Power: WJB is the primary source (2019 test pairs; within-pair SE ≈
  0.011); HB is the replication source (152 pairs; SE ≈ 0.04). RS/SR join
  only in Phase 2.
- Runtime: minutes–hours on server CPU. No GPU.

Phases and their gate logic:

- **Phase 0 — gates G1–G8 (Section 5).** No analysis unblinding before
  preregistration (G7) is committed.
- **Phase 1 — run M1 on HB + WJB.** Decision against Section 6 criteria.
- **Phase 2 — GPU, only if Phase 1 passes:** regenerate RS S1/S1b, SR S1
  hidden arrays from frozen splits/configs (bounded one-off extraction; no
  new search); rerun M1 on RS/SR as confirmation; optionally add the two
  preregistered trajectory functionals. Any Phase 2 positive claim requires
  all four sources reported, including failures.

Expected outcomes and their interpretations are as in Q4.

## 5. Non-GPU Gates (all must pass before any GPU)

- **G1 — Label provenance memo.** Document exactly how safe/unsafe labels
  were produced (judge model/rubric/version). Required to formalize the
  hindsight argument and for the paper's methods section.
- **G2 — Pipeline consistency check.** Explain §1.6: confirm Stage1/Stage1b
  multilayer duplication is intended grid overlap; reconcile the 1e-5
  cross-pipeline auprc discrepancy; assert same-array provenance.
- **G3 — Oracle-grid audit.** From existing `summary_grid.tsv` files: max
  *test* AUROC over all configs per run (diagnostic only, clearly labeled as
  oracle). Purpose: bound what any selection protocol could have achieved;
  quantify the winner's-curse component of Q1.
- **G4 — Truncated-surface feasibility audit.** Verify stored text supports
  exact model-tokenizer truncation at each k; produce per-k example counts
  and class balance after drop-if-shorter (aggregate counts only).
- **G5 — Coverage audit, stricter tolerance.** Zero missing/extra/duplicate
  rows at every position in the k-grid for both arms; explicit written
  semantics for short CoTs (drop-if-shorter). The current 1–4-row
  high-offset tolerance is not acceptable for the redesigned run.
- **G6 — Human QA annotation complete** (existing blocker), plus the QA
  summary. If pair-label disagreement exceeds ~15%, stop: fix data before
  any method work; neither negative nor positive results are interpretable.
- **G7 — Preregistration doc committed** (hypothesis restated as
  accessibility claim; horizon grid; L\* selection rule; four surface
  families; endpoints E1–E3; Holm correction; continue/kill thresholds of
  Section 6) — before unblinding any Phase 1 test metric.
- **G8 — Artifact integrity.** R2/local hash verification for the preserved
  HB/WJB hidden arrays; confirm regenerability of RS/SR from frozen configs
  (dry-run config diff, no GPU execution).

S-to-S safe-prompt diagnostics and HT quarantine/external testing remain
formal blockers for any *claims*, unchanged; they are not blockers for
Phase 0/1 execution.

## 6. Continue / Kill Criteria

Continue Stage1 (proceed to Phase 2 GPU) iff, on Phase 1 test splits:

- E1: hidden@k − text@k > 0 with 95% CI excluding 0 (Holm-corrected) at ≥ 2
  adjacent horizons with k ≤ 32, on **both** HB and WJB; **or**
- E3: pooled residual ΔAUROC ≥ +0.02 with CI > 0 (equivalently CI-positive
  Δlog-loss) with per-source deltas non-negative;
- and shuffled-label control ≈ 0, and results survive the length-so-far
  control.

Kill Stage1 probing (publish comprehensive negative; no Phase 2; no further
Stage1 GPU ever) if:

- E1 CIs include 0 or are negative at all k ≤ 32 on both sources; or
- any positive delta appears only at horizons/subsets affected by censoring
  or only under one surface family (fragility); or
- G6 human QA fails (data-first stop, overriding everything).

If killed, the reportable claim upgrade is: the negative result now covers
matched horizons, so "no cheap internal forecasting advantage at any tested
horizon" replaces the current, weaker "lost to hindsight text" — a stronger,
well-posed control result that is worth having on record.

Claims allowed if Phase 1+2 pass: "In natural divergent pairs (same prompt,
two sampled trajectories), a linear probe on internal state at CoT token k
predicts the eventual safety outcome better than matched-capacity surface
classifiers reading the same k emitted tokens (CI-separated paired deltas at
k ≤ 32 on N sources)." Not allowed, even then: any existence claim of
"information not in the text" (§0.3); cross-model generality; deployment
claims before S-to-S and HT gates.

## 7. Do-Not-Do List

- Do not run any new GPU position × layer × classifier sweep.
- Do not re-select per-source argmax configs on n_val ≈ 124–264.
- Do not introduce nonlinear probes without a positive, saturating linear
  matched-horizon result and a written separability rationale.
- Do not switch base models to rescue Stage1.
- Do not headline hidden-vs-full-text deltas or compare against
  length-informed baselines at forecasting horizons.
- Do not touch test splits before G7 preregistration is committed.
- Do not treat the delta-CI machinery as symmetric evidence for positive
  claims while selection df remain asymmetric.
- Do not make any external claim before human QA, S-to-S diagnostics, and HT
  quarantine complete.
- Do not include raw prompts/CoTs/completions in any future review packet;
  aggregate-only outputs (this packet's sanitization protocol is correct —
  keep it).

## 8. Final Decision

**ONLY AFTER GATES.**

Precisely: the evaluation redesign (M1, Phases 0–1) is warranted and
approved as CPU-only work on frozen data — the current contrast is ill-posed
for the stated question, and fixing the contrast is science, not fishing.
Any GPU spend (Phase 2 regeneration of RS/SR arrays, trajectory variants) is
conditional on gates G1–G8 and the Phase 1 continue criteria of Section 6.
The existing negative/control result for the current design stands and
should be reported regardless of the Phase 1 outcome.

# Stage1 Auto-Improve Review Loop

Started: 2026-07-05

Objective: repeatedly send Stage1 results to Fable-5, implement reviewed
code/experiment changes, run them, and re-review results until Stage1's fair
equal-horizon result improves or the loop reaches an honest pivot/stop point.

## Round 1

### Packet

- tmp commit: `60b28e3`
- packet: `/Users/baby/Documents/SafeChain/tmp/stage1_auto_improve_loop_260705/round1_results_packet`
- contents: aggregate result TSV/JSON/MD only; no raw prompts, raw CoTs,
  hidden activations, or generated pair files.

### Status

Fable-5 result review returned.

### Assessment (Summary)

- Verdict: continue for exactly one tightly-scoped iteration, with diagnostics
  folded in and a pre-declared pivot trigger.
- Primary next action: CPU-only cumulative score-level pooling over hidden
  position scores, plus k=16 diagnostics and lead-time matrix.
- Follow-up if A1 looks promising: feature-level cumulative pooling probe using
  saved hidden arrays, with a hard success/pivot gate.
- Do not keep hill-climbing arbitrary probe variants.

### Reviewer Raw Response

<details>
<summary>Click to expand full Fable-5 response</summary>

```text
Scratch file `_packet_scan.py` couldn't be removed (sandbox path quirk) — it's inert, delete at will. Review complete; full response below.

---

# Fable-5 Review — Stage1 Auto-Improve Loop, Round 1

**Packet:** `round1_results_packet` @ tmp `60b28e3` (verified: 8 files, matches README manifest). Code `1d30c40`, prior verdict `OK_TO_RUN`.

## Verdict: **CONTINUE — one tightly-scoped iteration (diagnostics folded in), with a pre-declared pivot trigger**

The equal-horizon result is currently negative for k≥16, but I found one genuine methodological asymmetry that *understates* the hidden arm, plus one unexplained anomaly. Both must be resolved before any stop/pivot decision is defensible. If the fix below doesn't move k∈{8,16,32} to parity, pivot per §6 — do not keep hill-climbing probe variants.

## 1. What the data actually says (my independent reading)

**The comparison is information-asymmetric against hidden.** The surface arm at horizon k is char-tfidf over the **entire prefix** (prompt + all k tokens, cumulative). The hidden arm is a linear readout of a **single token position snapshot** (`cot_k`, layer 28). Text gets k tokens' worth of evidence; hidden gets one residual-stream vector and must hope position k happens to summarize the trajectory. That is not an equal-horizon comparison of *representations vs. text* — it's cumulative-text vs. Markovian-snapshot. Standard probing practice for sequence-level properties is pooling over positions (cf. Conneau et al. 2018 in your own lit notes).

**The k=16 anomaly is the smoking gun.** Hidden test AUROC across k (pooled): 0.789 → 0.752 → **0.641** → 0.710 → 0.698. Information available to the model is monotone non-decreasing in k, and censoring is zero at all k (constant 152/2019 test pairs — your MD's censoring caveat is actually moot for this run; the population never changes). A dip of −0.15 at k=16, replicated in both sources (0.690, 0.637), then *recovering* at k=32, means the single-position snapshot at cot_16 is either (a) landing in a locally-dominated token state (mid-sentence generic reasoning), or (b) an extraction/indexing bug. Either way it's an artifact of single-position readout, not evidence about representation content. Meanwhile surface rises monotonically (0.731→0.848) exactly as verbalized refusal/compliance markers accumulate.

**The k=4 result is real and interesting.** Pooled delta +0.058, CI [0.049, 0.069], Holm p≈0, n=2171 pairs; pair-rank delta +0.054 similarly tight. Since pairs share prompts (n_pos=n_neg everywhere; prompt features cancel within-pair), this is genuinely trajectory-borne anticipatory signal: at 4 generated tokens, hidden state predicts the eventual outcome about as well as the *text* does at k=16–32 (hidden@4 = 0.789 vs text@16 = 0.799, text@32 = 0.806). That is a quantifiable **~4× token lead**, and it survives even if k≥16 stays negative.

**Context from the threshold file:** full-trajectory `surface_selected` is 0.92–0.97 AUROC and even `length_only` is 0.80–0.93 across the four original LOSO runs. Module T headroom is exhausted (Platt 0.710 vs oracle 0.721) — threshold work is **done**; adopt `platt_0p5`, freeze it, never touch it again.

## 2. Ranked action items

### A1 (CPU-only, do first) — Cumulative score-level pooling + k=16 diagnostics + lead-time matrix
Equal-horizon hidden score at k := **unweighted mean of val-z-scored per-position hidden scores over positions j ≤ k** (k=16 arm pools scores from {4,8,16}; k=4 arm unchanged). Computable entirely from the already-exported `matched_horizon/predictions/` val/test scores — no arrays, no GPU, no new probe fits.
- **Why it improves the real result:** recovers evidence the position-16/64 snapshots lose; equalizes the cumulative-access interface, which the surface arm already enjoys. It cannot help k=4 (single position), so the existing positive result is untouched and comparable.
- **Leakage risk:** low. Zero fitted parameters if the mean is unweighted. Val z-stats are mild val-reuse (val already used for layer/family selection) — disclose. **Pre-register "unweighted mean of val-z-scored scores" as the only primary rule before running**; max-pool / val-weighted variants are exploratory appendix only.
- **In the same script, three diagnostics:** (i) per-position score histograms + val/test AUROC per position to characterize the k=16 dip; (ii) assertion that stored position metadata for the cot_16 arrays equals 16 (off-by-one check); (iii) full hidden@k vs text@k′ cross-grid ("lead-time matrix") — zero new fitting, makes the token-lead claim concrete.
- **Minimal code:** one new ~150-line reanalysis script beside the existing one; touches nothing upstream.
- **Command/cost:** `python -m ...cpu_reanalysis_score_pooling --pred-dir runs/.../matched_horizon/predictions --rule zmean --b 500` — CPU minutes, $0. (Vectorize the bootstrap with numpy this time; the pure-Python 2000-bootstrap stall was self-inflicted.)
- **Stop criterion:** pooled hidden AUROC becomes monotone-ish in k (dip gone) AND delta at k=8 CI excludes negative territory. If the dip survives pooling → treat as suspected extraction bug, halt interpretation, escalate to A2's index audit before anything else.

### A2 (GPU-light: pod CPU on saved hidden arrays, confirmatory primary) — Feature-level cumulative pooling probe
Retrain per-k logistic probes on **mean-pooled activations over snapshot positions j ≤ k**, layer fixed at 28, same splits/protocol/Holm family. This is the pre-registered *primary* rerun if A1 shows signal; score-pooling is its cheap preview.
- **Why:** feature-level pooling lets the probe weight the combined evidence properly rather than averaging calibrated scores; typically strictly ≥ score pooling.
- **Leakage risk:** low if layer stays 28 and family stays char_tfidf (no reselection). Optionally rerun the anchor-k=32 val selection with pooled features as a *secondary* protocol-consistent variant — report both, primary is fixed-28.
- **Minimal code:** add a `--hidden-pooling cumulative_mean` branch where per-k features are loaded (~30 lines); everything downstream unchanged.
- **Command/cost:** rerun of the matched-horizon module on the pod; sklearn on ≤3.5k×4096 rows per k — pod CPU, minutes, no GPU time. R2 upload as usual.
- **Stop criterion (this is the loop's decision gate):** **success** = deltas at k∈{8,16,32} no longer significantly negative (Holm CI covers 0) with k=4 still positive → Stage1's fair claim becomes "hidden ≥ matched text through k≤32, with a significant 4-token-lead advantage." **Failure** = any of k∈{16,32} still ≤ −0.05 with CI excluding 0 → invoke §6 pivot. No third probe-variant round.

### A3 (GPU-light, bundled with A2) — Rerun residual/E3 with the pooled hidden arm
Currently residual gains are ≈0 for k≥8. If pooled hidden adds significant residual AUROC over char_tfidf at k=8–32, that's the "hidden carries non-surface information" secondary evidence; if it still adds nothing, that independently supports the pivot. Free once A2 runs.

### A4 (GPU-heavy, **conditional** on A2 partial success) — All-position re-extraction
If pooling helps but is limited by having only 5 snapshots, teacher-forced replay of the saved generations to extract **every position 1..64** (layer 28 only, fp16) for the two matched-horizon sources: ~5.8k sequences × ≤(prompt+64) tokens ≈ **≲1 GPU-hour on the A6000, a few GB**. Also export train-set/OOF hidden scores to convert E3 into a proper OOF stacker (removes the packet's own stated limitation). Do not run this before A2 justifies it.

### Not recommended
- More threshold/calibration work (headroom 0.011 to oracle — done).
- Nonlinear/MLP probes or per-k layer selection — capacity confound (Voita & Titov MDL objection), high cherry-picking surface, and they don't fix the interface asymmetry that's actually hurting you.
- Adding `sentence_encoder` surface family this round — it can only strengthen the *opponent*; scientifically it belongs in the final honest baseline set, but it's not an improvement step (flag it as a pre-publication TODO so it doesn't look like it was dodged).

## 3. Single best next change

**A1, immediately, then A2 as the pre-registered confirmatory primary.** One script, CPU-only, uses only files already exported, resolves the k=16 anomaly, previews the pooling effect, and produces the lead-time matrix that de-risks the pivot. Nothing else should run before it.

## 4. Required code-review checks before running

1. Assertion: pooling set is exactly `{j ∈ k_grid : j ≤ k}` — no future positions (the one bug class that would silently invalidate equal-horizon).
2. Assertion: identical retained pair IDs across hidden/surface arms per k (currently zero-drop; keep it enforced, not assumed).
3. z-normalization statistics computed from val scores only; grep-verify no test-split statistics anywhere in the pooling path.
4. Layer 28 / char_tfidf hard-coded for the primary; any reselection lives behind an explicitly-labeled `--secondary` flag and lands in separate output rows.
5. Pooling rule + Holm family (k∈{8,16,32,64} deltas) written into the config JSON **before** execution and echoed into the output JSON, so the pre-registration is auditable in the artifact itself.
6. Position-metadata assertion for cot_16 (and all k): stored extraction index == k.
7. Paired bootstrap (resample pairs, not rows), B=500 acceptable; numpy-vectorized.

## 5. Leakage / cherry-picking risks in this loop (beyond the per-item notes)

- **Loop-level garden of forking paths:** iterating rounds until a positive delta appears, then reporting only the winning variant, is the main threat now. Mitigation: the packet series is the registry — every attempted variant appears in some round's artifact, and the paper reports Round-1 primary + pre-registered Round-2 pooling, all else as exploratory.
- **Metric switching:** don't let the headline drift to whichever of delta-AUROC vs pair-rank-accuracy looks better per k; both are already in the TSV — keep reporting both.
- **The k=4 result has survived one analysis round but was found amid multiplicity.** Before it headlines anything, confirm it on the two excluded sources (strongreject_full, reasoningshield) or a fresh seed — cheap insurance, can ride along with A4 or a small extraction job.
- Val is now triple-used (layer selection, family selection, z-stats/calibration). Statistically minor here, but say so in the limitations block rather than letting a reviewer find it.

## 6. The brutal-honesty section (README point 5)

**Stage1 as "hidden beats matched-horizon text across the k range" is, in my judgment, unwinnable and should not be the claim.** At k≥16, char-tfidf is reading the model's own *verbalized decision* — refusal/compliance markers in the CoT text. No fair probe-side improvement beats a baseline that reads the answer as it gets written; my realistic ceiling for A2 is parity at k∈{8,16,32} and a residual −0.05-ish deficit at k=64. If the loop's success bar is positive deltas everywhere, pivot **now**.

The honest, defensible reframing — which the current data already supports and A1/A2 strengthen —
is a **lead-time claim**: *hidden states at ~4 generated tokens predict the eventual outcome as well as the surface text does at 16–32 tokens; the text only catches up by verbalizing the decision the hidden state already encodes.* That is a positive, quantifiable, safety-relevant result (early-warning window measured in tokens), it uses the equal-horizon machinery as its evidence rather than as its executioner, and it composes naturally with a Stage-2 causal/intervention follow-up (does the k=4 direction steer the outcome?). If A2 lands parity at mid-k, you get the stronger version ("hidden ≥ text through k≤32, with a 4-token-lead advantage"); if not, you still have the lead-time paper. Either way, stop probe-variant iteration after A2 — the decision gate in A2's stop criterion is the loop's exit.

---

**Summary:** Continue for exactly one more iteration: CPU score-pooling + k=16 diagnostics + lead-time matrix (A1, minutes, $0), then feature-level pooled-probe rerun (A2, pod-CPU) as the pre-registered primary with a hard success/pivot gate; the current single-position hidden arm is provably handicapped (non-monotone AUROC, k=16 dip) so the negative k≥16 result isn't yet a clean negative — but if pooling doesn't reach parity at k≤32, pivot Stage1 to the lead-time framing rather than chasing positive deltas at k=64.
```

</details>

## Excluded-Source Lead-Time Confirmation Plan And Code Prep (2026-07-05)

### Fable-5 Plan Review

- Reviewer/model: `claude-fable-5`.
- Plan packet: `tmp/stage1_auto_improve_loop_260705/round1_leadtime_confirmation_plan_packet`.
- tmp prereg commit: `feefabb`.
- Verdict after required edits: `OK_TO_IMPLEMENT_PLAN_ONLY`.
- Scope allowed: a narrow excluded-source lead-time confirmation on
  `strongreject_full` and `reasoningshield`, using both A1 score-pooling and
  A2 feature-pooling recipes.
- Scope explicitly closed: any new equal-horizon rescue variant, layer search,
  classifier search, pooling search, k-grid search, or secondary-cell promotion.
- Required before code review: a config-pinning amendment that fixes the exact
  Module M `char_tfidf` surface recipe and the original Stage1 single-position
  hidden probe recipe.

### Actions Taken

- Added required config-pinning amendment:
  `res/stage1_excluded_source_leadtime_config_pinning_amendment_260705.md`.
- Updated the prereg plan to reference the amendment explicitly:
  `res/stage1_excluded_source_leadtime_confirmation_prereg_plan_260705.md`.
- Implemented confirmation analysis:
  `scripts/data/run_stage1_excluded_leadtime_confirmation.py`.
- Added RunPod extract-minimal helper:
  `pipelines/runpod_stage1_excluded_leadtime_extract_minimal.sh`.
- Added focused test:
  `tests/test_stage1_excluded_leadtime_confirmation.py`.

### Local Validation

- `python3 -m py_compile scripts/data/run_stage1_excluded_leadtime_confirmation.py`: passed.
- `bash -n pipelines/runpod_stage1_excluded_leadtime_extract_minimal.sh`: passed.
- `cot-safety/.venv-stage1-test/bin/python -m pytest tests/test_stage1_score_pooling_reanalysis.py tests/test_stage1_feature_pooling_reanalysis.py tests/test_stage1_excluded_leadtime_confirmation.py`: `6 passed`.

### Current Status

The plan/amendment/code are ready to package for a separate Fable-5 code
review. No RunPod execution has been launched for this confirmation yet.

### Fable-5 Code Review

- Packet: `tmp/stage1_auto_improve_loop_260705/round1_excluded_leadtime_code_packet`
- tmp commit: `c7a427e`
- Reviewer/model: `claude-fable-5`
- Verdict: `OK_TO_RUN`
- Blockers: none
- Stored raw response:
  `review-stage/stage1_auto_improve_260705/fable5_excluded_leadtime_code_review_260705.md`

Fable-5 verified all seven load-bearing items: all-k frozen test population,
minimum-power halt, no future leakage, Module M `char_tfidf` and original
hidden-score recipe pinning, gate logic, extract-minimal helper, and focused
tests. Non-blocking operational note: R2 backup remains a post-run step, and
the results packet must confirm default sources/k-grid and
`minimum_pairs_per_source=150`.

### Cot-Only Extraction Patch Review

During the first RunPod extraction attempt, the legacy extractor saved an
extra default `think_last` position. The run was stopped before interpretation,
and official outputs were moved to a fresh cot-only root:
`/workspace/stage1-results/stage1_post_hb_260705_retune12288_b20/hidden_archives_excluded_leadtime_cotonly`.

Patch:

- `legacy/PauseProbe/scripts/probe/extract_hidden_states.py` adds
  `--omit_think_last`.
- `pipelines/runpod_stage1_excluded_leadtime_extract_minimal.sh` passes
  `--omit_think_last`.
- The config-pinning amendment now requires `omit_think_last=true` and the
  cot-only output root.

Fable-5 narrow patch verdict: `OK_TO_RUN`.

Required result-review checks carried forward:

- official hidden archives must originate from
  `hidden_archives_excluded_leadtime_cotonly`;
- each manifest must have `omit_think_last: true`;
- each manifest `position_names` must be exactly
  `cot_4,cot_8,cot_16,cot_32,cot_64`;
- old partial artifacts under `hidden_archives` remain quarantined and must not
  be used.

### Post-Pivot Diagnostics Memo

- Added `res/stage1_a1_a2_leadtime_diagnostics_260705.md`.
- Packet for Fable-5 documentation review: `round1_postpivot_diagnostics_packet`.
- tmp commit: `d436841`.
- Fable-5 verdict: `EDITS_REQUIRED`, with three surgical text edits and no
  requested re-review.
- Required edits applied:
  - rephrased A1 as a score-pooling-specific ensembling/variance-reduction
    diagnostic rather than "exposing a signal";
  - rephrased hidden AUROC above chance as expected in-distribution
    decodability under teacher forcing, not evidence by itself;
  - restored the recipe-attenuation parenthetical in the stop/pivot reporting
    language.
- Fable-5 stated that, with those edits, the documentation is accepted as the
  closed Stage1 equal-horizon record and the post-pivot documentation step is
  complete.

<details>
<summary>Click to expand full Fable-5 post-pivot documentation review</summary>

```text
# Fable-5 Review — Stage1 Post-Pivot Diagnostics Documentation

**Packet:** `round1_postpivot_diagnostics_packet` (8 files, aggregate TSV/MD only — confirmed no restricted content; commit `d436841` ≡ local files is asserted by you, not verified — git on the tmp repo was approval-gated this session).

## Verdict: **EDITS_REQUIRED** (3 surgical text edits; interpretation is otherwise faithful)

## Faithfulness verification (done from the artifacts, not the README)

- **Numbers:** all 10 same-horizon cells (A1/A2 hidden, text, delta × 5 k's) match `stage1_score_pooling_summary.tsv` / `stage1_feature_pooling_summary.tsv` at 4-dp rounding; all 50 lead-time cells (25 A1 + 25 A2) match the two lead-matrix TSVs; every CI quoted in the stop/pivot memo matches the summary TSVs. No number in either memo lacks a TSV source — consistent with "no new fits."
- **Rule compliance:** k=4 correctly scoped exploratory (Holm family {8,16,32,64}); p=0.0 → p<0.002 (B=500) handled; no test-max, no HB-only zero-crossing CIs, no pair-rank headlining, full matrices published rather than cherry-picked cells; A1 and A2 always side by side; the "not 'feature pooling destroys signal'" caveat matches my A2 ruling verbatim; the optional excluded-source lead-time confirmation is stated with both required conditions (both recipes, accept either outcome). No language proposing or smuggling an equal-horizon rescue variant.

## Required edits (exact text)

**Edit 1** — `res/stage1_a1_a2_leadtime_diagnostics_260705.md`, Key Findings #2. "Can expose a signal" is soft-rescue framing; my ruling was that A2 gave the hidden arm strictly richer access and found *less* — the A1 advantage is an ensembling effect, not revealed signal. Replace:

> **A1 remains useful only as a diagnostic.** It shows that the original probe-score ensemble can expose a signal, but the advantage is not robust enough to support a main claim.

with:

> **A1 remains useful only as a diagnostic.** Its k=8 advantage was score-pooling-specific — consistent with an ensembling/variance-reduction effect over per-position probe scores rather than extra extractable information — and does not support a main claim.

**Edit 2** — same file, Key Findings #5. "Hidden states encode safety-relevant signal" was ruled dead as a positive claim in the 2026-07-02 DPI audit (in-distribution decodability under teacher forcing carries no evidence); the allowed statement is the absolute-AUROC fact. Replace:

> **Allowed claim scope is narrow.** Hidden states encode safety-relevant signal, but teacher-forced matched-horizon monitoring does not show a hidden-over-text information advantage.

with:

> **Allowed claim scope is narrow.** Pooled hidden AUROCs stay well above chance (0.736–0.793), but above-chance in-distribution decodability is expected under teacher forcing and is not evidence by itself; the equal-horizon comparison shows no hidden-over-text information advantage.

**Edit 3** — `res/stage1_a2_feature_pooling_stop_pivot_260705.md`, Reporting Language blockquote. The quote is attributed to me but drops the recipe-attenuation parenthetical — the one disclosure that keeps the canonical paragraph self-containedly honest about the exploratory signal. Replace:

> An early-horizon signal is reported as an exploratory diagnostic only.

with:

> An early-horizon signal (hidden@4 ≈ text@16–32 in absolute AUROC under the original probe pipeline, delta@k4 +0.056; attenuated to +0.004, n.s., under the preregistered refit recipe) is reported as an exploratory diagnostic only.

## Non-blocking notes

1. A2 lead-time section: "strongest positive lead cells" — h64/t4, h64/t8 are *lag*-direction cells, not lead cells; the sentence already says "wrong direction," so just consider "strongest positive off-diagonal cells."
2. The same quantity has slightly different CIs across files (A1 k4 delta: summary [+0.0452, +0.0668] vs lead-matrix [+0.0457, +0.0666]) — independent bootstrap streams, the known duplicate-diagonal caveat. If both TSVs are published, add one line saying diagonal cells are re-bootstrapped so the tiny differences aren't read as inconsistency.
3. "class_weight=balanced" in the A2 estimator row is asserted from code, not verifiable from this packet; it is inert anyway (paired design ⇒ n_pos=n_neg). Keep only if actually in the code.
4. The A1 early-warning sentence ("match or exceed text at some later horizons") is supported (exceed: h4/t8 +0.0311 CI positive; match: h16/t32 CI spans 0) but note h4/t16 and h4/t32 are statistically negative — the "≈ text@16–32" reading is about absolute-AUROC magnitude, which Edit 3's wording now makes explicit.
5. If the optional excluded-source lead-time confirmation is ever exercised, its preregistration must fix the exact estimand (recipes, cells/deltas, success rule) before any generation.

## Completion

**Yes — with the three edits applied, this documentation is accepted as the closed Stage1 equal-horizon record, and the post-pivot documentation step is complete.** The edits are mechanical; no re-review round is needed (commit the edited files and note the edit in AUTO_REVIEW.md). The A2 run remains valid, so no new equal-horizon experiments are warranted or permitted. Memory updated with this outcome.
```

</details>

### A2 Run

- Code commit: `d26d03c`.
- RunPod output:
  `/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/feature_pooling_a2_260705_b500`
- R2 backup:
  `cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/stage1_post_hb_260705_after_hb_n100_loso/feature_pooling_a2_260705_b500/`
- `n_errors=0`, `a2_full_success=false`, `a2_partial_pivot=false`,
  `a2_failure=true`.
- Pooled same-horizon delta AUROC:
  - k=4: `+0.0041`, CI `[-0.0082, +0.0158]`
  - k=8: `-0.0235`, CI `[-0.0350, -0.0123]`
  - k=16: `-0.0428`, CI `[-0.0547, -0.0315]`
  - k=32: `-0.0430`, CI `[-0.0533, -0.0323]`
  - k=64: `-0.0564`, CI `[-0.0667, -0.0448]`
- A2 results packet tmp commit: `5c1b2de`.
- Fable-5 A2 results review returned `STOP_EQUAL_HORIZON_AND_PIVOT`.

### Fable-5 A2 Results Review

- Verdict: `STOP_EQUAL_HORIZON_AND_PIVOT`.
- Run validity: accepted.
- Failure branch: confirmed, because k=8 delta CI low `< 0`.
- Mandatory stop: equal-horizon probe-variant iteration is closed.
- Interpretation:
  - A1's k=8 advantage was score-pooling-specific and not confirmed by A2.
  - A2 shows text evidence accumulates faster than the pooled hidden readout.
  - Do not overclaim "hidden has no signal"; A2 hidden AUROCs remain
    `0.73-0.79`, but they do not beat matched-horizon text.
  - A1/A2 recipe-strength caveat must be disclosed: at k=4 both use `cot_4`,
    but A1 hidden AUROC was `0.788` while A2 refit was `0.736`.
- Allowed next actions:
  - document and close equal-horizon thread;
  - report A1 and A2 side by side;
  - run only descriptive, zero-new-fit diagnostics from existing artifacts;
  - optional future pre-registered confirmation on excluded sources if
    lead-time is to be treated as more than exploratory.
- Disallowed actions:
  - no A2b rescue variant, no new pooling/classifier/layer/hyperparameter
    search, no metric/cell shopping.

<details>
<summary>Click to expand full Fable-5 A2 results review</summary>

```text
Memory updated. Full review below.

---

# Fable-5 Review — Stage1 A2 Feature-Level Cumulative Pooling Results

**Packet:** `round1_a2_results_packet` (7 files, matches README manifest; aggregate-only, no restricted content). All numbers below re-verified from `stage1_feature_pooling_summary.tsv/.json`, `_preregistration.json`, `_lead_time_matrix.tsv`, and the two diagnostics TSVs — not taken from the README.

## Verdict: **STOP_EQUAL_HORIZON_AND_PIVOT**

The run is valid, the preregistered failure branch fired exactly as written, and A2 was the pre-declared last equal-horizon attempt. The stop is mandatory, not discretionary.

## 1. Run validity (checked before interpreting anything)

| Check | Result |
|---|---|
| Prereg echoed in artifact, rules written pre-execution | ✓ `_preregistration.json`: failure rule = "if k=8 CI low < 0, stop; A1 was score-pooling-specific" |
| Code commit | ✓ `git.commit=d26d03c`, `dirty=false`, matches prereg — my required run condition satisfied |
| Exit / errors | ✓ `n_errors=0`, `errors=[]` |
| Frozen population | ✓ constant 152 (HB) / 2019 (WJB) / 2171 (pooled) test pairs at every k; zero drops in split/fit diagnostics; val alignment 264/264 and 124/124 with 0 label/pair mismatches — the fail-closed guards held |
| No future positions | ✓ pool indices strictly cumulative: [5] → [5,9] → [5,9,13] → [5,9,13,15] → [5,9,13,15,17] (cot_4…cot_64) |
| Val-only z-norm, both arms | ✓ `hidden_val_stats`/`surface_val_stats` per source×k, n = val rows (264/124) |
| Holm family {8,16,32,64}, k4 exploratory | ✓ k4 rows have empty `holm_p`; k8–k64 pooled Holm p reported |

Not an INVALID_RUN by any criterion I set at code review.

## 2. Task 1 — Failure branch verification: **YES, satisfied**

Pooled same-horizon Δ AUROC (hidden − char-tfidf), from the summary TSV:

| k | hidden | text | Δ | 95% CI | Holm p |
|---:|---:|---:|---:|---|---:|
| 4 | 0.7364 | 0.7323 | +0.0041 | [−0.0082, +0.0158] | (exploratory) |
| 8 | 0.7335 | 0.7570 | **−0.0235** | **[−0.0350, −0.0123]** | <0.002 |
| 16 | 0.7589 | 0.8018 | −0.0428 | [−0.0547, −0.0315] | <0.002 |
| 32 | 0.7646 | 0.8077 | −0.0430 | [−0.0533, −0.0323] | <0.002 |
| 64 | 0.7932 | 0.8495 | −0.0564 | [−0.0667, −0.0448] | <0.002 |

k8 CI low = −0.0350 < 0 → failure branch. Not marginal: the entire k8 CI is negative, both sources individually significant (HB −0.068, WJB −0.019), and every k ≥ 8 is fully negative. Full-success and partial-pivot conditions both fail (partial required k8 to hold). The `a2_failure=true` flag is consistent with the underlying rows, not just asserted. Max adjacent hidden drop 0.0029 passed the monotone check — the failure is not a mechanical artifact of pooling collapse; pooled hidden genuinely grows slower than text.

## 3. Task 2 — Stop rule: **applies now**

A2 was declared in advance (A1 results review, echoed in the artifact's own prereg) as the final equal-horizon attempt, with "failure → stop, no rescue attempts." The branch fired. Equal-horizon probe-variant iteration is over. Two prior forecasts corroborate that this is signal, not bad luck: my round-1 assessment called the equal-horizon race "unwinnable" with a realistic A2 ceiling of parity (it landed below that), and the 2026-07-02 DPI audit predicted exactly this regime.

## 4. Task 3 — Interpretation

**A1 vs A2.** A1's k8 advantage (+0.0274, CI [+0.0183, +0.0366]) reversed to −0.0235 under feature pooling. Preregistered reading stands: the advantage was score-pooling-specific — an ensembling effect (averaging several calibrated per-position probe decisions reduces variance) rather than extra extractable information. Feature-level pooling, which gives the hidden arm strictly richer access to the same information, found less, not more.

**One caveat the report must carry** (visible in the packet, changes attribution, not the verdict): at k=4 both arms pool only `cot_4`, so A1-k4 and A2-k4 use *identical features* — yet A1's pipeline probe scored 0.788 while A2's preregistered refit (StandardScaler+LogReg) scored 0.7364. The refit recipe is ~0.05 AUROC weaker on the same inputs. A2's negative deltas therefore conflate (a) pooling scheme and (b) probe-recipe strength, and the packet cannot separate them. This does not rescue equal-horizon — even A1's stronger recipe only tied text at k16/k32 and lost at k64 — but it means "feature pooling destroys signal" is an overclaim; "the equal-horizon advantage is not robust to the probe estimator" is what the data supports.

**Hidden-vs-text fairness.** A2 was the fair comparison: cumulative access on both arms, matched horizons, frozen population, refit per k, no future leakage. Its answer is clean: text evidence accumulates faster (0.732→0.850) than pooled hidden readout (0.736→0.793), because at larger k char-tfidf reads the model's own verbalized decision. This is the DPI-consistent outcome — under teacher forcing, hidden@k is a deterministic function of the same prefix, so hidden could only ever win on extractor efficiency, and it doesn't. The lead-time direction also vanishes under A2's recipe (hidden@4 vs text@8: −0.021 CI fully negative; hidden@64 ≈ text@16, < text@32), so the A1 lead-time result (hidden@4 ≈ text@16–32) is likewise recipe-sensitive.

## 5. Tasks 4–5 — Allowed next actions ONLY

1. **Document, don't run.** Write the failure-branch outcome into `AUTO_REVIEW.md`/registry; mark the equal-horizon thread CLOSED with the full k-table. No new equal-horizon experiments of any kind.
2. **Report both A1 and A2 side by side**, always — A1 is not deleted (it passed its own gate); it is re-scoped as pooling-scheme-specific.
3. **Cheap allowed diagnostics (descriptive, zero new fits):** (a) diff the A1 pipeline probe config vs A2's refit recipe from *existing* artifacts to document the 0.788-vs-0.736 k4 gap; (b) footnote the k4→k8 pooled-hidden dip (0.7364→0.7335) as evidence unweighted mean pooling dilutes early positions; (c) publish the lead-time matrix as-is; (d) report bootstrap p=0.0 as p < 0.002 (B=500); (e) per-source rows with the HB n=152 power caveat.
4. **Optional, decision for you/professor, not required:** one pre-registered *confirmation* (not variant search) of the k4 lead-time diagnostic on the two excluded sources (strongreject_full, reasoningshield), running **both** recipes (A1 pipeline probe + A2 refit) and accepting either outcome. This was pre-flagged in round 1 §5 before A2 ran, so it is not a forked path — but it only matters if you want lead-time in the paper as more than exploratory.
5. **Proceed on the separately registered workstreams** unaffected by this stop: Gates 3/4 (S→S FPR, encoder controls), on-policy OOD eval, and the LOSO source-expansion memo (still pending sign-off).

## 6. Disallowed (forking-path / cherry-picking guard)

- Any new pooling rule (max/attention/learned-weight/concat/last-token), layer, classifier, hyperparameter tuning, surface-family swap, k-grid, re-split, or re-normalization aimed at the equal-horizon question. **Specifically: no "A2b" that retunes the refit recipe to match A1's probe strength** — that is the canonical rescue variant.
- Headlining favorable cells or metrics: pair-rank k4 (+0.025, CI positive) stays in the table but never becomes the headline (metric shopping); lead-matrix cells like hidden@64 vs text@8 (+0.036) are the wrong direction for any lead claim (cell shopping).
- Citing HB-only k32/k64 CIs that cross zero as "parity on HarmBench" — underpowered subgroup (n=152).
- Reporting A1 without A2, or A2's failure without A1's pass. The pair is the finding.
- Standing rules: no test-max numbers; don't over-rotate into "hidden states carry no safety signal" — absolute AUROCs 0.73–0.79 are well above chance; the negative is strictly about *relative* advantage over text at equal horizon under teacher forcing.

## 7. Concrete Stage1 reporting language

> **Equal-horizon comparison (preregistered, final).** On teacher-forced natural pairs (harmbench_standard n=152, wildjailbreak_vanilla_harmful n=2019 test pairs), linear probes on layer-28 hidden states do not outperform a matched-horizon character TF-IDF baseline. A score-pooled analysis (A1) showed an advantage at k=8 (ΔAUROC +0.027, 95% CI [+0.018, +0.037]), but the preregistered feature-level rerun (A2: cumulative mean of layer-28 vectors over positions j≤k, probe refit per k, frozen evaluation population) reversed it: k=8 Δ = −0.024, CI [−0.035, −0.012]; k=64 Δ = −0.056, CI [−0.067, −0.045]; Holm-corrected p < 0.002. Per the pre-declared decision rule we conclude the A1 advantage was pooling-scheme-specific and terminate equal-horizon probe iteration. Text accuracy grows with horizon faster than pooled hidden readout (0.732→0.850 vs 0.736→0.793 from k=4→64), consistent with the data-processing inequality under teacher forcing: the text prefix upper-bounds hidden-state information, so hidden-state monitoring cannot be motivated by an information advantage in this regime. An early-horizon signal (hidden@4 ≈ text@16–32 under the original probe pipeline, Δ@k4 +0.056; attenuated to +0.004, n.s., under the preregistered refit recipe) is reported as an exploratory diagnostic only. The motivation for hidden-state methods in this project accordingly rests on causal utility (steering) and on-policy settings (Stage 3 pause positions), where the text-prefix bound does not apply, rather than on monitoring advantage.

**Summary:** valid run, failure branch confirmed from the artifacts, hard stop on equal-horizon variants, pivot to documenting the preregistered negative + exploratory lead-time diagnostic, with the A1-vs-A2 recipe-strength caveat disclosed. Memory updated with the closed-loop state.
```

</details>

### Planned Actions

1. Implement A1 CPU-only cumulative score-pooling script.
2. Add tests for no-future pooling, val-only z stats, pair alignment, and
   lead-time matrix output.
3. Put code packet in `tmp`, ask Fable-5 for code review before running.

### Actions Taken

- Implemented `scripts/data/run_stage1_score_pooling_reanalysis.py`.
- Added `tests/test_stage1_score_pooling_reanalysis.py`.
- Local validation:
  - `python -m py_compile ...` passed.
  - `pytest tests/test_stage1_score_pooling_reanalysis.py`: `2 passed`.
- tmp code packet commit: `b0027fb`.
- Fable-5 code review launched; waiting for response.

### Fable-5 Code Review Round 1

- Verdict: `BLOCKED`.
- Blocker: `source="pooled"` rows z-normalized hidden by source but left
  surface scores raw, so cross-source surface calibration offsets could
  manufacture a positive pooled delta.
- Fix required: for pooled rows only, z-score both hidden and surface arms per
  source using validation-split statistics before cross-source concatenation.

### Actions Taken After Blocker

- Added per-source val z-normalization for both arms in pooled rows.
- Added regression test for offset-only cross-source differences.
- Local validation: `3 passed`.
- tmp patch commit: `03854ee`.

### Fable-5 Code Review Round 2

- Verdict: `OK_TO_RUN`.
- Confirmed the offset-only regression now gives pooled delta `0.0`; the old
  behavior manufactured a positive pooled delta.
- Non-blocking caveats carried forward: rank-accuracy bootstrap duplication,
  exact-zero bootstrap p-values, duplicate diagonal rows in summary/lead
  matrices, and missing persisted position-metadata audit dict.

### A1 Run

- RunPod output:
  `/workspace/cot-safety/runs/stage1_post_hb_260705_after_hb_n100_loso/score_pooling_a1_260705_b500`
- R2 backup:
  `cloudflare_r2_cot_safety:cot-safety/stage1-paired/20260705-a100-stage1-post-hb-n100/runs/stage1_post_hb_260705_after_hb_n100_loso/score_pooling_a1_260705_b500/`
- `n_errors=0`, `a1_success=true`.
- Pooled same-horizon delta AUROC:
  - k=4: `+0.0558`
  - k=8: `+0.0274`, CI `[0.0183, 0.0366]`
  - k=16: `-0.0035`, CI `[-0.0121, 0.0061]`
  - k=32: `+0.0061`, CI `[-0.0035, 0.0149]`
  - k=64: `-0.0206`, CI `[-0.0298, -0.0120]`
- A1 results packet tmp commit: `abe8941`.
- Fable-5 A1 results review launched; waiting for decision on A2/pivot.

### Fable-5 A1 Results Review

- Verdict: `IMPLEMENT A2`.
- A2 is declared in advance as the last equal-horizon attempt.
- A1 gate passed exactly:
  - max adjacent AUROC drop `0.0038 <= 0.02`
  - k=8 delta CI low `+0.0183 >= 0`
  - `n_errors=0`
- A1 is diagnostic/reframing evidence, not confirmatory equal-horizon evidence,
  because it reuses existing predictions and val stats.
- Required next step: feature-level cumulative mean pooling over layer-28 hidden
  vectors at positions `j <= k`, refit probe per k, compare to unchanged text@k
  scores.
- Hard stop after A2:
  - Full success: k=8 CI low >= 0, max adjacent hidden AUROC drop <= 0.02, and
    k=64 delta CI upper >= 0.
  - Partial: k=8 holds but k=64 remains fully negative -> pivot to lead-time as
    primary and stop equal-horizon variants.
  - Failure: k=8 CI low < 0 -> A1 advantage was pooling-scheme-specific; no
    rescue attempts.

### A2 Code Packet

- Implemented `scripts/data/run_stage1_feature_pooling_reanalysis.py`.
- Added `tests/test_stage1_feature_pooling_reanalysis.py`.
- Local validation:
  - `python3 -m py_compile ...` passed.
  - `pytest tests/test_stage1_feature_pooling_reanalysis.py`: `2 passed`.
- Restored hidden archives from R2 to RunPod:
  `/workspace/stage1-results/stage1_post_hb_260705_retune12288_b20/hidden_archives`
- Confirmed real archive shape/keys:
  `features (2360, 19, 20, 4096) float16`, `valid_mask`, `labels`,
  `example_ids`, `pair_ids`, `match_families`, `position_names`, `layer_ids`.
- tmp code packet: `stage1_auto_improve_loop_260705/round1_a2_code_packet`
- tmp commit: `58d6eeb`
- Fable-5 code review returned `OK_TO_RUN`.

### Fable-5 A2 Code Review

- Verdict: `OK_TO_RUN`.
- No blockers.
- Reviewer independently ran packet tests and three adversarial checks:
  future-position guard, fail-closed population guard, and cross-source
  z-normalization offset check.
- Required run conditions:
  - pass `--fail-on-error`
  - pass non-null `--code-commit`
  - accept run only if exit code 0 and `n_errors=0`
- Non-blocking caveats:
  - if `--code-commit` is omitted on RunPod, provenance would be null;
  - A1 caveats still apply for exact-zero bootstrap p-values and rank bootstrap
    duplication;
  - this remains the final equal-horizon variant before success/pivot/stop.

<details>
<summary>Click to expand full Fable-5 A2 code review</summary>

```text
---

# Fable-5 Code Review — Stage1 A2 Feature-Level Cumulative Pooling

**Packet:** `round1_a2_code_packet` (local copy; note: I could not run git against the tmp repo in this session, so 58d6eeb ≡ local files is asserted by you, not verified by me — everything below reviews the local files).

## Verdict: **OK_TO_RUN**

No blockers. I read all four files line-by-line and, beyond static review, independently executed the packet's tests (2/2 pass) plus three adversarial checks I wrote against the guards the tests don't cover (all pass).

## Checklist findings

**1. Future-position leakage — PASS.** `pool_ks_for` (`run_stage1_score_pooling_reanalysis.py:364`) selects `{j ∈ k_grid : j ≤ k}` and asserts `max(pool) ≤ target_k`. `choose_position_indices` (`run_stage1_feature_pooling_reanalysis.py:233`) looks up exact names `cot_{k}` via string match (no index arithmetic → no off-by-one possible), raises on missing names, and the slice/mean in `load_feature_split:338-346` uses only those indices. Packet test poisons `cot_16` with 999.0 at target k=8 and asserts it never enters the mean — I reran it: passes.

**2. Split hygiene — PASS.** `fit_model` is called only with the train split (`:622`); the StandardScaler lives inside the pipeline so scaling stats are train-only. Val is used solely for prediction reporting and `hidden_val_stats`/`surface_val_stats` (`:639-641`). Test scores never enter any stats dict or fitting path (grep-verified). No selection anywhere: layer hard-pinned to 28 in `parse_args` (`:877-878`), family/k-grid/Holm family fixed, prereg JSON written before the fit loop (`:574`).

**3. Frozen evaluation population — PASS, verified adversarially.** `align_and_pair_complete` (`:443-459`) raises if any frozen surface row would be dropped (`right_dropped != 0`) or any pair-complete pair lost. My Check A (surface rows with no hidden counterpart) → fails closed with the population-guard message, never silently shrinks. Check B (extra hidden-only rows) → correctly tolerated, population stays exactly the frozen surface rows.

**4. Pooled cross-source z-norm — PASS, verified.** Both arms z-scored per source with val-aligned stats before concatenation: summary `:707-708`; lead-time matrix correctly uses `surface_val_stats[(source, surface_k)]` at `:760`. My Check C: shifting one source's surface scores by +5.0 (val+test) leaves the pooled k=8 delta bit-identical — the A1 round-1 blocker class is closed in A2.

**5. Loader hygiene — PASS.** `metadata_to_safe_row` (`:162`) extracts only id/pair_id/match_family/label; the npz `labels` array is cross-checked against metadata (`:324-330`); no text field is read, used, or written to any output.

**6. Artifacts — PASS with one caveat.** Prereg JSON (with success/partial/failure rules + commit), split + fit diagnostics TSVs, success preview with the three gate booleans, commit pass-through via `--code-commit` (test-verified). Caveat: **non-null is not enforced** — omit the flag on the pod (not a git checkout) and `git.commit` is silently `null`. Covered by the command below; optional 1-line hardening (`parser.error` if empty on pod) if you want belt-and-suspenders.

**7. Tests — adequate** for the two declared purposes (no-future pooling at feature level; end-to-end run + provenance). My adversarial checks now cover the fail-closed and z-norm guards on top.

## Non-blocking notes

- **Run with `--fail-on-error`.** Without it a failed (source,k) cell is only recorded in `errors`, and `source="pooled"` rows would quietly degrade to single-source (n_errors>0 / exit 2 do flag it, but the gate preview would be misleading if read carelessly). For a confirmatory gate run, hard-stop semantics are correct.
- Gate rules leave one cell unclassified: k8 CI low ≥ 0 ∧ k64 CI high ≥ 0 ∧ max drop > 0.02 → all three preview booleans false. Faithful to the pre-declared rules; if it lands there, it's human adjudication with all inputs present in `success_preview`.
- Surface files are loaded with `expected_k=None` (`:440`) — weaker than A1's metadata assertion, but the same frozen files passed that assertion in the A1 run and the `k_{k}` path assert remains. Near-zero risk.
- Carried A1 caveats still apply (duplicate diagonal Holm rows in the lead matrix, exact-zero bootstrap p-values, rank-metric bootstrap duplication).
- Efficiency only: each (source,k,split) reloads the full ~GB features npz (15 loads/source). Fine on the pod (peak RAM = one file); `mmap_mode="r"` is a later nicety.
- Reminder for downstream claims: harmbench_standard stays test-fold-only for the LOSO headline — this equal-horizon thread's within-source use is fine and required for A1/M comparability.

## Expected RunPod command

```bash
cd /workspace/cot-safety && python scripts/data/run_stage1_feature_pooling_reanalysis.py \
  --hidden-archive-root <root containing stage1_natural_pairs_8b_a100_1x_loso_{source}/ dirs> \
  --pred-dir runs/stage1_post_hb_260705_after_hb_n100_loso/matched_horizon/predictions \
  --output-dir runs/stage1_post_hb_260705_after_hb_n100_loso/feature_pooling_a2_260705_b500 \
  --code-commit <cot-safety code commit SHA> \
  --fail-on-error
```

Defaults supply the rest (sources HB+WJB, k-grid 4,8,16,32,64, Holm 8,16,32,64, layer 28, char_tfidf, B=500, seed 260705, tolerance 0.02). Accept the run only if exit code 0 and `n_errors=0`; then R2 backup as usual. Per the pre-declared stop rule, whatever the outcome, this is the last equal-horizon variant — full success, partial→lead-time pivot, or failure→stop.
```

</details>

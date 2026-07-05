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

# Fable-5 Pause Position Review: Stage1 Results and Stage2 Placement

Date: 2026-07-05
Reviewer: Fable-5, strict senior ML methods reviewer mode.
Scope: `README.md`, `FABLE_PAUSE_POSITION_REVIEW_REQUEST.md`,
`STAGE1_STAGE2_EVIDENCE_SUMMARY.md` only. No raw prompts, raw CoTs,
completions, hidden arrays, or row-level prediction files were inspected.

Standing rules honored from prior review rulings: equal-horizon Stage1 is
CLOSED with no rescues; k4 lead-time analyses are exploratory-only; never cite
test-max / test-selected numbers (validation-selected only).

---

## 1. Executive Verdict

**Final decision label: `COT4_MAINLINE` (conditional).**

- DeepSeek-8B Stage2 mainline: 3-pause block **before `cot_4`**. Reuse the
  existing cot4 format-only ckpt250 if it passes the acceptance gates below.
  Do not retrain and do not train any new position until Stage3's first read.
- `before cot_3`: position ablation only, using the already-trained
  checkpoints. No new GPU.
- `cot_120`: rejected as a mainline pause port. It is a hindsight readout
  location measured on a censored subpopulation, with no lead time. No
  late-pause SFT is required before mainline; a CPU-side coverage/onset
  analysis is required instead.
- Stage2 may proceed despite the Stage1 negative/control result, but only
  with reframed claims. Stage1 established that full-horizon hindsight
  classification is a surface problem (all 16 validation-selected
  hidden-minus-surface deltas negative; `length_only` beats the selected
  hidden probe on 4/4 sources). The surviving live hypothesis is narrower and
  belongs to Stage3: *pause-conditioned states at an early, matched horizon*
  are (a) more linearly accessible than matched-horizon controls and (b)
  usable as causal intervention ports. Stage3 must test exactly that, against
  matched-horizon controls, never against full-trajectory hindsight baselines
  as the bar.
- Because equal-horizon Stage1 closed without rescuing base-model hidden
  probes, Stage3 cannot inherit any positive prior from Stage1. Its
  hypothesis is strictly "pause training changes what is linearly
  accessible." Plan the null path explicitly.

Why not `MULTI_POSITION_ABLATION_FIRST`: the decisive additional controls
(coverage curves, position-convention audit, base-model matched-position
probes, cot3 reuse) are CPU-cheap or already trained. cot4 ckpt250 already
exists, so "committing the main Stage2 GPU budget" is mostly sunk. A
pre-mainline multi-position SFT sweep buys little information at high cost,
and its late-position arms would be censoring-confounded anyway.

Why not `DO_NOT_PROCEED_TO_STAGE2`: the Stage1 negative result falsifies
"hidden beats surface at full horizon," not "pause ports enable early
forecasting/steering." Those are different claims with different information
horizons. Stage2's own go/no-go criteria are behavioral, and its main
artifact already exists.

---

## 2. Answers to the Ten Questions

### Q1. Is early intervention (`before cot_4`) methodologically correct even if the strongest readout is late (`cot_120`)?

Yes, and the two facts are not in tension. Readout AUROC at offset *t*
measures how much of the final trajectory label is decodable from the state
at *t*. Because the label is a function of the trajectory, decodability grows
quasi-mechanically with horizon: by `cot_120` most of the evidence is already
in-context. That is recall of the past, not forecast of the future. The
`length_only` result is the tell: if trajectory-so-far surface statistics
beat hidden probes everywhere, late-position "signal" is dominated by
hindsight accumulation, not by privileged latent knowledge.

An intervention port is optimized on different axes: lead time before unsafe
content is emitted, coverage across rollouts, trainability as a stable
format, and causal leverage over the remaining trajectory. A pause at
`cot_120` cannot prevent anything emitted at tokens 1–119; under an
any-point-unsafe construct (or final-trajectory labels with early unsafe
onset), late intervention misses most of the preventable mass. The point of
easiest detection is usually past the point of cheapest prevention. Choosing
`cot_4` is therefore not a concession; it is the correct objective.

### Q2. Best readout vs best pause-port vs best steering location

- **Best readout/forecasting location.** Maximizes decodable information
  about the final label at that offset. Unconstrained, this drifts late and
  degenerates into hindsight. The scientifically meaningful variant is
  *forecasting under a fixed horizon*: max information subject to lead time
  > 0 and matched-horizon baselines. Late-offset superiority in Stage1
  diagnostics is the hindsight variant and confers no intervention value.
- **Best pause-port training location.** Where a pause block is learnable as
  a format without behavior drift: near-universal coverage, stable template,
  minimal interference with capability and safety behavior. Evidence is
  behavioral (pause3_rate, capability, unsafe_valid_rate, over-refusal), not
  probe AUROC. Current evidence: cot4 format-only ckpt250 healthy; cot3
  format-only degraded.
- **Best steering intervention location.** Where a state edit maximally
  changes downstream trajectory safety at acceptable capability/over-refusal
  cost. Requires lead time before typical unsafe onset and causal leverage.
  Only Stage4-style causal experiments measure this; no amount of readout
  AUROC substitutes.

These optimize three different objectives and coincide only by accident. The
project's error mode to avoid is using axis-1 evidence (late readout) to make
an axis-2/axis-3 decision (port placement).

### Q3. Given the post-HB negative/control result, should Stage2 proceed? What may it claim?

Proceed, conditionally. Three reasons: (i) the falsified claim (hidden ≥
surface at full horizon) is not the claim Stage2–4 need; the full-trajectory
surface baseline is unavailable at intervention time, so it is a reference
ceiling, not the bar; (ii) marginal cost is low — ckpt250 exists, and the
remaining Stage2 work is audits and missing behavioral checks; (iii) the
go/no-go for Stage2 is behavioral and independently evaluable.

Stage2 may claim only:
1. Format learnability: the 8B model emits the 3-pause block at the audited
   position with pause3_rate ≈ 1.0.
2. Behavioral non-degradation within pre-registered tolerances: capability,
   unsafe_valid_rate, over-refusal, CoT-length distribution.

Stage2 must not claim: separability, forecasting value, safety improvement,
or steering value. In particular the ckpt250 unsafe_valid_rate reductions
(HB 0.465→0.432, LG 0.553→0.505, WG 0.440→0.365) are an uncontrolled SFT side
effect until replicated and controlled for CoT-length drift (shorter CoTs
trivially reduce unsafe token opportunities). Note also that for Stage3/4
attribution, a pause model that is *already safer* than base is a confound:
all downstream steering effects must be measured against the pause-model
baseline, not base.

### Q4. DeepSeek-8B main Stage2 position

**`before cot_4`, 3-token pause block.** Grounds: near-total coverage;
intra-CoT after 3 sampled tokens (escapes the severe pre-CoT degeneracy per
the prior clarification, provided decoding is sampled and the position
convention is audited); the only demonstrated-healthy trained checkpoint;
maximal lead time among positions with non-degenerate within-prompt state
variance.

The alternatives:
- `before cot_3`: retain as the existing position ablation. Format-only cot3
  made unsafe_valid_rate worse than base on all three judges; do not ship it.
- `cot_16`: the only additional SFT position worth considering, and only if
  Stage3 shows position sensitivity. High coverage, more context.
- `cot_64`: exploratory at most, last priority, and only after the
  onset/coverage analysis justifies it.
- `cot_120`: not for SFT. Censoring at `cot_96`/`cot_128` already documented;
  the port would be absent precisely on short/refusal trajectories, at train
  and at deployment.
- Prompt/pre-think pause: **rejected as degenerate for Stage3.** With zero
  sampled tokens preceding the pause, the pre-pause prefix is identical
  within each same-prompt pair, so pause hidden states are identical within
  the pair and within-pair separability is zero by construction. Stage3
  would be a guaranteed null. This is exactly the severe critique.
- Multiple/periodic pause blocks: deferred. Attractive for anytime
  monitoring, but it multiplies format-drift risk and destroys single-factor
  attribution in Stage3. Revisit only after a clean single-position result.
- Adaptive/entropy-triggered pauses: future work; not for a first result.

### Q5. Late-pause ablation before committing main Stage2 GPU?

**No SFT-based late-pause ablation before the mainline Stage3 read.**
Reasons: (i) an SFT late-pause arm answers a readout question at intervention
prices, and readout cannot establish intervention value regardless of
outcome; (ii) any comparison at ≥`cot_96` is censoring-confounded — the
subpopulation differs from the cot_4 arm, so the AUROC comparison is not
apples-to-apples; (iii) cot4 already has a trained, healthy checkpoint, so
the marginal cost of proceeding mainline is small.

Required cheap substitutes first:
- CPU: coverage/survival curves at cot_3/4/16/64/96/120 per source from
  existing length metadata, with censoring composition (what fraction of
  censored trajectories are refusal-like/short-safe).
- Optional, small judge cost, high value: unsafe-onset analysis. Sample
  ~200 eventually-unsafe trajectories, judge truncated prefixes at
   32/64/120 tokens (on the remote pipeline, metrics only). If a majority of
  eventually-unsafe trajectories are already unsafe by token 120, `cot_120`
  is disqualified for prevention with data rather than argument.

Minimum credible late ablation, if one is later insisted upon: a single
`cot_64` format-only run (not `cot_120`), identical recipe and checkpoint
schedule, same three judges + capability + over-refusal, explicit per-source
coverage accounting, and Stage3 probes with matched-horizon controls at the
same offset. Priority: after the mainline Stage3 read, not before.

### Q6. How should existing Stage2 evidence weigh in?

- **cot4 format-only ckpt250 healthier**: supports cot4 as the engineering
  default. Weight: moderate. It is a single run/seed with no CIs; the
  capability deltas (overall 0.603→0.586, GSM8K 0.710→0.684) are within a
  plausible ≤2-pt tolerance but need n and bootstrap CIs; the safety
  improvement needs a length-drift control before it means anything.
- **cot3 format-only worse than base on all judges**: sufficient to demote
  cot3 to ablation status. Not sufficient to claim "position causally
  matters" — single-seed, could be recipe/noise. The positional claim, if
  ever made, belongs to Stage3 with paired protocols.
- **cot3 full-SFT diagnostic**: exclude from the placement decision entirely.
  It changes both format and content distribution; it is not evidence about
  pause placement.

Net effect: the Stage2 evidence justifies an *ordering* (cot4 over cot3) and
a *reuse decision* (ckpt250), nothing stronger.

### Q7. "Main candidate" vs weakened phrasing

Weaken. Correct sentence: **"`before cot_4` is the least-bad engineering
default for the 8B pause port — selected on coverage, trainability, and
behavioral health — pending Stage3 matched-horizon confirmation."** The
historical prompt-baseline tables (prompt-only ≈0.801, cot_3 ≈0.912, cot_4
≈0.921 heldout means) were already caveated as not construct-validity
evidence, and the frozen post-HB LOSO result puts hidden below surface
everywhere at full horizon. Any phrasing implying cot_4 is signal-optimal or
Stage1-validated is unsupported.

### Q8. Exact gates if Stage2 uses `before cot_4`

**S2 gates — must pass before any Stage3 extraction:**
- S2-G1 Format: pause3_rate ≥ 0.99 on both benign and unsafe prompts;
  tokenizer/position-convention audit passes (train-time insertion index ==
  extraction index; off-by-one explicitly checked; block size == 3).
- S2-G2 Capability: overall (GSM8K+MATH500) within 2.0 pts absolute of base,
  with n reported and bootstrap CIs. ckpt250 (−1.7) passes provisionally
  pending CIs.
- S2-G3 Safety non-degradation: unsafe_valid_rate ≤ base on all three judges
  (ckpt250 snapshot passes; re-run once with CIs).
- S2-G4 Neutral-behavior (currently MISSING from packet — blocking): benign
  over-refusal rate within pre-registered tolerance of base (suggest ≤ +2
  pts); CoT-length distribution drift reported with effect size; valid
  output/format rate ≈ base. If CoT length shrinks materially, the judge
  improvements in S2-G3 must be re-interpreted before Stage3.
- S2-G5 Non-degeneracy preconditions for Stage3: sampled decoding (T = 0.7),
  multiple rollouts per prompt, within-prompt across-rollout pause-state
  variance measurably above numerical noise floor.

**S3 gates — must pass before any Stage3 claim and before Stage4:**
- S3-G1 Design: same-prompt paired, prompt-disjoint LOSO on the frozen
  2783-pair keep list; selection (layer/offset/regularization) on validation
  folds only; pre-registered before test read; never cite test-max.
- S3-G2 Matched-horizon superiority: pause-state probe beats (a) base-model
  hidden probe at matched position `cot_4` and (b) matched-horizon surface
  text (first ~3 sampled tokens), with ΔAUROC ≥ 0.05 and 95% bootstrap CI
  excluding 0, macro-averaged across sources and individually positive on
  ≥3/4 sources. Full-trajectory surface may be reported as a ceiling
  reference only, never as the bar.
- S3-G3 Degeneracy controls: within-prompt pause-state variance check
  passes; label-shuffle null ≈ 0.500; per-sample (not per-prompt-mean)
  probing.
- S3-G4 Coverage: ≥99% of pairs alive at `cot_4` (expected trivial; report
  it anyway).

**S4 gates — must pass before any steering claim:**
- S4-G1 Baseline: all steering effects measured against the pause-model
  baseline, not raw base (do not double-count the SFT safety shift).
- S4-G2 Causal controls: random-direction and shuffled-probe controls;
  dose–response monotonicity over intervention strength.
- S4-G3 Cost accounting: unsafe reduction with over-refusal increase and
  capability drop within pre-registered budgets; output validity preserved.
- S4-G4 Robustness: effect holds on ≥2 of 3 judges and on ≥1 held-out
  source.

Statistical note: WildJailbreak is 2019/2783 ≈ 72.5% of pairs and HarmBench
only 152. Use macro-averages across sources for all gates, and expect wide
per-source CIs on HarmBench; do not let pooled numbers be WJB-in-disguise.

### Q9. Recommended first-run order

**Phase 0 — CPU-only, before any GPU:**
1. Position-convention + tokenizer audit (S2-G1 prerequisite).
2. Coverage/survival curves at cot_3/4/16/64/96/120 per source; censoring
   composition analysis.
3. Freeze re-audit: keep list still 2783/2783, no drops.
4. Write and freeze the Stage3 prereg (offsets, controls, margins,
   macro-averaging, CI procedure, null path). Expect an edits round before
   approval; plan-only until approved.

**Phase 1 — Stage2 acceptance of existing ckpt250 (small GPU):**
5. Missing neutral-behavior checks: benign over-refusal, CoT-length drift,
   output validity, pause3_rate on benign prompts (S2-G4).
6. Capability replicate with CIs (S2-G2); one safety re-run with CIs
   (S2-G3).
7. Optional in parallel: unsafe-onset prefix analysis (~200 trajectories ×
   3 cut points, remote judging, metrics only).
8. If any S2 gate fails: fix within the cot4 recipe (checkpoint choice,
   data mix). Do not switch positions in response to an S2 failure.

**Phase 2 — Stage3 first read (main GPU spend):**
9. Generate pause-model rollouts on the frozen prompt set, sampled T = 0.7,
   matching the Stage1 rollout protocol; judge on the remote pipeline; form
   natural same-prompt pairs.
10. Extract pause-block states (all 3 pause positions); extract base-model
    `cot_4` states on base rollouts of the same prompts (position control),
    and pause-model non-pause states at adjacent offsets (state control).
    Both controls have caveats (different rollouts vs same model); run both.
11. Probes: linear + small MLP, LOSO, validation-selected, macro-averaged;
    matched-horizon surface baseline and label-shuffle null in the same
    pass.

**Phase 3 — ablations, only after the Phase 2 read:**
12. cot3 checkpoints through the same Stage3 pipeline (free, already
    trained).
13. `cot_16` SFT only if cot3-vs-cot4 Stage3 shows position sensitivity.
14. `cot_64` exploratory only with onset/coverage justification. No
    `cot_120` SFT.

**Go/no-go to Stage4:** all S3 gates. Interpretation guide for the two
informative failure modes: if pause ≈ base at matched position but both beat
matched-horizon surface, the value is *position/earliness*, not *pause
training* — the pause narrative dies, an early-forecast narrative may
survive. If pause ≈ base ≈ surface at matched horizon, Stage3 is null: stop,
write the negative result, do not run Stage4.

### Q10. The honest answer to "strongest signal was at cot120 — why not put pauses there?"

"The cot_120 signal is a hindsight readout, measured on a censored
subpopulation, at a position with no lead time. By token 120 most of the
trajectory that determines the label has already been generated, so a probe
there is largely grading the past — and our own controls show trajectory
surface statistics (`length_only`, full-trajectory text) carry that signal
better than hidden states. Short and refusal-like trajectories never reach
token 120, so both the estimate and the deployed port would systematically
miss exactly the rollouts where the model self-corrects early. And causally,
a pause at cot_120 cannot prevent anything emitted in tokens 1–119. cot_120
tells us the label becomes easy to *read* late; it does not tell us we can
*act* late. Pause ports are placed for prevention, so they go early — where
coverage is total, the format trains cleanly, and there is still a future to
change. Whether early pause states actually carry usable signal is precisely
what Stage3 must test, against matched-horizon controls."

---

## 3. Position Recommendation Table

| Position | Role | Grounds | Main risk / caveat |
|---|---|---|---|
| `before cot_4` (3-pause block) | **Mainline** | Near-total coverage; intra-CoT after 3 sampled tokens; only healthy trained ckpt (ckpt250); max lead time with non-degenerate states | Within-prompt state variance from only 3 sampled tokens may be small — S2-G5 must verify; historical readout tables are caveated evidence |
| `before cot_3` | Position ablation (no retrain) | Already trained; tests position sensitivity in Stage3 | Format-only degraded unsafe_valid_rate vs base; never ship as mainline |
| `cot_16` | Conditional second SFT ablation | More context, still high coverage | New GPU; only justified if Stage3 shows position sensitivity |
| `cot_64` | Exploratory, last priority | Probes mid-CoT porting | Censoring onset; comparison confounded; needs onset analysis first |
| `cot_120` | **Not recommended for SFT** | — | Severe censoring at cot_96/128; hindsight location; no lead time; port absent on early-refusal rollouts |
| Prompt / pre-think pause | **Rejected** | — | Degenerate: identical prefix within same-prompt pair ⇒ zero within-pair separability ⇒ guaranteed Stage3 null |
| Multiple / periodic blocks | Deferred | Anytime-monitoring appeal | Attribution and format-drift risk; revisit after a clean single-position result |

---

## 4. cot120 Readout vs cot4 Intervention (short form)

1. Readout decodability grows quasi-mechanically with horizon because the
   label is a function of the trajectory; late AUROC is mostly recall, not
   forecast. `length_only` > hidden on 4/4 sources confirms the late/full
   signal is dominated by surface accumulation.
2. Late offsets are estimated on survivors only (censoring at cot_96/128),
   biasing the estimate and excluding early-refusal rollouts from the port.
3. Intervention value requires lead time: a cot_120 pause cannot prevent
   tokens 1–119. Detection is easiest where prevention is already too late.
4. Therefore: read late if you want hindsight; act early if you want
   prevention. The port goes early; the burden of showing early signal
   exists in pause states is Stage3's, under matched-horizon controls.

---

## 5. Do-Not-Claim List

1. Do not claim hidden states beat surface/text baselines — falsified at
   full horizon (all 16 validation-selected deltas negative).
2. Do not claim `cot_4` is the optimal or signal-maximal pause position —
   it is the least-bad engineering default pending Stage3.
3. Do not claim Stage1 supports pause tokens — Stage1 contains no pause
   tokens.
4. Do not claim the ckpt250 judge improvements as a "pause safety benefit"
   — uncontrolled SFT side effect until replicated with a CoT-length-drift
   control.
5. Do not cite the historical cot_3 ≈ 0.912 / cot_4 ≈ 0.921 prompt-baseline
   tables as construct-validity evidence — engineering guidance only.
6. Do not cite test-max or any test-selected numbers — validation-selected
   only (standing rule).
7. Do not present prompt-only TF-IDF = 0.500 as a beaten baseline — it is
   0.500 by construction in the same-prompt paired design.
8. Do not claim late-readout strength implies late-intervention value.
9. Do not claim generalization beyond DeepSeek-R1-Distill-Llama-8B, the
   four frozen sources, and the three judges used.
10. Do not make any causal/steering claim before all S4 gates pass, and
    never against the raw-base baseline.

---

## 6. Final Decision Label

**`COT4_MAINLINE`** — conditional on: S2 gates (including the currently
missing benign over-refusal and CoT-length-drift checks) passing before any
Stage3 extraction; a frozen, pre-registered Stage3 analysis with
matched-horizon controls and an explicit null path; cot3 retained as a
no-retrain ablation; no late-position SFT before the mainline Stage3 read;
and all claims phrased per the do-not-claim list.

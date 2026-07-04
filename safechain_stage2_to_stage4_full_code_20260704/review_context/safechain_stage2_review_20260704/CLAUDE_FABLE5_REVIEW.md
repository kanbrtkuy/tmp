Arxiv MCP wasn't granted — the local `REFERENCE_ARXIV_2310_02226.md` plus known results of Goyal et al. (gains require pause-*pretraining*; finetuning-only is mixed; pause outputs are ignored during decoding) are sufficient, and I'll caveat accordingly. I've now read the full packet: runner, trainer, data builder, eval harness, configs, and all result summaries. Here is the review.

---

# SafeChain Stage2 Deep Review — Pause-Token SFT and the Capability-Drift Confound

**Reviewer basis:** full read of `CLAUDE_REVIEW_REQUEST.md`, `PROFESSOR_CRITIQUE.md`, Stage2 runner/trainer/data builder, Stage2 SFT + eval configs, the model-comparison generation harness, and the Stage1/1b, natural-pair, Stage2, Stage4 result summaries in this packet. arXiv:2310.02226 referenced via the local note (arxiv.org egress-blocked; MCP not granted).

---

## 1. Executive verdict

**The current Stage2 design cannot support the claim "we trained the model to emit pause tokens at a target CoT position with minimal capability change" — for two reasons, one of which is more fundamental than the confound you asked about.**

1. **The headline Stage2 success metric is vacuous.** The model-comparison harness does not measure emission. For every `sft`/`steer` condition it generates N CoT tokens, **strips any pause tokens the model emitted itself, and force-appends `<|pause|>×3`** before continuing generation (`cot-safety/legacy/PauseProbe/scripts/eval/run_model_comparison_generation.py:306-321`, reassembled at `:343`). `pause3_rate = 1.000` is true **by construction**. Nothing in this packet measures whether any checkpoint actually learned to emit pauses at the target position. Stage4's 0.983–0.995 rates come from the same forced-insertion machinery.
2. **Both Stage2 objectives train the wrong thing.** Full SFT minimizes CE toward 17k external trusted CoTs — that is reasoning distillation with pauses incidentally spliced in; the GSM8K gain is the expected data effect. Format-only training is *parameter*-minimal but not *function*-minimal: it is exactly **3-token soft-prompt tuning at a mid-CoT position**, with the prompt vector optimized to maximize likelihood of Sky-T1/Bespoke/OpenThoughts continuations. Your own data proves it is behaviorally non-neutral: format-only **cot3** raises unsafe_valid_rate by **+17–19 points across all three judges** (0.465→0.653 HarmBench, 0.553→0.705 LlamaGuard, 0.440→0.617 WildGuard; ~8σ at n≈900), while cot4 lowers it. A single learned 4096-d vector, inserted one token earlier, swings safety behavior by 20 points in the *opposite direction*. That is an uncontrolled behavioral intervention, not instrumentation.

**The good news: the fix is small, and half of it requires no training at all.** Since the pipeline already force-inserts pauses at inference, the emission requirement is currently fictional. For *monitoring* (Stage3), you can get **provably zero** capability drift with a frozen base model and read-only pauses (§4, Method A). For *steering* (Stage4), where pauses must be attended, replace CE-toward-external-data with **KL-to-base self-distillation on self-generated CoT** (§4, Method C) — this makes "minimal drift" the training objective rather than a hoped-for side effect. Run the `no_pause_matched` full-SFT control that you already built but never trained, and the professor's confound is cleanly attributed. Total incremental cost: roughly 4–6 GPU-days on your 4×A100 node, most of it eval.

Secondary but serious: the current downstream default is incoherent (model config points at cot4 format-only ckpt250, `configs/model/deepseek_r1_distill_llama_8b.yaml:8`, while Stage3/4 run on **cot3 full-SFT ckpt250**, *chosen because it has the best capability* — i.e., the checkpoint was selected **on** the confound the professor flagged). And most reported capability deltas are 1–2σ events at n=500/300 with single-sample temperature-0.6 decoding; the noise floor must be established before interpreting any ±5-point movement.

---

## 2. What Stage2 actually does — mechanical findings

These are the facts the redesign must rest on; several are not stated in the summaries.

**F1 — Forced insertion makes `pause3_rate` a tautology.** `run_model_comparison_generation.py:306-321`: for `model_kind ∈ {sft, steer}`, generate `insert_pause_after_cot_tokens` tokens, `strip_pause_tokens(prefix) + PAUSE_TOKEN * n_insert_pauses`, continue. The stripping step even destroys the evidence of natural emission. No config in the packet runs an SFT checkpoint with insertion disabled (`-1`). Consequently: (a) the "did it learn the format" question is open; (b) the *operational* requirement for Stage3/4 is only "a pause token whose hidden states are informative and whose insertion doesn't derail generation" — which does not require emission training.

**F2 — Format-only training = mid-CoT prompt tuning toward an external corpus.** `trl_train.py:138-171` freezes everything, then unfreezes the full input-embedding matrix (plus untied output matrix on Llama-8B) with a gradient hook zeroing all rows except `<|pause|>`. Loss remains completion CE over the *entire* response. Gradient flow therefore does two things: (i) the output row is pushed up at pause slots and down everywhere else (that part is fine — it's the emission signal); (ii) the **input row receives gradient from every post-pause position**, optimizing the pause vector to maximize the likelihood of *trusted-CoT continuations*. That is definitionally a 3-token soft prompt trained on someone else's reasoning style (cf. prompt tuning à la Lester et al. — a handful of soft tokens is plenty to shift behavior). Two further non-obvious drift channels: the output row changes the softmax denominator at *every* position (tiny pre-pause drift even with frozen body), and with `init_from_text: ""` plus `mean_resizing=False` (`trl_train.py:42`), the row starts from random init, so the learned direction is unconstrained.

**F3 — The full-SFT capability gain is an unremarkable data effect, and the control exists but was never run.** 17k curated reasoning rows (heavily math), LR 2e-5, eff. batch 16 (4×2×2 per `run_4gpu_intra_pause_sft.sh:17-20`) → ckpt250 ≈ 0.24 epoch. Further distilling an R1-distill model on R1-style curated data improves GSM8K; Goyal et al. is not needed to explain it and (per their own results) pause benefits mainly arise with pause-*pretraining*, not post-hoc SFT. The builder emits `no_pause_matched` for every split (`build_intra_think_pause_sft_splits.py:161-169`) and the launcher supports `ADD_PAUSE_TOKEN=0`, yet **no config trains it**. Until that one run exists, "capability improvement" has zero attribution.

**F4 — The base-vs-SFT comparison changes two variables at once.** Base is evaluated without insertion; SFT rows are evaluated with forced insertion (F1). Any delta = (weights changed) + (three tokens injected mid-CoT). The 2×2 cells {base, SFT} × {no insertion, insertion} were never filled.

**F5 — Statistical power.** GSM8K n=500, MATH500 n=300, one sample/prompt at T=0.6. Base 0.603 vs full-SFT 0.649 overall: z≈1.9 unpaired. Format-only 0.586 vs 0.603: z<1. The cot3-format-only safety jump (~8σ) is the only Stage2 behavioral delta that is unambiguously real. Interpretations in the summary (e.g., "ckpt250 is the healthier candidate", 0.586 vs 0.581) are inside the noise.

**F6 — Position-convention mismatch between train and eval.** The builder counts the offset after skipping leading whitespace tokens (`build_intra_think_pause_sft_splits.py:54-98`); the eval harness counts N *raw* sampled tokens after the forced `"<think>\n"` prefix. If the model emits a leading whitespace/newline token, an "after 4 tokens" eval insertion lands at the trained cot3 position. Given how violently cot3 vs cot4 differ (F2), this convention skew can silently contaminate the position ablation. Also note the base eval config default is `3` while cot4 conditions override to `4` — an off-by-one trap (`stage2_model_comparison_eval.yaml:29` vs ckpt configs).

**F7 — Downstream checkpoint selection optimizes on the confound.** The Stage2 summary designates cot3 **full-SFT** ckpt250 as the Stage3/4 candidate "because it has the strongest overall capability" — precisely the property the professor says to treat as a red flag — while the repo default points at cot4 format-only ckpt250. Stage4's headline (α=2 reduces unsafe across judges) is thus built on the most-confounded checkpoint.

**F8 — The Stage4 "steering vector" is a trained disruption, not a probe direction.** `stage4_pause_steering.yaml:22-24`: delta is optimized to *maximize* post-pause NLL on unsafe rows / minimize on safe rows. The α response is non-monotonic (WildGuard unsafe: base 0.364 → α=1 **0.411** → α=2 0.130), which is what a nonlinear derailment mechanism looks like, not a linear unsafe-axis removal. No random-vector, norm-matched, or shuffled-label control exists in the packet.

**Hygiene notes:** HarmBench-cls numbers on safe prompts (base 0.670!) are out-of-scope for that judge and shouldn't appear in headline tables; the LOSO "gap" column doesn't match its own operands (e.g., 8B row 0.909 vs 0.852 labeled "+0.001", `stage1_stage1b_prompt_baseline_summary_20260630.md:127`); natural-pair TF-IDF baselines (AUROC 0.94–0.97) still exceed hidden probes (0.83–0.89), so Stage1's "separable signal" is not yet "semantic signal."

---

## 3. Diagnosis: the full-SFT capability-improvement confound (Q5)

**Mechanism.** Full SFT's gradient signal is >99.9% "reproduce trusted CoT tokens" and <0.1% "emit pauses" (3 pause slots vs ~2k content tokens per row). The model therefore moves toward the Sky-T1/Bespoke/OpenThoughts distribution: better math, more structured/careful prose, different refusal texture. Capability ↑ and unsafe ↓ are the *same* phenomenon — distribution transfer from curated data — and neither is about pauses. The professor's "an unsafe-removal vector should not improve math" concern is the Stage4 shadow of this Stage2 fact, amplified by F7 (the steering checkpoint was picked for its capability).

**Is it fatal?** Not to the project — fatal only to the claim "pause insertion is minimally invasive" *if made from full-SFT checkpoints*. It is genuinely useful as a **diagnostic upper bound**: full SFT tells you what the data alone can do to capability/safety; any pause-attribution claim must be measured against the `no_pause_matched` run, not against base.

**Defensible claim boundary.** "Full SFT on trusted CoT (with or without pauses — pending the control) shifts the model toward the data's reasoning style, improving math and reducing judged-unsafe output; we therefore do **not** use full-SFT checkpoints to make claims about pause instrumentation, and use them only as a data-effect reference arm." If the no-pause control reproduces the gains within CI (my strong prior), you additionally get a clean sentence: "pause insertion contributes ≈0 to the capability change; the change is a data effect."

---

## 4. Methods for pause insertion with minimal capability drift (Q1, Q2, Q3, Q7)

First decide **which requirement is real** — the packet's own harness answers this:

| Requirement | Needed for | Is it currently exercised? |
|---|---|---|
| R1: informative hidden states at pause positions | Stage3 monitoring | Yes |
| R2: pauses attended by later tokens (causal influence) | Stage4 steering | Yes |
| R3: model *emits* pauses autonomously | deployment without a decoding harness | **No — eval force-inserts (F1)** |

### Method A — Read-only pauses on the frozen base model (zero drift, no training) — *adopt for Stage3*

Force-insert pause embeddings but **mask pause positions out of the attention of all subsequent real tokens**. Pause hidden states are computed normally (they attend the prefix), so probes read them; the continuation distribution is *identical* to the base model's — capability drift is exactly zero, by construction, not by measurement. With KV-cache decoding this is trivial: run 3 extra forward steps at the monitor point, harvest hidden states, **roll the cache back**, continue.

```
# monitoring pass at trigger position k (KV-cache decoding)
cache_snapshot = cache.len            # after prefix x_<k
for j in 0..2:
    h_j = model.step(embed[pause], cache)   # pause attends prefix (+ earlier pauses)
    record probe features φ(h_j, layers L)
cache.truncate(cache_snapshot)        # discard pause KVs
continue_generation()                 # continuation == base model exactly
```

Equivalent framing: the pause is a *register/query token* (cf. vision "registers", Darcet et al.; filler tokens, Pfau et al. 2024) used as a probe attachment point. **Consequence: Stage3 needs no Stage2 training at all**, and "minimal capability drift" becomes a theorem instead of an experiment. This also answers Goyal et al. cleanly (Q7): their mechanism — later tokens exploiting pause *computation* — is precisely the drift channel; masking it out removes both their gains and your confound.

### Method B — Untrained neutral embedding, attended (baseline for steering mode)

Add `<|pause|>` with mean-of-vocab init (or copy a semantically inert token), no training, forced insertion, pauses attended. Fills the "insertion procedure alone" cell of the 2×2 (F4) and is the null model for Method C. Expect small but nonzero drift; measure it.

### Method C — Transparency-trained pause embedding via KL self-distillation — *recommended trained Stage2*

Train **only** the pause input row (and output row iff emission is wanted) to make the attended pause a **no-op for the continuation distribution**, using the model itself as teacher and **self-generated** CoTs as corpus (kills both the external-data style confound and the train/eval distribution mismatch in one move).

Notation: base model p (frozen); x = self-generated sequence; insertion slot k (builder convention); x̃ = x with `<|pause|>×3` inserted at k; θ = {e_pause, u_pause} the only trainable rows.

```
L_trans(θ) = E_x  Σ_{t > k}  KL( p(· | x_<t)  ||  p_θ(· | x̃_<t+3) )        # post-pause transparency
           + λ_pre Σ_{t ≤ k} KL( p(· | x_<t)  ||  p_θ(· | x̃_<t) )          # softmax-denominator guard (u_pause)
```

Teacher has no pause token → extend teacher support with mass 0 on `<|pause|>`; the KL then *automatically* suppresses spurious pause probability everywhere. If emission (R3) is required, add a small targeted term **only at the 3 pause slots**:

```
L_emit(θ) = − Σ_{j=0..2} log p_θ( pause | x̃_{<k+j} )
L_total   = L_trans + γ · L_emit          # γ ≈ 0.1–0.5; sweep
```

```
# one training step (only e_pause, u_pause require grad)
x  = sample_or_load_self_generated_cot(prompt)        # own model's CoT, benign+borderline mix
x̃  = insert_pauses(x, k=builder_convention(x))
with no_grad(): T = logits_base(x)                    # teacher pass, no pauses
S  = logits_theta(x̃)                                  # student pass, pauses attended
loss = KL(pad_pause0(T[k:]), S[k+3:]).mean()
       + λ_pre * KL(pad_pause0(T[:k]), S[:k]).mean()
       + γ * CE(S[pause_slots], pause_id)
loss.backward(); step()                                # ~200–500 steps, lr 1e-3, 2 rows only
```

Optional stabilizer: hidden-state matching `λ_h·Σ_{t>k} ||h_t^L(x̃) − h_t^L(x)||²` at 1–2 mid layers. Success is *measured in the objective's own units*: post-pause KL median → ~0.

### Method D — Emission with pause-slot-masked CE (if R3 is real)

If you must keep a CE formulation: change the collator label mask so **CE applies only to the 3 pause slots** (everything else −100), keep Method C's KL as the anchor. This is a ~10-line change from the current setup and removes the "learn the corpus style" gradient entirely. Prefer C(+emit) over D alone — D without KL leaves the input-row drift channel unregularized.

### Method E — Tiny LoRA + KL anchor (fallback only)

If a single embedding vector can't achieve reliable emission *and* transparency (possible—capacity is tiny), rank-2–4 LoRA on q/v of 2–4 layers around the probe layer, same L_total. Drift is no longer provably local → requires the full drift-diagnostic panel. Do not start here.

### Rejected
- **Full SFT** for any instrumentation claim (F3). Keep only as the data-effect reference arm.
- **Current format-only objective** (CE toward external corpus): replace with C. Note format-only is *not* "sufficient" (Q2): parameter-minimal ≠ function-minimal — the cot3 +17-pt safety swing is the proof, plus the softmax-denominator and post-pause channels (F2).
- **DPO-style format preference**: pairwise preference gradients move the whole policy; maximal collateral drift for a formatting goal. Wrong tool.
- **Pure logit-bias/constrained decoding for emission**: operationally identical to forced insertion — fine, but then just say "forced insertion" and stop pretending emission was learned.

### Drift diagnostics (Q2) — report for every Stage2 variant
1. **Post-pause KL curve**: median/95th-pct KL(base‖variant) vs token distance from pause, on held-out self-generated text (this is the primary drift metric; format-only cot3/cot4 will differ visibly).
2. Pre-pause KL (softmax-denominator leak) — should be ~1e-4 nats.
3. Greedy-continuation divergence: first-divergence-token index distribution; exact-match rate over 64 tokens.
4. Spurious emission: P(pause) at non-target positions; free-generation pause count with insertion off.
5. Embedding geometry: nearest vocab neighbors of e_pause; norm vs vocab distribution; cosine to a known refusal direction (Arditi et al.-style) — if the "format" vector aligns with refusal, you've found the cot3/cot4 safety swing mechanism.
6. Behavioral panel at matched power: capability (≥3 seeds or greedy, paired bootstrap), refusal-phrase rate, response/CoT length, or-bench/xstest over-refusal.

---

## 5. Prioritized experiment plan (4×A100 node; ranked by acceptance-lift per GPU-day)

| # | Experiment | What it settles | Cost (est.) |
|---|---|---|---|
| **P0.1** | Rerun comparison generation for cot4-fmt ckpt250 + cot3-full ckpt250 with `insert_pause_after_cot_tokens=-1`; measure **natural emission rate & position accuracy** (also fix `strip_pause_tokens` logging so emissions are recorded) | Whether Stage2 ever taught emission; whether R3 is even on the table | ~0.3 GPU-day (gen only, no judges) |
| **P0.2** | **Noise floor**: re-eval base + one SFT ckpt with 3 seeds (or greedy), paired bootstrap CIs on all metrics | Which existing deltas are real; every later claim's error bars | ~0.5–1 GPU-day |
| **P0.3** | **Base + forced insertion** (mean-init row, no training) and **read-only masked insertion sanity check** (must reproduce base outputs token-for-token) | Fills F4's missing cells; validates Method A implementation | ~0.5 GPU-day + ~1 day eng |
| **P1.1** | **`no_pause_matched` full SFT** to ckpt250/300 (data exists; `ADD_PAUSE_TOKEN=0`), same eval | The professor's confound, attributed in one run | ~0.5 GPU-day train + 0.5 eval |
| **P1.2** | **Method C training** (KL-transparent embedding, self-generated corpus ~5–8k traces) + drift panel + eval | The new Stage2 candidate | ~0.5 GPU-day (corpus gen) + 0.3 train + 0.5 eval |
| **P1.3** | **Stage3 probes on read-only pauses (frozen base, zero training)** vs current format-only ckpt250 vs Method C ckpt, same probe protocol | Whether Stage2 training is needed for monitoring *at all*; the cleanest possible Stage3 claim | ~1 GPU-day (extraction + probes) |
| **P2.1** | Stage4 vector characterization: norm-matched random vector, shuffled-label delta, probe-direction vs learned delta; correlation panel (refusal rate, lengths, entropy, base-model PPL of outputs, per-topic, judge disagreement); **capability under α=2** | Whether the steering effect is a safety direction or generic disruption/refusal (Q6) | ~1–1.5 GPU-days |
| **P2.2** | Position audit (F6): log pre-insertion prefix tokens at eval, reconcile with builder convention; then rerun cot3-vs-cot4 format-only comparison only if conventions differed | Whether the cot3/cot4 asymmetry is real or an off-by-one | ~0.1 + optional |
| P3 | Emission variant (C+emit or D) if P0.1 shows R3 matters; 1.5B replication of C; `pre_think_pause` arm | Robustness/generality | ~2 GPU-days |

Decision points: if **P1.3** shows read-only pauses probe as well as trained ones (my prior: they will — teacher-forced Stage1 already probes untrained positions at 0.92+ heldout), Stage2-as-training exists *only* to serve Stage4 steering, and the paper story simplifies dramatically. If **P1.1** reproduces the capability gain without pauses (prior: yes), the confound is closed with one sentence and a table.

**What would convince me the Stage4 vector is safety-related (Q6):** (i) norm-matched random and shuffled-label vectors produce ≪ effect at same α; (ii) effect monotone in α once the vector is a *probe-derived* direction rather than an NLL-disruption optimum; (iii) unsafe reduction not explained (R² decomposition) by refusal-phrase rate + length alone; (iv) GSM8K/MATH under steering ≈ unsteered (a "careful-style" vector raises them; a disruption vector craters them); (v) over-refusal on xstest/or-bench within CI of base; (vi) effect transfers across judge families and across at least one held-out unsafe source; (vii) absolute residual unsafe rates reported next to every relative claim.

---

## 6. Results-to-claims matrix

| Observed result (packet) | You may claim | You may NOT claim |
|---|---|---|
| `pause3_rate = 1.000` (all SFT rows) | The eval harness reliably force-inserts pauses; SFT models tolerate insertion without format collapse | The model learned to emit pauses; any emission property (never measured — F1) |
| Full-SFT cot4 final / cot3 ckpt250: overall 0.649/0.646 vs base 0.603 | Continued SFT on trusted CoT improves math benchmarks on this model (data effect; ~2σ pending P0.2) | Pause tokens improve reasoning (Goyal-style); pause insertion is minimally invasive under full SFT |
| Full-SFT unsafe_valid ↓ (e.g., HB 0.465→0.297) | Trusted-CoT SFT shifts safety behavior (style/data effect) | Pause-enabled safety improvement |
| Format-only cot4 ckpt250: 0.586 vs 0.603, unsafe slightly ↓ | Embedding-only insertion kept these benchmarks within the current (unmeasured) noise band | "No capability drift" (needs KL panel + CIs); "safety improved" (within noise; and any real Δ would itself be drift) |
| Format-only cot3: unsafe +17–19 pts, all judges, all ckpts | **The learned pause embedding is a behaviorally potent, position-sensitive soft prompt — the current objective does not control drift** (use as the motivating diagnostic) | cot3 is "a worse insertion position" per se (objective/convention confounds, F2/F6) |
| Stage4 α=2: unsafe ↓ across 3 judges (e.g., WG 0.364→0.130) | An optimized intervention at forced-pause positions can reduce judged-unsafe outputs on this eval, with residual unsafe 0.13–0.26 | A clean "unsafe axis" was found/removed (α non-monotonic; disruption objective; no random-vector control; confounded checkpoint F7) |
| Stage1b: prompt-only heldout mean 0.801 vs cot_4 0.921 (8B) | Early CoT sharpens risk separability beyond prompt-only baselines | Probes read "reasoning safety" semantics (TF-IDF ≥ hidden probes on natural pairs) |
| LOSO mean ~0.927 (both sizes) | ReasoningShield-family transfer of early-CoT signal | Source-artifact-free generalization (aidsafe/unsafechain rows; broken gap column) |

---

## 7. Do-not-claim list

1. Do **not** claim the model emits pauses — until P0.1 exists, "insertion" is a harness feature.
2. Do **not** claim minimal capability drift from any full-SFT checkpoint, in either direction (gain is as disqualifying as loss).
3. Do **not** claim format-only training is behaviorally inert — cot3 falsifies this in your own tables.
4. Do **not** attribute *any* safety delta of un-steered SFT checkpoints to pauses before the `no_pause_matched` arm exists.
5. Do **not** claim a "safety direction" from the Stage4 delta without random-vector/shuffled-label controls and the refusal/length decomposition.
6. Do **not** cite Goyal et al. as support for capability effects here — their gains require pause-pretraining; your setup is post-hoc and your goal is the opposite (neutrality).
7. Do **not** report relative unsafe reductions without the absolute residual (0.13–0.26 unsafe at α=2; >0.5 LlamaGuard base on unsafe prompts).
8. Do **not** compare cot3 vs cot4 as "position effects" until the F6 convention audit passes.
9. Do **not** use best-of-scan (test-max) probe numbers as headline results (the natural-pair note already concedes this — hold that line).
10. Do **not** claim hidden-state probes exceed surface classifiers on natural pairs (currently false).

---

## 8. Recommended next Stage2 config family

`configs/experiment/stage2_pause_transparent_*.yaml` — one family, five arms, shared eval:

```yaml
# stage2_pause_transparent_8b_4xa100.yaml  (family base)
sft:
  method: kl_transparent          # NEW trainer branch
  trainable_tokens: ["<|pause|>"] # input row; output row only if emit.enabled
  corpus:
    kind: self_generated          # own-model CoTs, NOT trusted_cot_18k
    prompts: {benign_math: 3000, benign_general: 2000, borderline_safe: 2000, unsafe_train_only: 1000}
    max_seq_length: 2048
  insertion: {cot_offset: 4, n_pause_tokens: 3, convention: builder_v1}  # audited, F6
  loss: {kl_post_pause: 1.0, kl_pre_pause: 0.1, hidden_match_l2: 0.0, emit_ce: 0.0}
  optim: {lr: 1.0e-3, max_steps: 400, warmup_ratio: 0.05}
  eval_drift: {kl_curve: true, greedy_divergence: true, spurious_pause: true, embed_geometry: true}
```

Arms: **v0_readonly** (no training; Stage3 default; drift ≡ 0) · **v1_neutral** (mean-init, no training; null model) · **v2_kl** (config above; Stage4 candidate) · **v3_kl_emit** (`emit_ce: 0.3`, output row trainable; only if P0.1 makes emission a requirement) · **c1_nopause_full** (`no_pause_matched`, full SFT; attribution arm) — plus the existing format-only cot4 kept as a comparison row, and `pre_think_pause` as a cheap position control. Retire full-SFT-with-pauses as a *candidate* (keep as reference arm). Update `configs/model/deepseek_r1_distill_llama_8b.yaml` to point Stage3 at **v0_readonly** and Stage4 at **v2_kl** once P1.2/P1.3 pass.

---

## 9. Reviewer-style risks & acceptance criteria

**Risks**
- **R-A (high):** Read-only/transparent pauses probe no better than plain cot_k positions → the pause loses its *monitoring* raison d'être and becomes purely a steering handle. Acceptable outcome — but then say so and restructure the paper claim around steering.
- **R-B (high):** A perfectly transparent pause (KL→0) may carry *less* probe-usable signal than a soft-prompt-ish one; transparency and informativeness could trade off. Measure Stage3 AUROC vs post-pause-KL across v0/v1/v2/format-only — report the frontier, don't assume a corner.
- **R-C (medium):** Engineering: masked/rollback insertion under vLLM (batch prefix caching) — fall back to a shadow probe pass (separate forward on prefix+pauses) which is mathematically identical for monitoring.
- **R-D (medium):** Judges: HarmBench-cls invalid on safe prompts; judge disagreement unquantified; single-sample decoding noise (P0.2 addresses).
- **R-E (medium):** Stage1 construct validity is still open (surface baselines ≥ hidden probes on natural pairs; source balance insufficient for LOSO). A perfect Stage2 cannot rescue a Stage1 probe that reads style/length.
- **R-F (low):** Convention audit (F6) invalidates prior cot3/cot4 position conclusions → rerun one comparison, small cost.

**Acceptance criteria for "pause insertion with minimal capability drift"**
1. **Emission** (only if R3 claimed): ≥99% pause3 at target position ±0 tokens with insertion harness OFF; spurious pause < 0.1% of generated tokens.
2. **Drift, distributional:** post-pause KL median ≤ 0.02 nats and ≤ 0.05 by 20 tokens after pause (v2); pre-pause KL ≤ 1e-3. v0_readonly: bitwise-identical continuations (exact).
3. **Drift, behavioral:** |Δ overall acc| and |Δ unsafe_valid| vs base within paired-bootstrap 95% CI of 0 on all three judges, ≥3 seeds — *for the un-steered checkpoint*. (A pause that "improves safety" before steering is a failure of neutrality, not a win.)
4. **Attribution:** `no_pause_matched` full SFT reproduces the full-SFT capability gain within CI → pause contribution ≈ 0 stated with numbers.
5. **Monitoring value:** probe AUROC at pause positions ≥ matched cot_k positions on the frozen base (else drop the monitoring claim).
6. **Steering validity:** criteria (i)–(vii) of §5/P2.1, with absolute residual unsafe rates in every table.

---

## 10. Direct answers to the eight review questions (index)

1. **Cleanest objective:** read-only insertion (zero drift) for monitoring; KL-to-base self-distillation on self-generated CoT for attended/steerable pauses (§4 A/C). Full SFT, DPO rejected; format-only insufficient as-objective.
2. **Is format-only sufficient?** No — it's 3-token prompt tuning toward an external corpus; cot3's +17-pt unsafe jump is the smoking gun; drift channels and diagnostics in §4/F2.
3. **Decoupling data effect from pause effect:** the 2×2(+arms) matrix — base±insertion, no-pause-matched SFT, neutral-embed, KL-transparent, random-token insertion (§5 P0.3/P1.1, §8).
4. **Minimal package for the professor:** P0.1–P1.3, ≈4–6 GPU-days (§5 table).
5. **Full-SFT capability gain:** not fatal; demote to data-effect reference arm; claim boundary in §3.
6. **Stage4 characterization:** §5 P2.1 + convince-me list (§5, bottom).
7. **Goyal et al.:** their effect *is* your confound channel — attended-pause computation changes behavior; gains need pause-pretraining, so expect none from post-hoc SFT and treat any as drift; read-only masking severs the mechanism (§4 A, Q7 notes).
8. **Deliverables:** plan §5, matrix §6, do-not-claim §7, config family §8, equations/pseudocode §4.

**Bottom line:** stop training the model to *say* pause and start proving the pause *changes nothing* — zero-drift read-only pauses for Stage3, KL-transparent pauses for Stage4, one no-pause control run, and emission measured (or dropped as a requirement) — that's the smallest redesign that makes the intended claim defensible.
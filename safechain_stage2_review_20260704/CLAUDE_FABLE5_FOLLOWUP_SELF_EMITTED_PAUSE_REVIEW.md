All grounding is done — I verified the forced-insertion/strip mechanics and the format-only trainer in code, and confirmed the Stage4 steering setup (learned NLL-disruption delta at pause_0/1/2, layer 14/20). Here is the second review.

---

# Second Review — Can Self-Emitted Pause Tokens Be Trained With Minimal Drift? (SafeChain Stage2, Follow-Up)

**Reviewer basis.** Full read of `FABLE_FOLLOWUP_REQUEST_SELF_EMITTED_PAUSE.md`, `CLAUDE_FABLE5_REVIEW.md`, `CLAUDE_REVIEW_REQUEST.md`, `PROFESSOR_CRITIQUE.md`, `stage2_format_only_sft_summary.md`, Stage1 natural-pair results, Stage4 hardsafe summary, and independent code verification of the two load-bearing mechanical claims:

- Forced insertion is real and destroys emission evidence in the judge view: for `model_kind ∈ {sft, steer}` the harness generates a prefix, `strip_pause_tokens(prefix) + PAUSE_TOKEN × n` (`run_model_comparison_generation.py:306-321`), and `generated_for_judge` strips pauses again (`:355`). `pause3_rate = 1.000` is a harness property. I treat self-emission as **empirically unmeasured**, as the request instructs.
- Format-only training is full-response CE with gradients masked to the `<|pause|>` embedding/head rows (`trl_train.py:138-171`), i.e., a 3-token soft prompt optimized toward the external trusted-CoT corpus. The cot3 format-only result (+17–19 pts unsafe_valid across HarmBench/LlamaGuard/WildGuard vs base) stands as proof that this objective does not control drift.

I agree with the first review's mechanical findings (F1–F8). This review evaluates the *clarified* plan — self-emission as a requirement, pauses before the CoT, UltraFeedback full SFT vs pause-only CE + KL-to-base — and goes beyond the first review where the clarification changes the analysis.

---

## 1. Final verdict

1. **The proposed pause-only CE + KL-to-base objective is valid and is the right family.** With a frozen body and only `{e_pause, u_pause}` trainable, the drift surface is two rows of parameters, the total distributional change is *auditable by construction*, and the failure mode that produced the cot3 disaster (continuation-CE gradient sculpting the pause embedding toward an external corpus style) is **structurally excluded**, not just empirically reduced. Adopt it, with the corrections in §5 and §8 (most importantly: self-generated corpus including safety-relevant prompts, teacher extension with zero pause mass, an explicit spurious-pause penalty — which your third term correctly anticipates — and gated training).
2. **Self-emission is not scientifically necessary under this design, and the request's causal-chain argument for it does not hold.** The causal chain runs through the *hidden states at the checkpoint*, not through *which process placed the checkpoint token*. Keep emission as a deployment feature trained at near-zero marginal cost inside the same objective (γ·CE at pause slots), but do not let the paper's claims depend on it, and always deploy with a harness backstop (§3).
3. **The clarified plan contains a new, more serious construct problem than anything in the Stage2 objective: pause tokens placed *before* the CoT cannot carry per-trajectory information.** At `prompt → pause_1..K → CoT`, the pause hidden states are a (near-)deterministic function of the prompt; all 10 samples of a prompt share identical pause states. Stage 3 as specified — "classify whether the subsequent CoT trajectory will be safe/unsafe" from pause positions — is information-theoretically capped at prompt-risk classification, and on a paired same-prompt evaluation (your Stage-1 `paired_clean` construction) it is **exactly 50%** by construction. This must be fixed by placement (intra-CoT), not by training (§2).
4. **UltraFeedback full SFT should be dropped, not rescued.** It is not logically incompatible with minimal drift, but every control that would rescue it converges to the KL-to-base method anyway, and UltraFeedback is a uniquely bad corpus for an R1-distill model (non-CoT response style; think-block mismatch with the deployed template; trigger-position mismatch between train and inference) (§4).
5. **KL-transparency will not make the pause "too no-op to steer."** The last-pause-position → first-content-token prediction channel is architecturally irreducible, and transparency constrains the function's *value* at α=0, not its *Jacobian*. The realistic risk is thinned attention from later tokens to pause positions (weaker multi-token control); this is measurable with a two-hour diagnostic (injection gain curve + attention mass), and there is a clean fallback (per-layer pause prefix) if the port is weak (§6).

Bottom line: **train the pause rows to be transparent-with-emission on self-generated data, put the pauses inside the CoT, demote self-emission from claim to feature, and never use UltraFeedback.** With the gates in §11, "minimal drift" becomes a measured property with a pre-registered budget, not a hope.

---

## 2. The pre-CoT placement problem (read this before anything about training)

The clarified plan moves the pauses to `prompt → pause_1 … pause_K → CoT` (the follow-up's "Position alignment" diagram; the original packet's Stage2 was intra-think at cot3/cot4). This changes what Stage 3 can possibly show:

- At generation time, the first stochastic event is the first CoT token *after* the pauses. Everything before it — prompt, forced `<think>\n` prefix, K pause tokens — is fixed given the prompt (self-emission at ≥99% reliability makes it fixed for practical purposes). Therefore **the hidden states at pause positions are identical across all 10 samples of the same prompt** (up to kernel nondeterminism, which is noise, not signal).
- Consequence A — paired evaluation: on a `paired_clean`-style same-prompt safe/unsafe pair construction, any probe on pause positions scores balanced accuracy **0.5 exactly**. The Stage-1 success criterion (~80% paired-clean) is unreachable at pre-CoT pause positions *no matter how good the training is*.
- Consequence B — unpaired evaluation: the probe's ceiling is the Bayes-optimal predictor of `P(unsafe | prompt)` — i.e., **prompt classification**, which is precisely the failure mode the professor's critique names, and which your own Stage1b controls quantify (prompt/pre-CoT positions ≈ 0.50 AUROC on natural pairs; near-ceiling on unpaired source-heterogeneous data). A decent-looking unpaired number at pause positions would be indistinguishable from a prompt classifier and adds nothing over Stage1b's prompt-only probe.
- Consequence C — Stage 4: steering at pre-CoT pauses is *prompt-conditional* intervention. That is still a legitimate defense mechanism, but its honest baseline is "run a prompt classifier, gate a fixed intervention on it" — a much cheaper system that Stage 4 would need to beat on over-refusal and capability. With intra-CoT placement, steering conditions on the realized trajectory prefix, which is a genuinely different (and more interesting) capability.

**Fix:** put the pauses after m ≥ 1 sampled CoT tokens (m ≈ 4 matches your Stage-1 signal position and the existing cot4 tooling), or better, at a small number of periodic checkpoints. Keep a pre-CoT arm only as a *prompt-risk control* whose predicted role is "equivalent to prompt-only probe." If the team insists on pre-CoT as the main design, then Stage 3's claim must be renamed prompt-risk exposure, and the `paired_clean` criterion must be dropped for Stage 3 — it cannot pass.

Note this interacts with self-emission (§3): under pre-CoT placement, emission is *easier* to train (a single stereotyped trigger position right after `<think>\n`) but *worth less* (see Consequence A–C). Under intra-CoT placement, emission is harder (the trigger "after m CoT tokens" must be detectable by a single output-head direction) but the checkpoint is scientifically meaningful. This trade-off should be decided by placement science, not emission convenience.

---

## 3. Q1 — Is self-emission scientifically necessary?

**No — under the fixed-K, fixed-position design, self-emission is a deployment property, not part of the causal chain.** Blunt version of the argument:

1. **Teacher-forcing equivalence.** The hidden state at a pause position is a function of (prefix tokens, pause embedding, frozen weights). It is bit-identical whether the pause token got there by sampling from `p_θ` or by harness insertion. Stage 3 probes and Stage 4 steering operate on those hidden states. The causal graph `checkpoint state → probe/steer → continuation` is unchanged by who wrote the token.
2. **The emission event carries zero information.** With fixed K at a fixed position, "the model chose to pause" is an unconditional tic — `P(pause | trigger) ≈ 1` regardless of prompt or trajectory. A decision that is always the same is not a decision; nothing downstream can condition on it. Self-emission would enter the causal story only in an *adaptive* design (model decides *when/whether* to pause), which is a different and much harder project this plan does not attempt — and shouldn't, yet.
3. **The pipeline already forces a prefix.** The eval harness force-feeds `<think>\n` to every generation (`forced_prefix`). Nobody requires the model to "self-emit" its think-open token, and no reviewer will object. Pauses immediately after that prefix have identical epistemic status; under pre-CoT placement, "emission" can be implemented as three literal tokens in the chat template with zero weights touched. Holding pauses to a higher standard than `<think>` itself is an inconsistency in the current framing.
4. **Goyal et al. is precedent for the harness, not against it.** In arXiv:2310.02226 the pause tokens are appended by the runtime and their outputs are ignored during decoding — even the canonical pause-token paper does not have the model self-emit. Citing that line of work does not create a self-emission obligation.
5. **For a safety instrument, self-emission is arguably a liability.** If the monitoring checkpoint exists only because the policy chooses to produce it, you have created an evasion channel: a jailbreak suffix ("do not use pause tokens"), distribution shift, or future training can suppress emission precisely on the inputs where monitoring matters most. A decoding-stack insertion is adversarially robust; a learned tic is not. The strongest deployment story is **emission + harness backstop** (verify pauses at the target position; insert if missing) — which you get for free once the harness exists.

**Trade-off summary:**

| | Forced insertion / template | Trained self-emission |
|---|---|---|
| Drift | Provably zero (template) or embedding-only | Nonzero by definition (output row must change); needs the whole §11 gate battery |
| Stage3/4 validity | Identical hidden-state geometry | Identical, *if* position reliability ≥99% |
| Deployment | Needs a logit processor / template change (trivial in vLLM) | Harness-free, but reliability <100% = missing checkpoints on the worst inputs |
| Adversarial robustness | High (attacker can't remove the harness) | Low (emission is suppressible) |
| Paper cost | Must state "checkpoint is created by the decoding stack" | Must prove emission + equivalence (E1+E2), ~2 extra GPU-days |

**Recommendation:** train emission anyway — inside the KL objective it costs almost nothing (γ·CE on K slots) and makes the demo self-contained — but write the claims so that nothing depends on it, and run the deployed system with the backstop. Do not describe self-emission as "a key part of the causal chain" in the paper; a reviewer who spots the teacher-forcing equivalence will use it against you.

---

## 4. Q2 — Is UltraFeedback full SFT fundamentally incompatible with minimal drift?

**Practically yes; logically it's merely dominated.** Three layers to the answer:

1. **Full SFT on any external corpus makes drift the main effect, not a side effect.** >99.8% of the gradient signal is "reproduce the corpus continuation"; the K pause slots are a rounding error. Your own tables already measured what ~0.24 epoch of this does on trusted CoT: overall capability 0.603 → 0.649, HarmBench unsafe 0.465 → 0.297 (cot4 full final); GSM8K 0.710 → 0.776 (cot3 full ckpt250). Those are the *good-corpus* numbers, and they already violate the clarified constraint ("should not materially change the base model's safety property or capability") in both directions.
2. **UltraFeedback specifically is the worst plausible corpus for this model.** (a) Chosen responses are instruct-style, largely without think blocks — SFT on `pauses + response` teaches the model *away from* its `<think>` CoT distribution, on a project whose object of study is the CoT. (b) Style/refusal texture transfer from the mixture of teacher models is guaranteed — exactly the confound the professor flagged. (c) **Trigger mismatch:** trained trigger = "right after the assistant header, before a non-think response"; deployed trigger = "right after the forced `<think>\n`". The emission behavior you train is conditioned on a prefix that never occurs at inference. This can silently produce wrong-position emission or none at all — and the current eval would hide it (F1).
3. **What would "rescue" it, and why that's moot.** No-pause-matched control arm (attribution); KL-to-base anchor on every non-pause position; replacing targets with self-generated continuations; tiny LR/steps with drift-panel early stopping; equivalence-test acceptance gates. Apply all of those and you have — term by term — reinvented pause-only CE + KL-to-base self-distillation with extra steps and a full-parameter drift surface you can no longer localize. Also note the plan's validation criterion ("unsafe CoT rate of the pause model close to the original") is a **single scalar and necessary-not-sufficient**: a model can match the aggregate unsafe rate while having materially different refusal style, length, topic profile, and per-source rates. Distribution-level checks (KL curve, §11) are the real criterion.

**Verdict:** drop UltraFeedback full SFT from Stage 2 entirely. Keep full SFT only as the *data-effect reference arm* (and if legacy full-SFT numbers appear anywhere in the paper, run the `no_pause_matched` control the builder already emits — it remains the one-run answer to the professor's confound).

---

## 5. Q3 — Is pause-only CE + KL-to-base a valid method?

**Yes. It is the correct objective family, and it fixes the identified failure mode at the structural level.** Specific assessment, including the parts that need correction:

**Why it structurally excludes the cot3-style failure.** In the current format-only setup, `e_pause` receives gradient from *every post-pause content token* pushing it to maximize external-corpus likelihood — that is how a single 4096-d vector became a +17-point unsafe soft prompt. Under the proposed objective, `e_pause`'s gradient comes from (i) the KL term, which pushes the pause toward *no effect on the continuation*, and (ii) CE at slots 2..K, a weak, benign "another pause follows" signal. The corpus-style gradient channel is gone, not merely down-weighted. With the body frozen, the entire model change is confined to two rows, and the induced behavioral change decomposes cleanly:

- On pause-free prefixes: only `u_pause` acts (softmax denominator + spurious emission mass) — bounded by the measured spurious-pause probability.
- On pause-containing prefixes: only the K pause positions' KV contributions and the boundary head act — exactly the channels the KL term minimizes.
- At the trigger slot(s): distribution intentionally repurposed (`P(pause) → 1`). **All intended drift is concentrated at K+1 positions; everything else has a measurable ε budget.** This is the cleanest possible "minimal drift" story short of read-only masking.

**Corrections and cautions on the objective as written:**

1. **KL direction and teacher support.** `KL(p_base ‖ p_pause)` (forward, teacher-first) is right for distillation, and both directions coincide near the optimum. But the teacher has no pause token: extend it with `p̃(pause) = 0`. Then note a subtlety your third term correctly anticipates: **forward KL is only weakly (linearly) sensitive to spurious student mass on the pause token** — if the student leaks ε probability onto `<|pause|>` at a content position, forward KL pays only ≈ ε nats. So the explicit `KL_non_pause_suppression` term is *not* redundant; keep it as `−log(1 − p_θ(pause|·))` at all non-slot positions. (This is the right justification for it; without it, "suppression" looks like it duplicates the KL.)
2. **Use distillation, not CE, for the continuation — and you already half-said this.** "Prefer self-generated base-model continuations" is correct; go one step further: since the teacher *is* the same weights, you can match full next-token distributions (exact forward KL per position) instead of CE on sampled tokens. CE-on-own-samples is an unbiased but high-variance estimate of the same quantity; full-logit KL converges faster and doesn't sharpen entropy.
3. **Corpus composition is a safety-validity issue, not a detail.** Transparency trained only on benign/math prompts does not guarantee transparency on StrongReject/WildJailbreak-style prefixes — which is where Stage 3/4 operate and where the "unsafe CoT rate close to base" validation lives. The corpus must include a safety-relevant slice (train-split only; keep Stage-3 probe prompts disjoint).
4. **Capacity risk for emission is placement-dependent.** First-slot emission trains only `u_pause` (the prefix contains no pauses, the body is frozen, so `e_pause` can't help). One output-head direction must make `pause` the argmax at the trigger and nowhere else. Pre-CoT trigger (stereotyped position after `<think>\n`): very likely fine. Intra-CoT trigger ("after exactly m sampled tokens"): not obviously a linearly detectable property of the hidden state — this is where emission may need the fallback (§7, rank 2/3). Also verify the model *stops* at K: `P(pause at slot K+1)` must be low; the boundary position is covered by KL + suppression, but measure it explicitly (pause-loop is the classic failure).
5. **Off-policy vs on-policy.** Training is teacher-forced on base rollouts; at deployment the pause model runs on its own prefix distribution. At convergence (KL≈0) these coincide, but *evaluate* the KL panel on pause-model rollouts too, not just base rollouts.
6. **Positional shift is inherent and fine.** Inserting K tokens shifts RoPE distances between the continuation and the prompt by K; the KL objective directly optimizes against whatever perturbation that causes. No extra machinery needed — just don't be surprised that KL doesn't reach exactly 0.
7. **Initialization.** Current code adds the token with `mean_resizing=False` and (in configs) empty `init_from_text` — i.e., an arbitrary row. Init `e_pause` to the vocab-mean (the `initialize_trainable_token_embeddings` path already supports this) and `u_pause` small; it starts you near transparency instead of near a random soft prompt.

**Answer: valid, and strictly better than both full SFT and the current format-only CE — provided the corrections above, the placement fix (§2), and the acceptance gates (§11).**

---

## 6. Q4 — Does KL-transparency kill Stage-4 steerability?

**No, and here is the mechanistic argument, plus the one real risk and its diagnostic.**

1. **An irreducible causal channel exists.** The first content token after the pauses is predicted *from the last pause position's residual stream*. Steering adds `α·δ` at layer 14/20 at pause positions; the perturbation at pause_K propagates through the remaining ~half of the network into the next-token logits exactly as it would at any position. No amount of embedding-row training can sever this — the LM head and the layers above L are frozen. And once the first post-pause token changes, autoregressive feedback propagates the intervention. So "too no-op to steer" is false in the strong sense.
2. **Transparency const

2. **Transparency constrains the value, not the Jacobian.** The KL term forces `p_θ(· | prefix+pauses) ≈ p_base(· | prefix)` *at the unsteered operating point* (δ=0). It says nothing about how fast the continuation distribution moves as you perturb the pause hidden states. A function can pass through the base point and still have large directional derivatives there. Transparency-training would reduce steerability only if the optimizer happens to find a *robustly* flat solution (e.g., pause keys that repel all later queries). With only `e_pause` trainable and the body frozen, its control over deep-layer pause keys/values is limited — pause residual streams at depth are dominated by attention *from* the pauses *to* the prefix, which `e_pause` cannot switch off.
3. **The one real risk: attention thinning.** The optimizer could shape `e_pause` so later tokens attend weakly to pause positions, making steering act mostly through the first-token channel (one-shot redirection) rather than sustained multi-token control. Given Stage 4's stated goal — redirect the trajectory *before* the unsafe CoT unfolds — first-token-dominant influence is arguably the intended mechanism, but measure it:
   - **Injection gain curve** (~2 GPU-hours): inject norm-controlled random δ at pause positions, plot continuation KL vs ‖δ‖; compare against the same injection at matched cot_k positions. If `dKL/dα` at pauses ≥ at ordinary positions, the port is live.
   - **Attention mass to pauses**: mean attention from continuation tokens to pause positions, per layer, v2-KL vs untrained-embedding arm.
4. **Fallbacks if the port is weak:** (a) lower the KL weight / stop training earlier along the transparency–steerability frontier (measure the frontier across λ, don't assume a corner — this is the steering analogue of the first review's R-B); (b) upgrade the port to a **per-layer pause prefix** (prefix-tuning-style K×L×d parameters living only at pause positions, trained under the same KL) — a much richer intervention surface, still body-frozen, still position-localized drift.
5. **Independent of Stage 2:** the Stage-4 δ is trained *after* the pause model is frozen, so it will find whatever channel remains. Replace the current NLL-disruption δ (`stage4_pause_steering.yaml:22-24`, which produced the non-monotone α response: WildGuard unsafe 0.364 → 0.411 @α=1 → 0.130 @α=2) with a probe-derived / difference-of-means (CAA-style) direction — monotone dose-response is itself evidence the channel is a direction, not a derailment.

---

## 7. Q5 + Q7 — Method ranking and better ideas

Ranked for the clarified requirement (self-emission wanted, minimal drift mandatory):

| Rank | Method | Trains | Verdict |
|---|---|---|---|
| **1** | **Rows-only KL-transparency + pause-slot CE** on self-generated corpus (§8) | `e_pause`, `u_pause` (2×d) | **Recommended.** Drift provably localized; emission included; cot3-style failure structurally excluded |
| 2 | Rank-1 + **per-layer pause prefix** (prefix-tuning port), same KL | + K×L×d at pause positions only | Adopt only if the §6 gain curve shows a dead port or intra-CoT emission fails; drift still position-localized |
| 3 | Full-model **pure-KL self-distillation** (no corpus CE) | all params | More capacity for transparency+emission, but drift no longer provable — needs the full panel; use only if 1–2 fail |
| 4 | **Template/logit-processor emission + read-only masked pauses** (no training) | nothing | Scientifically the cleanest for Stage 3 (drift ≡ 0); drops self-emission; run it as a control regardless — it deflates the emission requirement (§3.3) |
| 5 | Current format-only (full-response CE) | 2 rows | Reject as objective — cot3 falsified neutrality; keep existing ckpts as comparison rows |
| 6 | UltraFeedback full SFT | all params | Reject (§4) |
| — | DPO-style format preference | all params | Reject: whole-policy movement for a formatting goal |

**Better ideas beyond the proposal (Q7):**

- **Periodic / sentence-boundary checkpoints** (pauses every N tokens or at sentence ends, same KL recipe). Fixes §2 at the root, gives Stage 3 a *trajectory-conditioned* monitoring story ("risk read as reasoning unfolds") and Stage 4 a last-checkpoint-before-divergence intervention. Emission trigger (punctuation) is learnable by a single head direction. This is the strongest version of the project; cost is one more arm.
- **Harness backstop as a design element, not a workaround**: emission + insert-if-missing. Converts the emission-reliability liability (§3.5) into a non-issue and lets you report emission rate as a metric rather than defend it as an assumption.
- **Hidden-state stabilizer**: small L2 `‖h_t^L(x̃) − h_t^L(x)‖²` at 1–2 mid layers on post-pause tokens — helps KL converge and directly bounds what Stage-3 probes see.
- Not recommended: Quiet-STaR-style thought tokens (trains a mixing head — maximally invasive), soft-prompt-only emission without KL (that's the cot3 failure), logit-bias-only "emission" (fine, but call it forced insertion).

---

## 8. Recommended Stage-2 objective (concrete)

Setup: frozen base `p`; student `p_θ`, θ = {e_pause, u_pause} only. Corpus: 5–8k **self-generated** base rollouts `y ~ p(·|q)` at T=0.7, prompts = {benign math 3k, benign general 2k, borderline-safe 2k, unsafe-train-only 1k} (Stage-3 probe prompts disjoint). Insertion at m sampled CoT tokens (m=4 recommended; m=0 pre-CoT arm as prompt-risk control): `x̃ = q ⊕ y_{1:m} ⊕ P^K ⊕ y_{m+1:T}`. Teacher extended with `p̃(P|·) = 0` (equivalently: teacher logits over the original vocab only).

```text
L_KL   = (1/|T_post|) Σ_{t>m}   KL( p̃(·|q,y_<t) ‖ p_θ(·|x̃_<t+K) )      # transparency (full-logit forward KL)
L_pre  = (1/|T_pre|)  Σ_{t≤m}   KL( p̃(·|q,y_<t) ‖ p_θ(·|x̃_<t) )        # softmax-denominator guard
L_sup  = (1/|T_post∪pre|) Σ_t  −log(1 − p_θ(P | ·))                      # spurious-pause penalty (forward KL is only
                                                                          #  linearly sensitive to this — keep explicit)
L_emit = −(1/K) Σ_{j=1..K} log p_θ(P | x̃_<m+j)                          # emission CE, pause slots only
L      = L_KL + 0.1·L_pre + 1.0·L_sup + γ·L_emit                         # γ: 0 → 0.25 ramp; sweep {0.1, 0.3, 1.0}
```

```python
# one step; single weight copy, two forwards
y  = load_self_generated(q)                       # base rollout, T=0.7
xt = insert_pauses(q, y, m=builder_convention)    # audited convention (F6)
with no_grad(): T = logits(model, cat(q, y))[:, :V_base]        # teacher = same weights, no pauses
S  = logits(model, xt)                                          # student, pauses attended
loss = (kl(T[m:], S[m+K:]).mean()
        + 0.1 * kl(T[:m], S[:m]).mean()
        + 1.0 * spurious(S, pause_id, exclude=slots).mean()
        + gamma * ce(S[slots], pause_id))
loss.backward(); opt.step()                       # only e_pause, u_pause have grad
```

Hyperparameters: lr 5e-4–2e-3, 300–800 steps, batch 16–32 seqs, bf16; init `e_pause` = vocab mean, `u_pause` small. **Gated curriculum:** train γ=0 first → must pass the transparency gate (§11.2) → enable γ → must pass emission gate without breaking transparency. Fits on 1 GPU for 8B; develop on 1.5B first (matches the clarified Stage-1 model), replicate on 8B. Eval KL on held-out base rollouts *and* on-policy pause-model rollouts.

---

## 9. Q6 — Minimal experiment package (four proof obligations)

| # | Obligation | Design | Criterion | Cost |
|---|---|---|---|---|
| **E1** | Self-emission at correct position | Harness **OFF** (`insert_pause_after_cot_tokens=-1`), 500 prompts × 4 samples, T∈{0, 0.7}, math + SafeChain slices; ALSO run this on the existing cot4-fmt ckpt250 / cot3-full ckpt250 (their CE did include pause slots — they may already emit; currently unmeasured) | exact-K at target ≥99%; spurious pause <0.1% of tokens; P(pause) at slot K+1 <5% | ~0.3 GPU-day |
| **E2** | Unsteered ≈ base | (i) KL panel: median/p95 KL vs token-distance-from-pause, 1k held-out traces + 500 on-policy rollouts; greedy first-divergence index. (ii) Behavioral: GSM8K/MATH500 (3 seeds or greedy, paired bootstrap), 3-judge unsafe_valid on SafeChain eval, XSTest/OR-Bench over-refusal, refusal-phrase rate, CoT-length KS — as **equivalence tests** against pre-registered budgets, not "failed to reject" | §11.2–.3 | ~1.5 GPU-days |
| **E3** | Pause positions improve separability | Intra-CoT placement; same-prompt paired protocol; probes at {pause_j} vs matched {cot_k} vs same positions on frozen base (read-only arm); report against TF-IDF/length surface baselines | pause ≥ matched cot_k AND > prompt-only; paired (not unpaired) metric | ~1 GPU-day |
| **E4** | Steering reduces unsafe CoT without over-refusal/breakage | Probe-derived δ (diff-of-means) + learned δ; α∈{0,.5,1,2,4}; controls: norm-matched random ×3, shuffled-label δ; hardsafe set + one held-out unsafe source; metrics: 3-judge unsafe (absolute residuals), over-refusal, GSM8K under chosen α, format-validity/parse rate; mechanistic: gain curve + attention mass (§6) | §11.5 | ~2 GPU-days |

Total ≈ **5 GPU-days** on the 4×A100 node. E1 on existing checkpoints is the cheapest decisive experiment in the whole program — run it first.

---

## 10. Ablation matrix

| Arm | Trains | Corpus | Emission | Isolates | Priority |
|---|---|---|---|---|---|
| A0 base | — | — | — | reference | P0 |
| A1 base + forced pauses, vocab-mean embed, attended | nothing | — | forced | "insertion procedure alone" (missing F4 cell) | P0 |
| A2 read-only masked pauses (rollback) | nothing | — | forced | monitoring ceiling at drift ≡ 0 | P0 |
| A3 rows-only KL, γ=0 | 2 rows | self-gen | forced | transparency without emission | P1 |
| **A4 rows-only KL + emit CE** | 2 rows | self-gen | **self** | **the recommended Stage 2** | **P1** |
| A5 A4 + per-layer pause prefix | + prefix | self-gen | self | steerability port (only if §6 gate fails) | P2 |
| C1 no_pause_matched full SFT | all | trusted-CoT | — | legacy data-effect attribution | P2 (only if legacy numbers published) |
| C2 random-token ×3 insertion | nothing | — | forced | pause-identity vs any-token | P2 |
| Position sub-ablation | m ∈ {0 (pre-CoT control), 4, periodic} on A4 | | | §2 placement question | P1 |

Shared metric battery for every arm: E1 emission stats, KL panel, behavioral panel, E3 probe AUROC, E4 gain curve.

---

## 11. Acceptance criteria (pre-register these)

1. **Emission (A4):** ≥99% exact-K at target ±0 tokens, harness OFF, T=0.7, on both benign and safety prompt sets; spurious <0.1%; boundary P(pause) <5%.
2. **Transparency:** post-pause KL median ≤0.02 nats (p95 ≤0.1) at 5 tokens and median ≤0.05 by 20 tokens, on base *and* on-policy rollouts; pre-pause KL ≤1e-3; A2 continuations bitwise-identical to base.
3. **Behavioral equivalence (unsteered):** |Δ capability| ≤1.5 pts with 95% CI ⊂ ±2.5; |Δ unsafe_valid| ≤2 pts with CI ⊂ ±4, per judge; length KS D ≤0.05; refusal-phrase Δ ≤1 pt. A safety *improvement* fails this gate just like a regression.
4. **Monitoring value:** paired probe AUROC at pause positions ≥ matched cot_k AND > prompt-only baseline by ≥5 pts (intra-CoT arms only; pre-CoT arm is expected ≈ prompt-only and reported as such).
5. **Steering validity:** monotone α response for the probe-derived δ; random/shuffled controls ≤20% of the effect at matched norm; over-refusal within CI of base; capability under chosen α within the §11.3 budget; transfers across the 3 judges and one held-out source; absolute residual unsafe rates in every table.
6. **Port liveness:** injection gain `dKL/dα` at pause positions ≥ matched cot_k positions.

---

## 12. Do-not-claim list

1. No emission claim from any existing result — `pause3_rate = 1.000` is forced by the harness (verified at `run_model_comparison_generation.py:306-321`), and `generated_for_judge` strips whatever the model emitted.
2. "Unsteered ≈ base" cannot rest on the single scalar "unsafe CoT rate close" — necessary, not sufficient; the KL + behavioral panel is the claim.
3. **No trajectory-monitoring claim at pre-CoT pause positions** — those states are constant across samples of a prompt; any signal there is prompt classification (§2).
4. Stage-1 ~80% paired-clean does not establish *latent semantic* separability while TF-IDF/length baselines (0.94–0.97 / 0.83–0.88 AUROC) exceed hidden probes; report probes relative to surface baselines. Also: any-position Llama-Guard labeling makes labels length-correlated — keep the length-only baseline in every table.
5. Don't cite Goyal et al. for self-emission (their pauses are runtime-appended, outputs ignored) or for capability hopes (gains need pause-pretraining).
6. No "safety direction" claim from the current NLL-disruption δ — non-monotone α response, no random/shuffled controls, trained on the checkpoint selected *for* capability (the professor's confound).
7. Capability *improvement* under any Stage-2 variant is drift, not a bonus — both directions disqualify the neutrality claim.
8. If deployment uses the harness backstop or a template prefix, say so — do not present it as pure self-emission.

---

## 13. Highest-priority implementation tasks

1. **Eval-harness fix (no GPU):** log the raw response before `strip_pause_tokens`; add natural-emission metrics (position histogram, K-count) to every run; keep the F6 position-convention audit.
2. **E1 emission audit on existing checkpoints** (cot4-fmt ckpt250, cot3-full ckpt250), harness OFF — cheapest decisive result in the program.
3. **A1/A2 zero-training arms** (forced neutral-embed; read-only rollback with bitwise-identity check).
4. **New trainer branch** `kl_transparent_emit` (rows-only, teacher-mask util, drift-panel module) + self-generated corpus build (5–8k, mixed families, manifest).
5. **Train A3 → gate → A4** on 1.5B, then 8B; γ sweep.
6. **E3 probe comparison** {A2, A4, legacy format-only cot4} at intra-CoT positions, paired protocol, surface baselines.
7. **E4 steering rerun** on A4 with probe-derived δ + controls; gain-curve precondition first.
8. Retire UltraFeedback from Stage 2; run C1 `no_pause_matched` only if legacy full-SFT numbers will be published.

---

## 14. Where this review differs from the first Fable review

- **Agreements, independently verified:** forced insertion (F1), format-only-as-soft-prompt (F2), full-SFT-as-data-effect (F3), the missing 2×2 cells, the Stage-4 δ concerns. The proposed objective in the follow-up is essentially the first review's Method C + emit — I endorse that convergence.
- **New here:** the pre-CoT information-theoretic ceiling (§2) — the first review analyzed the intra-CoT design, where this cannot arise; the clarified plan's placement makes it the top-priority issue, above any training question. Also new: the teacher-forcing-equivalence and adversarial-evasion arguments against self-emission-as-causal-claim (§3), the forward-KL insensitivity justification for the suppression term (§5.1), the trigger-position mismatch in the UltraFeedback plan (§4.2), and the value-vs-Jacobian argument plus gain-curve gate for Q4 (§6).
- **Mild disagreement:** the first review's read-only Method A is the right Stage-3 default but should not be framed as competing with the trained arm for Stage 4 — it cannot serve steering at all (masked pauses have no causal channel). The two are complements: A2 for monitoring claims, A4 for steering claims, with §11.6 deciding whether A4's port is live.

**One-sentence summary:** the proposed pause-only CE + KL-to-base objective is the right Stage-2 design and should be adopted with a self-generated safety-inclusive corpus, gated training, and rows-only parameterization — but move the pauses inside the CoT or Stage 3 cannot demonstrate anything beyond prompt classification, and treat self-emission as a deployable feature with a harness backstop rather than a scientific requirement.


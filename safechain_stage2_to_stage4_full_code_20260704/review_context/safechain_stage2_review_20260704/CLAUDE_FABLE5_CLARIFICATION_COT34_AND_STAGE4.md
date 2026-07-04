I have everything needed. Here are the answers.

---

# Answers to Clarification Request (cot3/cot4 Placement & Stage4 Algorithm)

## Clarification 1 — Does the pre-CoT critique apply to cot3/cot4?

**1. Are cot3/cot4 the recommended intra-CoT placement?** Yes. The pre-CoT critique (§2 of the second review) was aimed at the *clarified follow-up plan's* diagram `prompt → pause_1..K → CoT` (m=0), not at the existing experiments. cot3/cot4 insert the pause block after 3/4 model-generated CoT tokens (`--insert_pause_after_cot_tokens 3` in `run_intra_pause_full_steering_eval.sh:297`; "before `cot_3`/`cot_4`" in the Stage2 summary), i.e., m=3/4 sampled tokens precede the pauses. The second review says this explicitly: §8 recommends "m ≈ 4 matches your Stage-1 signal position and the existing cot4 tooling," and §14 notes the first review analyzed the intra-CoT design "where this cannot arise." cot3/cot4 are instances of the recommendation, not of the problem.

**2. Exact condition for the severe critique.** The "only prompt classification" ceiling applies iff there are **zero stochastic events before the pause positions** — formally, I(pause hidden states; trajectory label | prompt) = 0. That happens when (a) m=0 (pauses immediately after the forced `<think>\n`), or (b) any placement evaluated under deterministic decoding (greedy/T=0) up to the pause, or (c) the pre-pause prefix is teacher-forced identically across compared samples. Then pause states are a function of the prompt alone: paired same-prompt accuracy = 0.5 exactly; unpaired accuracy is capped at Bayes-optimal `P(unsafe | prompt)`.

**3. Can Stage3 legitimately test trajectory separability at cot3/cot4 pauses?** Yes, in principle. With m=3/4 sampled tokens (T=0.7), pause states differ across the 10 samples of a prompt, so a paired probe is not degenerate. Whether the tokens are self-emitted or harness-inserted is irrelevant for hidden-state validity (teacher-forcing equivalence, §3.1): the state at a pause position depends only on the prefix, the pause embedding, and the weights — not on which process wrote the token. The information ceiling is I(prompt + first m sampled tokens; label), and Stage-1's ~80% paired result at early-CoT positions is your own evidence that this ceiling is well above chance.

**4. Remaining caveats — four real ones:**

- **Position convention (F6).** "before `cot_3`" vs `insert_pause_after_cot_tokens 3` vs Stage-1's `cot_4` indexing must be audited for 0- vs 1-indexing and whether the forced `<think>\n` counts as CoT tokens. Probe positions, training insertion, and steering targets must share one audited convention.
- **Forced insertion.** At eval the harness strips and re-inserts pauses (`run_model_comparison_generation.py:306-321`), so cot3/cot4 support **no emission claims** (E1 still unmeasured on these checkpoints), and steering deployed via these checkpoints is honestly "harness-inserted checkpoint," not self-emitted.
- **Teacher-forced vs self-generated distributions.** Stage2 training inserted pauses into external trusted-CoT continuations; Stage3/4 run on model-generated prefixes. For probing this is fine (probes are fit on actual rollout states), but `e_pause` was optimized off-policy, and — more important — the format-only cot3 checkpoints carry **+17–19 pt unsafe drift** and `cot4 full final` shows large safety/capability drift. The placement is conceptually aligned; the specific checkpoints fail the Stage-2 neutrality gate, so Stage3 results on them describe a drifted model. Prefer the A2/A4 arms for paper claims; report cot3/cot4 results with the drift caveat.
- **Low prefix entropy.** R1-distill openers are stereotyped ("Okay, so…"), so 3–4 tokens may carry limited entropy. Sanity-check that pause-state variance across the 10 samples exceeds numerical noise, and use sampled (never greedy) decoding for the paired protocol. m=4 or periodic checkpoints strictly dominate m=3 here (consistent with cot3 behaving worse in your summary).

## Clarification 2 — Reusing the previous Stage4 algorithm on KL-transparent pauses

**1. Is the general Stage4 idea still valid?** Yes, conditionally. KL transparency pins the continuation distribution at δ=0 (the value), not its sensitivity to perturbations at pause positions (the Jacobian). The last-pause → first-content-token channel is architecturally irreducible with a frozen body, and autoregressive feedback amplifies any first-token change. Condition: pass the port-liveness gate (§11.6) before committing compute.

**2. Reusable as-is?** No. The **infrastructure** is reusable: pause_0/1/2 targeting, layer hook, alpha-sweep/judge/summarize scaffolding, the eval battery in `stage4_pause_steering.yaml` (xstest/or-bench, GSM8K/MATH500, three judges — keep all of it). The **δ and its evidence standard** are not: (a) the delta checkpoint (`zero_l14_steps80/learned_delta.pt`) was trained against the legacy cot3 SFT model — a δ is model-specific and is invalid for a new KL-transparent model regardless of objective; (b) the NLL-disruption loss (`stage4_pause_steering.yaml:22-24`) is a derailment objective with degenerate optima (breaking text raises NLL) and produced the non-monotone α response (WildGuard 0.364 → 0.411 @α=1 → 0.130 @α=2); (c) no norm-matched random or shuffled-label controls exist anywhere in the config/script; (d) α grid {0,1,2} is too coarse to establish dose-response.

**3. Minimal changes required:**
1. Re-derive δ on the frozen A4 model (mandatory — old .pt is for a different model).
2. Primary δ = probe-derived difference-of-means (CAA-style) from Stage3 pause hidden states at the same layer/positions; keep learned-δ at most as a comparison arm.
3. Add control arms: norm-matched random δ ×3 and shuffled-label δ; pre-register "controls ≤20% of effect at matched norm."
4. Densify α ∈ {0, 0.5, 1, 2, 4}; require monotone response for the primary δ.
5. Run the gain-curve + attention-mass diagnostic first as a gating precondition (~2 GPU-hours).
6. Report absolute residual unsafe rates, over-refusal, capability at the chosen α, and format-validity/parse rate; don't select the checkpoint for capability and then claim safety (the professor's confound).

**4. Probe-derived / diff-of-means instead of NLL-disruption?** Yes, as the primary method. It tests the actual hypothesis (a safety *direction* exists at the checkpoint); monotone dose-response is itself evidence the channel is a direction rather than a derailment. NLL-disruption can remain as a secondary arm under the same controls, but no "safety direction" claim can rest on it (do-not-claim §12.6).

**5. What the diagnostics decide.** They are a go/no-go gate before the ~2-GPU-day E4:
- **Injection gain curve:** inject norm-controlled random δ at pause positions (layer 14/20), plot continuation KL vs ‖δ‖, compare `dKL/d‖δ‖` against the same injection at matched `cot_k` positions. Decision rule: pause slope ≥ matched-cot slope → port live → run full Stage4. Substantially below → don't run E4; go to fallback.
- **Attention mass:** mean attention from continuation tokens to pause positions, per layer, A4 vs untrained-embedding arm (A1). Decides the *mechanism claim*: healthy mass → sustained multi-token control is on the table; thinned mass → expect first-token one-shot redirection only, and scope Stage4 claims (and α range) accordingly.

**6. Fallback order if the port is weak.**
1. **Frontier scan first** (cheapest, no new parameters): sweep KL weight λ / earlier stopping — these are the same lever — and measure the transparency–steerability frontier; don't assume a corner. Constraint: only accept operating points that still pass the pre-registered transparency gates (§11.2–3). Dropping KL below that budget is not an acceptable fallback — it forfeits the "unsteered ≈ base" property that motivates the whole design.
2. **Per-layer pause prefix** (rank-2 method) if no frontier point passes both gates: K×L×d parameters living only at pause positions, trained under the same KL — a structurally richer port that keeps drift position-localized and transparency intact.

---

## Final table

| Question | Answer |
|---|---|
| cot3/cot4 affected by pre-CoT critique? | **No** — intra-CoT (m=3/4 sampled tokens precede pauses); critique targets m=0 / deterministic-prefix designs only |
| Stage3 valid on cot3/cot4? | **Conditional yes** — requires sampled decoding, F6 position-convention audit, per-sample state-variance check, paired protocol + surface baselines; existing checkpoints carry Stage-2 drift, so prefer A2/A4 arms for claims |
| Previous Stage4 *idea* valid on KL-transparent pauses? | **Conditional yes** — transparency constrains value, not Jacobian; must pass the port-liveness gate first |
| Previous Stage4 *algorithm* reusable as-is? | **No** — infrastructure/eval battery yes; δ must be re-derived on the new model, NLL-disruption objective replaced as primary, controls and denser α grid added |
| Minimum required changes before claiming Stage4 evidence? | Probe-derived (diff-of-means) δ on frozen A4; random + shuffled-label controls ≤20% of effect; monotone response over α∈{0,.5,1,2,4}; gain-curve/attention precondition passed; absolute unsafe rates + over-refusal + capability + format at chosen α; no capability-selected checkpoint |

**Bottom line:** your cot3/cot4 experiments are conceptually aligned with the intra-CoT recommendation — the pre-CoT critique does not apply to them — and the Stage4 *site and harness* carry over to a KL-transparent pause model, but the δ derivation and evidence standard must be upgraded (probe-derived direction, controls, monotone dose-response, port-liveness gate) before any Stage4 claim.
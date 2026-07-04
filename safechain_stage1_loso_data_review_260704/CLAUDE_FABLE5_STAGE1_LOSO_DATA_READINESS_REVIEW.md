# Fable5 Review — SafeChain Stage 1 LOSO Data-Readiness (packet 260704)

Reviewed from the local mirror (`FABLE_STAGE1_LOSO_DATA_READINESS_REQUEST.md` + README + prior-advice note), cross-checked against the pre-registration memo `rp_source_expansion_memo.md` in the repo root.

---

## 1. Verdict

**Conditional NO. The source counts are sufficient; the freeze is not.**

You have fixed the problem I blocked you on last round (LOSO impossible because eligible data was ReasoningShield-only): four families each clear 150 pairs, so a four-fold LOSO is arithmetically runnable. But "enough rows" != "data-ready." Three integrity problems must close before launch, and two of them are not on your own worry list:

**Blocker 1 — Adaptive sampling drift from the pre-registration.**
The memo froze the sampler at **k=50 rollouts/prompt** and a **WJB sample of 800 prompts** (~70k rollouts total, explicitly sized against the known-feasible 1,276-prompt run). The committed run shows:

- 163,210 candidate rows for the mixed run — more than 2x the pre-registered budget;
- ~3,000 WJB prompts implied (2,043 pairs + share of the 1,039 both-sides-missing drops), vs. the pre-registered 800;
- "Last committed sample range 105-109" in the mixed run and HarmThoughts actively on "round 20-24" — i.e., **multi-round, per-source-unequal, still-running sampling**, not a frozen k.

Why this is fatal *specifically for LOSO*: pair eligibility requires >=1 judged-safe and >=1 judged-unsafe rollout for the same prompt. Under adaptive budgets, "a pair exists" partly encodes *how long you kept sampling*, and the per-prompt unsafe-rollout prevalence distribution becomes a **source-correlated artifact**. A probe can then win "cross-source transfer" by keying on generation-budget signatures rather than anything about the trajectories. The yields already hint at this: old RS yield ~37% (467/1,276), WJB yield ~68% — either genuine source bistability or a budget difference; you currently cannot distinguish the two, and that ambiguity sits directly under your headline claim.

**Remedy:** at freeze, re-select pairs using **only the first N rollouts per prompt, identical N for every source** (N=50 per the memo, or pick one N and file a written pre-registration amendment). Report per source: prompts attempted, rollout-count distribution, yield under fixed budget, per-prompt unsafe-prevalence distribution. If a family drops below floor under fixed N, *that* is your real missing-data number — see Section 2.

**Blocker 2 — "0 duplicate edges" is a red flag, not a clean bill.**
Across 2,928 pairs the audit reports zero duplicate clusters, zero cross-source clusters, zero edges. Your own memo says ReasoningShield is aggregator-derived (SALAD-lineage contamination is why you dropped AdvBench) and that StrongReject's curation "drew partly on older sets -> *must* pass through quarantine." Some cross-source hits were *expected*. A flat zero is what a misconfigured dedup looks like (wrong field compared, embedding failure, thresholds never actually applied) — or it means the pre-generation quarantine already removed the hits, in which case this audit **proves nothing** and the evidence lives in the upstream quarantine log, which the packet does not report.

**Remedy:** before freeze, produce (a) the full pairwise-similarity histogram, (b) the count of cross-source pairs in the 0.80-0.90 cosine band (just below threshold — a zero *there too* would confirm misconfiguration), (c) a manual eyeball of the top-50 cross-source nearest neighbors with scores, (d) upstream pre-generation quarantine counts per source.

**Blocker 3 — The promised human QA is absent.**
The memo commits to ~50 stratified human spot-checks per source with an agreement table. The packet reports none. This is not bureaucratic: ~154k judge-passed rows were mined for comparatively rare unsafe positives. When you mine that many candidates with a noisy judge, the selected "unsafe" pair-members are enriched for **judge false positives**, and the enrichment scales with per-prompt budget — which (Blocker 1) differs by source. A probe separating "real unsafe CoT" from "judge noise labeled unsafe" at different rates per source would pass your gates while measuring nothing you care about. **No spot-check agreement table, no freeze.**

Also: yes, the stale audit (2,461 vs 2,474 mixed-source pairs) must be rerun on the exact final selection. You flagged this yourselves; it is necessary but the *least* of the four items.

## 2. Missing Data By Source

**No mandatory new collection**, with explicit contingencies:

| Source | Current | Target | Ruling |
|---|---:|---:|---|
| ReasoningShield | 335 | 335 | Sufficient — **if** you prove the old 467-export used the same sampler config, judge, and comparable effective budget as the new runs. If not, regenerate RS prompts under the frozen protocol rather than papering over a source-by-protocol confound. |
| StrongReject | 277 | 277 | Sufficient. 277/313 prompts = 88% yield — there is no headroom to collect more even if you wanted it. |
| HarmBench | 154 | 154 | Do not collect more. 154 pairs from 200 standard behaviors = 77% of the entire universe. More HarmBench means importing contextual/copyright behaviors, a distribution change strictly worse than tolerating a +/-0.06-0.07 CI. |
| WildJailbreak | 2,043 | <=700 per training split | Over-supplied. The problem is imbalance, not volume. |
| HarmThoughts | 315+, growing | irrelevant to launch | Not needed for the four-fold LOSO. Its ongoing generation must not delay the mixed-source freeze. |

**The one scenario that creates a real gap:** if the fixed-budget re-selection drops any family below 150 pairs, downgrade that family to the 100-pair pilot floor and label the entire run **pilot-grade** — do not attach paper claims to a sub-150 fold. If a family falls below 100 under fixed N, LOSO is re-blocked for that family and the deficit is the collection target, generated under the frozen protocol only.

## 3. LOSO Fold Construction

Keep F1-F4 as planned (hold out RS / SR / WJB / HB respectively), with these hard requirements:

1. **Group by prompt everywhere.** The packet constrains only eval to one pair per prompt; state explicitly whether training has one pair per prompt globally or prompt-grouped split disjointness between train/val/test.
2. **Validation stratified per family**, grouped by prompt, drawn from training families only. Additionally report per-family validation AUC, or WJB mass silently dominates model selection in F1/F2/F4.
3. **F3 is the weakest-power fold**: trains on only ~612 pairs (RS 335 + SR 277), tests on 2,030. Write now that interpretation weights F1/F2/F4 more heavily, so it cannot look post hoc.
4. **F4 trains on three families** while F1-F3 train on two; report per-fold train composition.
5. Fixed seeds; fold manifests with pair IDs, per-fold per-family counts, and hashes, committed before the first probe run.
6. Text/surface baselines must use identical fold manifests and validation-selection discipline as the probe.

## 4. HarmBench And HarmThoughts

**HarmBench: keep test-fold-only. Do not promote to symmetric training family.**

Reasons: its imperative house style is a trainable shortcut; putting it in training risks teaching a style detector; and 154 pairs add no meaningful training mass. n=154 gives AUC SE about 0.03, so 95% CI is about +/-0.06-0.07 against a 0.68 gate: wide but workable. Symmetric inclusion buys nothing and adds risk.

**HarmThoughts: reserve as the permanently unseen final external test. Do not put it in LOSO. Do not wait for it.**

- It is your only zero-adaptive-exposure eval set. Spending it in fold rotation destroys that value.
- Its ongoing generation does not block launch; freeze HT separately later under the same fixed-budget rule and frozen judge, and never peek until final confirmatory run.
- The 132 dropped ambiguous `harmthoughts+reasoningshield` pairs are direct evidence of HT-RS prompt overlap. Before HT is ever scored, it must be deduped/quarantined against all four LOSO families, not just RS.

## 5. WildJailbreak Imbalance Handling

**Yes — cap it, and pre-register the cap before any result exists.**

- **Primary:** prompt-level, fixed-seed subsample of WJB to **<=700 pairs in any training split** (~2x RS, keeps WJB <= ~60% of training mass in every fold). Commit the subsample manifest at freeze.
- **Sensitivity:** full WJB with inverse-family-frequency weighting in probe/baseline training objective. Report this, but never as the headline.
- **F3 test side:** evaluate on full 2,030 held-out WJB pairs.
- **Validation:** cap applies to validation composition too, because validation is drawn from training families.

Pick one primary and freeze it. Uncapped, F1 training is 88% WJB and "held-out RS transfer" becomes "does a WJB probe transfer," which is a weaker claim.

## 6. Required Final Freeze/Audit Checklist

All items complete before the first probe or baseline run; freeze declared in a single commit; no re-freeze after any fold result is observed.

1. **Fixed-budget re-selection:** first-N rollouts/prompt, identical N across all sources; per-source table of prompts attempted, rollout distribution, yield, per-prompt unsafe-prevalence; written amendment for every deviation from the memo.
2. **Dedup/quarantine audit rerun** on the exact final selection: similarity histogram, 0.80-0.90 band count, top-50 cross-source nearest-neighbor manual review, upstream pre-generation quarantine counts.
3. **Human spot-check agreement table:** ~50/source, stratified approximately safe/unsafe, with per-source judge-agreement rates. Pre-set an acceptability bar, e.g. >=90% agreement on the unsafe side.
4. **Provenance hashes:** judge model/prompt/threshold hash, generator/sampler config hash; explicit confirmation the old RS export matches, or RS regenerated.
5. **Fold manifests:** pair IDs, per-fold per-family train/val/test counts, seeds, WJB-cap subsample manifest — committed and hashed.
6. **Per-fold token-window and caliper counts:** k = 4...160 coverage per fold; caliper decision fixed now. With only 260 pairs at 0.9 caliper across all sources, per-fold length-matched eval is underpowered, so pre-register 0.8 caliper (549 pairs) as matched eval or replace with length-stratified reporting against the length-only baseline.
7. **HT quarantined against all four LOSO families**; HT freeze explicitly deferred and documented as untouched.
8. **Baseline parity:** surface baselines bound to the same fold manifests and validation-selection rules as the probe.

## 7. Gate/Claim Changes

- **Raise per-fold floor from 80 to 150.** You meet 150 everywhere today; leaving the floor at 80 creates unearned slack. If fixed-budget re-selection forces it, 100 = pilot floor and the run is labeled pilot-grade.
- **Add Gate 0 (freeze integrity):** checklist items 1-5 of Section 6 pass before any probe/baseline number is admissible. Gates A/B on unfrozen data are void.
- **Gate A** and **Gate B** unchanged. The asymmetry — validation-selected probe vs. test-max surface adversary — is stacked against you, correctly.
- **Truncation curves must run under LOSO folds** with probe and surface both validation-selected, same tokenizer, k = 4...160.
- Gates 2/3/4 unchanged, but must run on the frozen dataset; results from earlier compositions do not carry over.
- **Allowed claim:** add a load-bearing qualifier: "held-out prompt-source separability beyond strongest surface baselines **under a frozen generation budget and frozen judge**, under this protocol."
- Forbidden claims unchanged: no safety semantics, no pooled/test-max evidence, no 0.926/0.927 revival.

## 8. Minimal Next Actions

1. Stop treating 2,474 as final; run fixed-budget re-selection (first-N rollouts, same N all sources) and produce the per-source budget/yield table. This decides whether any real data gap exists.
2. Verify dedup: histogram + 0.80-0.90 band + top-50 neighbors + upstream quarantine counts.
3. Run 50/source human spot-checks and commit the agreement table with a pre-set acceptability bar.
4. File the pre-registration amendment: budget deviation, WJB cap <=700 primary, floor 150, caliper 0.8, truncation-under-LOSO, F3 down-weighting; then rerun the freeze audit on the final selection.
5. Commit fold manifests + hashes in one freeze commit, then launch probes and baselines together.
6. Let HarmThoughts finish in the background; quarantine it against all four families; do not look at it until final confirmatory run.

**Bottom line:** the collection effort worked — you are one honest freeze away from a runnable LOSO, not more scraping. But the run drifted from its own pre-registration, the dedup audit currently proves nothing, and the promised human QA is missing. Close those three and launch. Launch without them, and the LOSO numbers inherit exactly the credibility problems that killed the 0.926 era.


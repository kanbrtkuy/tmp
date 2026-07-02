Review complete. I read the full script, the summary JSON, and audited the request file for truncation (counts only, no content). Direct answers below, plus two findings you didn't flag.

## Two findings you didn't mention

**F1 — 70% of "original unsafe trajectory" fields were truncated at 520 words (1833/2617), and 19% of paraphrases too.** This directly undermines two judgments:
- `added_dangerous_detail` compares paraphrase vs original. When the original is clipped, detail that exists past word 520 of the original looks "new" → false positives. Your 40 `added_dangerous_detail` drops are the single largest drop reason and are exactly the judgments most exposed to this artifact.
- `overcompressed_semantic_loss` = 0/2617 is not reassuring when the judge never saw the full original for 70% of rows — it *can't* detect loss of content it never saw.

**F2 — the 1–5 alignment gates are dead code in practice.** All three alignment means are exactly 5.0 with zero rows ≤3, and `template_dominance` hugs the example value 1. That's a degenerate distribution: the judge is copying the schema example values (5, 5, 5, 1). So your effective keep rule is really just four booleans. This is the same anchoring pathology as `usable_for_*`, just in the "agreeable" direction, so it silently passed everything instead of failing everything.

Also: `deterministic_metrics` computes `safe_refusalish`/`unsafe_refusalish` per row (audit_openai_full_ab.py:236-237) but `summarize_deterministic` only aggregates the five numeric keys — the refusal-marker booleans never appear in any report. That's the cheapest arbiter you have for the safe-mode dispute and it's currently discarded.

## Your questions

**1. Dropping `usable_for_primary_A/B` as gates — correct?** Yes, unambiguously. The explicit fields agreed 187–200/200 across two independent prompts; the holistic fields flipped 150/200→14/200. A field that unstable measures prompt design, not data. One requirement: this was a post-hoc rule change made after seeing the data, so record it explicitly — version the keep rule (v1.1), note the change and its justification in `manifest_hashes.json` or a sidecar decision log. The comment in `keep_decision` is good but not provenance.

**2. Is the revised keep rule reasonable?** Structurally yes, but understand what it actually is: four booleans (the alignment/`≥4` gates never fire, per F2). Given F1, the boolean most likely to be wrong is `added_dangerous_detail` on truncated rows. Don't loosen or tighten the thresholds — instead fix the inputs (see repair plan). Not gating on `major_asymmetry` is right; but note your deterministic stats show the asymmetry is *systematic*, not rare: unsafe/safe word ratio mean ≈1.4, line ratio mean ≈0.48 (safe side has ~2× the lines), 29% of rows outside the 0.5–2 sentence-ratio band. Length/format-matched controls at probe time are mandatory, not optional.

**3. Trust the combined prompt's safe-mode labels?** No — and don't trust the earlier sample audit either. 93% `refusal_style` vs 80% `safe_completion_style` on overlapping data means at least one judge run is measuring the prompt, not the text. Two plausible mechanisms: (a) enum order — `refusal_style` is listed first in both the instruction and the schema string; (b) contrast effect — in the combined prompt the judge reads the safe rewrite immediately after an unsafe trajectory, making anything safety-oriented look "refusal-like." Arbitrate cheaply: (i) surface the already-computed refusal-marker stats (though your `REFUSALISH_RE` includes "instead"/"safe alternative", which safe-completions also trigger — split the regex into hard-refusal markers vs redirect markers); (ii) you personally read ~30 safe rewrites and label mode by hand — that's the ground truth both judge runs get scored against. This matters more than the keep/drop dispute: if the safe class really is 93% refusal register, any separability result is at risk of being a refusal-register detector.

**4. Combined vs split prompts for the self-consistency check?** Split — but run **both** on the same rows: (a) split single-task prompts (unsafe_quality / safe_mode / pair_alignment), (b) a rerun of the exact combined prompt. That decomposes disagreement into test-retest instability (combined vs combined, temp is already 0 so this measures API nondeterminism only) vs prompt-design bias (combined vs split). Sampling: 50 is thin for three constructs; go to ~100–150 and deliberately oversample (a) all 60 dropped rows, (b) truncated rows, (c) the 3 `major_asymmetry` rows. Random keep rows tell you almost nothing — the boundary rows are what the keep rule acts on. Report per-field Cohen's κ, not raw agreement (with 99% base rates, raw agreement is inflated).

**5. Remove default values from the JSON schema?** Yes — your own data is the evidence: every example value came back as the modal answer (all alignments = 5, `usable_* ≈ false`, template_dominance ≈ 1). Replace with placeholders (`"<true|false>"`, `"<1-5>"`). Better: use structured outputs (Responses API `text.format` with a JSON schema) so you never ship a filled example at all, and the enum order lives in a schema rather than prose. Also randomize or alphabetize enum listing order in the instructions.

**6. Manifest sufficiency?** Rows are good (per-field sha256 + stable payload hash + deterministic metrics). Gaps:
- No judge provenance in `manifest_hashes.json`: add batch_id, model string, sha256 of `SYSTEM_PROMPT + COMBINED_AUDIT_TEMPLATE`, keep-rule version, git commit, and the truncation params (220/520/120).
- No hash of the **original** unsafe trajectory — you hash the paraphrase but the original is the provenance anchor for the whole counterfactual design.
- Add a per-row `audit_input_truncated: {original, paraphrase, prompt}` flag (derivable from word counts, no API needed) so downstream analyses can stratify by F1 exposure.
- `row_payload_sha256` includes `audit_keep`/`audit_reject_reasons`, so hashes churn whenever the keep rule changes — fine, but that's another reason to version the rule.

**7. Exact next actions before CPU baselines, in order:**
1. **Local, no API:** add truncation flags + judge provenance to manifests; surface refusalish counts (split into hard-refusal vs redirect markers) in the summary; version the keep rule.
2. **Human spot-check:** ~30 safe rewrites for mode labels (you, not a model). Half an hour, resolves Q3's ground truth.
3. **Self-consistency batch (one batch, ~100–150 rows):** split prompts with schema-anchoring fixed + combined rerun, oversampled as in Q4. Score κ per field against both prior runs and your hand labels.
4. **Conditional re-audit:** if `added_dangerous_detail` is unstable on truncated rows, re-audit only the long rows with `--max-text-words` raised to ~2000 (gpt-4.1-mini handles it; cost is trivial), then re-freeze as v1.2. If stable, freeze v1.1 as-is.
5. Then CPU baselines on the frozen manifests — and given the deterministic asymmetry stats, the first CPU baseline should include a length/format/refusal-marker-only classifier as the shortcut floor before any semantic baseline.

## Is the combined audit "contaminated"?

Partially, not wholesale. Three strata: (1) **explicit booleans** — near-perfect cross-prompt agreement, trustworthy, keep; (2) **scale fields** — degenerate/anchored, uninformative but harmless (they gate nothing in practice); (3) **holistic `usable_*` and `rewrite_mode`** — anchoring-driven, treat as unusable until re-measured. The repair is the targeted plan above, not a full redo. The one thing I would **not** do is start CPU baselines before step 2–3 resolve the safe-mode question, because it determines whether your A′ safe class has a register confound baked in.

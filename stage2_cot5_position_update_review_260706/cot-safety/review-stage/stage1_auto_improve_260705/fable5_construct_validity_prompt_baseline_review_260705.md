# Fable-5 Review: Construct-Validity Prompt-Baseline Packet

Reviewer: `claude-fable-5`

Date: 2026-07-05

Packet:

- `tmp/stage1_auto_improve_loop_260705/round4_construct_validity_prompt_baseline_packet/`
- tmp commit: `4039bdf`

Verdict:

```text
CONSTRUCT_VALIDITY_CONCERN_SUBSTANTIALLY_RESOLVED_IN_DOWNGRADED_FORM
STOP_CURRENT_STAGE1: UNCHANGED
FRESH_PREREG_ONLY: UNCHANGED
```

## Raw Fable-5 Response

```text
All four documents and the review state are read. Here is the Fable-5 review.

---

# Fable-5 Review — Round 4: Construct-Validity Prompt-Baseline Packet (2026-07-05)

VERDICT: CONSTRUCT_VALIDITY_CONCERN_SUBSTANTIALLY_RESOLVED_IN_DOWNGRADED_FORM
STOP_CURRENT_STAGE1: UNCHANGED (not reversed)
FRESH_PREREG_ONLY: UNCHANGED
WRITE_UP_ONLY actions authorized below. No runs.

## 1. Sub-item rulings

| Professor sub-item | Ruling | Basis |
|---|---|---|
| Add prompt-only / pre-CoT probe baseline | Solved | Stage1b implements `last_prompt_token` / `pre_think` / `think_last` as first-class positions; natural runs add prompt-only TF-IDF. |
| "Pre-`<think>` probe might already hit ~0.97" | Solved (refuted), with a disclosure obligation | In the same-prompt natural pairs the feared mode cannot and does not occur: prompt-only TF-IDF 0.5000, best prompt/pre-CoT hidden 0.5000–~0.51. But in the old source-family Stage1b, prompt-only is only modest on heldout (mean 0.785/0.801) while in-domain test mean is 0.92–0.93 and test best 0.945–0.952 — close to the professor's feared number. The packet quotes only heldout means; the paper must disclose the in-domain level too. |
| Trajectory adds signal beyond prompt | Solved for the weak claim in the same-prompt setting; partial on the frozen set | Natural pairs: pre-CoT at chance, CoT-position hidden validation-selected 0.6983 / 0.7188 / 0.7890 (early-CoT Stage1b), up to 0.83 late-CoT Stage1. Frozen source-family set: heldout mean increment +0.06–0.12, in-domain only ~+0.04, and prompt-label correlation is built into that set. |
| Trajectory-monitoring framing intact | Not solved — framing stays downgraded | Ruling out "prompt classification" does not rule in "latent trajectory monitoring." On the same natural CoTs, length-only is 0.83–0.88 and word TF-IDF 0.94–0.97 AUROC; hidden never beats surface anywhere in Stage1 (all 16 deltas negative; lead-time confirmations null/negative). |

Two points of rigor the packet under-weights:

- The 0.5000 is by construction, not a discovery. In same-prompt pairs the two arms share identical prompt tokens, so pre-CoT teacher-forced hidden states are (numerics aside) identical and prompt-only classifiers are at chance definitionally. This is the strongest possible control — a design guarantee rather than an empirical baseline — but the paper must say "at chance by construction; empirical 0.50–0.51 confirms correct implementation," not present it as a surprising empirical win.
- Test-max discipline. The "0.73–0.81" range in the question is test-max. The citable early-CoT range is validation-selected: ~0.70–0.79. Several natural runs show large test-max vs val-selected gaps (e.g., 0.8841 → 0.7488), i.e., scan overfitting is real. Per the standing rule: never cite test-max in the main claim.

Also flag for correction in write-up: the 260630 LOSO table's "trajectory - prompt gap" column does not equal the difference of its adjacent columns (e.g., RS-test 0.932 vs 0.933 listed as +0.079), and the "prompt-baseline module best" column contains CoT positions (`cot_7`, `cot_4`, `assistant_last`). Either define that column precisely or drop it; as printed it will confuse exactly the professor it is meant to answer.

## 2. Effect on STOP_CURRENT_STAGE1

No change. STOP_CURRENT_STAGE1 and FRESH_PREREG_ONLY stand. I do not reverse them.

The STOP verdict was driven by the failed strong claim — hidden vs surface/length superiority (all 16 deltas negative, length-only wins on all four sources, matched-horizon negative k>=8, k=4 hint null under excluded-source confirmation, M1 kill rule fired). The prompt-baseline evidence addresses a different threat: whether the surviving weak positive (above-chance decodable signal) is an artifact of prompt classification. It shows that threat is not realized. That protects the weak claim from one more objection; it supplies zero evidence toward the strong claim, and therefore cannot reopen the frozen set. If anything it sharpens the closure: signal beyond prompt exists, and it still loses to surface text.

Consequently, no improvement-seeking runs on frozen Stage1 are recommended or permitted.

## 3. Remaining limitations (must appear in write-up)

1. Setting mismatch. The primary evidence (natural same-prompt pairs) is a different data construction from the frozen source-family Stage1 set the professor's question targeted. On the frozen set itself, prompt-conditioned signal is substantial (heldout mean ~0.79–0.80; in-domain ~0.93); the correct claim is "not reducible to prompt classification," not "prompt-free."
2. CoT-internal confounds unresolved. Length, style, refusal-pattern, and generation/selection artifacts remain: length-only 0.83–0.88, word TF-IDF 0.94–0.97 on the same pairs. Cannot claim clean latent safety semantics.
3. Small n, no CIs quoted. Natural test splits are 36–67 pairs; val-selected AUROCs of ~0.70–0.79 carry wide intervals. Report n and paired bootstrap CIs.
4. Provenance gap. Natural 8B gen/gen collapses to ReasoningShield-only after dropping 132 ambiguous pairs (335 eligible) — no LOSO within the natural setting; cross-source generality of the construct-validity result is unknown.
5. Teacher-forced only. On-policy behavior is untested (Stage2 territory).
6. Diagnostic rows stay quarantined. A-prime rewrite (~0.98) and `cot_0`≈1.000 families are artifact diagnostics, not evidence.

## 4. Safe paper wording

Executor's proposal is acceptable in substance; tightened version:

> In natural same-prompt pairs, the safe and unsafe arms share an identical prompt, so prompt identity carries no label information by construction; empirically, prompt-only TF-IDF and the strongest prompt/pre-CoT hidden probes are at chance (0.50–0.51), confirming the control. Validation-selected CoT-position hidden probes on the same pairs reach approximately 0.70–0.79 AUROC. The early-CoT signal is therefore not attributable to prompt classification alone. However, surface controls on the same CoTs remain strong (length-only 0.83–0.88; word-level TF-IDF 0.94–0.97), and hidden probes do not outperform matched or full surface baselines anywhere in Stage 1. We therefore report the Stage 1 result as above-chance decodable signal beyond the prompt — not as latent safety semantics, hidden-state superiority, or a stable trajectory-monitoring advantage. In the earlier source-family setting, where arms have different prompts, strict prompt-only probes are themselves strong (held-out mean AUROC 0.785–0.801; in-domain test mean 0.92–0.93), and early-CoT positions add roughly +0.06–0.12 held-out mean AUROC; we treat that setting as secondary because prompt and label are correlated there.

Forbidden phrasings: "proves the probe reads reasoning, not the prompt"; "trajectory monitoring works"; any number from a test-max cell; quoting the 260630 heldout means without the in-domain levels.

## 5. Allowed actions on current frozen Stage1 (exhaustive; write-up scope only)

1. Assemble the construct-validity mini-table from existing aggregates: natural same-prompt as primary (prompt-only TF-IDF 0.5000; pre-CoT hidden 0.5000/~0.51; val-selected CoT cells 0.6983 / 0.7188 / 0.7890), 260630 source-family results to appendix with both in-domain and heldout prompt-only levels. Missing cells are marked N/A — never filled by new runs.
2. Paired bootstrap CIs on already-produced predictions for exactly the prespecified val-selected cells — symmetric reporting only, no reselection of position/layer, results reported regardless of favorability.
3. Documentation corrections: fix/define the LOSO gap column and the mislabeled "prompt-baseline module" positions; add the by-construction note; add n per split.
4. Coverage-gap documentation: record the natural 8B provenance collapse (335 eligible, RS-only) and the small-n limitation.
5. Update the professor-status doc to the rulings above (sub-items 1–3 solved/solved-with-disclosure/partial; framing stays downgraded) and the final write-up with the wording in §4.

Explicitly not authorized: any new probe family/position/layer/head/pooling on the frozen set; folding the new generation inventories (406/607 32B pairs, 746 8B pairs) into Stage1 post hoc; any new GPU scan framed as Stage1 evidence. Those belong exclusively to the fresh preregistered path (Stage2/on-policy, or a preregistered source-balanced natural-pair confirmatory design), per the standing verdict.

---

Bottom line: The prompt-classification failure mode is refuted where it can be tested cleanly, so the professor's third concern is substantially resolved — but only in the downgraded form "above-chance signal beyond the prompt." It does not touch the failed strong claim, so STOP_CURRENT_STAGE1 stands, and the only permitted response is table assembly, CI reporting, and write-up as itemized above.
```


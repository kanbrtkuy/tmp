# Stage1 Decision Addendum

Date: 2026-07-05

This addendum is a follow-up review request for the existing sanitized packet:

- GitHub repo: `https://github.com/kanbrtkuy/tmp`
- Packet path: `stage1_fable_review_retune12288_b20_260705/`
- Previous packet commit: `ec5af66`
- Current packet adds this decision request only; it still contains no raw
  prompts, raw CoTs, completions, or row-level prediction JSONL files.

## Why This Follow-Up

The previous review judged the post-GPU execution acceptable and the result a
robust negative/control result for the current Stage1 hidden-probe design. We
now need a sharper decision:

1. Should we accept Stage1 as a negative/control finding and stop spending GPU
   on this exact design?
2. Or is there enough evidence that a redesigned probe is scientifically worth
   another GPU run?
3. Is the poor Stage1 performance more likely caused by data quality / dataset
   construction, or by the current method / probe design?

## Relevant Aggregate Facts

- Fixed-budget LOSO freeze uses N=100 sample budget where possible.
- Four LOSO holdout sources are included: HarmBench, ReasoningShield,
  StrongReject, and WildJailbreak.
- Prepared splits are balanced within label for each source/split.
- Post-GPU row audit checked 4464 prediction files.
- Row audit found 133 mismatching prediction files, localized to Stage1 linear
  high-CoT-offset positions (`cot_96`, `cot_128`).
- Stage1b and multilayer outputs did not show the same mismatch pattern.
- Hidden-minus-surface delta CI completed all 16 comparisons with
  `n_bootstrap_valid=2000`.
- All 16 validation-selected hidden probes underperform the validation-selected
  surface baseline on test AUROC.
- Representative hidden-minus-surface AUROC deltas:
  - HB Stage1 linear: -0.1259, CI [-0.1677, -0.0853]
  - RS Stage1 linear: -0.2141, CI [-0.2492, -0.1814]
  - SR Stage1 linear: -0.1229, CI [-0.1552, -0.0944]
  - WJB Stage1 linear: -0.0927, CI [-0.1031, -0.0816]
- Human QA remains unannotated, so all scientific claims are still provisional.
- S-to-S safe-prompt diagnostics and HT quarantine/external testing remain
  outstanding formal checks.

## Files To Inspect

- `fable_response.md` for the prior review conclusion.
- `delta_ci/hidden_surface_delta_ci_summary.tsv` for all 16 hidden-vs-surface
  deltas.
- `val_fixed/val_fixed_probe_report.tsv` for validation-only selected hidden
  probes.
- `row_audit/stage1_prediction_row_audit_summary.json` for split sizes and row
  audit status.
- `text_baselines/<source>/summary.tsv` and `metrics.json` for surface
  baseline behavior.
- `docs/stage1_post_hb_retune12288_b20_gap_audit_260705.md` for the local
  audit narrative.

## Specific Questions For Fable

Please answer as a strict senior ML reviewer.

1. Given the current evidence, should the project accept Stage1 as a
   negative/control result for the current hidden-probe design, or redesign the
   probe and spend another GPU run?
2. If you recommend redesign, what is the minimum credible redesign that could
   change the conclusion? Please distinguish changes that are scientifically
   motivated from blind hyperparameter fishing.
3. Is the weak hidden-probe performance more likely a data problem, a probe
   design/method problem, or a claims/expectation problem? Please assign rough
   probabilities if useful.
4. What evidence in this packet argues against "the data are simply bad"?
5. What evidence, if any, argues that the data still need fixing before any
   redesign run?
6. If we choose to accept the negative/control result, what claims are still
   allowed after human QA + S-to-S + HT checks?
7. If we choose to redesign, what exact non-GPU audits should gate the next GPU
   run?

Please be decisive. A useful answer should end with one of:

- "Accept negative/control; no more GPU for current Stage1 design."
- "Redesign probe, but only after gates X/Y/Z."
- "Data are too compromised; fix data before interpreting Stage1."

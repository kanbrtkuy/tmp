# Fable-5 Final Verdict on the E1--E6 Amendments

Date: 2026-07-13

## Final verdict

`READY_TO_IMPLEMENT`

## Per-stage status

- **Stage2: READY_TO_IMPLEMENT.** E6 is closed: mean-direction/median-norm
  pause-row initialization with checksums; pause positions explicitly unmasked
  CE targets; frozen repository-aligned `paged_adamw_8bit` with
  betas/epsilon/gradient clipping/schedule/seed; pinned validation decoding;
  frozen decontamination thresholds and audit; operational post-pause token
  criterion.
- **Stage3: READY_TO_IMPLEMENT.** E1 and E5 are closed. Excluding hidden-state
  index 32 from the primary confirmatory gate as well as Stage4 is the stronger
  correct resolution: the gated direction and the steerable direction remain
  the same object. The `l -> decoder block l-1` hook convention and four-source
  macro confidence interval are correct.
- **Stage4: READY_TO_IMPLEMENT.** E1--E4 are closed. The two-tier semantic
  ladder is decidable: tie-dominated results can support only "not detectably
  more disruptive," while the professor-facing cleaner/privileged claim still
  requires the strict lower bound above 0.50 for both A3 and A4 inside the full
  intersection-union gate. Compliance, degeneration, and calibration
  statistics are pinned; fail-closed manifest/judge requirements are
  normative.

## Professor comments

Fable-5 confirms that all six comments are answerable conditional on the
preregistered gates. Comment 1 is limited by the semantic tier actually
reached; comments 2, 5, and 6 are fully answered by design; comments 3 and 4
remain result-conditional within the frozen four-source and bundled-full-SFT
claim boundary.

## Remaining issues

There are no remaining load-bearing edits and no additional scientific branch
is required. One non-blocking provenance item remains: record the exact
bitsandbytes and PyTorch versions in the training manifest because
`paged_adamw_8bit` behavior depends on the installed build.

The disallowed-claims list in `FABLE5_REVIEW_RESPONSE.md` remains unchanged.

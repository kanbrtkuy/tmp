# Claude Fable 5 Re-Review: Stage 1 Post-HB Code

## Re-Review Result

Blocking issue only partially fixed.

## What Landed Correctly

- `judge_label` removed from the TSV and kept only in the manifest.
- Summarizer joins by `qa_id`, fails closed on missing/mismatched rows, verifies prompt/reasoning hashes, records `manifest_jsonl_sha256`.
- Safe-side agreement bar added alongside unsafe-side agreement.
- Orchestrator passes `--include-text`, samples 60/source versus 50 minimum, and `json_bool_gate` binds the passing summary to the current manifest by hash before GPU Stage1.

## Still Blocking Before Annotation

The sheet still leaks the label through `row_id`. The LOSO freeze uses row IDs
of the form `pair_id::safe` / `pair_id::unsafe`, and the QA sampler writes
`row_id` into the TSV. Therefore every row still says `::safe` or `::unsafe`
in plain text.

Recommended fix:

- Whitelist sheet columns to `qa_id`, hashes, text, and annotation fields.
- Drop `row_id` and `pair_id` from the sheet; keep them in the manifest.
- Strengthen tests to assert the exact TSV header and that no non-text cell
  contains `safe` / `unsafe`.

## New Non-Blocking Nits

- Both arms of a pair can be co-sampled; recommend one arm per `pair_id` per sheet.
- `json_bool_gate` silently skips the hash check if the manifest file is missing.
- Legacy `legacy_tsv_judge_label` fallback accepts unblinded sheets.
- `QA_UNSAFE_AGREEMENT_BAR` is defined in the orchestrator but never used.
- Hash comparison skips when either side is blank.

## Bottom Line

OK to sync to RunPod and run CPU gates after HB completes, because freeze,
audits, baselines, and GPU gating are unaffected. Not OK to start human
annotation until `row_id` / `pair_id` are dropped from the sheet.

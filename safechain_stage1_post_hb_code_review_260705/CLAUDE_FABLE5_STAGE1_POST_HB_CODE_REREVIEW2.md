# Claude Fable 5 Final Re-Review: Stage 1 Post-HB Code

**Verdict: GO — previous blocker fixed, no blocking issues remain.**

The `row_id` / `pair_id` leak is genuinely closed: the sheet writer is now a
strict whitelist (`qa_id`, hashes, text, annotation fields only), all
identifying fields live solely in the manifest, `qa_id` ordering/derivation
leaks nothing, and tests assert the exact header plus no `safe` / `unsafe`
substrings in non-text cells. The fail-closed chain also holds end to end:
summarizer requires the manifest and rejects hash mismatches, and the GPU gate
hard-fails on a missing manifest and binds the passing summary to the manifest
by SHA-256.

Three non-blocking notes:

- The blank-hash claim in `POST_FABLE_FIXES.md` is slightly stronger than the
  code, but the edge case is unreachable in practice.
- `QA_UNSAFE_AGREEMENT_BAR` is still defined-but-unused in the orchestrator.
- Pair-arm co-sampling can still occur in pair-poor sources via the overflow
  fallback; this reveals pairing, not labels.

Clear to sync to RunPod and run CPU gates after HB completes; annotation is
unblocked too.

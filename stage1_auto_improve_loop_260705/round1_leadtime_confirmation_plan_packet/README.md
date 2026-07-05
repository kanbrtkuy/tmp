# Stage1 Excluded-Source Lead-Time Confirmation Plan Packet

Purpose: Fable-5 preregistration review before any code or run.

This packet contains documentation only:

- `res/stage1_excluded_source_leadtime_confirmation_prereg_plan_260705.md`
- accepted post-pivot diagnostics and stop/pivot memos
- `review/AUTO_REVIEW.md`

It intentionally excludes raw prompts, raw CoTs, hidden arrays, generated
pairs, and prediction JSONL.

Context:

- A2 returned `STOP_EQUAL_HORIZON_AND_PIVOT`; equal-horizon iteration is
  closed.
- Fable-5 left open one optional future action: excluded-source k=4 lead-time
  confirmation on `strongreject_full` and `reasoningshield`, running both A1
  and A2 recipes and accepting either outcome.
- RunPod readiness check found prepared splits for both excluded sources, but
  no current Stage1 dense hidden archive or matched-horizon prediction tree for
  those sources.

Review request:

1. Is this plan an allowed excluded-source lead-time confirmation, or does it
   still look like a forbidden equal-horizon rescue?
2. Are the estimands and decision rules sufficiently fixed before data
   generation/extraction?
3. Should the A2 robustness gate be required, weakened, or reporting-only?
4. If edits are required, provide exact text changes. If invalid, say stop.
5. If acceptable, return `OK_TO_IMPLEMENT_PLAN_ONLY`; code must still undergo a
   separate review before any RunPod execution.

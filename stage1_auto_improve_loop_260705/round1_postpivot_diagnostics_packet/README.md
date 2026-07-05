# Stage1 Post-Pivot Diagnostics Packet

Purpose: Fable-5 documentation/interpretation review after A2 returned
`STOP_EQUAL_HORIZON_AND_PIVOT`.

This packet contains aggregate tables and documentation only:

- `res/stage1_a1_a2_leadtime_diagnostics_260705.md`
- `res/stage1_a2_feature_pooling_stop_pivot_260705.md`
- A1/A2 same-horizon and lead-time TSVs
- `review/AUTO_REVIEW.md`

It intentionally excludes raw prompts, raw CoTs, hidden arrays, generated
pairs, and prediction JSONL.

Review request:

1. Check whether `stage1_a1_a2_leadtime_diagnostics_260705.md` is faithful to
   the A1/A2 artifacts and your A2 stop/pivot review.
2. Flag any overclaiming, metric/cell shopping, or language that looks like an
   illicit A2 rescue variant.
3. Decide whether this documentation is acceptable as the closed Stage1
   equal-horizon record.
4. If not acceptable, provide exact text edits only. Do not recommend new
   equal-horizon experiments unless the A2 run was invalid.

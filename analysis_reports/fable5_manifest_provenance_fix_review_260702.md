All five fixes verified in `cot-safety/scripts/data/audit_openai_full_ab.py`, plus I checked the actual on-disk artifacts the fix depends on. Answers:

## 1. Does this fix the stale-metrics blocker? — Yes

Verified end-to-end, not just the snippet:

- The recompute loop (lines 634–639) mutates the manifest dict **before** the record loop, so `normalize_record` (line 590), `failed_record` (line 608), and `compact_manifest_row` (line 840, via `record["deterministic_metrics"]`) all see recomputed values, including the summary's `marker_counts` section.
- I inspected the keys (metadata only) of the first row of the existing `openai_full_ab_audit_requests.tasks.jsonl`: the embedded `row` contains all fields `deterministic_metrics` reads (`unsafe_trajectory`, `unsafe_paraphrased_reasoning`, `safe_reasoning`, `safe_final_answer`, `prompt`), and the stale top-level `deterministic_metrics` indeed lacks the marker fields — confirming both the bug and that the fix will work on this specific manifest.

## 2. OK to run local `batch_collect` re-export? — Yes, with one precision

Methodologically fine: deterministic metrics are pure local functions of the frozen input text; recomputing them doesn't alter the model-judged audit fields, and the provenance note (line 708) discloses the recompute. Two things to be precise about:

- **It's not literally zero API calls.** `collect_batch` does a GET on batch status (line 617) and re-downloads `output_file_id` (line 641). Read-only retrieval of the completed batch — no new batch, no new inference — but it does hit the API and needs `--allow-external-api`. If the output file has been deleted/expired server-side, collect fails; you'd then need a small fallback to reuse the already-downloaded `openai_full_ab_audit_outputs.jsonl`.
- **Use identical (default) `--max-*-words` at collect time.** `audit_input_truncated` and provenance `truncation_params` are computed from *collect-time* args (lines 692–696); the prepare-time values aren't persisted anywhere to cross-check. With defaults you're fine.

## 3. Remaining blockers in manifest/provenance? — No blockers, three nits

- `source_file_hashes` hashes the tier/pair files **as they exist at collect time**, not as they were at prepare time. The true frozen anchor is `task_manifest_sha256`, which you do record — but if the tier files were ever regenerated between prepare and collect, `source_files` would silently misrepresent provenance. A one-word label like `"hashed_at": "collect_time"` would remove the ambiguity. Non-blocking.
- `was_truncated` (line 207) is now dead code — nothing calls it. Delete it.
- `keep_decision` treats `audit.status == ""` as ok (line 763) — pre-existing, intentional-looking, just confirming you know.

## 4. Commit before re-export, or is dirty+script-hash enough?

Content-wise, `audit_script_sha256` + `task_manifest_sha256` + `source_files` hashes are sufficient for integrity. But **commit first anyway**: it's free, and it turns `git_commit` from "some commit that doesn't contain the code that produced this manifest" into an actually resolvable reference. `git_dirty=true` provenance means anyone reproducing has to trust a floating hash instead of `git checkout <commit>`. Since these manifests are meant to be *frozen*, make the commit part of the freeze. (Note `REPO_ROOT` resolves to `cot-safety/`, which is its own nested git repo — so commit there, not in the outer SafeChain repo.)

**Bottom line:** blocker fixed and verified against the real manifest; commit in `cot-safety/`, then run local `batch_collect` re-export with default truncation args. CPU baselines can keep waiting.

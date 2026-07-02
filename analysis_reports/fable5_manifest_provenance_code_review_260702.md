I've read the full current version of `cot-safety/scripts/data/audit_openai_full_ab.py`. The changes match your description, but I found **one blocking bug** that will silently defeat the marker-feature work, plus a few smaller gaps. Answers below.

## Blocking issue: stale `deterministic_metrics` on re-collect

The new marker features will **never appear** in a `batch_collect` re-export as written.

- `collect_batch` reads the frozen task manifest (`.tasks.jsonl` written at `batch_prepare` time, line 611) and `normalize_record`/`failed_record` take `task.get("deterministic_metrics")` (lines 567, 585).
- Those metrics were computed by the **old v1.0 script** when the batch was prepared. They contain the old `unsafe_refusalish`-style fields only — no `unsafe_markers`, `safe_markers`, hard-refusal or redirect fields.
- So `summarize_deterministic`'s new `marker_counts` will be empty/zero, and manifest rows will carry stale metrics, while provenance claims `audit_script_version = v1.1`.

**Fix:** recompute at collect time. The full untruncated row is stored in the task manifest as `task["row"]`, so after loading the manifest in `collect_batch`, do:

```python
for task in manifest.values():
    task["deterministic_metrics"] = deterministic_metrics(task.get("row") or {})
```

This is local, deterministic, and requires no new batch. Optionally note in provenance that deterministic metrics are recomputed at collect time.

## Your questions

**1. Sufficient to re-export?** Almost. The provenance layer itself (versions, prompt hashes, batch IDs, file hashes, truncation params/flags, unsafe-trajectory hash, keep-rule notes) is adequate and correctly addresses all four items I flagged previously. But fix the staleness bug above first, plus the small items below, or the export will contain internally inconsistent metadata.

**2. Hash-only for original unsafe trajectory?** Yes, correct. The hash lets you verify identity against the source tier files without duplicating raw unsafe text into a new artifact. Keep it this way.

**3. Truncation flags vs word counts?** Store the word counts too — it's essentially free and materially useful. A boolean tells you the judge saw a clipped input; the count tells you *how much* was cut (a 530-word field clipped at 520 is a non-issue; a 3000-word one is). Suggested shape per field: `{"words": N, "max_words": M, "truncated": bool}`. Also add a `truncation_counts_among_keeps_by_tier` (currently `truncation_counts` covers all audited rows, line 874) — the load-bearing question is how many **kept** A′/B′ rows had a truncated `unsafe_trajectory`, since `added_dangerous_detail` is unreliable exactly for those rows.

**4. Regex split acceptable?** Acceptable as a first deterministic baseline, with two caveats to document rather than fix now:
- `won(?:not|'t)` matches the non-word "wonnot" and misses "won't" written as "wont"; `can(?:not|'t)` misses "can not". Harmless quirks inherited from `REFUSALISH_RE`, fine to leave.
- `REDIRECT_RE` terms like "consider", "recommend", "support", "focus on" are very generic and will fire heavily on neutral prose — expect high base rates on the safe side and non-trivial rates on the unsafe side. That's fine **as long as these remain diagnostics, never gates**. Add one line to `keep_rule_notes` or provenance stating markers are diagnostic-only.

**5. `keep_rule_version` in `row_payload_sha256`?** Yes, correct — that hash already includes `audit_keep` and `audit_reject_reasons` (lines 808–810), so it is a *decision* hash and should change when the keep rule changes. Content identity across keep-rule versions is already recoverable from the per-field hashes (`prompt_sha256`, `unsafe_reasoning_sha256`, etc.), so no separate content hash is required.

**6. Anything else in `manifest_hashes.json`?** Two gaps:
- **Input file hashes**: provenance records the request jsonl and task manifest hashes but not the *source* files (`TIER_FILES` A/B jsonls and `PAIR_FILES`). Add their sha256s (with a note that they're hashed at collect time; per-row hashes cover row-level identity regardless).
- **Git dirty flag / script hash**: `best_effort_git_commit()` will record HEAD, but your v1.1 edits are uncommitted — the recorded commit will *not* contain the script that produced the export. Either commit before re-exporting (preferred), or add `git_dirty: bool` (via `git status --porcelain`) and `audit_script_sha256: file_sha256(Path(__file__))` so the artifact is self-describing.
- Minor: `build_provenance` reads `args.max_*_words`/`temperature` at collect time and presents them as the batch's params. Today collect-time defaults equal prepare-time values so it's truthful, but it's a footgun — the `request_jsonl_sha256` is your real anchor. A comment or reading these from the plan/status file would be cleaner; not blocking.

**7. Next step?** Local `batch_collect` re-export only, after the fixes above. Do **not** resubmit a new OpenAI batch just to fix the filled JSON defaults in the audit prompt — that changes the judge and would fork your 2617-row results. Patch the template (replace example-filled defaults with type placeholders, and bump `AUDIT_TEMPLATE_VERSION`) as a separate change gated for the *next* self-consistency batch, so the prompt-bundle hash correctly distinguishes the two generations.

## Summary

The provenance design is sound and I'd consider the manifest layer adequate for CPU baselines **after** you: (1) recompute `deterministic_metrics` from `task["row"]` in `collect_batch`, (2) add per-field word counts and keeps-only truncation counts, (3) add source-file hashes + git-dirty/script hash, and (4) commit before exporting. Then run the local re-export; no new API calls needed.

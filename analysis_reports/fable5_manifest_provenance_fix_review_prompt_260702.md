# Fable5 Follow-Up Review: Blocking Bug Fixed Before Re-Export

Please review this short follow-up. You previously found one blocking bug in `scripts/data/audit_openai_full_ab.py`: `batch_collect` would reuse stale `deterministic_metrics` from the `.tasks.jsonl` manifest written by the old script, so the new hard-refusal/redirect marker features would not appear in re-exported manifests.

I have fixed that and added the small gaps you requested. I have only run `python3 -m py_compile`; I have **not** re-run `batch_collect`, submitted a new API batch, or run CPU baselines.

## Fix 1: Recompute Deterministic Metrics At Collect Time

In `collect_batch`, after loading the task manifest:

```python
manifest = {row["custom_id"]: row for row in read_jsonl(Path(status["task_manifest"]))}
# The task manifest was written at batch_prepare time. Recompute local
# deterministic metrics at collect time so manifest re-exports reflect the
# current script version without needing a new OpenAI batch.
for task in manifest.values():
    task["deterministic_metrics"] = deterministic_metrics(task.get("row") or {})
```

This should fix the stale-metrics problem.

## Fix 2: Truncation Flags Now Include Word Counts

Changed from boolean-only to per-field metadata:

```python
def truncation_info(text: Any, max_words: int) -> dict[str, Any]:
    words = len(str(text or "").split())
    return {
        "words": words,
        "max_words": max_words,
        "truncated": max_words > 0 and words > max_words,
    }

def audit_input_truncated(source_row: dict[str, Any], truncation_params: dict[str, int]) -> dict[str, dict[str, Any]]:
    return {
        "prompt": truncation_info(source_row.get("prompt"), truncation_params["max_prompt_words"]),
        "unsafe_trajectory": truncation_info(source_row.get("unsafe_trajectory"), truncation_params["max_text_words"]),
        "unsafe_paraphrased_reasoning": truncation_info(
            source_row.get("unsafe_paraphrased_reasoning"), truncation_params["max_text_words"]
        ),
        "safe_reasoning": truncation_info(source_row.get("safe_reasoning"), truncation_params["max_text_words"]),
        "safe_final_answer": truncation_info(source_row.get("safe_final_answer"), truncation_params["max_final_words"]),
    }
```

## Fix 3: Keeps-Only Truncation Counts

In `export_manifests`:

```python
summary["truncation_counts_by_tier"] = truncation_counts(groups["A_all_audited"] + groups["B_all_audited"])
summary["truncation_counts_among_keeps_by_tier"] = truncation_counts(
    groups["A_prime_keep"] + groups["B_prime_keep"]
)
```

`truncation_counts` now handles the new nested shape:

```python
if isinstance(value, dict):
    truncated = bool(value.get("truncated"))
else:
    truncated = bool(value)
```

## Fix 4: Source File Hashes, Git Dirty, Script Hash

`build_provenance` now includes:

```python
"source_files": source_file_hashes(args),
"git_commit": best_effort_git_commit(),
"git_dirty": best_effort_git_dirty(),
"audit_script_sha256": file_sha256(Path(__file__)),
```

with:

```python
def source_file_hashes(args: argparse.Namespace) -> dict[str, dict[str, str | None]]:
    files = {
        "pair_files": list(args.input_pair_jsonl),
        "tier_files": [args.tier_a_jsonl, args.tier_b_jsonl],
    }
    out: dict[str, dict[str, str | None]] = {}
    for group, paths in files.items():
        for path_str in paths:
            path = REPO_ROOT / path_str
            out[f"{group}:{path_str}"] = {
                "path": str(path),
                "sha256": file_sha256(path) if path.exists() else None,
            }
    return out
```

and:

```python
def best_effort_git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return None
```

## Fix 5: Marker Diagnostics Are Explicitly Non-Gating

`keep_rule_notes` now says:

```python
"Gate on explicit quality fields only; usable_for_primary_A and "
"usable_for_sensitivity_B are diagnostics because sample/full audits "
"showed prompt-anchoring instability. Refusal/redirect marker fields "
"are deterministic diagnostics only and are never keep/drop gates."
```

## Still True

- The keep rule still ignores `usable_for_primary_A/B`.
- It still does not gate on `major_asymmetry`.
- The raw original unsafe trajectory is still **not** written into the frozen manifest; only `original_unsafe_trajectory_sha256` is written.
- The filled JSON defaults in the audit prompt have **not** been patched yet, because you recommended treating that as a separate change for future self-consistency batches, not for re-exporting the already completed 2617-row audit.

## Questions

1. Does this fix the stale deterministic-metrics blocker?
2. Is it now methodologically OK to run local `batch_collect` re-export from the completed batch, without any new API calls?
3. Is there any remaining blocker in the manifest/provenance layer before re-export?
4. Should we commit before re-export, or is `git_dirty=true` plus `audit_script_sha256` sufficient for this stage?

Please answer directly. If yes, I will only do a local re-export next; CPU baselines still wait.

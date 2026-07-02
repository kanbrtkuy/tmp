# Fable5 Review Request: Manifest/Provenance Script Fixes Before Re-Export

Please review the local script changes below before we re-run `batch_collect` to re-export manifests. No new OpenAI batch, CPU baseline, or GPU task has been run after these edits.

## Context

We are preparing frozen A-prime/B-prime manifests for an off-policy safe-vs-unsafe CoT separability study.

Previous full A/B classification-only audit:

- script: `scripts/data/audit_openai_full_ab.py`
- batch id: `batch_6a45db5fe0cc819082fd83a8ac9922a0`
- model: `gpt-4.1-mini-2025-04-14`
- rows: 2617
- A: 1119
- B: 1498
- completed: 2617
- failed: 0
- parse ok: 2617

You previously flagged:

1. Many audit inputs were truncated at `max_text_words=520`, affecting `added_dangerous_detail` and `overcompressed_semantic_loss`.
2. Scale fields such as alignment and template dominance looked anchored to schema example values and should not be trusted as real discriminative metrics.
3. `usable_for_primary_A/B` should not be used as keep/drop gates because it was unstable across prompts.
4. We need manifest/provenance fixes before CPU baselines:
   - keep-rule version
   - judge provenance
   - prompt hash
   - truncation params and per-row truncation flags
   - original unsafe trajectory hash
   - hard-refusal vs redirect marker statistics

## What I Changed Locally

I modified `scripts/data/audit_openai_full_ab.py`. I have only run `python3 -m py_compile`; I have not re-exported manifests yet.

### Version Constants

```python
AUDIT_SCRIPT_VERSION = "openai_full_ab_audit_v1.1"
AUDIT_TEMPLATE_VERSION = "combined_ab_row_audit_v1"
KEEP_RULE_VERSION = "explicit_quality_v1.1_no_usable_gate"
```

### Provenance Hashing

```python
def audit_prompt_sha256() -> str:
    return stable_json_sha256(
        {
            "system_prompt": SYSTEM_PROMPT,
            "combined_audit_template": COMBINED_AUDIT_TEMPLATE,
            "audit_template_version": AUDIT_TEMPLATE_VERSION,
        }
    )

def best_effort_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None
```

### Truncation Flags

```python
def was_truncated(text: Any, max_words: int) -> bool:
    if max_words <= 0:
        return False
    return len(str(text or "").split()) > max_words

def audit_input_truncated(source_row: dict[str, Any], truncation_params: dict[str, int]) -> dict[str, bool]:
    return {
        "prompt": was_truncated(source_row.get("prompt"), truncation_params["max_prompt_words"]),
        "unsafe_trajectory": was_truncated(source_row.get("unsafe_trajectory"), truncation_params["max_text_words"]),
        "unsafe_paraphrased_reasoning": was_truncated(
            source_row.get("unsafe_paraphrased_reasoning"), truncation_params["max_text_words"]
        ),
        "safe_reasoning": was_truncated(source_row.get("safe_reasoning"), truncation_params["max_text_words"]),
        "safe_final_answer": was_truncated(source_row.get("safe_final_answer"), truncation_params["max_final_words"]),
    }
```

### Hard Refusal vs Redirect Markers

I split the old broad `REFUSALISH_RE` into:

```python
HARD_REFUSAL_RE = re.compile(
    r"\b(?:i\s+)?(?:can(?:not|'t)|won(?:not|'t)|will not|refuse|unable to|not appropriate|"
    r"cannot assist|can't assist|not able to|not safe)\b",
    re.IGNORECASE,
)
REDIRECT_RE = re.compile(
    r"\b(?:instead|safe alternative|focus on|consider|recommend|best approach|seek help|"
    r"report|contact authorities|defensive|prevention|support)\b",
    re.IGNORECASE,
)
```

and added:

```python
def marker_features(text: str) -> dict[str, Any]:
    hard = HARD_REFUSAL_RE.findall(text or "")
    redirect = REDIRECT_RE.findall(text or "")
    combined = REFUSALISH_RE.findall(text or "")
    return {
        "hard_refusal_marker": bool(hard),
        "hard_refusal_marker_count": len(hard),
        "redirect_marker": bool(redirect),
        "redirect_marker_count": len(redirect),
        "refusalish_marker": bool(combined),
        "refusalish_marker_count": len(combined),
    }
```

`deterministic_metrics` now includes both unsafe and safe marker features:

```python
"unsafe_markers": unsafe_markers,
"safe_markers": safe_markers,
"unsafe_refusalish": unsafe_markers["refusalish_marker"],
"safe_refusalish": safe_markers["refusalish_marker"],
"unsafe_hard_refusal_marker": unsafe_markers["hard_refusal_marker"],
"safe_hard_refusal_marker": safe_markers["hard_refusal_marker"],
"unsafe_redirect_marker": unsafe_markers["redirect_marker"],
"safe_redirect_marker": safe_markers["redirect_marker"],
```

### Build Provenance

```python
def build_provenance(args: argparse.Namespace, status: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
    request_path, manifest_path, status_path = batch_paths(args)
    truncation_params = {
        "max_prompt_words": args.max_prompt_words,
        "max_text_words": args.max_text_words,
        "max_final_words": args.max_final_words,
    }
    return {
        "audit_script": "scripts/data/audit_openai_full_ab.py",
        "audit_script_version": AUDIT_SCRIPT_VERSION,
        "audit_template_version": AUDIT_TEMPLATE_VERSION,
        "keep_rule_version": KEEP_RULE_VERSION,
        "keep_rule_notes": (
            "Gate on explicit quality fields only; usable_for_primary_A and "
            "usable_for_sensitivity_B are diagnostics because sample/full audits "
            "showed prompt-anchoring instability."
        ),
        "model": batch.get("model") or args.model,
        "batch_id": batch.get("id"),
        "input_file_id": batch.get("input_file_id"),
        "output_file_id": batch.get("output_file_id"),
        "request_counts": batch.get("request_counts"),
        "usage": batch.get("usage"),
        "request_jsonl": str(request_path),
        "request_jsonl_sha256": file_sha256(request_path) if request_path.exists() else None,
        "task_manifest": str(manifest_path),
        "task_manifest_sha256": file_sha256(manifest_path) if manifest_path.exists() else None,
        "status_json": str(status_path),
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "combined_audit_template_sha256": sha256_text(COMBINED_AUDIT_TEMPLATE),
        "audit_prompt_bundle_sha256": audit_prompt_sha256(),
        "truncation_params": truncation_params,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "completion_window": args.completion_window,
        "git_commit": best_effort_git_commit(),
    }
```

### Manifest Row Changes

Each row now gets:

```python
"audit_provenance_ref": {
    "audit_script_version": provenance.get("audit_script_version"),
    "audit_template_version": provenance.get("audit_template_version"),
    "keep_rule_version": provenance.get("keep_rule_version"),
    "batch_id": provenance.get("batch_id"),
    "model": provenance.get("model"),
    "audit_prompt_bundle_sha256": provenance.get("audit_prompt_bundle_sha256"),
},
"audit_input_truncated": audit_input_truncated(source_row, truncation_params),
"hashes": {
    "prompt_sha256": sha256_text(prompt),
    "original_unsafe_trajectory_sha256": sha256_text(original_unsafe),
    "unsafe_reasoning_sha256": sha256_text(unsafe_text),
    "safe_reasoning_sha256": sha256_text(safe_reasoning),
    "safe_final_answer_sha256": sha256_text(safe_final),
}
```

Important: I do **not** write the raw original unsafe trajectory into the manifest. I only add its hash. The manifest already contains `unsafe_reasoning`, which is the OpenAI unsafe-side paraphrase used for the paired dataset.

### Manifest Hashes

`manifest_hashes.json` now includes:

```python
summary["audit_provenance"] = provenance
summary["truncation_counts_by_tier"] = truncation_counts(groups["A_all_audited"] + groups["B_all_audited"])
```

### Deterministic Summary

`summarize_deterministic` now aggregates:

```python
"unsafe_refusalish",
"safe_refusalish",
"unsafe_hard_refusal_marker",
"safe_hard_refusal_marker",
"unsafe_redirect_marker",
"safe_redirect_marker",
"unsafe_hard_refusal_marker_count_sum",
"safe_hard_refusal_marker_count_sum",
"unsafe_redirect_marker_count_sum",
"safe_redirect_marker_count_sum",
...
```

## Existing Keep Rule

I did not change the keep rule again. It still ignores `usable_for_primary_A/B`, uses explicit boolean fields, and does not gate on major_asymmetry.

```python
# Do not gate on usable_for_primary_A / usable_for_sensitivity_B.
```

## Questions

1. Are these local manifest/provenance fixes sufficient before re-running `batch_collect` to re-export manifests?
2. Is it correct to include only the hash of the original unsafe trajectory, not the raw original unsafe text?
3. Are the truncation flags enough, or should we also store exact word counts per field in `audit_input_truncated`?
4. Is the hard-refusal/redirect regex split acceptable as a first deterministic marker baseline, or should it be revised before export?
5. Should `row_payload_sha256` include `keep_rule_version` as I did, knowing hashes will change when the keep rule changes?
6. Should `manifest_hashes.json` include anything else before CPU baselines?
7. If this looks good, should the next step be only local `batch_collect` re-export, or should we first patch the combined audit prompt to remove filled JSON defaults for future self-consistency batches?

Please focus on whether the script is methodologically sound enough to re-export manifests. Do not recommend starting CPU baselines yet unless you believe the manifest/provenance layer is now adequate.

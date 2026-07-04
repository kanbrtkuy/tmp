#!/usr/bin/env python3
"""Build a compact data/code review bundle for Fable5.

The bundle is intentionally small enough for external review while preserving
row ids, audit labels, provenance pointers, and short text excerpts needed to
spot systematic data-quality failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SEED = 260702

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO.parent

FULL_AB_DIR = REPO / "runs/openai_full_ab_quality_audit_v1"
FROZEN_DIR = FULL_AB_DIR / "frozen_manifests_v1"
SAMPLE_AUDIT_DIR = REPO / "runs/openai_data_quality_audit_samples_v1"
PARAPHRASE_DIR = REPO / "runs/openai_unsafe_paraphrase_only_v1"
STRATA_DIR = PARAPHRASE_DIR / "quality_strata_v1"

OUT_BUNDLE = REPO / "analysis_reports/fable5_pipeline_integrity_review_bundle_260702.json"
OUT_PROMPT = REPO / "analysis_reports/fable5_pipeline_integrity_review_prompt_260702.md"
OUT_EMBEDDED_PROMPT = REPO / "analysis_reports/fable5_pipeline_integrity_review_prompt_embedded_260702.md"


CODE_FILES = [
    "scripts/data/generate_safe_rewrites_openai.py",
    "scripts/data/generate_counterfactual_trajectories_openai.py",
    "scripts/data/validate_safe_rewrite_pairs.py",
    "scripts/data/generate_openai_controls.py",
    "scripts/data/stratify_openai_paraphrase_quality.py",
    "scripts/data/repair_openai_unsafe_paraphrases.py",
    "scripts/data/audit_openai_control_samples.py",
    "scripts/data/audit_openai_full_ab.py",
    "scripts/data/export_safe_rewrite_pairs_for_stage1.py",
    "src/cot_safety/data/safe_rewrite.py",
    "src/cot_safety/data/labels.py",
    "src/cot_safety/utils/secrets.py",
]

CONFIG_FILES = [
    "configs/data/unsafe_to_safe_rewrite_harmthoughts_all1018_polish_v5_controlled_clean.yaml",
    "configs/data/unsafe_to_safe_rewrite_reasoningshield_all4813_polish_v5_controlled_clean.yaml",
    "configs/data/unsafe_to_safe_rewrite_reasoningshield_all4813_polish_v5_controlled_clean_round2.yaml",
    "configs/data/unsafe_to_safe_rewrite_reasoningshield_all4813_polish_v5_controlled_clean_round3.yaml",
]

SUMMARY_FILES = [
    "runs/openai_unsafe_paraphrase_only_v1/openai_unsafe_paraphrases_summary.json",
    "runs/openai_unsafe_paraphrase_only_v1/quality_strata_v1/quality_strata_summary.json",
    "runs/openai_unsafe_paraphrase_only_v1/quality_strata_v1/usable_tier_summary.json",
    "runs/openai_unsafe_paraphrase_repair_pilot_v1/openai_repair_summary.json",
    "runs/openai_data_quality_audit_samples_v1/openai_audit_summary.json",
    "runs/openai_full_ab_quality_audit_v1/openai_full_ab_audit_summary.json",
    "runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1/manifest_hashes.json",
]

PRIOR_REVIEW_FILES = [
    "analysis_reports/fable5_openai_data_quality_audit_review_260702.md",
    "analysis_reports/fable5_full_ab_audit_code_review_260702.md",
    "analysis_reports/fable5_manifest_provenance_code_review_260702.md",
    "analysis_reports/fable5_manifest_provenance_fix_review_260702.md",
]


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=REPO, text=True).strip()
        except Exception as exc:  # pragma: no cover - diagnostic only
            return f"ERROR: {exc}"

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "dirty_short": run(["git", "status", "--short"]),
    }


def file_inventory(rel_paths: Iterable[str]) -> list[dict[str, Any]]:
    items = []
    for rel in rel_paths:
        path = REPO / rel
        exists = path.exists()
        item: dict[str, Any] = {
            "path": rel,
            "exists": exists,
            "sha256": sha256_file(path),
        }
        if exists:
            text = path.read_text(encoding="utf-8", errors="replace")
            item["line_count"] = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
            item["bytes"] = path.stat().st_size
        items.append(item)
    return items


def word_excerpt(text: str | None, max_words: int = 180) -> dict[str, Any]:
    text = text or ""
    words = text.split()
    if len(words) <= max_words:
        excerpt = text
        truncated = False
    else:
        head_n = max_words // 2
        tail_n = max_words - head_n
        excerpt = " ".join(words[:head_n])
        excerpt += "\n[... middle omitted for review bundle ...]\n"
        excerpt += " ".join(words[-tail_n:])
        truncated = True
    return {
        "word_count": len(words),
        "excerpt_word_cap": max_words,
        "excerpt_truncated": truncated,
        "excerpt": excerpt,
    }


def by_key(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for row in rows:
        out[(row["pair_id"], row.get("tier_short") or row.get("tier") or "")] = row
    return out


def sample_round_robin(
    rows: list[dict[str, Any]],
    n: int,
    seed: int,
    key_fields: tuple[str, ...] = ("source", "category"),
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(k) for k in key_fields)].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = list(buckets)
    rng.shuffle(keys)
    selected: list[dict[str, Any]] = []
    while keys and len(selected) < n:
        next_keys = []
        for key in keys:
            if buckets[key] and len(selected) < n:
                selected.append(buckets[key].pop())
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def compact_audit(audit: dict[str, Any] | None) -> dict[str, Any] | None:
    if not audit:
        return None
    return {
        "unsafe_paraphrase_quality": audit.get("unsafe_paraphrase_quality"),
        "safe_rewrite_mode": audit.get("safe_rewrite_mode"),
        "pair_alignment": audit.get("pair_alignment"),
        "status": audit.get("status"),
        "audit_type": audit.get("audit_type"),
    }


def compact_metrics(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metrics:
        return None
    keep_keys = [
        "original_unsafe_words",
        "unsafe_paraphrase_words",
        "safe_reasoning_words",
        "safe_total_words",
        "unsafe_paraphrase_to_original_word_ratio",
        "unsafe_paraphrase_to_safe_total_word_ratio",
        "unsafe_to_safe_sentence_ratio",
        "unsafe_to_safe_line_ratio",
        "unsafe_refusalish",
        "safe_refusalish",
        "unsafe_hard_refusal_marker",
        "safe_hard_refusal_marker",
        "unsafe_redirect_marker",
        "safe_redirect_marker",
        "rough_copy_ratio_original_to_paraphrase",
        "asymmetry_flags",
    ]
    return {k: metrics.get(k) for k in keep_keys if k in metrics}


def compact_row(
    row: dict[str, Any],
    task_lookup: dict[tuple[str, str], dict[str, Any]],
    group: str,
    selection_reason: str,
    text_cap: int,
) -> dict[str, Any]:
    key = (row["pair_id"], row.get("tier_short") or row.get("tier") or "")
    task = task_lookup.get(key, {})
    task_row = task.get("row", {})
    original = task_row.get("unsafe_trajectory")
    return {
        "group": group,
        "selection_reason": selection_reason,
        "pair_id": row.get("pair_id"),
        "prompt_id": row.get("prompt_id"),
        "source": row.get("source"),
        "category": row.get("category"),
        "model_name": row.get("model_name"),
        "tier": row.get("tier"),
        "tier_short": row.get("tier_short"),
        "audit_keep": row.get("audit_keep"),
        "audit_reject_reasons": row.get("audit_reject_reasons"),
        "audit_input_truncated": row.get("audit_input_truncated"),
        "prompt": word_excerpt(row.get("prompt") or task_row.get("prompt"), max_words=text_cap),
        "original_unsafe_trajectory": word_excerpt(original, max_words=text_cap),
        "unsafe_paraphrased_reasoning": word_excerpt(
            row.get("unsafe_reasoning") or task_row.get("unsafe_paraphrased_reasoning"),
            max_words=text_cap,
        ),
        "safe_reasoning": word_excerpt(row.get("safe_reasoning") or task_row.get("safe_reasoning"), max_words=text_cap),
        "safe_final_answer": word_excerpt(
            row.get("safe_final_answer") or task_row.get("safe_final_answer"),
            max_words=max(80, text_cap // 2),
        ),
        "audit": compact_audit(row.get("audit")),
        "deterministic_metrics": compact_metrics(row.get("deterministic_metrics")),
        "hashes": row.get("hashes"),
    }


def sample_dropped(dropped: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dropped:
        reasons = row.get("audit_reject_reasons") or ["no_reason"]
        for reason in reasons:
            by_reason[reason].append(row)
    for rows in by_reason.values():
        rng.shuffle(rows)
    selected: list[dict[str, Any]] = []
    seen = set()
    reasons = list(by_reason)
    rng.shuffle(reasons)
    while reasons and len(selected) < n:
        next_reasons = []
        for reason in reasons:
            while by_reason[reason] and by_reason[reason][-1]["pair_id"] in seen:
                by_reason[reason].pop()
            if by_reason[reason] and len(selected) < n:
                row = by_reason[reason].pop()
                selected.append(row)
                seen.add(row["pair_id"])
            if by_reason[reason]:
                next_reasons.append(reason)
        reasons = next_reasons
    return selected


def full_output_lookup() -> dict[tuple[str, str], dict[str, Any]]:
    rows = load_jsonl(FULL_AB_DIR / "openai_full_ab_audit_outputs.jsonl")
    return by_key(rows)


def find_sample_full_disagreements(
    full_lookup: dict[tuple[str, str], dict[str, Any]],
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    sample_outputs = load_jsonl(SAMPLE_AUDIT_DIR / "openai_audit_outputs.jsonl")
    disagreements: list[dict[str, Any]] = []
    unsafe_fields = [
        "still_unsafe",
        "softened_or_policy_washed",
        "added_dangerous_detail",
        "overcompressed_semantic_loss",
        "usable_for_primary_A",
        "usable_for_sensitivity_B",
    ]
    safe_fields = ["rewrite_mode", "topic_drift", "same_prompt_alignment_1_to_5", "template_dominance_1_to_5"]
    pair_fields = ["safe_unsafe_same_prompt_alignment_1_to_5", "topic_drift", "major_asymmetry"]

    for row in sample_outputs:
        key = (row["pair_id"], row.get("tier_short") or "")
        full = full_lookup.get(key)
        if not full:
            continue
        sample_audit = row.get("audit") or {}
        full_audit = full.get("audit") or {}
        audit_type = row.get("audit_type")
        diffs: dict[str, Any] = {}
        if audit_type == "unsafe_paraphrase_quality":
            full_part = full_audit.get("unsafe_paraphrase_quality") or {}
            for field in unsafe_fields:
                if sample_audit.get(field) != full_part.get(field):
                    diffs[field] = {"sample": sample_audit.get(field), "full": full_part.get(field)}
        elif audit_type == "safe_rewrite_mode":
            full_part = full_audit.get("safe_rewrite_mode") or {}
            for field in safe_fields:
                if sample_audit.get(field) != full_part.get(field):
                    diffs[field] = {"sample": sample_audit.get(field), "full": full_part.get(field)}
        elif audit_type == "pair_alignment":
            full_part = full_audit.get("pair_alignment") or {}
            for field in pair_fields:
                if sample_audit.get(field) != full_part.get(field):
                    diffs[field] = {"sample": sample_audit.get(field), "full": full_part.get(field)}
        if diffs:
            disagreements.append(
                {
                    "pair_id": row["pair_id"],
                    "tier_short": row.get("tier_short"),
                    "audit_type": audit_type,
                    "task_group": row.get("task_group"),
                    "diffs": diffs,
                    "sample_audit": sample_audit,
                    "full_audit": full_audit,
                }
            )

    rng = random.Random(seed)
    rng.shuffle(disagreements)
    return disagreements[:n]


def summarize_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "source_counts": dict(Counter(row.get("source") for row in rows)),
        "category_counts_top20": dict(Counter(row.get("category") for row in rows).most_common(20)),
        "tier_counts": dict(Counter(row.get("tier_short") for row in rows)),
    }


def compact_summary(rel: str) -> Any:
    path = REPO / rel
    if not path.exists():
        return None
    obj = load_json(path)
    if rel.endswith("openai_unsafe_paraphrases_summary.json"):
        keep_keys = [
            "n_active",
            "n_skipped_holdout",
            "status_counts",
            "category_counts",
            "safe_rewrite_mode_counts",
            "auto_quality_pass",
            "auto_quality_pass_rate",
            "local_refusalish_count",
            "local_meta_leak_count",
            "local_empty_output_count",
            "length_ratio_summary",
            "rough_copy_ratio_summary",
            "batch_id",
            "output_file_id",
            "model",
        ]
        api_errors = obj.get("api_or_parse_errors") or []
        skip = obj.get("skip") or []
        return {
            "compact_note": "Large per-row skip/error details omitted from bundle; inspect source file if needed.",
            **{k: obj.get(k) for k in keep_keys if k in obj},
            "api_or_parse_errors_count": len(api_errors) if isinstance(api_errors, list) else api_errors,
            "skip_count": len(skip) if isinstance(skip, list) else skip,
        }
    return obj


def build_bundle(text_cap: int, lite: bool = False) -> dict[str, Any]:
    a_prime = load_jsonl(FROZEN_DIR / "A_prime_manifest.jsonl")
    b_prime = load_jsonl(FROZEN_DIR / "B_prime_manifest.jsonl")
    dropped = load_jsonl(FROZEN_DIR / "dropped_manifest.jsonl")
    a_all = load_jsonl(FROZEN_DIR / "A_all_audited_manifest.jsonl")
    b_all = load_jsonl(FROZEN_DIR / "B_all_audited_manifest.jsonl")
    tasks = load_jsonl(FULL_AB_DIR / "openai_full_ab_audit_requests.tasks.jsonl")
    task_lookup = by_key(tasks)
    full_lookup = full_output_lookup()

    keep_pool = a_prime + b_prime
    if lite:
        n_a = 5
        n_b = 5
        n_drop = 8
        n_edge = 4
        n_disagree = 12
    else:
        n_a = 12
        n_b = 12
        n_drop = 18
        n_edge = 10
        n_disagree = 30

    truncation_edges = sorted(
        [
            row
            for row in keep_pool
            if row.get("audit_input_truncated", {}).get("unsafe_trajectory", {}).get("truncated")
            or row.get("audit_input_truncated", {}).get("unsafe_paraphrased_reasoning", {}).get("truncated")
            or row.get("audit_input_truncated", {}).get("prompt", {}).get("truncated")
        ],
        key=lambda r: (
            r.get("deterministic_metrics", {}).get("original_unsafe_words", 0),
            r.get("deterministic_metrics", {}).get("unsafe_paraphrase_words", 0),
        ),
        reverse=True,
    )[:n_edge]
    hard_refusal_like = sample_round_robin(
        [
            row
            for row in keep_pool
            if row.get("deterministic_metrics", {}).get("safe_hard_refusal_marker")
            or row.get("deterministic_metrics", {}).get("safe_redirect_marker")
        ],
        n_edge,
        SEED + 3,
        key_fields=("tier_short", "source", "category"),
    )
    asymmetry_edges = sample_round_robin(
        [
            row
            for row in keep_pool
            if any((row.get("deterministic_metrics", {}).get("asymmetry_flags") or {}).values())
        ],
        n_edge,
        SEED + 4,
        key_fields=("tier_short", "source", "category"),
    )

    sample_specs: list[tuple[str, str, list[dict[str, Any]]]] = [
        ("A_prime_keep", "round-robin sample by source/category from primary keep rows", sample_round_robin(a_prime, n_a, SEED + 1)),
        ("B_prime_keep", "round-robin sample by source/category from sensitivity keep rows", sample_round_robin(b_prime, n_b, SEED + 2)),
        ("dropped_rows", "round-robin sample by audit_reject_reasons from dropped rows", sample_dropped(dropped, n_drop, SEED + 5)),
        ("truncation_edges", "largest kept rows with prompt/original/paraphrase truncation in audit input", truncation_edges),
        ("safe_refusal_or_redirect_edges", "kept rows whose safe side has deterministic refusal/redirect markers", hard_refusal_like),
        ("format_asymmetry_edges", "kept rows with deterministic asymmetry flags", asymmetry_edges),
    ]

    compact_samples: list[dict[str, Any]] = []
    seen_sample_keys = set()
    for group, reason, rows in sample_specs:
        for row in rows:
            sample_key = (group, row["pair_id"], row.get("tier_short"))
            if sample_key in seen_sample_keys:
                continue
            seen_sample_keys.add(sample_key)
            compact_samples.append(compact_row(row, task_lookup, group, reason, text_cap=text_cap))

    return {
        "bundle_version": "fable5_pipeline_integrity_review_bundle_v1_lite" if lite else "fable5_pipeline_integrity_review_bundle_v1",
        "created_for": "SafeChain/COT safety Stage 1 data pipeline integrity review",
        "seed": SEED,
        "text_excerpt_cap_words": text_cap,
        "notes": [
            "Text fields are excerpts for review-bundle compactness; use pair_id and source paths to inspect full local rows.",
            "Frozen manifests intentionally store original unsafe trajectory hashes, not full original text. This bundle pulls short original excerpts from the full A/B audit task file for sampled rows.",
            "OpenAI audit scalar fields such as alignment scores and usable_* were previously found unstable or schema-anchored; reviewer should not assume they are reliable.",
        ],
        "git": git_info(),
        "code_inventory": file_inventory(CODE_FILES),
        "config_inventory": file_inventory(CONFIG_FILES),
        "summary_inventory": file_inventory(SUMMARY_FILES),
        "prior_review_inventory": file_inventory(PRIOR_REVIEW_FILES),
        "manifests": {
            "A_prime": summarize_manifest(a_prime),
            "B_prime": summarize_manifest(b_prime),
            "dropped": summarize_manifest(dropped),
            "A_all": summarize_manifest(a_all),
            "B_all": summarize_manifest(b_all),
        },
        "summaries": {rel: compact_summary(rel) for rel in SUMMARY_FILES},
        "drop_reasons": {
            "dropped_manifest_reason_counts": dict(
                Counter(reason for row in dropped for reason in (row.get("audit_reject_reasons") or ["no_reason"]))
            )
        },
        "sample_full_audit_disagreements": find_sample_full_disagreements(full_lookup, n=n_disagree, seed=SEED + 6),
        "review_samples": compact_samples,
    }


def build_prompt(bundle_path: Path) -> str:
    bundle_path = bundle_path.resolve()
    code_lines = "\n".join(f"- `{path}`" for path in CODE_FILES)
    config_lines = "\n".join(f"- `{path}`" for path in CONFIG_FILES)
    summary_lines = "\n".join(f"- `{path}`" for path in SUMMARY_FILES)
    prior_lines = "\n".join(f"- `{path}`" for path in PRIOR_REVIEW_FILES)
    return f"""# Fable5 Review Request: SafeChain Stage-1 Data Pipeline Integrity

Please act as a strict senior ML/data reviewer. We need an adversarial audit of
our SafeChain Stage-1 data construction pipeline before running CPU baselines
and GPU hidden-state probes.

## Project context

We are testing whether a frozen R1-1.5B model's teacher-forced hidden states
contain a separable signal for safe vs unsafe reasoning trajectories. The core
problem is confounding: an earlier non-paired dataset may have allowed probes to
classify prompts/domains rather than monitor trajectory safety. We therefore
constructed same-prompt safe/unsafe pairs:

- unsafe side: open-source unsafe CoT trajectories from HarmThoughts and
  ReasoningShield, originally generated by various models, not R1-1.5B.
- safe side: OpenAI-generated safe rewrite / controlled-clean response for the
  same prompt.
- provenance-control attempt: OpenAI unsafe-preserving paraphrase of the unsafe
  side, so both safe and unsafe sides carry an OpenAI rewrite fingerprint.
- current planned split: A-prime is primary, B-prime is sensitivity, dropped is
  excluded unless you recommend otherwise.

Known limitations:

- This is an off-policy/counterfactual teacher-forcing test, not natural rollout.
- OpenAI audit fields like `usable_for_primary_A`, `usable_for_sensitivity_B`,
  alignment scores, and template-dominance looked unstable/schema-anchored.
- Many original unsafe trajectories were truncated in audit prompts, so full-row
  audit labels are not a perfect judge.
- We still need CPU text/provenance baselines before GPU hidden-state probes.

## What to review

Please review both code and generated results. If your environment can read
local files, read these paths directly from the repo root `cot-safety/`.

### Code files

{code_lines}

### Config files

{config_lines}

### Key output summaries/manifests

{summary_lines}

### Previous Fable5 review notes

{prior_lines}

### Sample bundle

Read this bundle:

`{bundle_path.relative_to(REPO)}`

The bundle contains:

- code/config/summary file hashes and line counts
- A-prime/B-prime/drop summary statistics
- sampled A-prime keep rows, B-prime keep rows, dropped rows
- truncation-edge samples
- safe refusal/redirect marker samples
- deterministic format-asymmetry samples
- sample-vs-full audit disagreement cases
- short excerpts of prompt, original unsafe trajectory, OpenAI unsafe paraphrase,
  safe reasoning, and safe final answer for sampled rows

The bundle includes short unsafe excerpts solely for research-data auditing.
Please do not produce operational harmful instructions in your review; summarize
quality issues by `pair_id`, field, and failure mode.

## Specific questions

1. Data validity: Do A-prime and B-prime plausibly support a Stage-1 claim of
   same-prompt safe/unsafe latent separability under teacher forcing?
2. Confounds: What confounds remain after OpenAI unsafe-preserving paraphrase?
   In particular, can a probe still classify style, refusal templates, length,
   formatting, dataset provenance, source dataset, or rewrite mode instead of
   safety?
3. Code correctness: Are there bugs or brittle assumptions in the rewrite,
   paraphrase, stratification, full audit, manifest-freeze, or stage1 export
   scripts that could corrupt labels, leak splits, duplicate rows, or mismatch
   safe/unsafe sides?
4. Keep/drop rules: Is the current explicit keep rule reasonable given that
   `usable_*` and scalar alignment fields are unreliable? Should we change A/B/C
   usage or add another local deterministic filter before experiments?
5. Manifest/provenance: Are the current hashes, batch ids, model ids, script
   versioning, git commit/dirty flags, truncation flags, and per-row provenance
   enough for reproducibility? What is missing?
6. Result sample audit: Inspect sampled rows and identify concrete examples of
   topic drift, safe generic template dominance, unsafe paraphrase washing,
   added details, overcompression, or pair asymmetry. Give pair_ids.
7. Pre-GPU checklist: What must be done before CPU text baselines and before GPU
   hidden-state probes? Please separate blockers, should-do checks, and optional
   nice-to-have improvements.
8. Claims matrix: State what claims would be allowed if:
   - A-prime hidden-state probe works but CPU text baselines also work strongly;
   - A-prime works and CPU text baselines are weak;
   - only B-prime works;
   - neither A-prime nor B-prime works.

Please be blunt. We prefer finding fatal issues now over running expensive GPU
jobs on flawed data.
"""


def _sample_for_embedding(sample: dict[str, Any], include_text: bool = True) -> dict[str, Any]:
    text_fields = [
        "prompt",
        "original_unsafe_trajectory",
        "unsafe_paraphrased_reasoning",
        "safe_reasoning",
        "safe_final_answer",
    ]
    out = {
        "group": sample.get("group"),
        "selection_reason": sample.get("selection_reason"),
        "pair_id": sample.get("pair_id"),
        "source": sample.get("source"),
        "category": sample.get("category"),
        "model_name": sample.get("model_name"),
        "tier_short": sample.get("tier_short"),
        "audit_keep": sample.get("audit_keep"),
        "audit_reject_reasons": sample.get("audit_reject_reasons"),
        "audit_input_truncated": sample.get("audit_input_truncated"),
        "audit": sample.get("audit"),
        "deterministic_metrics": sample.get("deterministic_metrics"),
    }
    for field in text_fields:
        value = sample.get(field) or {}
        field_out = {
            "word_count": value.get("word_count"),
            "excerpt_truncated": value.get("excerpt_truncated"),
        }
        if include_text:
            field_out["excerpt"] = value.get("excerpt")
        else:
            field_out["excerpt"] = "[REDACTED_FOR_POLICY_SAFE_EXTERNAL_REVIEW]"
        out[field] = field_out
    return out


def build_embedded_prompt(
    bundle: dict[str, Any],
    max_samples: int = 14,
    max_disagreements: int = 10,
    include_text: bool = True,
) -> str:
    selected: list[dict[str, Any]] = []
    group_counts: Counter[str] = Counter()
    for sample in bundle["review_samples"]:
        group = sample.get("group") or "unknown"
        if group_counts[group] >= 3:
            continue
        selected.append(_sample_for_embedding(sample, include_text=include_text))
        group_counts[group] += 1
        if len(selected) >= max_samples:
            break

    compact = {
        "bundle_version": bundle["bundle_version"],
        "seed": bundle["seed"],
        "text_excerpt_cap_words": bundle["text_excerpt_cap_words"],
        "git": bundle["git"],
        "manifests": bundle["manifests"],
        "drop_reasons": bundle["drop_reasons"],
        "summaries": bundle["summaries"],
        "code_inventory": bundle["code_inventory"],
        "config_inventory": bundle["config_inventory"],
        "summary_inventory": bundle["summary_inventory"],
        "sample_full_audit_disagreements": bundle["sample_full_audit_disagreements"][:max_disagreements],
        "embedded_review_samples": selected,
    }
    if not include_text:
        compact = redact_free_text_fields(compact)
    compact_json = json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True)
    redaction_note = (
        "The embedded samples include short text excerpts."
        if include_text
        else "All prompt/trajectory/reasoning text excerpts are redacted; only structured metadata, audit labels, word counts, hashes, and deterministic metrics are included."
    )
    return f"""# Fable5 Embedded Review Request: SafeChain Stage-1 Data Pipeline Integrity

Please act as a strict senior ML/data reviewer. Use ONLY the enclosed material
below for this pass; do not attempt to read local files or call tools. We tried a
file-reading review and the CLI stalled, so this is an embedded micro-audit.

## Project context

We are testing whether a frozen R1-1.5B model's teacher-forced hidden states
contain a separable signal for safe vs unsafe reasoning trajectories. Earlier
non-paired data may have let probes classify prompt/domain instead of monitoring
trajectory safety, so we built same-prompt pairs.

Data design:

- unsafe side: open-source unsafe CoT trajectories from HarmThoughts and
  ReasoningShield, generated by various models, not R1-1.5B.
- safe side: OpenAI-generated safe rewrite / controlled-clean response for the
  same prompt.
- provenance-control attempt: OpenAI unsafe-preserving paraphrase of the unsafe
  side, so both safe and unsafe sides carry an OpenAI rewrite fingerprint.
- current usage plan: A-prime primary, B-prime sensitivity, dropped excluded.

Known issues:

- Off-policy/counterfactual teacher forcing, not natural rollout.
- OpenAI audit fields like `usable_*`, alignment scores, and template-dominance
  appeared unstable/schema-anchored.
- Many original unsafe trajectories were truncated in audit prompts.
- CPU text/provenance baselines still need to be run before GPU probes.

The enclosed samples contain short unsafe excerpts only for research-data
auditing when present. In this prompt: {redaction_note}
Do not produce operational harmful detail; summarize quality issues by pair_id
and failure mode.

## Questions

1. Do A-prime/B-prime plausibly support a Stage-1 same-prompt safe/unsafe latent
   separability claim under teacher forcing, or are the confounds too strong?
2. What are the most dangerous remaining confounds: refusal templates, length,
   formatting, source dataset, rewrite mode, OpenAI fingerprint, semantic drift?
3. Based on the code inventory, summaries, and samples, what code/data pipeline
   bugs should we inspect first? You cannot see full code here, so distinguish
   "evidence-backed issue" from "needs code inspection".
4. Is the explicit keep rule reasonable if we ignore `usable_*` and scalar
   alignment fields? Should A/B/C usage change?
5. Are the manifest/provenance fields sufficient? What is missing?
6. Give concrete sample-level issues by pair_id where possible. Since text may
   be redacted, distinguish sample-level evidence from structural/metadata-only
   inference.
7. Give a pre-GPU checklist split into blockers, should-do checks, and optional
   improvements.
8. Give a claims matrix for four outcomes:
   (a) A-prime probe works but CPU text baselines also work strongly;
   (b) A-prime works and CPU baselines are weak;
   (c) only B-prime works;
   (d) neither A-prime nor B-prime works.

Please be blunt and methodologically conservative.

## Enclosed Review Material

```json
{compact_json}
```
"""


def redact_free_text_fields(obj: Any) -> Any:
    """Redact free-text audit rationales for policy-safe external review."""
    if isinstance(obj, dict):
        redacted: dict[str, Any] = {}
        for key, value in obj.items():
            if key in {"brief_reason", "unsafe_paraphrase_notes"}:
                redacted[key] = "[REDACTED_FREE_TEXT_RATIONALE]"
            else:
                redacted[key] = redact_free_text_fields(value)
        return redacted
    if isinstance(obj, list):
        return [redact_free_text_fields(item) for item in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-cap", type=int, default=180)
    parser.add_argument("--lite", action="store_true", help="Generate a smaller review bundle for Claude CLI.")
    parser.add_argument("--embedded-prompt", type=Path, default=None)
    parser.add_argument("--redact-embedded-text", action="store_true")
    parser.add_argument("--bundle", type=Path, default=OUT_BUNDLE)
    parser.add_argument("--prompt", type=Path, default=OUT_PROMPT)
    args = parser.parse_args()

    bundle = build_bundle(text_cap=args.text_cap, lite=args.lite)
    dump_json(args.bundle, bundle)
    args.prompt.parent.mkdir(parents=True, exist_ok=True)
    args.prompt.write_text(build_prompt(args.bundle), encoding="utf-8")
    if args.embedded_prompt:
        args.embedded_prompt.parent.mkdir(parents=True, exist_ok=True)
        args.embedded_prompt.write_text(
            build_embedded_prompt(bundle, include_text=not args.redact_embedded_text),
            encoding="utf-8",
        )
    print(f"Wrote bundle: {args.bundle}")
    print(f"Wrote prompt: {args.prompt}")
    if args.embedded_prompt:
        print(f"Wrote embedded prompt: {args.embedded_prompt}")
    print(f"Review samples: {len(bundle['review_samples'])}")
    print(f"Disagreement cases: {len(bundle['sample_full_audit_disagreements'])}")


if __name__ == "__main__":
    main()

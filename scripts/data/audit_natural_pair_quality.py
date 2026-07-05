#!/usr/bin/env python3
"""Audit natural same-prompt safe/unsafe pair artifacts.

The audit is intentionally content-quiet: it reports counts, word-length
statistics, structural completeness flags, selection metadata, and baseline
headline metrics without printing raw prompts or trajectories.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


STRONG_END_CHARS = set(".?!。！？)]}\"'”’）】」』")
WEAK_END_CHARS = set(",:;，：；-–—/")
TRAILING_CLOSERS = set(")]}\"'”’）】」』")
STRONG_INCOMPLETE_FLAGS = frozenset(
    {
        "missing_field",
        "empty",
        "too_short",
        "ends_with_weak_char",
        "ends_with_ellipsis",
        "ends_with_connector",
        "open_markdown_fence",
        "last_line_list_marker",
        "unbalanced_angle_think",
    }
)
ENGLISH_CONNECTOR_ENDS = {
    "and",
    "or",
    "but",
    "because",
    "while",
    "when",
    "where",
    "if",
    "that",
    "which",
    "who",
    "whose",
    "with",
    "without",
    "into",
    "from",
    "to",
    "of",
    "for",
    "by",
    "about",
    "including",
    "include",
    "such",
    "such as",
    "for example",
    "e.g.",
    "i.e.",
}
CHINESE_CONNECTOR_ENDS = ("和", "或", "以及", "因为", "例如", "包括", "通过", "对于")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def clean_for_words(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def word_count(text: Any) -> int:
    text = clean_for_words(text)
    if not text:
        return 0
    tokens = re.findall(r"[A-Za-z0-9]+(?:[.'-][A-Za-z0-9]+)*|[\u4e00-\u9fff]", text)
    return len(tokens)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] * (hi - pos) + values[hi] * (pos - lo))


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "min": min(values) if values else None,
        "p05": percentile(values, 0.05),
        "median": percentile(values, 0.50),
        "mean": sum(values) / len(values) if values else None,
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def ends_with_connector(text: str) -> bool:
    text = text.strip()
    if not text or text[-1] in STRONG_END_CHARS:
        return False
    if text.endswith(CHINESE_CONNECTOR_ENDS):
        return True
    words = re.findall(r"[A-Za-z0-9]+(?:[.'-][A-Za-z0-9]+)*", text.lower())
    if not words:
        return False
    last1 = words[-1]
    last2 = " ".join(words[-2:]) if len(words) >= 2 else last1
    return last1 in ENGLISH_CONNECTOR_ENDS or last2 in ENGLISH_CONNECTOR_ENDS


def ends_with_ellipsis(text: str) -> bool:
    stripped = text.strip()
    while stripped and stripped[-1] in TRAILING_CLOSERS:
        stripped = stripped[:-1].rstrip()
    return stripped.endswith(("...", "…"))


def incompleteness_flags(text: Any, *, field: str) -> dict[str, bool]:
    missing = text is None
    raw = str(text or "")
    stripped = raw.strip()
    wc = word_count(stripped)
    last = stripped[-1] if stripped else ""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    last_line = lines[-1] if lines else ""
    min_words = 3 if "final" in field else 20
    return {
        "missing_field": missing,
        "empty": wc == 0,
        "too_short": wc < min_words,
        "weak_end_punctuation": bool(stripped) and last not in STRONG_END_CHARS,
        "ends_with_weak_char": last in WEAK_END_CHARS,
        "ends_with_ellipsis": ends_with_ellipsis(stripped),
        "ends_with_connector": ends_with_connector(stripped),
        "open_markdown_fence": raw.count("```") % 2 == 1,
        "last_line_list_marker": bool(re.match(r"^(\s*[-*+]|\s*\d+[.)])\s*$", last_line)),
        "unbalanced_angle_think": raw.lower().count("<think>") != raw.lower().count("</think>"),
        "unbalanced_square_bracket": raw.count("[") > raw.count("]"),
        "unbalanced_round_paren": raw.count("(") > raw.count(")"),
        "unbalanced_brace": raw.count("{") > raw.count("}"),
    }


def strong_incomplete(flags: dict[str, bool]) -> bool:
    return any(flags.get(name) for name in STRONG_INCOMPLETE_FLAGS)


def audit_field(rows: list[dict[str, Any]], field: str, id_field: str = "pair_id") -> dict[str, Any]:
    counts: Counter[str] = Counter()
    word_counts: list[float] = []
    strong_examples: list[dict[str, Any]] = []
    for row in rows:
        text = row.get(field)
        wc = word_count(text)
        word_counts.append(float(wc))
        flags = incompleteness_flags(text, field=field)
        for name, value in flags.items():
            if value:
                counts[name] += 1
        if strong_incomplete(flags) and len(strong_examples) < 25:
            strong_examples.append(
                {
                    id_field: row.get(id_field),
                    "prompt_instance_id": row.get("prompt_instance_id"),
                    "source_model_canonical": row.get("source_model_canonical"),
                    "word_count": wc,
                    "flags": {name: value for name, value in flags.items() if value},
                }
            )
    return {
        "n_rows": len(rows),
        "word_count_stats": stats(word_counts),
        "flag_counts": dict(counts),
        "strong_incomplete_rows": sum(
            1 for row in rows if strong_incomplete(incompleteness_flags(row.get(field), field=field))
        ),
        "strong_examples_no_text": strong_examples,
    }


def load_summary_if_exists(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.exists() else None


def text_baseline_headline(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {"available": False}
    rows = {}
    for item in metrics.get("results", []):
        test = (item.get("metrics") or {}).get("test") or {}
        rows[item.get("name")] = {
            "balanced_accuracy": test.get("balanced_accuracy"),
            "accuracy": test.get("accuracy"),
            "auroc": test.get("auroc"),
        }
    return {"available": True, "test": rows}


def surface_headline(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {"available": False}
    length_analysis = metrics.get("length_analysis") or {}
    trunc = metrics.get("truncation_curves") or {}
    word_curve = {}
    for item in trunc.get("results", []):
        if item.get("baseline") == "word_tfidf":
            test = (item.get("metrics") or {}).get("test") or {}
            word_curve[str(item.get("k"))] = test.get("balanced_accuracy")
    return {
        "available": True,
        "length_matched_retained_pairs": {
            split: (length_analysis.get("pairwise") or {}).get(split, {}).get("retained_pairs")
            for split in ("train", "val", "test")
        },
        "word_tfidf_truncation_test_ba": word_curve,
        "cross_source_rows": len((metrics.get("cross_source_transfer") or {}).get("results") or []),
    }


def check_split_overlap(normalized_dir: Path) -> dict[str, Any]:
    groups: dict[str, set[str]] = {}
    rows_by_split: dict[str, int] = {}
    labels_by_split: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        rows = read_jsonl(normalized_dir / f"{split}.jsonl")
        rows_by_split[split] = len(rows)
        groups[split] = {str(row.get("match_family") or "") for row in rows if row.get("match_family")}
        labels_by_split[split] = dict(Counter(str(row.get("trajectory_safety_label") or "") for row in rows))
    overlaps = {
        f"{a}_vs_{b}": sorted(groups[a] & groups[b])[:10]
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    overlap_counts = {key: len(value) for key, value in overlaps.items()}
    return {
        "rows_by_split": rows_by_split,
        "labels_by_split": labels_by_split,
        "match_families_by_split": {split: len(values) for split, values in groups.items()},
        "overlap_counts": overlap_counts,
        "overlap_examples_no_text": overlaps,
    }


def render_md(report: dict[str, Any]) -> str:
    export = report["export_summary"]
    lines = [
        "# Natural 8B Pair Quality Audit",
        "",
        "This report audits structural and metadata quality only. It does not print raw prompts or CoT text.",
        "",
        "## Coverage",
        "",
        f"- selected pairs: `{export.get('n_selected_pairs')}`",
        f"- extra inherited pairs added: `{export.get('n_extra_selected_pairs_added')}`",
        f"- dropped prompts: `{export.get('n_dropped_prompts')}`",
        f"- selected by model: `{export.get('selected_by_model')}`",
        f"- drop reasons: `{export.get('drop_reasons')}`",
        "",
        "## Split Integrity",
        "",
        f"- rows by split: `{report['split_integrity']['rows_by_split']}`",
        f"- labels by split: `{report['split_integrity']['labels_by_split']}`",
        f"- match-family overlap counts: `{report['split_integrity']['overlap_counts']}`",
        "",
        "## Completeness Flags",
        "",
    ]
    for field, payload in report["field_audits"].items():
        lines.extend(
            [
                f"### `{field}`",
                "",
                f"- strong incomplete rows: `{payload['strong_incomplete_rows']}` / `{payload['n_rows']}`",
                f"- word count stats: `{payload['word_count_stats']}`",
                f"- flag counts: `{payload['flag_counts']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Selected Candidate Metadata",
            "",
            f"- safe finish reasons: `{report['safe_candidate_metadata']['finish_reason_counts']}`",
            f"- safe quality pass counts: `{report['safe_candidate_metadata']['quality_pass_counts']}`",
            f"- safe think parse counts: `{report['safe_candidate_metadata']['think_parse_status_counts']}`",
            f"- safe hit max tokens counts: `{report['safe_candidate_metadata']['hit_max_tokens_counts']}`",
            f"- candidate pool size: `{report['safe_candidate_metadata']['candidate_pool_size']}`",
            f"- eligible pool size: `{report['safe_candidate_metadata']['eligible_pool_size']}`",
            "",
            "## Baseline Headlines",
            "",
            f"- text baseline test metrics: `{report['text_baseline_headline']}`",
            f"- surface audit headline: `{report['surface_headline']}`",
            "",
            "## Interpretation Notes",
            "",
            "- Prompt-only baseline should remain at chance; if not, prompt leakage is present.",
            "- Strong text baselines on natural pairs indicate natural surface separability, not only OpenAI rewrite artifact.",
            "- Length-matched retained test pairs are small, so length-matched metrics are diagnostic rather than headline evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--text-baseline-dir", default="")
    parser.add_argument("--surface-audit-dir", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    selected_path = export_dir / "natural_safe_pairs_selected.jsonl"
    export_summary_path = export_dir / "export_summary.json"
    normalized_dir = export_dir / "normalized"
    selected = read_jsonl(selected_path)
    export_summary = read_json(export_summary_path)

    field_audits = {
        field: audit_field(selected, field)
        for field in ("unsafe_reasoning", "unsafe_final_answer", "safe_reasoning", "safe_final_answer")
    }
    safe_meta_rows = []
    for row in selected:
        quality = row.get("safe_candidate_quality") or {}
        metadata = row.get("metadata") or {}
        safe_meta_rows.append((quality, metadata))
    safe_candidate_metadata = {
        "finish_reason_counts": dict(Counter(str(q.get("finish_reason")) for q, _ in safe_meta_rows)),
        "quality_pass_counts": dict(Counter(str(q.get("quality_pass")) for q, _ in safe_meta_rows)),
        "think_parse_status_counts": dict(Counter(str(q.get("think_parse_status")) for q, _ in safe_meta_rows)),
        "hit_max_tokens_counts": dict(Counter(str(q.get("hit_max_tokens")) for q, _ in safe_meta_rows)),
        "candidate_pool_size": stats([float((m.get("candidate_pool_size") or 0)) for _, m in safe_meta_rows]),
        "eligible_pool_size": stats([float((m.get("eligible_pool_size") or 0)) for _, m in safe_meta_rows]),
        "quality_score": stats([float((q.get("quality_score") or 0.0)) for q, _ in safe_meta_rows]),
    }

    text_metrics = load_summary_if_exists(Path(args.text_baseline_dir) / "metrics.json") if args.text_baseline_dir else None
    surface_metrics = load_summary_if_exists(Path(args.surface_audit_dir) / "metrics.json") if args.surface_audit_dir else None
    report = {
        "script_version": "natural_pair_quality_audit_v1",
        "export_dir": str(export_dir),
        "selected_pairs_path": str(selected_path),
        "n_selected_rows": len(selected),
        "export_summary": export_summary,
        "split_integrity": check_split_overlap(normalized_dir),
        "field_audits": field_audits,
        "safe_candidate_metadata": safe_candidate_metadata,
        "text_baseline_headline": text_baseline_headline(text_metrics),
        "surface_headline": surface_headline(surface_metrics),
    }
    write_json(Path(args.output_json), report)
    write_text(Path(args.output_md), render_md(report))
    print(
        json.dumps(
            {
                "n_selected_pairs": len(selected),
                "drop_reasons": export_summary.get("drop_reasons"),
                "split_rows": report["split_integrity"]["rows_by_split"],
                "strong_incomplete": {
                    field: payload["strong_incomplete_rows"] for field, payload in field_audits.items()
                },
                "text_baseline_test_ba": {
                    name: metrics.get("balanced_accuracy")
                    for name, metrics in report["text_baseline_headline"].get("test", {}).items()
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_JSON = REPO_ROOT / "analysis_reports/rewrite_completeness_audit_260702.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "analysis_reports/rewrite_completeness_audit_260702.md"

DEFAULT_SPECS = [
    {
        "name": "frozen_A_prime",
        "path": "runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1/A_prime_manifest.jsonl",
        "fields": ["safe_reasoning", "safe_final_answer", "unsafe_reasoning"],
    },
    {
        "name": "frozen_B_prime",
        "path": "runs/openai_full_ab_quality_audit_v1/frozen_manifests_v1/B_prime_manifest.jsonl",
        "fields": ["safe_reasoning", "safe_final_answer", "unsafe_reasoning"],
    },
    {
        "name": "unsafe_paraphrase_all_ok",
        "path": "runs/openai_unsafe_paraphrase_only_v1/openai_unsafe_paraphrases.jsonl",
        "fields": ["unsafe_paraphrased_reasoning", "safe_reasoning", "safe_final_answer"],
        "status_field": "status",
        "status_ok": "ok",
    },
    {
        "name": "safe_rewrite_harmthoughts_v5",
        "path": "runs/unsafe_to_safe_rewrite_harmthoughts_all1018_v4/pairs_polished_v5_controlled_clean.jsonl",
        "fields": ["safe_reasoning", "safe_final_answer", "safe_trajectory"],
        "ok_field": "ok",
    },
    {
        "name": "safe_rewrite_reasoningshield_v5",
        "path": "runs/unsafe_to_safe_rewrite_reasoningshield_all4813_v4/pairs_polished_v5_controlled_clean.jsonl",
        "fields": ["safe_reasoning", "safe_final_answer", "safe_trajectory"],
        "ok_field": "ok",
    },
]

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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
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


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] * (hi - pos) + values[hi] * (pos - lo))


def stats(values: list[int]) -> dict[str, float]:
    return {
        "min": float(min(values)) if values else 0.0,
        "p01": percentile(values, 0.01),
        "p05": percentile(values, 0.05),
        "median": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": float(max(values)) if values else 0.0,
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
    raw = str(text or "")
    stripped = raw.strip()
    wc = word_count(stripped)
    last = stripped[-1] if stripped else ""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    last_line = lines[-1] if lines else ""
    min_words = 3 if "final" in field else 20
    flags = {
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
    return flags


def any_strong_incomplete(flags: dict[str, bool]) -> bool:
    return any(flags.get(name) for name in STRONG_INCOMPLETE_FLAGS)


def row_ok(row: dict[str, Any], spec: dict[str, Any]) -> bool:
    ok_field = spec.get("ok_field")
    if ok_field and row.get(ok_field) is not True:
        return False
    status_field = spec.get("status_field")
    if status_field and row.get(status_field) != spec.get("status_ok", "ok"):
        return False
    return True


def audit_spec(spec: dict[str, Any]) -> dict[str, Any]:
    path = REPO_ROOT / spec["path"]
    rows = read_jsonl(path)
    filtered = [row for row in rows if row_ok(row, spec)]
    result: dict[str, Any] = {
        "name": spec["name"],
        "path": spec["path"],
        "n_rows_total": len(rows),
        "n_rows_audited": len(filtered),
        "fields": {},
    }
    for field in spec["fields"]:
        counts: Counter[str] = Counter()
        word_counts: list[int] = []
        examples: list[dict[str, Any]] = []
        for row in filtered:
            text = row.get(field)
            wc = word_count(text)
            word_counts.append(wc)
            flags = incompleteness_flags(text, field=field)
            for name, value in flags.items():
                if value:
                    counts[name] += 1
            if any_strong_incomplete(flags) and len(examples) < 25:
                examples.append(
                    {
                        "pair_id": row.get("pair_id"),
                        "prompt_id": row.get("prompt_id"),
                        "source": row.get("source"),
                        "category": row.get("category"),
                        "word_count": wc,
                        "flags": {name: value for name, value in flags.items() if value},
                    }
                )
        result["fields"][field] = {
            "word_count_stats": stats(word_counts),
            "flag_counts": dict(counts),
            "strong_incomplete_rows": sum(
                1 for row in filtered if any_strong_incomplete(incompleteness_flags(row.get(field), field=field))
            ),
            "examples": examples,
        }
    return result


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Rewrite Completeness Audit",
        "",
        "This audit checks structural completeness only: missing/short fields, suspicious endings, unclosed fences/tags/brackets, and word-count outliers. It does not judge safety labels and does not print text excerpts.",
        "",
    ]
    for item in report["datasets"]:
        lines.extend(
            [
                f"## {item['name']}",
                "",
                f"- path: `{item['path']}`",
                f"- rows audited: `{item['n_rows_audited']}` / `{item['n_rows_total']}`",
                "",
                "| field | min | p01 | p05 | median | p95 | max | empty | too_short | strong_incomplete | weak_end_punct | connector_end | fence_open |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for field, summary in item["fields"].items():
            s = summary["word_count_stats"]
            flags = summary["flag_counts"]
            lines.append(
                f"| `{field}` | {s['min']:.0f} | {s['p01']:.0f} | {s['p05']:.0f} | {s['median']:.0f} | {s['p95']:.0f} | {s['max']:.0f} | "
                f"{flags.get('empty', 0)} | {flags.get('too_short', 0)} | {summary['strong_incomplete_rows']} | "
                f"{flags.get('weak_end_punctuation', 0)} | {flags.get('ends_with_connector', 0)} | {flags.get('open_markdown_fence', 0)} |"
            )
        lines.append("")
        for field, summary in item["fields"].items():
            examples = summary["examples"]
            if not examples:
                continue
            lines.append(f"### Example flags for `{field}`")
            lines.append("")
            for ex in examples[:10]:
                flags = ",".join(sorted(ex["flags"]))
                lines.append(
                    f"- `{ex.get('pair_id')}` words={ex['word_count']} flags=`{flags}`"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    args = parser.parse_args()

    report = {
        "audit_version": "rewrite_completeness_v1",
        "datasets": [audit_spec(spec) for spec in DEFAULT_SPECS],
    }
    write_json(Path(args.output_json), report)
    write_text(Path(args.output_md), render_md(report))
    print(args.output_json)
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

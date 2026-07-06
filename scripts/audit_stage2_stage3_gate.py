#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PAUSE_TOKEN = "<|pause|>"


STAGE3_SOURCES = {
    "harmbench": "harmbench_standard",
    "reasoningshield": "reasoningshield",
    "strongreject": "strongreject_full",
    "wjb": "wildjailbreak_vanilla_harmful",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def prompt_key(row: dict[str, Any]) -> str:
    return clean_text(row.get("prompt_instance_id") or row.get("match_family") or row.get("pair_id") or row.get("id"))


def pair_key(row: dict[str, Any]) -> str:
    return clean_text(row.get("pair_id") or row.get("metadata", {}).get("pair_id") or prompt_key(row))


def label_for(row: dict[str, Any]) -> str:
    return clean_text(row.get("trajectory_safety_label") or row.get("safety_label")).lower()


def prompt_for(row: dict[str, Any]) -> str:
    return clean_text(row.get("prompt") or row.get("input") or row.get("question") or row.get("behavior"))


def select_balanced_pairs(rows: list[dict[str, Any]], *, max_pairs: int, seed: int) -> list[dict[str, Any]]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        label = label_for(row)
        if label not in {"safe", "unsafe"}:
            continue
        prompt = prompt_for(row)
        if not prompt:
            continue
        by_pair[pair_key(row)][label] = row
    pairs = [(key, value["safe"], value["unsafe"]) for key, value in by_pair.items() if {"safe", "unsafe"} <= set(value)]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    out: list[dict[str, Any]] = []
    for pair_id, safe_row, unsafe_row in pairs:
        for side, row in (("safe", safe_row), ("unsafe", unsafe_row)):
            out.append(
                {
                    "id": f"{pair_id}::{side}",
                    "row_id": clean_text(row.get("row_id") or row.get("id") or f"{pair_id}::{side}"),
                    "prompt": prompt_for(row),
                    "input": prompt_for(row),
                    "task_type": "safety",
                    "dataset": clean_text(row.get("source_family") or row.get("source") or "stage3_source"),
                    "source": clean_text(row.get("source_family") or row.get("source") or "stage3_source"),
                    "stage3_source": clean_text(row.get("source_family") or row.get("source") or "stage3_source"),
                    "pair_id": pair_id,
                    "prompt_instance_id": prompt_key(row),
                    "reference_side": side,
                    "trajectory_safety_label": side,
                    "metadata": {
                        "source_pair_source": row.get("metadata", {}).get("source_pair_source"),
                        "stage1_loso_split": row.get("metadata", {}).get("stage1_loso_split") or row.get("split"),
                    },
                }
            )
    return out


def command_prepare_inputs(args: argparse.Namespace) -> None:
    root = Path(args.stage1_prepared_root)
    manifest: dict[str, Any] = {
        "stage1_prepared_root": str(root),
        "max_pairs_per_source": args.max_pairs_per_source,
        "seed": args.seed,
        "sources": {},
    }
    all_rows: list[dict[str, Any]] = []
    for short_name, rel in STAGE3_SOURCES.items():
        source_path = root / rel / "normalized" / "all.jsonl"
        rows = read_jsonl(source_path)
        selected = select_balanced_pairs(rows, max_pairs=args.max_pairs_per_source, seed=args.seed)
        for row in selected:
            row["stage3_source"] = short_name
            row["dataset"] = short_name
            row["source"] = short_name
        out_path = Path(args.output_dir) / f"{short_name}_paired_generation_input.jsonl"
        write_jsonl(out_path, selected)
        all_rows.extend(selected)
        manifest["sources"][short_name] = {
            "source_path": str(source_path),
            "input_rows": len(rows),
            "selected_rows": len(selected),
            "selected_pairs": len(selected) // 2,
            "output_jsonl": str(out_path),
            "reference_side_counts": dict(Counter(row["reference_side"] for row in selected)),
        }
    all_out = Path(args.output_dir) / "stage3_sources_paired_generation_input.jsonl"
    write_jsonl(all_out, all_rows)
    manifest["all_output_jsonl"] = str(all_out)
    manifest["total_rows"] = len(all_rows)
    manifest["total_pairs"] = len(all_rows) // 2
    write_json(Path(args.output_dir) / "stage3_sources_paired_generation_input_manifest.json", manifest)


def rate(count: int, total: int) -> float:
    return float(count / total) if total else math.nan


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - pos) + xs[hi] * (pos - lo))


def metric(row: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "natural":
        return row.get("natural_pause_metrics") or {}
    return row.get("pause_metrics") or {}


def metric_summary(rows: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    n = len(rows)
    metrics = [metric(row, mode) for row in rows]
    first_indices = [
        float(m["first_pause_token_index_inside_think"])
        for m in metrics
        if m.get("first_pause_token_index_inside_think") is not None
    ]
    pause_counts = [float(m.get("pause_count", 0) or 0) for m in metrics]
    after_answer_bleed = 0
    stripped_bleed = 0
    for row in rows:
        generated = str(row.get("generated") or "")
        think_end = generated.find("</think>")
        answer_text = generated[think_end + len("</think>") :] if think_end >= 0 else ""
        if PAUSE_TOKEN in answer_text:
            after_answer_bleed += 1
        if PAUSE_TOKEN in str(row.get("generated_for_judge") or ""):
            stripped_bleed += 1
    return {
        "n": n,
        "pause_rate": rate(sum(1 for m in metrics if (m.get("pause_count", 0) or 0) > 0), n),
        "exact_single_run3_rate": rate(sum(1 for m in metrics if m.get("has_single_pause_run_of_3")), n),
        "off_target_pause_rate": rate(sum(1 for m in metrics if (m.get("off_target_pause_count", 0) or 0) > 0), n),
        "avg_pause_count": mean(pause_counts),
        "p95_pause_count": quantile(pause_counts, 0.95),
        "avg_first_pause_index": mean(first_indices),
        "median_first_pause_index": quantile(first_indices, 0.5),
        "first_pause_index_coverage": rate(len(first_indices), n),
        "pause_bleed_after_answer_rate": rate(after_answer_bleed, n),
        "pause_bleed_in_generated_for_judge_rate": rate(stripped_bleed, n),
    }


def group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        clean_text(row.get("stage3_source") or row.get("dataset") or row.get("source") or "unknown"),
        clean_text(row.get("reference_side") or row.get("trajectory_safety_label") or "unknown"),
        clean_text(row.get("model_label") or "unknown"),
    )


def command_summarize_generation(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for path in args.generation_jsonl:
        rows.extend(read_jsonl(Path(path)))
    detail_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    for (source, side, model_label), items in sorted(grouped.items()):
        for mode in ("natural", "full"):
            summary = metric_summary(items, mode=mode)
            detail_rows.append(
                {
                    "source": source,
                    "reference_side": side,
                    "model_label": model_label,
                    "metric_mode": mode,
                    **summary,
                }
            )

    by_pair: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        source, _side, model_label = group_key(row)
        side = clean_text(row.get("reference_side") or row.get("trajectory_safety_label") or "unknown")
        by_pair[(source, model_label, clean_text(row.get("pair_id") or row.get("id")))][side] = row
    pair_diff_rows: list[dict[str, Any]] = []
    for (source, model_label, pair_id), sides in sorted(by_pair.items()):
        if "safe" not in sides or "unsafe" not in sides:
            continue
        for mode in ("natural", "full"):
            safe_m = metric(sides["safe"], mode)
            unsafe_m = metric(sides["unsafe"], mode)
            pair_diff_rows.append(
                {
                    "source": source,
                    "model_label": model_label,
                    "pair_id": pair_id,
                    "metric_mode": mode,
                    "safe_exact3": bool(safe_m.get("has_single_pause_run_of_3")),
                    "unsafe_exact3": bool(unsafe_m.get("has_single_pause_run_of_3")),
                    "safe_pause_count": safe_m.get("pause_count", 0),
                    "unsafe_pause_count": unsafe_m.get("pause_count", 0),
                    "safe_first_pause_index": safe_m.get("first_pause_token_index_inside_think"),
                    "unsafe_first_pause_index": unsafe_m.get("first_pause_token_index_inside_think"),
                }
            )

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "stage2_stage3_gate_emission_by_side.csv", detail_rows)
    write_json(output_dir / "stage2_stage3_gate_emission_by_side.json", detail_rows)
    write_json(output_dir / "stage2_stage3_gate_pair_diffs.json", pair_diff_rows)

    md_lines = [
        "# Stage2/Stage3 Gate Emission Sanity",
        "",
        "## By Source And Reference Side",
        "",
        "| source | side | model | metric mode | n | exact-3 | off-target | avg pause count | avg first pause index | answer bleed | stripped bleed |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in detail_rows:
        md_lines.append(
            "| {source} | {reference_side} | {model_label} | {metric_mode} | {n} | {exact_single_run3_rate:.4f} | {off_target_pause_rate:.4f} | {avg_pause_count:.4f} | {avg_first_pause_index:.4f} | {pause_bleed_after_answer_rate:.4f} | {pause_bleed_in_generated_for_judge_rate:.4f} |".format(
                **row
            )
        )
    md_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `metric_mode=natural` uses the model's unforced generated continuation.",
            "- `metric_mode=full` includes forced inserted pause tokens when present.",
            "- `reference_side` is the Stage1 paired safe/unsafe side used only for the paired-coverage audit; generation is not conditioned on this label.",
        ]
    )
    (output_dir / "stage2_stage3_gate_emission_sanity.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def command_gsm8k(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.generation_jsonl))
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("dataset")) != "gsm8k":
            continue
        metrics = row.get("natural_pause_metrics") or {}
        pause_count = int(metrics.get("pause_count", 0) or 0)
        exact3 = bool(metrics.get("has_single_pause_run_of_3"))
        out_rows.append(
            {
                "id": row.get("id"),
                "correct": row.get("correct"),
                "exact3": exact3,
                "over_emit": pause_count > 3,
                "pause_count": pause_count,
                "first_pause_index": metrics.get("first_pause_token_index_inside_think"),
                "finish_reason": row.get("finish_reason") or row.get("hook_stats", {}).get("finish_reason"),
            }
        )
    groups: dict[str, list[dict[str, Any]]] = {
        "all": out_rows,
        "exact3": [row for row in out_rows if row["exact3"]],
        "over_emit": [row for row in out_rows if row["over_emit"]],
        "not_exact3": [row for row in out_rows if not row["exact3"]],
    }
    summary = {
        name: {
            "n": len(items),
            "accuracy": rate(sum(1 for row in items if row.get("correct") is True), len(items)),
            "avg_pause_count": mean([float(row["pause_count"]) for row in items]),
            "avg_first_pause_index": mean(
                [float(row["first_pause_index"]) for row in items if row.get("first_pause_index") is not None]
            ),
        }
        for name, items in groups.items()
    }
    output_dir = Path(args.output_dir)
    write_json(output_dir / "gsm8k_over_emission_summary.json", summary)
    write_csv(output_dir / "gsm8k_over_emission_rows.csv", out_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage2->Stage3 gate audits without changing training code.")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare-inputs")
    prep.add_argument("--stage1_prepared_root", required=True)
    prep.add_argument("--output_dir", required=True)
    prep.add_argument("--max_pairs_per_source", type=int, default=64)
    prep.add_argument("--seed", type=int, default=260707)

    summ = sub.add_parser("summarize-generation")
    summ.add_argument("--generation_jsonl", action="append", required=True)
    summ.add_argument("--output_dir", required=True)

    gsm = sub.add_parser("gsm8k")
    gsm.add_argument("--generation_jsonl", required=True)
    gsm.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare-inputs":
        command_prepare_inputs(args)
    elif args.command == "summarize-generation":
        command_summarize_generation(args)
    elif args.command == "gsm8k":
        command_gsm8k(args)
    else:
        raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected_json_object:{path}")
    return payload


def target_from_path(path: Path, run_root: Path) -> dict[str, str] | None:
    try:
        rel = path.relative_to(run_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 6:
        return None
    condition, direction, dataset, target = parts[:4]
    mode = "missing"
    if len(parts) >= 7 and parts[4].startswith("mode_"):
        mode, seed, alpha = parts[4:7]
    else:
        seed, alpha = parts[4:6]
    return {
        "condition": condition,
        "direction": direction,
        "dataset": dataset,
        "target": target,
        "mode": mode.replace("mode_", "", 1),
        "seed_alpha": f"{seed}/{alpha}",
    }


def row_relative_norms(row: dict[str, Any]) -> list[float]:
    stats = row.get("hook_stats") or {}
    values = stats.get("applied_relative_norms") or []
    return [float(item) for item in values if item is not None]


def summarize(run_root: Path, *, condition: str, direction: str) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    manifest_checks: list[dict[str, Any]] = []
    row_checks: list[dict[str, Any]] = []
    files = sorted(run_root.glob(f"condition_{condition}/direction_{direction}/**/generations.jsonl"))
    for path in files:
        parsed = target_from_path(path, run_root)
        if parsed is None:
            continue
        dataset = parsed["dataset"]
        target = parsed["target"]
        seed_alpha = parsed["seed_alpha"]
        groups.setdefault((dataset, seed_alpha, target), [])
        manifest_path = path.with_suffix(".manifest.json")
        manifest_status = "pass"
        manifest_strength_mode = None
        if not manifest_path.exists():
            manifest_status = "missing_manifest"
        else:
            manifest = read_json(manifest_path)
            manifest_strength_mode = manifest.get("strength_mode")
        manifest_checks.append(
            {
                "path": str(path),
                "manifest_path": str(manifest_path),
                "dataset": dataset,
                "seed_alpha": seed_alpha,
                "target": target,
                "path_strength_mode": parsed["mode"],
                "manifest_strength_mode": manifest_strength_mode,
                "status": manifest_status,
            }
        )
        for row in read_jsonl(path):
            if row.get("skip_judge"):
                continue
            values = row_relative_norms(row)
            stats = row.get("hook_stats") or {}
            alpha0 = str(seed_alpha).endswith("alpha_0p0")
            if not alpha0:
                expected_count = int(stats.get("num_target_tokens") or 0)
                status = "pass"
                if str(stats.get("strength_mode") or "") != str(manifest_strength_mode or ""):
                    status = "row_wrong_strength_mode"
                elif len(values) != expected_count or expected_count <= 0:
                    status = "row_norm_count_mismatch"
                row_checks.append(
                    {
                        "path": str(path),
                        "dataset": dataset,
                        "seed_alpha": seed_alpha,
                        "target": target,
                        "row_id": row.get("id"),
                        "n_values": len(values),
                        "num_target_tokens": expected_count,
                        "row_strength_mode": stats.get("strength_mode"),
                        "manifest_strength_mode": manifest_strength_mode,
                        "status": status,
                    }
                )
            groups[(dataset, seed_alpha, target)].extend(values)
    summaries = []
    for (dataset, seed_alpha, target), values in sorted(groups.items()):
        summaries.append(
            {
                "dataset": dataset,
                "seed_alpha": seed_alpha,
                "target": target,
                "n_values": len(values),
                "mean_applied_relative_norm": mean(values) if values else None,
                "max_applied_relative_norm": max(values) if values else None,
                "min_applied_relative_norm": min(values) if values else None,
                "_values": values,
            }
        )
    return {
        "run_root": str(run_root),
        "condition": condition,
        "direction": direction,
        "targets": summaries,
        "manifest_checks": manifest_checks,
        "row_checks": row_checks,
    }


def max_relative_gap(values: list[float], reference: float) -> float | None:
    if not values:
        return None
    denom = max(abs(float(reference)), 1e-12)
    return max(abs(float(value) - float(reference)) / denom for value in values)


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "_values"}


def compare(
    summary: dict[str, Any],
    *,
    reference_target: str,
    compare_targets: list[str],
    tolerance_ratio: float,
    min_nonzero_mean: float,
    expected_strength_mode: str,
    include_alpha0: bool = False,
) -> dict[str, Any]:
    by_key = {
        (row["dataset"], row["seed_alpha"], row["target"]): row
        for row in summary["targets"]
        if row["mean_applied_relative_norm"] is not None
    }
    checks = []
    present_keys = {
        (str(row["dataset"]), str(row["seed_alpha"]), str(row["target"]))
        for row in summary["targets"]
        if include_alpha0 or not str(row["seed_alpha"]).endswith("alpha_0p0")
    }
    reference_keys = sorted(
        {
            (str(row["dataset"]), str(row["seed_alpha"]))
            for row in summary["targets"]
            if str(row["target"]) == reference_target
            and row["mean_applied_relative_norm"] is not None
            and (include_alpha0 or not str(row["seed_alpha"]).endswith("alpha_0p0"))
        }
    )
    for row in summary["targets"]:
        target = str(row["target"])
        if target not in compare_targets:
            continue
        if not include_alpha0 and str(row["seed_alpha"]).endswith("alpha_0p0"):
            continue
        ref = by_key.get((row["dataset"], row["seed_alpha"], reference_target))
        if ref is None:
            checks.append({**public_row(row), "reference_target": reference_target, "status": "missing_reference"})
            continue
        if row["mean_applied_relative_norm"] is None:
            checks.append({**public_row(row), "reference_target": reference_target, "status": "no_values"})
            continue
        ref_mean = float(ref["mean_applied_relative_norm"])
        cmp_mean = float(row["mean_applied_relative_norm"])
        if abs(ref_mean) <= min_nonzero_mean:
            checks.append({**public_row(row), "reference_target": reference_target, "reference_mean": ref_mean, "status": "reference_zero"})
            continue
        if abs(cmp_mean) <= min_nonzero_mean:
            checks.append(
                {
                    **public_row(row),
                    "reference_target": reference_target,
                    "reference_mean": ref_mean,
                    "target_mean": cmp_mean,
                    "status": "target_zero",
                }
            )
            continue
        denom = max(abs(ref_mean), 1e-12)
        ratio = abs(cmp_mean - ref_mean) / denom
        ref_values = [float(value) for value in ref.get("_values", [])]
        cmp_values = [float(value) for value in row.get("_values", [])]
        ref_token_gap = max_relative_gap(ref_values, ref_mean)
        target_token_gap = max_relative_gap(cmp_values, ref_mean)
        status = "pass"
        if ratio > tolerance_ratio:
            status = "fail_mean_gap"
        if ref_token_gap is None or ref_token_gap > tolerance_ratio:
            status = "fail_reference_token_gap"
        if target_token_gap is None or target_token_gap > tolerance_ratio:
            status = "fail_target_token_gap"
        checks.append(
            {
                "dataset": row["dataset"],
                "seed_alpha": row["seed_alpha"],
                "reference_target": reference_target,
                "target": target,
                "reference_mean": ref_mean,
                "target_mean": cmp_mean,
                "relative_gap": ratio,
                "reference_token_max_relative_gap": ref_token_gap,
                "target_token_max_relative_gap": target_token_gap,
                "tolerance_ratio": tolerance_ratio,
                "status": status,
            }
        )
    for dataset, seed_alpha in reference_keys:
        for target in compare_targets:
            if (dataset, seed_alpha, target) not in present_keys:
                checks.append(
                    {
                        "dataset": dataset,
                        "seed_alpha": seed_alpha,
                        "reference_target": reference_target,
                        "target": target,
                        "status": "missing_target_arm",
                    }
                )
    targets = [public_row(row) for row in summary["targets"]]
    manifest_checks = []
    for row in summary.get("manifest_checks", []):
        status = str(row.get("status") or "")
        if status == "pass":
            if str(row.get("path_strength_mode") or "") != expected_strength_mode:
                status = "wrong_path_strength_mode"
            elif str(row.get("manifest_strength_mode") or "") != expected_strength_mode:
                status = "wrong_manifest_strength_mode"
        manifest_checks.append({**row, "expected_strength_mode": expected_strength_mode, "status": status})
    row_checks = []
    for row in summary.get("row_checks", []):
        status = str(row.get("status") or "")
        if status == "pass" and str(row.get("row_strength_mode") or "") != expected_strength_mode:
            status = "row_wrong_strength_mode"
        row_checks.append({**row, "expected_strength_mode": expected_strength_mode, "status": status})
    manifest_pass = bool(manifest_checks) and all(row["status"] == "pass" for row in manifest_checks)
    row_pass = all(row["status"] == "pass" for row in row_checks)
    return {
        **summary,
        "targets": targets,
        "manifest_checks": manifest_checks,
        "row_checks": row_checks,
        "matched_strength_checks": checks,
        "matched_strength_pass": manifest_pass and row_pass and bool(checks) and all(c["status"] == "pass" for c in checks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Stage4 matched-strength norms across steering target arms.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--condition", default="gprs")
    parser.add_argument("--direction", default="main")
    parser.add_argument("--reference_target", default="pause_all3")
    parser.add_argument("--compare_targets", default="content_pre_pause_2_4,post_pause_1_3")
    parser.add_argument("--expected_strength_mode", required=True)
    parser.add_argument("--tolerance_ratio", type=float, default=0.01)
    parser.add_argument("--min_nonzero_mean", type=float, default=1e-8)
    parser.add_argument("--include_alpha0", action="store_true")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    summary = summarize(Path(args.run_root), condition=args.condition, direction=args.direction)
    payload = compare(
        summary,
        reference_target=args.reference_target,
        compare_targets=[piece.strip() for piece in args.compare_targets.split(",") if piece.strip()],
        tolerance_ratio=float(args.tolerance_ratio),
        min_nonzero_mean=float(args.min_nonzero_mean),
        expected_strength_mode=str(args.expected_strength_mode),
        include_alpha0=bool(args.include_alpha0),
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    if not payload["matched_strength_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

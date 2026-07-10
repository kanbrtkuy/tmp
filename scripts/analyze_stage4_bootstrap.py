#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


UNSAFE_LABELS = {"unsafe", "unsafe_valid", "harmful", "1", "true", "yes"}


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_stage4_path(path: Path, run_root: Path) -> dict[str, str] | None:
    try:
        rel = path.relative_to(run_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 8 or not parts[4].startswith("mode_"):
        return None
    condition, direction, dataset, target, mode, seed, alpha = parts[:7]
    return {
        "condition": condition.replace("condition_", "", 1),
        "direction": direction.replace("direction_", "", 1),
        "dataset": dataset,
        "target": target,
        "strength_mode": mode.replace("mode_", "", 1),
        "seed": seed.replace("seed_", "", 1),
        "alpha": alpha.replace("alpha_", "", 1),
    }


def judge_label(row: dict[str, Any]) -> str:
    for key in ("binary_safety_label", "judge_label", "normalized_label", "safety_label", "label"):
        value = row.get(key)
        if value is not None:
            return str(value).strip().lower()
    return "unlabeled"


def is_unsafe(row: dict[str, Any]) -> float:
    label = judge_label(row)
    return 1.0 if label in UNSAFE_LABELS else 0.0


def load_arm(
    gen_path: Path,
    *,
    normalized_filename: str,
    fail_on_skip_judge: bool,
) -> dict[str, Any]:
    norm_path = gen_path.parent / normalized_filename
    if not norm_path.exists():
        raise SystemExit(f"Missing normalized judge file: {norm_path}")
    gen_rows = read_jsonl(gen_path)
    if fail_on_skip_judge:
        skipped = [row for row in gen_rows if row.get("skip_judge")]
        if skipped:
            raise SystemExit(f"Refusing bootstrap with skip_judge rows: {gen_path} n={len(skipped)}")
    norm_rows = read_jsonl(norm_path)
    labels_by_id = {str(row.get("id")): row for row in norm_rows}
    values: dict[str, float] = {}
    missing_labels = []
    for row in gen_rows:
        row_id = str(row.get("id"))
        if row.get("skip_judge"):
            continue
        if row_id not in labels_by_id:
            missing_labels.append(row_id)
            continue
        source_key = str(row.get("source_row_index"))
        values[source_key] = is_unsafe(labels_by_id[row_id])
    if missing_labels:
        raise SystemExit(f"Missing normalized labels for {gen_path}: n={len(missing_labels)}")
    return {"path": str(gen_path), "norm_path": str(norm_path), "values": values, "n": len(values)}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def paired_delta(
    left: dict[str, float],
    right: dict[str, float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    left_keys = set(left)
    right_keys = set(right)
    if left_keys != right_keys:
        raise SystemExit(
            "Stage4 paired bootstrap row-set mismatch: "
            f"left_only={len(left_keys - right_keys)} right_only={len(right_keys - left_keys)}"
        )
    keys = sorted(left_keys)
    if not keys:
        raise SystemExit("Stage4 paired bootstrap needs at least one paired row.")
    diffs = [float(left[key]) - float(right[key]) for key in keys]
    rng = random.Random(seed)
    boot = []
    for _ in range(int(samples)):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boot.append(mean(sample))
    return {
        "n": len(keys),
        "left_rate": mean(float(left[key]) for key in keys),
        "right_rate": mean(float(right[key]) for key in keys),
        "left_minus_right": mean(diffs),
        "ci_low": percentile(boot, 0.025),
        "ci_high": percentile(boot, 0.975),
        "bootstrap_samples": int(samples),
    }


def arm_key(meta: dict[str, str]) -> tuple[str, str, str, str, str, str, str]:
    return (
        meta["dataset"],
        meta["seed"],
        meta["alpha"],
        meta["condition"],
        meta["direction"],
        meta["target"],
        meta["strength_mode"],
    )


def load_arms(run_root: Path, *, normalized_filename: str, fail_on_skip_judge: bool) -> dict[tuple[str, str, str, str, str, str, str], dict[str, Any]]:
    arms = {}
    for gen_path in sorted(run_root.glob("condition_*/direction_*/*/*/mode_*/*/alpha_*/generations.jsonl")):
        meta = parse_stage4_path(gen_path, run_root)
        if meta is None:
            continue
        arm = load_arm(gen_path, normalized_filename=normalized_filename, fail_on_skip_judge=fail_on_skip_judge)
        arm["meta"] = meta
        arms[arm_key(meta)] = arm
    return arms


def analyze(run_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    arms = load_arms(run_root, normalized_filename=args.normalized_filename, fail_on_skip_judge=not args.allow_skip_judge)
    by_dataset_seed_mode: dict[tuple[str, str, str], list[tuple[tuple[str, str, str, str, str, str, str], dict[str, Any]]]] = defaultdict(list)
    for key, arm in arms.items():
        dataset, seed, _alpha, _condition, _direction, _target, mode = key
        by_dataset_seed_mode[(dataset, seed, mode)].append((key, arm))
    comparisons = []
    for (dataset, seed, mode), items in sorted(by_dataset_seed_mode.items()):
        lookup = {key[2:6]: arm for key, arm in items}
        a2 = lookup.get(("0p0", "ppc", "main", "pause_all3"))
        if a2 is None:
            continue
        alphas = sorted({key[2] for key, _arm in items if key[3] == "gprs" and key[5] == "pause_all3" and key[2] != "0p0"})
        for alpha in alphas:
            a3 = lookup.get((alpha, "gprs", "main", "pause_all3"))
            a5 = lookup.get((alpha, "gprs", "random", "pause_all3"))
            a4_candidates = [
                lookup.get((alpha, "gprs", "main", "content_pre_pause_2_4")),
                lookup.get((alpha, "gprs", "main", "post_pause_1_3")),
            ]
            a4_candidates = [arm for arm in a4_candidates if arm is not None]
            if a3 is None or a5 is None or not a4_candidates:
                raise SystemExit(f"Missing Stage4 comparison arms: dataset={dataset} seed={seed} mode={mode} alpha={alpha}")
            a4_best = min(a4_candidates, key=lambda arm: mean(arm["values"].values()))
            comparisons.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "strength_mode": mode,
                    "alpha": alpha,
                    "a3_minus_a2": paired_delta(a3["values"], a2["values"], samples=args.bootstrap_samples, seed=args.seed),
                    "a3_minus_best_a4": paired_delta(a3["values"], a4_best["values"], samples=args.bootstrap_samples, seed=args.seed + 17),
                    "a3_minus_a5": paired_delta(a3["values"], a5["values"], samples=args.bootstrap_samples, seed=args.seed + 29),
                    "best_a4_path": a4_best["path"],
                }
            )
    if not comparisons:
        raise SystemExit(f"No complete Stage4 comparison sets found under {run_root}")
    return {
        "run_root": str(run_root),
        "normalized_filename": args.normalized_filename,
        "n_arms": len(arms),
        "n_comparisons": len(comparisons),
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap analysis for clean Stage4 A0-A5 battery.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--normalized_filename", default="open_judges_normalized.jsonl")
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=260710)
    parser.add_argument("--allow_skip_judge", action="store_true")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()
    payload = analyze(Path(args.run_root), args)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        write_json(Path(args.output_json), payload)
    print(text)


if __name__ == "__main__":
    main()

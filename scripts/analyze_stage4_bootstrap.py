#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


SAFE_LABELS = {"safe", "safe_refusal", "harmless", "benign", "refusal", "0", "false", "no"}
UNSAFE_LABELS = {"unsafe", "unsafe_valid", "harmful", "1", "true", "yes"}
UNKNOWN_LABELS = {
    "",
    "partial",
    "ambiguous_partial",
    "garbage",
    "low_quality_garbage",
    "unlabeled",
    "unknown",
    "none",
    "null",
}


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


def label_to_unsafe(label: str, *, allow_unknown_labels: bool) -> float | None:
    if label in UNSAFE_LABELS:
        return 1.0
    if label in SAFE_LABELS:
        return 0.0
    if label in UNKNOWN_LABELS and allow_unknown_labels:
        return None
    raise ValueError(f"unusable_judge_label:{label or '<empty>'}")


def load_arm(
    gen_path: Path,
    *,
    normalized_filename: str,
    fail_on_skip_judge: bool,
    allow_unknown_labels: bool,
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
    label_counts: dict[str, int] = defaultdict(int)
    dropped_unknown: list[str] = []
    for row in gen_rows:
        row_id = str(row.get("id"))
        if row.get("skip_judge"):
            continue
        if row_id not in labels_by_id:
            missing_labels.append(row_id)
            continue
        source_key = str(row.get("source_row_index"))
        label = judge_label(labels_by_id[row_id])
        label_counts[label] += 1
        try:
            unsafe_value = label_to_unsafe(label, allow_unknown_labels=allow_unknown_labels)
        except ValueError as exc:
            raise SystemExit(
                f"Refusing bootstrap with non-binary judge label in {norm_path}: id={row_id} label={label!r}. "
                "Partial/garbage/unlabeled rows can make degeneration look falsely safe; rerun judges or pass "
                "--allow_unknown_labels to drop them explicitly."
            ) from exc
        if unsafe_value is None:
            dropped_unknown.append(row_id)
            continue
        values[source_key] = unsafe_value
    if missing_labels:
        raise SystemExit(f"Missing normalized labels for {gen_path}: n={len(missing_labels)}")
    if dropped_unknown and not allow_unknown_labels:
        raise SystemExit(f"Internal error: dropped unknown labels without permission: {gen_path}")
    return {
        "path": str(gen_path),
        "norm_path": str(norm_path),
        "values": values,
        "n": len(values),
        "n_generated": len(gen_rows),
        "n_normalized": len(norm_rows),
        "label_counts": dict(sorted(label_counts.items())),
        "n_dropped_unknown_labels": len(dropped_unknown),
    }


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


def load_arms(
    run_root: Path,
    *,
    normalized_filename: str,
    fail_on_skip_judge: bool,
    allow_unknown_labels: bool,
) -> dict[tuple[str, str, str, str, str, str, str], dict[str, Any]]:
    arms = {}
    for gen_path in sorted(run_root.glob("condition_*/direction_*/*/*/mode_*/*/alpha_*/generations.jsonl")):
        meta = parse_stage4_path(gen_path, run_root)
        if meta is None:
            continue
        arm = load_arm(
            gen_path,
            normalized_filename=normalized_filename,
            fail_on_skip_judge=fail_on_skip_judge,
            allow_unknown_labels=allow_unknown_labels,
        )
        arm["meta"] = meta
        arms[arm_key(meta)] = arm
    return arms


def analyze(run_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    arms = load_arms(
        run_root,
        normalized_filename=args.normalized_filename,
        fail_on_skip_judge=not args.allow_skip_judge,
        allow_unknown_labels=bool(args.allow_unknown_labels),
    )
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
                    "arm_paths": {
                        "a2_ppc_no_steer": a2["path"],
                        "a3_pause_gprs": a3["path"],
                        "a4_best_diagnostic_control": a4_best["path"],
                        "a5_random_direction": a5["path"],
                    },
                    "arm_label_counts": {
                        "a2_ppc_no_steer": a2["label_counts"],
                        "a3_pause_gprs": a3["label_counts"],
                        "a4_best_diagnostic_control": a4_best["label_counts"],
                        "a5_random_direction": a5["label_counts"],
                    },
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
        "allow_unknown_labels": bool(args.allow_unknown_labels),
        "arms": [
            {
                "path": arm["path"],
                "norm_path": arm["norm_path"],
                "meta": arm["meta"],
                "n": arm["n"],
                "n_generated": arm["n_generated"],
                "n_normalized": arm["n_normalized"],
                "label_counts": arm["label_counts"],
                "n_dropped_unknown_labels": arm["n_dropped_unknown_labels"],
            }
            for _key, arm in sorted(arms.items())
        ],
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap analysis for clean Stage4 A0-A5 battery.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--normalized_filename", default="open_judges_normalized.jsonl")
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=260710)
    parser.add_argument("--allow_skip_judge", action="store_true")
    parser.add_argument(
        "--allow_unknown_labels",
        action="store_true",
        help="Drop partial/garbage/unlabeled judge rows explicitly instead of failing closed.",
    )
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()
    payload = analyze(Path(args.run_root), args)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        write_json(Path(args.output_json), payload)
    print(text)


if __name__ == "__main__":
    main()

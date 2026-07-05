#!/usr/bin/env python3
"""Train MLP probes from restored Intra-Pause Probe hidden states.

This launcher is for the post-linear ablation stage. It assumes that the
Intra-Pause Probe data/hidden tarballs have already been restored to their
original /workspace paths and trains MLP probes on exactly the same NPZ splits
used by the linear runs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LAYERS = (7, 14, 17, 21, 22, 28)
POSITIONS = (
    "pause_0",
    "pause_1",
    "pause_2",
    "pre_pause_1",
    "pre_pause_2",
    "pre_pause_3",
    "post_pause_1",
    "post_pause_2",
    "post_pause_3",
    "control_cot_3",
    "control_cot_4",
    "cot_3",
    "cot_4",
    "cot_7",
    "cot_8",
)


@dataclass(frozen=True)
class Recipe:
    name: str
    hidden_dir: Path
    prefix: str
    single_out: Path
    pooled_out: Path
    log_dir: Path


@dataclass(frozen=True)
class PooledSpec:
    name: str
    positions: tuple[str, ...]
    layers: tuple[int, ...]
    layer_combine: str
    position_pool: str


def parse_csv(value: str) -> list[str]:
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def parse_layers(value: str) -> list[int]:
    return [int(piece.strip()) for piece in value.split(",") if piece.strip()]


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def run_logged(cmd: list[str], log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_rows(rows: list[dict[str, Any]], path_prefix: Path) -> None:
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    with path_prefix.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    keys = list(rows[0]) if rows else []
    with path_prefix.with_suffix(".tsv").open("w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for row in rows:
            f.write("\t".join(fmt(row.get(key)) for key in keys) + "\n")


def recipe_for(name: str) -> Recipe:
    if name == "3to1":
        hidden_dir = Path("/workspace/hidden_states/intra_pause_probe_3to1_260616")
        return Recipe(
            name="3to1",
            hidden_dir=hidden_dir,
            prefix="intra_pause_3to1_final1600",
            single_out=Path("runs/probes/intra_pause_probe_3to1_mlp_single_260618"),
            pooled_out=Path("runs/probes/intra_pause_probe_3to1_mlp_pooled_260618"),
            log_dir=Path("/workspace/logs/intra_pause_probe_3to1_mlp_260618"),
        )
    if name == "1to1":
        hidden_dir = Path("/workspace/hidden_states/intra_pause_probe_1to1_260616")
        return Recipe(
            name="1to1",
            hidden_dir=hidden_dir,
            prefix="intra_pause_1to1_final1600",
            single_out=Path("runs/probes/intra_pause_probe_1to1_mlp_single_260618"),
            pooled_out=Path("runs/probes/intra_pause_probe_1to1_mlp_pooled_260618"),
            log_dir=Path("/workspace/logs/intra_pause_probe_1to1_mlp_260618"),
        )
    raise ValueError(f"Unknown recipe: {name}")


def split_npz(recipe: Recipe, split: str, layers: list[int]) -> Path:
    suffix = "_".join(str(layer) for layer in layers)
    return recipe.hidden_dir / f"{recipe.prefix}_{split}_layers_{suffix}.npz"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def pooled_specs(layers: list[int]) -> list[PooledSpec]:
    pause = ("pause_0", "pause_1", "pause_2")
    pre = ("pre_pause_1", "pre_pause_2", "pre_pause_3")
    post = ("post_pause_1", "post_pause_2", "post_pause_3")
    specs: list[PooledSpec] = []
    for layer in layers:
        specs.extend(
            [
                PooledSpec(f"pause_mean_l{layer}", pause, (layer,), "mean", "mean"),
                PooledSpec(f"pause_concat_l{layer}", pause, (layer,), "mean", "concat"),
                PooledSpec(f"pre_mean_l{layer}", pre, (layer,), "mean", "mean"),
                PooledSpec(f"post_mean_l{layer}", post, (layer,), "mean", "mean"),
            ]
        )
    all_layers = tuple(layers)
    specs.extend(
        [
            PooledSpec("pause_mean_layers_mean", pause, all_layers, "mean", "mean"),
            PooledSpec("pause_mean_layers_concat", pause, all_layers, "concat", "mean"),
            PooledSpec("pause_concat_layers_concat", pause, all_layers, "concat", "concat"),
            PooledSpec(
                "control_cot3_cot4_concat_layers_concat",
                ("control_cot_3", "control_cot_4"),
                all_layers,
                "concat",
                "concat",
            ),
        ]
    )
    return specs


def run_single(args: argparse.Namespace, recipe: Recipe, layers: list[int], positions: list[str]) -> None:
    cmd = [
        args.python,
        "scripts/probe/run_position_scan_pilot.py",
        "--train_npz",
        str(split_npz(recipe, "train", layers)),
        "--val_npz",
        str(split_npz(recipe, "val", layers)),
        "--test_npz",
        str(split_npz(recipe, "test", layers)),
        "--eval_npz",
        f"reasoningshield_test={split_npz(recipe, 'source_heldout_reasoningshield_test', layers)}",
        "--out_root",
        str(recipe.single_out),
        "--log_dir",
        str(recipe.log_dir),
        "--positions",
        ",".join(positions),
        "--layers",
        ",".join(str(layer) for layer in layers),
        "--model_kinds",
        "mlp",
        "--jobs",
        str(args.single_jobs),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch_size",
        str(args.single_batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--threshold_max_fpr",
        str(args.threshold_max_fpr),
        "--sample_weight_mode",
        args.sample_weight_mode,
        "--device",
        args.device,
        "--python",
        args.python,
    ]
    if args.skip_existing:
        cmd.append("--skip_existing")
    run_logged(cmd, recipe.log_dir / "single_mlp_grid.log", args.repo_dir)


def train_pooled_cmd(args: argparse.Namespace, recipe: Recipe, spec: PooledSpec, layers: list[int]) -> list[str]:
    return [
        args.python,
        "scripts/probe/train_probe.py",
        "--train_npz",
        str(split_npz(recipe, "train", layers)),
        "--val_npz",
        str(split_npz(recipe, "val", layers)),
        "--test_npz",
        str(split_npz(recipe, "test", layers)),
        "--output_dir",
        str(recipe.pooled_out / spec.name),
        "--positions",
        ",".join(spec.positions),
        "--layers",
        ",".join(str(layer) for layer in spec.layers),
        "--layer_combine",
        spec.layer_combine,
        "--position_pool",
        spec.position_pool,
        "--hidden_sizes",
        "clear_default",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.pooled_batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--sample_weight_mode",
        args.sample_weight_mode,
        "--threshold_max_fpr",
        str(args.threshold_max_fpr),
        "--patience",
        str(args.patience),
        "--device",
        args.device,
    ]


def eval_pooled_cmd(args: argparse.Namespace, recipe: Recipe, spec: PooledSpec, layers: list[int]) -> list[str]:
    return [
        args.python,
        "scripts/probe/evaluate_probe.py",
        "--probe_pt",
        str(recipe.pooled_out / spec.name / "probe.pt"),
        "--input_npz",
        str(split_npz(recipe, "source_heldout_reasoningshield_test", layers)),
        "--output_dir",
        str(recipe.pooled_out / f"eval_reasoningshield_test_{spec.name}"),
        "--batch_size",
        str(args.eval_batch_size),
        "--device",
        args.device,
    ]


def run_pooled_one(args: argparse.Namespace, recipe: Recipe, spec: PooledSpec, layers: list[int]) -> str:
    run_dir = recipe.pooled_out / spec.name
    log_path = recipe.log_dir / f"pooled_mlp_{spec.name}.log"
    metrics_path = run_dir / "metrics.json"
    probe_path = run_dir / "probe.pt"
    if not (args.skip_existing and metrics_path.exists() and probe_path.exists()):
        run_logged(train_pooled_cmd(args, recipe, spec, layers), log_path, args.repo_dir)
    require_file(probe_path)
    eval_metrics = recipe.pooled_out / f"eval_reasoningshield_test_{spec.name}" / "metrics.json"
    if not (args.skip_existing and eval_metrics.exists()):
        run_logged(eval_pooled_cmd(args, recipe, spec, layers), log_path, args.repo_dir)
    return spec.name


def nested_metric(path: Path, split: str) -> dict[str, Any]:
    return read_json(path)["metrics"][split]


def plain_metric(path: Path) -> dict[str, Any]:
    return read_json(path)["metrics"]


def summarize_pooled(recipe: Recipe, specs: list[PooledSpec]) -> None:
    rows = []
    for spec in specs:
        metrics_path = recipe.pooled_out / spec.name / "metrics.json"
        eval_path = recipe.pooled_out / f"eval_reasoningshield_test_{spec.name}" / "metrics.json"
        if not metrics_path.exists() or not eval_path.exists():
            continue
        val = nested_metric(metrics_path, "val")
        test = nested_metric(metrics_path, "test")
        heldout = plain_metric(eval_path)
        row = {
            "name": spec.name,
            "positions": ",".join(spec.positions),
            "layers": ",".join(str(layer) for layer in spec.layers),
            "layer_combine": spec.layer_combine,
            "position_pool": spec.position_pool,
            "val_auroc": val.get("auroc"),
            "val_recall": val.get("recall"),
            "val_fpr": val.get("fpr"),
            "test_auroc": test.get("auroc"),
            "test_auprc": test.get("auprc"),
            "test_recall": test.get("recall"),
            "test_fpr": test.get("fpr"),
            "reasoningshield_test_auroc": heldout.get("auroc"),
            "reasoningshield_test_auprc": heldout.get("auprc"),
            "reasoningshield_test_recall": heldout.get("recall"),
            "reasoningshield_test_fpr": heldout.get("fpr"),
        }
        rows.append(row)
    write_rows(rows, recipe.pooled_out / "summary_grid")
    ranked_test = sorted(rows, key=lambda row: float(row.get("test_auroc") or float("nan")), reverse=True)
    ranked_heldout = sorted(
        rows,
        key=lambda row: float(row.get("reasoningshield_test_auroc") or float("nan")),
        reverse=True,
    )
    write_rows(ranked_test, recipe.pooled_out / "summary_by_test_auroc")
    write_rows(ranked_heldout, recipe.pooled_out / "summary_by_reasoningshield_test_auroc")


def run_pooled(args: argparse.Namespace, recipe: Recipe, layers: list[int]) -> None:
    specs = pooled_specs(layers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.pooled_jobs, len(specs))) as executor:
        futures = {executor.submit(run_pooled_one, args, recipe, spec, layers): spec for spec in specs}
        for future in concurrent.futures.as_completed(futures):
            spec = futures[future]
            run_name = future.result()
            print(f"finished pooled {recipe.name} {run_name} ({','.join(spec.positions)})", flush=True)
    summarize_pooled(recipe, specs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", default="3to1,1to1")
    parser.add_argument("--phases", default="single,pooled")
    parser.add_argument("--layers", default=",".join(str(layer) for layer in LAYERS))
    parser.add_argument("--positions", default=",".join(POSITIONS))
    parser.add_argument("--repo_dir", type=Path, default=Path("/workspace/PauseProbe"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--single_jobs", type=int, default=12)
    parser.add_argument("--pooled_jobs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--single_batch_size", type=int, default=2048)
    parser.add_argument("--pooled_batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--threshold_max_fpr", type=float, default=0.05)
    parser.add_argument("--sample_weight_mode", choices=("none", "label", "source", "source_label"), default="source_label")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    positions = parse_csv(args.positions)
    recipes = [recipe_for(name) for name in parse_csv(args.recipes)]
    phases = set(parse_csv(args.phases))

    for recipe in recipes:
        for split in ("train", "val", "test", "source_heldout_reasoningshield_test"):
            require_file(split_npz(recipe, split, layers))
        recipe.log_dir.mkdir(parents=True, exist_ok=True)
        if "single" in phases:
            run_single(args, recipe, layers, positions)
        if "pooled" in phases:
            run_pooled(args, recipe, layers)


if __name__ == "__main__":
    main()

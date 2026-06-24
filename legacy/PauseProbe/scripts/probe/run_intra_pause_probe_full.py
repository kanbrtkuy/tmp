#!/usr/bin/env python3
"""Run the full Intra-Pause Probe experiment.

This launcher is tuned for a single high-memory GPU such as an A6000:

1. prepare external normalized trajectories when needed
2. apply source-label caps and rewrite rows with intra-CoT pauses before cot_3
3. extract hidden states with the final intra-pause SFT model
4. run a position x layer heatmap over pause/pre/post/control positions
5. run pause-span pooled and multi-layer ablations

The heavy lifting stays in the existing lower-level scripts so that this file is
mostly orchestration and resumability glue.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "/workspace/outputs/deepseek_intra_pause_cot3_trusted_cot_18k_lr2e5_260615/final"
DEFAULT_BASE_DATA_DIR = "data/external_probe_v0"
DEFAULT_DATA_DIR = "data/intra_pause_probe_v0"
DEFAULT_HIDDEN_DIR = "data/hidden"
DEFAULT_SOURCES = (
    "reasoningshield_train_sft",
    "reasoningshield_train_dpo",
    "reasoningshield_test",
    "star41k",
    "star1",
    "aidsafe_beavertails",
    "aidsafe_dataadvisor",
    "unsafechain_selected",
    "harmthoughts",
)
DEFAULT_HELDOUT_SOURCES = ("reasoningshield_test",)
DEFAULT_LAYERS = (7, 14, 17, 21, 22, 28)
DEFAULT_SINGLE_POSITIONS = (
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
DEFAULT_COT_OFFSETS = (0, 1, 2, 3, 4, 5, 7, 8)


@dataclass(frozen=True)
class SplitSpec:
    name: str
    input_json: Path
    output_npz: Path
    metadata_jsonl: Path
    manifest_json: Path


@dataclass(frozen=True)
class PooledSpec:
    name: str
    positions: tuple[str, ...]
    layers: tuple[int, ...]
    layer_combine: str
    position_pool: str
    model_kind: str = "linear"


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def parse_layers(value: str) -> list[int]:
    return [int(piece.strip()) for piece in value.split(",") if piece.strip()]


def layer_suffix(layers: list[int] | tuple[int, ...]) -> str:
    return "_".join(str(layer).replace("-", "m") for layer in layers)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_rows(rows: list[dict[str, Any]], path_prefix: Path) -> None:
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    with path_prefix.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, default=str)
    keys = list(rows[0]) if rows else []
    with path_prefix.with_suffix(".tsv").open("w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for row in rows:
            f.write("\t".join(fmt(row.get(key)) for key in keys) + "\n")


def run_logged(cmd: list[str], log_path: Path, dry_run: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "$ " + " ".join(cmd)
    print(rendered)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(rendered + "\n")
        log.flush()
        if dry_run:
            log.write("[dry-run] skipped\n")
            return
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def model_hidden_sizes(model_kind: str) -> str:
    if model_kind == "linear":
        return ""
    if model_kind == "mlp":
        return "clear_default"
    raise ValueError(f"Unknown model kind: {model_kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--base_data_dir", default=DEFAULT_BASE_DATA_DIR)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--hidden_dir", default=DEFAULT_HIDDEN_DIR)
    parser.add_argument("--hidden_prefix", default="intra_pause")
    parser.add_argument("--log_dir", default="logs/intra_pause_probe")
    parser.add_argument("--single_scan_out_root", default="runs/probes/intra_pause_probe_single")
    parser.add_argument("--pooled_out_root", default="runs/probes/intra_pause_probe_pooled")
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument("--heldout_source", action="append", default=list(DEFAULT_HELDOUT_SOURCES))
    parser.add_argument("--recipe", choices=("pilot", "full", "full_1to1"), default="pilot")
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--star_min_score", type=float, default=8.0)
    parser.add_argument("--max_prompt_words", type=int, default=800)
    parser.add_argument("--max_reasoning_words", type=int, default=1600)
    parser.add_argument("--max_final_words", type=int, default=1600)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    parser.add_argument("--positions", default=",".join(DEFAULT_SINGLE_POSITIONS))
    parser.add_argument("--cot_offsets", default=",".join(str(x) for x in DEFAULT_COT_OFFSETS))
    parser.add_argument("--pre_pause_window", type=int, default=3)
    parser.add_argument("--post_pause_window", type=int, default=3)
    parser.add_argument("--extract_batch_size", type=int, default=8)
    parser.add_argument("--extract_max_length", type=int, default=4096)
    parser.add_argument("--extract_jobs", type=int, default=1)
    parser.add_argument("--extract_devices", default="cuda")
    parser.add_argument("--torch_dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--save_dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--scan_jobs", type=int, default=12)
    parser.add_argument("--pooled_jobs", type=int, default=6)
    parser.add_argument("--scan_epochs", type=int, default=30)
    parser.add_argument("--scan_patience", type=int, default=8)
    parser.add_argument("--scan_batch_size", type=int, default=2048)
    parser.add_argument("--scan_eval_batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--threshold_max_fpr", type=float, default=0.05)
    parser.add_argument("--sample_weight_mode", choices=("none", "label", "source", "source_label"), default="source_label")
    parser.add_argument("--probe_device", default="cuda")
    parser.add_argument("--skip_base_data_prep", action="store_true")
    parser.add_argument("--skip_intra_data_prep", action="store_true")
    parser.add_argument("--skip_hidden_extraction", action="store_true")
    parser.add_argument("--skip_single_scan", action="store_true")
    parser.add_argument("--skip_pooled", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.extract_jobs < 1:
        parser.error("--extract_jobs must be >= 1")
    if args.scan_jobs < 1:
        parser.error("--scan_jobs must be >= 1")
    if args.pooled_jobs < 1:
        parser.error("--pooled_jobs must be >= 1")
    return args


def split_specs(args: argparse.Namespace, layers: list[int]) -> dict[str, SplitSpec]:
    data_dir = Path(args.data_dir)
    hidden_dir = Path(args.hidden_dir)
    suffix = layer_suffix(layers)
    specs = {
        "train": SplitSpec(
            "train",
            data_dir / "cotpause" / "train.json",
            hidden_dir / f"{args.hidden_prefix}_train_layers_{suffix}.npz",
            hidden_dir / f"{args.hidden_prefix}_train_layers_{suffix}.metadata.jsonl",
            hidden_dir / f"{args.hidden_prefix}_train_layers_{suffix}.manifest.json",
        ),
        "val": SplitSpec(
            "val",
            data_dir / "cotpause" / "val.json",
            hidden_dir / f"{args.hidden_prefix}_val_layers_{suffix}.npz",
            hidden_dir / f"{args.hidden_prefix}_val_layers_{suffix}.metadata.jsonl",
            hidden_dir / f"{args.hidden_prefix}_val_layers_{suffix}.manifest.json",
        ),
        "test": SplitSpec(
            "test",
            data_dir / "cotpause" / "test.json",
            hidden_dir / f"{args.hidden_prefix}_test_layers_{suffix}.npz",
            hidden_dir / f"{args.hidden_prefix}_test_layers_{suffix}.metadata.jsonl",
            hidden_dir / f"{args.hidden_prefix}_test_layers_{suffix}.manifest.json",
        ),
    }
    for source in args.heldout_source:
        name = f"source_heldout_{source}"
        specs[name] = SplitSpec(
            name,
            data_dir / "cotpause" / f"source_heldout_{source}.json",
            hidden_dir / f"{args.hidden_prefix}_source_heldout_{source}_layers_{suffix}.npz",
            hidden_dir / f"{args.hidden_prefix}_source_heldout_{source}_layers_{suffix}.metadata.jsonl",
            hidden_dir / f"{args.hidden_prefix}_source_heldout_{source}_layers_{suffix}.manifest.json",
        )
    return specs


def run_base_data_prep(args: argparse.Namespace) -> None:
    cmd = [
        args.python,
        "scripts/data/prepare_external_trajectories.py",
        "--output_dir",
        args.base_data_dir,
        "--sources",
        *args.sources,
        "--star_min_score",
        str(args.star_min_score),
        "--max_prompt_words",
        str(args.max_prompt_words),
        "--max_reasoning_words",
        str(args.max_reasoning_words),
        "--max_final_words",
        str(args.max_final_words),
        "--train_ratio",
        str(args.train_ratio),
        "--val_ratio",
        str(args.val_ratio),
        "--split_strategy",
        "source_label_prompt_group",
        "--dedupe_strategy",
        "none",
        "--seed",
        str(args.seed),
    ]
    for source in args.heldout_source:
        cmd.extend(["--heldout_source", source])
    run_logged(cmd, Path(args.log_dir) / "intra_pause_base_data_prep.log", args.dry_run)


def run_intra_data_prep(args: argparse.Namespace) -> None:
    tokenizer = args.tokenizer or args.model
    cmd = [
        args.python,
        "scripts/data/prepare_intra_pause_probe_data.py",
        "--input_dir",
        args.base_data_dir,
        "--output_dir",
        args.data_dir,
        "--tokenizer",
        tokenizer,
        "--recipe",
        args.recipe,
        "--seed",
        str(args.seed),
        "--train_ratio",
        str(args.train_ratio),
        "--val_ratio",
        str(args.val_ratio),
        "--trust_remote_code",
        "--dedupe_strategy",
        "none",
        "--split_strategy",
        "source_label_prompt_group",
    ]
    for source in args.heldout_source:
        cmd.extend(["--heldout_source", source])
    run_logged(cmd, Path(args.log_dir) / "intra_pause_data_rewrite.log", args.dry_run)


def extraction_cmd(args: argparse.Namespace, spec: SplitSpec, layers: list[int], device: str) -> list[str]:
    tokenizer = args.tokenizer or args.model
    return [
        args.python,
        "scripts/probe/extract_hidden_states.py",
        "--model",
        args.model,
        "--tokenizer",
        tokenizer,
        "--input_file",
        str(spec.input_json),
        "--output_npz",
        str(spec.output_npz),
        "--metadata_jsonl",
        str(spec.metadata_jsonl),
        "--manifest_json",
        str(spec.manifest_json),
        "--label_field",
        "trajectory_safety_label",
        "--pause_layout",
        "intra_cot",
        "--pre_pause_window",
        str(args.pre_pause_window),
        "--post_pause_window",
        str(args.post_pause_window),
        "--layers",
        ",".join(str(x) for x in layers),
        "--cot_offsets",
        args.cot_offsets,
        "--batch_size",
        str(args.extract_batch_size),
        "--max_length",
        str(args.extract_max_length),
        "--device",
        device,
        "--torch_dtype",
        args.torch_dtype,
        "--save_dtype",
        args.save_dtype,
        "--trust_remote_code",
        "--compressed",
    ]


def run_extraction_one(args: argparse.Namespace, spec: SplitSpec, layers: list[int], device: str) -> str:
    if args.skip_existing and spec.output_npz.exists() and spec.manifest_json.exists():
        print(f"skip existing extraction: {spec.name} -> {spec.output_npz}")
        return spec.name
    if not args.dry_run:
        require_file(spec.input_json)
    run_logged(
        extraction_cmd(args, spec, layers, device),
        Path(args.log_dir) / f"intra_pause_extract_{spec.name}.log",
        args.dry_run,
    )
    return spec.name


def run_hidden_extraction(args: argparse.Namespace, specs: dict[str, SplitSpec], layers: list[int]) -> None:
    devices = parse_csv(args.extract_devices) or ["cuda"]
    jobs = list(specs.values())
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.extract_jobs, len(jobs))) as executor:
        futures = {}
        for idx, spec in enumerate(jobs):
            device = devices[idx % len(devices)]
            futures[executor.submit(run_extraction_one, args, spec, layers, device)] = (spec, device)
        for future in concurrent.futures.as_completed(futures):
            spec, device = futures[future]
            result = future.result()
            print(f"finished extraction {result} on {device}")


def eval_npz_args(specs: dict[str, SplitSpec]) -> list[str]:
    out: list[str] = []
    for name, spec in specs.items():
        if not name.startswith("source_heldout_"):
            continue
        eval_name = name.removeprefix("source_heldout_")
        out.extend(["--eval_npz", f"{eval_name}={spec.output_npz}"])
    return out


def run_single_scan(args: argparse.Namespace, specs: dict[str, SplitSpec], positions: list[str], layers: list[int]) -> None:
    if not args.dry_run:
        for name in ("train", "val", "test"):
            require_file(specs[name].output_npz)
        for spec in specs.values():
            if spec.name.startswith("source_heldout_"):
                require_file(spec.output_npz)
    cmd = [
        args.python,
        "scripts/probe/run_position_scan_pilot.py",
        "--train_npz",
        str(specs["train"].output_npz),
        "--val_npz",
        str(specs["val"].output_npz),
        "--test_npz",
        str(specs["test"].output_npz),
        *eval_npz_args(specs),
        "--out_root",
        args.single_scan_out_root,
        "--log_dir",
        args.log_dir,
        "--positions",
        ",".join(positions),
        "--layers",
        ",".join(str(x) for x in layers),
        "--model_kinds",
        "linear",
        "--jobs",
        str(args.scan_jobs),
        "--epochs",
        str(args.scan_epochs),
        "--patience",
        str(args.scan_patience),
        "--batch_size",
        str(args.scan_batch_size),
        "--eval_batch_size",
        str(args.scan_eval_batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--threshold_max_fpr",
        str(args.threshold_max_fpr),
        "--sample_weight_mode",
        args.sample_weight_mode,
        "--device",
        args.probe_device,
        "--python",
        args.python,
    ]
    if args.skip_existing:
        cmd.append("--skip_existing")
    run_logged(cmd, Path(args.log_dir) / "intra_pause_single_scan.log", args.dry_run)


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
            PooledSpec("control_cot3_cot4_concat_layers_concat", ("control_cot_3", "control_cot_4"), all_layers, "concat", "concat"),
        ]
    )
    return specs


def train_pooled_cmd(args: argparse.Namespace, spec: PooledSpec, out_dir: Path, splits: dict[str, SplitSpec]) -> list[str]:
    return [
        args.python,
        "scripts/probe/train_probe.py",
        "--train_npz",
        str(splits["train"].output_npz),
        "--val_npz",
        str(splits["val"].output_npz),
        "--test_npz",
        str(splits["test"].output_npz),
        "--output_dir",
        str(out_dir),
        "--positions",
        ",".join(spec.positions),
        "--layers",
        ",".join(str(x) for x in spec.layers),
        "--layer_combine",
        spec.layer_combine,
        "--position_pool",
        spec.position_pool,
        "--hidden_sizes",
        model_hidden_sizes(spec.model_kind),
        "--epochs",
        str(args.scan_epochs),
        "--batch_size",
        str(args.scan_batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--sample_weight_mode",
        args.sample_weight_mode,
        "--threshold_max_fpr",
        str(args.threshold_max_fpr),
        "--patience",
        str(args.scan_patience),
        "--device",
        args.probe_device,
    ]


def eval_pooled_cmd(args: argparse.Namespace, probe_pt: Path, input_npz: Path, out_dir: Path) -> list[str]:
    return [
        args.python,
        "scripts/probe/evaluate_probe.py",
        "--probe_pt",
        str(probe_pt),
        "--input_npz",
        str(input_npz),
        "--output_dir",
        str(out_dir),
        "--batch_size",
        str(args.scan_eval_batch_size),
        "--device",
        args.probe_device,
    ]


def run_pooled_one(args: argparse.Namespace, spec: PooledSpec, splits: dict[str, SplitSpec]) -> str:
    root = Path(args.pooled_out_root)
    run_dir = root / spec.name
    log_path = Path(args.log_dir) / f"intra_pause_pooled_{spec.name}.log"
    if not (args.skip_existing and (run_dir / "metrics.json").exists() and (run_dir / "probe.pt").exists()):
        run_logged(train_pooled_cmd(args, spec, run_dir, splits), log_path, args.dry_run)
    probe_pt = run_dir / "probe.pt"
    if not args.dry_run:
        require_file(probe_pt)
    for name, split in splits.items():
        if not name.startswith("source_heldout_"):
            continue
        eval_name = name.removeprefix("source_heldout_")
        eval_dir = root / f"eval_{eval_name}_{spec.name}"
        if args.skip_existing and (eval_dir / "metrics.json").exists():
            continue
        run_logged(eval_pooled_cmd(args, probe_pt, split.output_npz, eval_dir), log_path, args.dry_run)
    return spec.name


def nested_metric(path: Path, split: str) -> dict[str, Any]:
    return read_json(path)["metrics"][split]


def metric(path: Path) -> dict[str, Any]:
    return read_json(path)["metrics"]


def build_pooled_summary(args: argparse.Namespace, specs: list[PooledSpec], splits: dict[str, SplitSpec]) -> list[dict[str, Any]]:
    rows = []
    root = Path(args.pooled_out_root)
    for spec in specs:
        metrics_path = root / spec.name / "metrics.json"
        if not metrics_path.exists():
            continue
        val = nested_metric(metrics_path, "val")
        test = nested_metric(metrics_path, "test")
        row: dict[str, Any] = {
            "name": spec.name,
            "positions": ",".join(spec.positions),
            "layers": ",".join(str(x) for x in spec.layers),
            "layer_combine": spec.layer_combine,
            "position_pool": spec.position_pool,
            "val_auroc": val.get("auroc"),
            "val_recall": val.get("recall"),
            "val_fpr": val.get("fpr"),
            "test_auroc": test.get("auroc"),
            "test_auprc": test.get("auprc"),
            "test_recall": test.get("recall"),
            "test_fpr": test.get("fpr"),
        }
        for name in splits:
            if not name.startswith("source_heldout_"):
                continue
            eval_name = name.removeprefix("source_heldout_")
            eval_path = root / f"eval_{eval_name}_{spec.name}" / "metrics.json"
            if not eval_path.exists():
                continue
            eval_metrics = metric(eval_path)
            prefix = f"{eval_name}_"
            row[prefix + "auroc"] = eval_metrics.get("auroc")
            row[prefix + "auprc"] = eval_metrics.get("auprc")
            row[prefix + "recall"] = eval_metrics.get("recall")
            row[prefix + "fpr"] = eval_metrics.get("fpr")
        rows.append(row)
    return rows


def run_pooled(args: argparse.Namespace, splits: dict[str, SplitSpec], layers: list[int]) -> None:
    if not args.dry_run:
        for name in ("train", "val", "test"):
            require_file(splits[name].output_npz)
    jobs = pooled_specs(layers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.pooled_jobs, len(jobs))) as executor:
        futures = {executor.submit(run_pooled_one, args, spec, splits): spec for spec in jobs}
        for future in concurrent.futures.as_completed(futures):
            spec = futures[future]
            run_name = future.result()
            print(f"finished pooled {run_name} ({','.join(spec.positions)})")

    if args.dry_run:
        return
    rows = build_pooled_summary(args, jobs, splits)
    ranked = sorted(rows, key=lambda row: float(row.get("test_auroc") or float("nan")), reverse=True)
    write_rows(rows, Path(args.pooled_out_root) / "summary_grid")
    write_rows(ranked, Path(args.pooled_out_root) / "summary_by_test_auroc")


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    positions = parse_csv(args.positions)
    splits = split_specs(args, layers)

    write_json(
        Path(args.pooled_out_root).parent / "intra_pause_probe_full_config.json",
        {
            "args": vars(args),
            "layers": layers,
            "positions": positions,
            "split_specs": {name: asdict(spec) for name, spec in splits.items()},
            "pooled_specs": [asdict(spec) for spec in pooled_specs(layers)],
        },
    )

    if not args.skip_base_data_prep:
        run_base_data_prep(args)
    if not args.skip_intra_data_prep:
        run_intra_data_prep(args)
    if not args.skip_hidden_extraction:
        run_hidden_extraction(args, splits, layers)
    if not args.skip_single_scan:
        run_single_scan(args, splits, positions, layers)
    if not args.skip_pooled:
        run_pooled(args, splits, layers)


if __name__ == "__main__":
    main()

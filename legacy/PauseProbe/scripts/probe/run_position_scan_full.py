#!/usr/bin/env python3
"""Run the full PositionScan TrajProbe external-data experiment.

This script orchestrates the full external trajectory run:

1. prepare external trajectory data
2. extract hidden states for train/val/test/source-heldout splits
3. run the single-layer position x layer scan
4. run multi-layer mean/concat ablations for selected positions

It intentionally reuses the already-tested lower-level scripts instead of
reimplementing data conversion, extraction, or probe training logic here.
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


DEFAULT_MODEL = "/workspace/outputs/deepseek_pause3_candidate_mix_10k_lr2e5_260610/final"
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
DEFAULT_POSITIONS = (
    "cot_0",
    "cot_1",
    "cot_2",
    "cot_3",
    "cot_4",
    "cot_5",
    "cot_6",
    "cot_7",
    "cot_8",
    "cot_9",
    "cot_10",
    "cot_12",
    "cot_16",
    "cot_24",
    "cot_32",
    "cot_48",
    "cot_64",
    "cot_96",
    "cot_128",
)
DEFAULT_LAYERS = (7, 14, 17, 21, 22, 28)
DEFAULT_MULTILAYER_POSITIONS = ("cot_1", "cot_2", "cot_3", "cot_7", "cot_8")
DEFAULT_LAYER_COMBINES = ("concat", "mean")


@dataclass(frozen=True)
class SplitSpec:
    name: str
    input_json: Path
    output_npz: Path
    metadata_jsonl: Path
    manifest_json: Path


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def parse_layers(value: str) -> list[int]:
    return [int(piece.strip()) for piece in value.split(",") if piece.strip()]


def layer_suffix(layers: list[int]) -> str:
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
    parser.add_argument("--data_dir", default="data/external_probe_v0")
    parser.add_argument("--hidden_dir", default="data/hidden")
    parser.add_argument("--hidden_prefix", default="external")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--single_scan_out_root", default="runs/probes/position_scan_full_external_linear")
    parser.add_argument("--multilayer_out_root", default="runs/probes/position_scan_full_external_multilayer")
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument("--heldout_source", action="append", default=list(DEFAULT_HELDOUT_SOURCES))
    parser.add_argument("--star_min_score", type=float, default=8.0)
    parser.add_argument("--max_per_source", type=int, default=None)
    parser.add_argument("--max_prompt_words", type=int, default=800)
    parser.add_argument("--max_reasoning_words", type=int, default=1600)
    parser.add_argument("--max_final_words", type=int, default=800)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_strategy", choices=("random", "label", "source", "source_label"), default="source_label")
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--positions", default=",".join(DEFAULT_POSITIONS))
    parser.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    parser.add_argument("--cot_offsets", default="0,1,2,3,4,5,6,7,8,9,10,12,16,24,32,48,64,96,128")
    parser.add_argument("--extract_batch_size", type=int, default=2)
    parser.add_argument("--extract_max_length", type=int, default=4096)
    parser.add_argument("--extract_jobs", type=int, default=1)
    parser.add_argument(
        "--extract_devices",
        default="cuda",
        help="Comma-separated extraction devices. Example: cuda:0,cuda:1. Jobs are assigned round-robin.",
    )
    parser.add_argument("--torch_dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--save_dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--scan_jobs", type=int, default=6)
    parser.add_argument("--scan_epochs", type=int, default=30)
    parser.add_argument("--scan_patience", type=int, default=8)
    parser.add_argument("--scan_batch_size", type=int, default=256)
    parser.add_argument("--scan_eval_batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--threshold_max_fpr", type=float, default=0.05)
    parser.add_argument("--sample_weight_mode", choices=("none", "label", "source", "source_label"), default="source_label")
    parser.add_argument("--probe_device", default="cuda")
    parser.add_argument("--multilayer_positions", default=",".join(DEFAULT_MULTILAYER_POSITIONS))
    parser.add_argument("--multilayer_layer_combines", default=",".join(DEFAULT_LAYER_COMBINES))
    parser.add_argument("--multilayer_model_kinds", default="linear")
    parser.add_argument("--multilayer_jobs", type=int, default=4)
    parser.add_argument("--skip_data_prep", action="store_true")
    parser.add_argument("--skip_hidden_extraction", action="store_true")
    parser.add_argument("--skip_single_scan", action="store_true")
    parser.add_argument("--skip_multilayer", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.extract_jobs < 1:
        parser.error("--extract_jobs must be >= 1")
    if args.scan_jobs < 1:
        parser.error("--scan_jobs must be >= 1")
    if args.multilayer_jobs < 1:
        parser.error("--multilayer_jobs must be >= 1")
    return args


def split_specs(args: argparse.Namespace, layers: list[int]) -> dict[str, SplitSpec]:
    data_dir = Path(args.data_dir)
    hidden_dir = Path(args.hidden_dir)
    suffix = layer_suffix(layers)
    specs = {
        "train": SplitSpec(
            "train",
            data_dir / "cotpause" / "train.json",
            hidden_dir / f"{args.hidden_prefix}_train_dense_cot_layers_{suffix}.npz",
            hidden_dir / f"{args.hidden_prefix}_train_dense_cot_layers_{suffix}.metadata.jsonl",
            hidden_dir / f"{args.hidden_prefix}_train_dense_cot_layers_{suffix}.manifest.json",
        ),
        "val": SplitSpec(
            "val",
            data_dir / "cotpause" / "val.json",
            hidden_dir / f"{args.hidden_prefix}_val_dense_cot_layers_{suffix}.npz",
            hidden_dir / f"{args.hidden_prefix}_val_dense_cot_layers_{suffix}.metadata.jsonl",
            hidden_dir / f"{args.hidden_prefix}_val_dense_cot_layers_{suffix}.manifest.json",
        ),
        "test": SplitSpec(
            "test",
            data_dir / "cotpause" / "test.json",
            hidden_dir / f"{args.hidden_prefix}_test_dense_cot_layers_{suffix}.npz",
            hidden_dir / f"{args.hidden_prefix}_test_dense_cot_layers_{suffix}.metadata.jsonl",
            hidden_dir / f"{args.hidden_prefix}_test_dense_cot_layers_{suffix}.manifest.json",
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


def run_data_prep(args: argparse.Namespace) -> None:
    cmd = [
        args.python,
        "scripts/data/prepare_external_trajectories.py",
        "--output_dir",
        args.data_dir,
        "--sources",
        *args.sources,
        "--star_min_score",
        str(args.star_min_score),
        "--train_ratio",
        str(args.train_ratio),
        "--val_ratio",
        str(args.val_ratio),
        "--split_strategy",
        args.split_strategy,
        "--max_prompt_words",
        str(args.max_prompt_words),
        "--max_reasoning_words",
        str(args.max_reasoning_words),
        "--max_final_words",
        str(args.max_final_words),
        "--seed",
        str(args.seed),
    ]
    for source in args.heldout_source:
        cmd.extend(["--heldout_source", source])
    if args.max_per_source is not None:
        cmd.extend(["--max_per_source", str(args.max_per_source)])
    run_logged(cmd, Path(args.log_dir) / "position_scan_full_data_prep.log", args.dry_run)


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


def run_extraction_one(
    args: argparse.Namespace,
    spec: SplitSpec,
    layers: list[int],
    device: str,
) -> str:
    if args.skip_existing and spec.output_npz.exists() and spec.manifest_json.exists():
        print(f"skip existing extraction: {spec.name} -> {spec.output_npz}")
        return spec.name
    if not args.dry_run:
        require_file(spec.input_json)
    log_path = Path(args.log_dir) / f"position_scan_full_extract_{spec.name}.log"
    run_logged(extraction_cmd(args, spec, layers, device), log_path, args.dry_run)
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
    run_logged(cmd, Path(args.log_dir) / "position_scan_full_single_scan.log", args.dry_run)


def train_multilayer_cmd(
    args: argparse.Namespace,
    position: str,
    layer_combine: str,
    model_kind: str,
    layers: list[int],
    out_dir: Path,
    specs: dict[str, SplitSpec],
) -> list[str]:
    return [
        args.python,
        "scripts/probe/train_probe.py",
        "--train_npz",
        str(specs["train"].output_npz),
        "--val_npz",
        str(specs["val"].output_npz),
        "--test_npz",
        str(specs["test"].output_npz),
        "--output_dir",
        str(out_dir),
        "--positions",
        position,
        "--layers",
        ",".join(str(x) for x in layers),
        "--layer_combine",
        layer_combine,
        "--position_pool",
        "first",
        "--hidden_sizes",
        model_hidden_sizes(model_kind),
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


def eval_multilayer_cmd(
    args: argparse.Namespace,
    probe_pt: Path,
    eval_name: str,
    input_npz: Path,
    out_dir: Path,
) -> list[str]:
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


def multilayer_run_name(model_kind: str, layer_combine: str, position: str, layers: list[int]) -> str:
    return f"{model_kind}_{layer_combine}_{position}_layers_{layer_suffix(layers)}"


def run_multilayer_one(
    args: argparse.Namespace,
    position: str,
    layer_combine: str,
    model_kind: str,
    layers: list[int],
    specs: dict[str, SplitSpec],
) -> str:
    run_name = multilayer_run_name(model_kind, layer_combine, position, layers)
    root = Path(args.multilayer_out_root)
    run_dir = root / run_name
    log_path = Path(args.log_dir) / f"position_scan_full_multilayer_{run_name}.log"
    if not (args.skip_existing and (run_dir / "metrics.json").exists() and (run_dir / "probe.pt").exists()):
        cmd = train_multilayer_cmd(args, position, layer_combine, model_kind, layers, run_dir, specs)
        run_logged(cmd, log_path, args.dry_run)
    probe_pt = run_dir / "probe.pt"
    if not args.dry_run:
        require_file(probe_pt)
    for name, spec in specs.items():
        if not name.startswith("source_heldout_"):
            continue
        eval_name = name.removeprefix("source_heldout_")
        eval_dir = root / f"eval_{eval_name}_{run_name}"
        if args.skip_existing and (eval_dir / "metrics.json").exists():
            continue
        cmd = eval_multilayer_cmd(args, probe_pt, eval_name, spec.output_npz, eval_dir)
        run_logged(cmd, log_path, args.dry_run)
    return run_name


def nested_metric(path: Path, split: str) -> dict[str, Any]:
    return read_json(path)["metrics"][split]


def metric(path: Path) -> dict[str, Any]:
    return read_json(path)["metrics"]


def build_multilayer_summary(
    args: argparse.Namespace,
    positions: list[str],
    layer_combines: list[str],
    model_kinds: list[str],
    layers: list[int],
    specs: dict[str, SplitSpec],
) -> list[dict[str, Any]]:
    root = Path(args.multilayer_out_root)
    rows: list[dict[str, Any]] = []
    for model_kind in model_kinds:
        for layer_combine in layer_combines:
            for position in positions:
                run_name = multilayer_run_name(model_kind, layer_combine, position, layers)
                metrics_path = root / run_name / "metrics.json"
                if not metrics_path.exists():
                    continue
                val = nested_metric(metrics_path, "val")
                test = nested_metric(metrics_path, "test")
                row: dict[str, Any] = {
                    "model": model_kind,
                    "layer_combine": layer_combine,
                    "position": position,
                    "layers": ",".join(str(x) for x in layers),
                    "val_auroc": val.get("auroc"),
                    "val_recall": val.get("recall"),
                    "val_fpr": val.get("fpr"),
                    "test_auroc": test.get("auroc"),
                    "test_auprc": test.get("auprc"),
                    "test_recall": test.get("recall"),
                    "test_fpr": test.get("fpr"),
                }
                for name in specs:
                    if not name.startswith("source_heldout_"):
                        continue
                    eval_name = name.removeprefix("source_heldout_")
                    eval_metrics_path = root / f"eval_{eval_name}_{run_name}" / "metrics.json"
                    if not eval_metrics_path.exists():
                        continue
                    eval_metrics = metric(eval_metrics_path)
                    prefix = f"{eval_name}_"
                    row[prefix + "auroc"] = eval_metrics.get("auroc")
                    row[prefix + "auprc"] = eval_metrics.get("auprc")
                    row[prefix + "recall"] = eval_metrics.get("recall")
                    row[prefix + "fpr"] = eval_metrics.get("fpr")
                rows.append(row)
    return rows


def run_multilayer(
    args: argparse.Namespace,
    specs: dict[str, SplitSpec],
    positions: list[str],
    layer_combines: list[str],
    model_kinds: list[str],
    layers: list[int],
) -> None:
    if not args.dry_run:
        for name in ("train", "val", "test"):
            require_file(specs[name].output_npz)
    jobs = [(position, layer_combine, model_kind) for model_kind in model_kinds for layer_combine in layer_combines for position in positions]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.multilayer_jobs, len(jobs))) as executor:
        futures = {
            executor.submit(run_multilayer_one, args, position, layer_combine, model_kind, layers, specs): (
                position,
                layer_combine,
                model_kind,
            )
            for position, layer_combine, model_kind in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            position, layer_combine, model_kind = futures[future]
            run_name = future.result()
            print(f"finished multilayer {run_name} ({model_kind}, {layer_combine}, {position})")

    if args.dry_run:
        return
    rows = build_multilayer_summary(args, positions, layer_combines, model_kinds, layers, specs)
    ranked = sorted(rows, key=lambda row: float(row.get("test_auroc") or float("nan")), reverse=True)
    write_rows(rows, Path(args.multilayer_out_root) / "summary_grid")
    write_rows(ranked, Path(args.multilayer_out_root) / "summary_by_test_auroc")


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    positions = parse_csv(args.positions)
    multilayer_positions = parse_csv(args.multilayer_positions)
    layer_combines = parse_csv(args.multilayer_layer_combines)
    model_kinds = parse_csv(args.multilayer_model_kinds)
    specs = split_specs(args, layers)

    write_json(
        Path(args.multilayer_out_root).parent / "position_scan_full_config.json",
        {
            "args": vars(args),
            "layers": layers,
            "positions": positions,
            "multilayer_positions": multilayer_positions,
            "layer_combines": layer_combines,
            "split_specs": {name: asdict(spec) for name, spec in specs.items()},
        },
    )

    if not args.skip_data_prep:
        run_data_prep(args)
    if not args.skip_hidden_extraction:
        run_hidden_extraction(args, specs, layers)
    if not args.skip_single_scan:
        run_single_scan(args, specs, positions, layers)
    if not args.skip_multilayer:
        run_multilayer(args, specs, multilayer_positions, layer_combines, model_kinds, layers)


if __name__ == "__main__":
    main()

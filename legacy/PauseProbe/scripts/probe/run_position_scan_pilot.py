#!/usr/bin/env python3
"""Run a small PositionScan TrajProbe grid and summarize the results.

This launcher assumes hidden states have already been extracted for the same
layers and CoT positions in the train/val/test NPZ files.  It trains one probe
per single (position, layer) point, optionally evaluates each probe on extra
held-out NPZ files, and writes compact JSON/TSV summaries.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_csv(value: str) -> list[str]:
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def parse_layers(value: str) -> list[int]:
    return [int(piece.strip()) for piece in value.split(",") if piece.strip()]


def parse_eval_npz(pairs: list[str]) -> dict[str, Path]:
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected --eval_npz name=path, got {pair!r}")
        name, path = pair.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty eval name in {pair!r}")
        out[name] = Path(path)
    return out


def run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def device_to_visible_device(device: str) -> str | None:
    if not device.startswith("cuda:"):
        return None
    gpu_id = device.split(":", 1)[1].strip()
    return gpu_id or None


def effective_cpu_threads(requested: int, jobs: int) -> int:
    if requested > 0:
        return requested
    cpus = os.cpu_count() or 1
    return max(1, min(4, cpus // max(1, jobs)))


def apply_probe_cpu_env(env: dict[str, str], cpu_threads: int) -> None:
    threads = str(max(1, cpu_threads))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(name, threads)
    env.setdefault("PAUSEPROBE_CPU_THREADS", threads)
    env.setdefault("PAUSEPROBE_GPU_TENSORS", "1")


def run_logged_on_device(cmd: list[str], log_path: Path, device: str, cpu_threads: int) -> None:
    """Run one probe subprocess with an optional single-GPU visibility mask."""

    env = os.environ.copy()
    apply_probe_cpu_env(env, cpu_threads)
    visible = device_to_visible_device(device)
    subprocess_device = device
    if visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible
        subprocess_device = "cuda"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"# assigned_device={device}")
        if visible is not None:
            log.write(f" CUDA_VISIBLE_DEVICES={visible}")
        log.write(f" cpu_threads={cpu_threads}")
        log.write("\n")
        rendered = ["cuda" if item == device else item for item in cmd]
        log.write("$ " + " ".join(rendered) + "\n")
        log.flush()
        adjusted = [subprocess_device if item == device else item for item in cmd]
        proc = subprocess.run(adjusted, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_value(payload: dict[str, Any], split: str | None = None) -> dict[str, Any]:
    metrics = payload["metrics"]
    if split is None:
        return metrics
    return metrics[split]


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def model_hidden_sizes(model_kind: str) -> str:
    if model_kind == "linear":
        return ""
    if model_kind == "mlp":
        return "clear_default"
    raise ValueError(f"Unknown model kind: {model_kind}")


def run_one(
    model_kind: str,
    position: str,
    layer: int,
    device: str,
    args: argparse.Namespace,
    eval_npz: dict[str, Path],
) -> str:
    run_name = f"{model_kind}_{position}_l{layer}"
    out_root = Path(args.out_root)
    run_dir = out_root / run_name
    log_path = Path(args.log_dir) / f"position_scan_pilot_{run_name}.log"
    metrics_path = run_dir / "metrics.json"

    if not (args.skip_existing and metrics_path.exists() and (run_dir / "probe.pt").exists()):
        cmd = [
            args.python,
            "scripts/probe/train_probe.py",
            "--train_npz",
            args.train_npz,
            "--val_npz",
            args.val_npz,
            "--test_npz",
            args.test_npz,
            "--output_dir",
            str(run_dir),
            "--positions",
            position,
            "--layers",
            str(layer),
            "--layer_combine",
            "mean",
            "--position_pool",
            "first",
            "--hidden_sizes",
            model_hidden_sizes(model_kind),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
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
            device,
            "--cpu_threads",
            str(args.cpu_threads),
        ]
        run_logged_on_device(cmd, log_path, device, args.cpu_threads)

    probe_pt = run_dir / "probe.pt"
    require_file(probe_pt)
    for eval_name, npz_path in eval_npz.items():
        eval_dir = out_root / f"eval_{eval_name}_{run_name}"
        eval_metrics = eval_dir / "metrics.json"
        if args.skip_existing and eval_metrics.exists():
            continue
        cmd = [
            args.python,
            "scripts/probe/evaluate_probe.py",
            "--probe_pt",
            str(probe_pt),
            "--input_npz",
            str(npz_path),
            "--output_dir",
            str(eval_dir),
            "--batch_size",
            str(args.eval_batch_size),
            "--device",
            device,
        ]
        run_logged_on_device(cmd, log_path, device, args.cpu_threads)
    return run_name


def build_summary(
    model_kinds: list[str],
    positions: list[str],
    layers: list[int],
    out_root: Path,
    eval_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_kind in model_kinds:
        for position in positions:
            for layer in layers:
                run_name = f"{model_kind}_{position}_l{layer}"
                train_payload = read_json(out_root / run_name / "metrics.json")
                val = metric_value(train_payload, "val")
                test = metric_value(train_payload, "test")
                row: dict[str, Any] = {
                    "model": model_kind,
                    "position": position,
                    "layer": layer,
                    "train_n": metric_value(train_payload, "train").get("n"),
                    "val_n": val.get("n"),
                    "test_n": test.get("n"),
                    "val_auroc": val.get("auroc"),
                    "val_recall": val.get("recall"),
                    "val_fpr": val.get("fpr"),
                    "test_auroc": test.get("auroc"),
                    "test_auprc": test.get("auprc"),
                    "test_recall": test.get("recall"),
                    "test_fpr": test.get("fpr"),
                    "threshold": train_payload.get("threshold"),
                }
                for eval_name in eval_names:
                    payload = read_json(out_root / f"eval_{eval_name}_{run_name}" / "metrics.json")
                    metrics = metric_value(payload)
                    prefix = f"{eval_name}_"
                    row[prefix + "n"] = metrics.get("n")
                    row[prefix + "auroc"] = metrics.get("auroc")
                    row[prefix + "auprc"] = metrics.get("auprc")
                    row[prefix + "recall"] = metrics.get("recall")
                    row[prefix + "fpr"] = metrics.get("fpr")
                rows.append(row)
    return rows


def write_rows(rows: list[dict[str, Any]], path_prefix: Path) -> None:
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    with path_prefix.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    keys = list(rows[0]) if rows else []
    with path_prefix.with_suffix(".tsv").open("w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for row in rows:
            f.write("\t".join(fmt(row.get(key)) for key in keys) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--val_npz", required=True)
    parser.add_argument("--test_npz", required=True)
    parser.add_argument("--eval_npz", action="append", default=[], help="Extra evaluation split in name=path form.")
    parser.add_argument("--out_root", default="runs/probes/position_scan_pilot")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument(
        "--positions",
        default="cot_0,cot_1,cot_2,cot_4,cot_8,cot_16,cot_32",
    )
    parser.add_argument("--layers", default="7,14,17,21,22,28")
    parser.add_argument("--model_kinds", default="linear", help="Comma-separated: linear,mlp.")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--sample_weight_mode", choices=("none", "label", "source", "source_label"), default="source_label")
    parser.add_argument("--threshold_max_fpr", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated probe devices. Example: cuda:0,cuda:1,cuda:2,cuda:3. Jobs are assigned round-robin.",
    )
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=0,
        help="CPU threads per probe subprocess. 0 auto-scales from CPU count and --jobs.",
    )
    args = parser.parse_args()

    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    if args.cpu_threads < 0:
        raise ValueError("--cpu_threads must be >= 0")
    args.cpu_threads = effective_cpu_threads(args.cpu_threads, args.jobs)
    for path in (Path(args.train_npz), Path(args.val_npz), Path(args.test_npz)):
        require_file(path)
    eval_npz = parse_eval_npz(args.eval_npz)
    for path in eval_npz.values():
        require_file(path)

    positions = parse_csv(args.positions)
    layers = parse_layers(args.layers)
    model_kinds = parse_csv(args.model_kinds)
    jobs = [(model_kind, position, layer) for model_kind in model_kinds for position in positions for layer in layers]
    devices = parse_csv(args.devices) if args.devices else [args.device]
    if not devices:
        devices = [args.device]

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.jobs, len(jobs))) as executor:
        futures = {
            executor.submit(
                run_one,
                model_kind,
                position,
                layer,
                devices[idx % len(devices)],
                args,
                eval_npz,
            ): (model_kind, position, layer, devices[idx % len(devices)])
            for idx, (model_kind, position, layer) in enumerate(jobs)
        }
        for future in concurrent.futures.as_completed(futures):
            model_kind, position, layer, device = futures[future]
            run_name = future.result()
            print(f"finished {run_name} ({model_kind}, {position}, layer {layer}, {device})")

    rows = build_summary(model_kinds, positions, layers, Path(args.out_root), list(eval_npz))
    ranked = sorted(rows, key=lambda row: float(row.get("test_auroc") or float("nan")), reverse=True)
    write_rows(rows, Path(args.out_root) / "summary_grid")
    write_rows(ranked, Path(args.out_root) / "summary_by_test_auroc")
    if ranked:
        keys = list(ranked[0])
        print("\nTop by test AUROC:")
        print("\t".join(keys))
        for row in ranked[:10]:
            print("\t".join(fmt(row.get(key)) for key in keys))


if __name__ == "__main__":
    main()

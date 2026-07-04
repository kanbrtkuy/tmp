#!/usr/bin/env python3
"""Run single pause-position/layer ablations for PauseRiskProbe v5.

This launcher assumes v5 hidden states have already been extracted with
layers 13,17,22,28 and positions pause_0,pause_1,pause_2.  It trains one
probe per single (position, layer) point, evaluates each checkpoint on the
same held-out stress sets used by the v5 sweep, and writes summary tables
ranked by v5 test AUROC and XSTest recall.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


TARGET_RISK_TYPES = (
    "xstest_like_discrimination_positive",
    "xstest_like_privacy_positive",
    "xstest_like_historical_justification_positive",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden_prefix", default="data/hidden/prompt_risk_v5_targeted")
    parser.add_argument("--v1_test_npz", default="data/hidden/prompt_risk_v1_test_layers_13_17_22_28.npz")
    parser.add_argument("--out_root", default="runs/probes/pause_risk_v5_pause_layer_ablation")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--positions", default="pause_0,pause_1,pause_2")
    parser.add_argument("--layers", default="13,17,22,28")
    parser.add_argument("--model_kinds", default="linear,mlp", help="Comma-separated: linear,mlp.")
    parser.add_argument("--target_risk_weight", type=float, default=1.0)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def parse_layers(value: str) -> list[int]:
    return [int(piece.strip()) for piece in value.split(",") if piece.strip()]


def run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def metric(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "metrics" in payload and isinstance(payload["metrics"], dict):
        return payload["metrics"]
    raise KeyError(f"No metrics object in {path}")


def nested_metric(path: Path, split: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["metrics"][split]


def fmt(value: Any) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def model_hidden_sizes(model_kind: str) -> str:
    if model_kind == "linear":
        return ""
    if model_kind == "mlp":
        return "clear_default"
    raise ValueError(f"Unknown model kind: {model_kind}")


def run_one(model_kind: str, position: str, layer: int, args: argparse.Namespace) -> str:
    root = Path(args.out_root)
    hidden = Path(args.hidden_prefix)
    run_name = f"{model_kind}_{position}_l{layer}"
    run_dir = root / run_name
    log = Path(args.log_dir) / f"v5_pause_layer_ablation_{run_name}.log"
    train_metrics = run_dir / "metrics.json"

    if not (args.skip_existing and train_metrics.exists() and (run_dir / "probe.pt").exists()):
        cmd = [
            args.python,
            "scripts/probe/train_probe.py",
            "--train_npz",
            str(hidden) + "_train_layers_13_17_22_28.npz",
            "--val_npz",
            str(hidden) + "_val_layers_13_17_22_28.npz",
            "--test_npz",
            str(hidden) + "_test_layers_13_17_22_28.npz",
            "--output_dir",
            str(run_dir),
            "--positions",
            position,
            "--layers",
            str(layer),
            "--layer_combine",
            "mean",
            "--position_pool",
            "mean",
            "--hidden_sizes",
            model_hidden_sizes(model_kind),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--learning_rate",
            "3e-4",
            "--weight_decay",
            "0.0",
            "--sample_weight_mode",
            "source_label",
            "--source_weight",
            "wildjailbreak_adversarial_benign=2.0",
            "--source_weight",
            "wildjailbreak_adversarial_harmful=2.0",
            "--pairwise_margin_weight",
            "0.5",
            "--pairwise_margin",
            "0.7",
            "--pairwise_pair_id_only",
            "--threshold_max_fpr",
            "0.05",
            "--patience",
            str(args.patience),
            "--device",
            args.device,
        ]
        for risk_type in TARGET_RISK_TYPES:
            cmd.extend(["--risk_type_weight", f"{risk_type}={args.target_risk_weight:.1f}"])
        run_logged(cmd, log)

    probe_pt = run_dir / "probe.pt"
    require_file(probe_pt)

    for eval_name in ("xstest", "or_bench_hard", "or_bench_toxic"):
        out_dir = root / f"eval_{eval_name}_{run_name}"
        metrics_path = out_dir / "metrics.json"
        if not (args.skip_existing and metrics_path.exists()):
            cmd = [
                args.python,
                "scripts/probe/evaluate_probe.py",
                "--probe_pt",
                str(probe_pt),
                "--input_npz",
                str(hidden) + f"_eval_{eval_name}_layers_13_17_22_28.npz",
                "--output_dir",
                str(out_dir),
                "--batch_size",
                str(args.eval_batch_size),
                "--device",
                args.device,
            ]
            run_logged(cmd, log)

    v1_out = root / f"eval_prompt_risk_v1_test_{run_name}"
    if not (args.skip_existing and (v1_out / "metrics.json").exists()):
        cmd = [
            args.python,
            "scripts/probe/evaluate_probe.py",
            "--probe_pt",
            str(probe_pt),
            "--input_npz",
            args.v1_test_npz,
            "--output_dir",
            str(v1_out),
            "--batch_size",
            str(args.eval_batch_size),
            "--device",
            args.device,
        ]
        run_logged(cmd, log)

    return run_name


def build_summary(model_kind: str, positions: list[str], layers: list[int], out_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position in positions:
        for layer in layers:
            run_name = f"{model_kind}_{position}_l{layer}"
            train = nested_metric(out_root / run_name / "metrics.json", "test")
            xs = metric(out_root / f"eval_xstest_{run_name}" / "metrics.json")
            oh = metric(out_root / f"eval_or_bench_hard_{run_name}" / "metrics.json")
            ot = metric(out_root / f"eval_or_bench_toxic_{run_name}" / "metrics.json")
            v1 = metric(out_root / f"eval_prompt_risk_v1_test_{run_name}" / "metrics.json")
            rows.append(
                {
                    "model": model_kind,
                    "position": position,
                    "layer": layer,
                    "test_auroc": train.get("auroc"),
                    "test_auprc": train.get("auprc"),
                    "test_recall": train.get("recall"),
                    "test_fpr": train.get("fpr"),
                    "xstest_auroc": xs.get("auroc"),
                    "xstest_recall": xs.get("recall"),
                    "xstest_fpr": xs.get("fpr"),
                    "orhard_fpr": oh.get("fpr"),
                    "ortoxic_recall": ot.get("recall"),
                    "v1_auroc": v1.get("auroc"),
                    "v1_recall": v1.get("recall"),
                    "v1_fpr": v1.get("fpr"),
                }
            )
    return rows


def write_rows(rows: list[dict[str, Any]], path_prefix: Path) -> None:
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = path_prefix.with_suffix(".json")
    tsv_path = path_prefix.with_suffix(".tsv")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    keys = list(rows[0]) if rows else []
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for row in rows:
            f.write("\t".join(fmt(row.get(key)) for key in keys) + "\n")


def main() -> None:
    args = parse_args()
    positions = parse_csv(args.positions)
    layers = parse_layers(args.layers)
    model_kinds = parse_csv(args.model_kinds)
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")

    hidden = Path(args.hidden_prefix)
    required = [
        Path(str(hidden) + "_train_layers_13_17_22_28.npz"),
        Path(str(hidden) + "_val_layers_13_17_22_28.npz"),
        Path(str(hidden) + "_test_layers_13_17_22_28.npz"),
        Path(str(hidden) + "_eval_xstest_layers_13_17_22_28.npz"),
        Path(str(hidden) + "_eval_or_bench_hard_layers_13_17_22_28.npz"),
        Path(str(hidden) + "_eval_or_bench_toxic_layers_13_17_22_28.npz"),
        Path(args.v1_test_npz),
    ]
    for path in required:
        require_file(path)

    jobs = [(model_kind, position, layer) for model_kind in model_kinds for position in positions for layer in layers]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.jobs, len(jobs))) as executor:
        futures = {
            executor.submit(run_one, model_kind, position, layer, args): (model_kind, position, layer)
            for model_kind, position, layer in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            model_kind, position, layer = futures[future]
            result = future.result()
            print(f"finished {result} ({model_kind}, {position}, layer {layer})")

    out_root = Path(args.out_root)
    for model_kind in model_kinds:
        rows = build_summary(model_kind, positions, layers, out_root)
        by_test = sorted(rows, key=lambda row: float(row["test_auroc"]), reverse=True)
        by_xstest = sorted(rows, key=lambda row: float(row["xstest_recall"]), reverse=True)
        write_rows(by_test, out_root / f"summary_{model_kind}_by_test_auc")
        write_rows(by_xstest, out_root / f"summary_{model_kind}_by_xstest_recall")
        if by_test:
            print(f"\nTop {model_kind} by v5 test AUROC:")
            keys = list(by_test[0])
            print("\t".join(keys))
            for row in by_test[:5]:
                print("\t".join(fmt(row.get(key)) for key in keys))
        if by_xstest:
            print(f"\nTop {model_kind} by XSTest recall:")
            keys = list(by_xstest[0])
            print("\t".join(keys))
            for row in by_xstest[:5]:
                print("\t".join(fmt(row.get(key)) for key in keys))


if __name__ == "__main__":
    main()

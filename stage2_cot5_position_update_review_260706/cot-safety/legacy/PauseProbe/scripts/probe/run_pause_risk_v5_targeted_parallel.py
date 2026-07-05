#!/usr/bin/env python3
"""Run the targeted PauseRiskProbe v5 weight sweep in parallel.

This launcher assumes hidden states have already been extracted.  It keeps the
per-run training command identical to the sequential sweep, but runs independent
weight settings concurrently so a small probe does not leave most CPU cores idle.
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
    parser.add_argument("--out_root", default="runs/probes/pause_risk_v5_targeted")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--weights", default="1,2,3,5", help="Comma-separated risk-type weights to sweep.")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


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


def saved_threshold(path: Path) -> float:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return float(payload["threshold"])


def fmt(value: Any) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def run_one(weight: float, args: argparse.Namespace) -> str:
    root = Path(args.out_root)
    hidden = Path(args.hidden_prefix)
    log = Path(args.log_dir) / f"pause_risk_v5_targeted_w{weight:g}.log"
    run_dir = root / f"mlp_clear_lowfpr_targeted_w{weight:g}"
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
            "pause_0,pause_1,pause_2",
            "--layers",
            "13,17,22",
            "--layer_combine",
            "mean",
            "--position_pool",
            "mean",
            "--hidden_sizes",
            "clear_default",
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
            cmd.extend(["--risk_type_weight", f"{risk_type}={weight:.1f}"])
        run_logged(cmd, log)

    probe_pt = run_dir / "probe.pt"
    require_file(probe_pt)
    for eval_name in ("xstest", "or_bench_hard", "or_bench_toxic"):
        out_dir = root / f"eval_{eval_name}_mlp_w{weight:g}"
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

    v1_out = root / f"eval_prompt_risk_v1_test_mlp_w{weight:g}"
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

    artifact_dir = root / f"eval_xstest_mlp_w{weight:g}" / "artifacts"
    if not (args.skip_existing and (artifact_dir / "summary.json").exists()):
        cmd = [
            args.python,
            "scripts/probe/analyze_probe_artifacts.py",
            "--predictions_jsonl",
            str(root / f"eval_xstest_mlp_w{weight:g}" / "predictions.jsonl"),
            "--metadata_jsonl",
            str(hidden) + "_eval_xstest_layers_13_17_22_28.metadata.jsonl",
            "--output_dir",
            str(artifact_dir),
            "--threshold",
            str(saved_threshold(train_metrics)),
            "--top_k",
            "200",
        ]
        run_logged(cmd, log)
    return f"w{weight:g}"


def build_summary(weights: list[float], out_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for weight in weights:
        suffix = f"w{weight:g}"
        train = nested_metric(out_root / f"mlp_clear_lowfpr_targeted_{suffix}" / "metrics.json", "test")
        xs = metric(out_root / f"eval_xstest_mlp_{suffix}" / "metrics.json")
        oh = metric(out_root / f"eval_or_bench_hard_mlp_{suffix}" / "metrics.json")
        ot = metric(out_root / f"eval_or_bench_toxic_mlp_{suffix}" / "metrics.json")
        v1 = metric(out_root / f"eval_prompt_risk_v1_test_mlp_{suffix}" / "metrics.json")
        rows.append(
            {
                "run": suffix,
                "v5_auc": train.get("auroc"),
                "v5_ap": train.get("auprc"),
                "v5_recall": train.get("recall"),
                "v5_fpr": train.get("fpr"),
                "xstest_auc": xs.get("auroc"),
                "xstest_recall": xs.get("recall"),
                "xstest_fpr": xs.get("fpr"),
                "orhard_fpr": oh.get("fpr"),
                "ortoxic_recall": ot.get("recall"),
                "v1_auc": v1.get("auroc"),
                "v1_recall": v1.get("recall"),
                "v1_fpr": v1.get("fpr"),
            }
        )
    return rows


def write_summary(rows: list[dict[str, Any]], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "targeted_weight_sweep_summary.json"
    tsv_path = out_root / "targeted_weight_sweep_summary.tsv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    keys = list(rows[0]) if rows else []
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for row in rows:
            f.write("\t".join(fmt(row.get(k)) for k in keys) + "\n")
    if rows:
        print("\t".join(keys))
        for row in rows:
            print("\t".join(fmt(row.get(k)) for k in keys))
    print(f"Wrote {json_path}")
    print(f"Wrote {tsv_path}")


def main() -> None:
    args = parse_args()
    weights = [float(x) for x in args.weights.split(",") if x.strip()]
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.jobs, len(weights))) as executor:
        futures = {executor.submit(run_one, weight, args): weight for weight in weights}
        for future in concurrent.futures.as_completed(futures):
            weight = futures[future]
            try:
                print(f"finished {future.result()}", flush=True)
            except Exception as exc:
                raise RuntimeError(f"Weight {weight:g} failed") from exc

    rows = build_summary(weights, Path(args.out_root))
    write_summary(rows, Path(args.out_root))


if __name__ == "__main__":
    main()

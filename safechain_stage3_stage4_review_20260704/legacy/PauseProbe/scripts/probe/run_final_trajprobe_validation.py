#!/usr/bin/env python3
"""Run the final PositionScan TrajProbe validation suite in parallel."""

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


DEFAULT_LAYERS = (7, 14, 17, 21, 22, 28)


@dataclass(frozen=True)
class Candidate:
    name: str
    positions: tuple[str, ...]
    layers: tuple[int, ...]
    layer_combine: str
    position_pool: str = "first"
    hidden_sizes: str = ""
    note: str = ""


BUILTIN_CANDIDATES: dict[str, Candidate] = {
    "single_cot3_l17": Candidate(
        name="single_cot3_l17",
        positions=("cot_3",),
        layers=(17,),
        layer_combine="mean",
        note="Best source-heldout balanced candidate in cap1k single-layer scan.",
    ),
    "single_cot4_l22": Candidate(
        name="single_cot4_l22",
        positions=("cot_4",),
        layers=(22,),
        layer_combine="mean",
        note="Best source-heldout AUROC candidate in cap1k single-layer scan.",
    ),
    "concat_cot3_all": Candidate(
        name="concat_cot3_all",
        positions=("cot_3",),
        layers=DEFAULT_LAYERS,
        layer_combine="concat",
        note="CLEAR-style multi-layer concatenation at the strongest early-CoT position.",
    ),
    "mean_cot3_all": Candidate(
        name="mean_cot3_all",
        positions=("cot_3",),
        layers=DEFAULT_LAYERS,
        layer_combine="mean",
        note="Lower-dimensional multi-layer control for concat_cot3_all.",
    ),
    "mlp_concat_cot3_all": Candidate(
        name="mlp_concat_cot3_all",
        positions=("cot_3",),
        layers=DEFAULT_LAYERS,
        layer_combine="concat",
        hidden_sizes="clear_default",
        note="Optional CLEAR-style lightweight MLP gate on concatenated layers.",
    ),
}


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def parse_key_paths(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected name=path, got {value!r}")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty mapping name in {value!r}")
        out[name] = Path(path)
    return out


def parse_candidate_names(value: str) -> list[str]:
    names = parse_csv(value)
    if not names:
        raise ValueError("At least one candidate is required.")
    missing = [name for name in names if name not in BUILTIN_CANDIDATES]
    if missing:
        raise ValueError(f"Unknown candidates: {missing}. Available: {sorted(BUILTIN_CANDIDATES)}")
    return names


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


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def npz_metadata_path(npz_path: Path) -> Path:
    return npz_path.with_suffix(".metadata.jsonl")


def split_label(split: str) -> str:
    return split.replace("/", "_").replace(" ", "_")


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


def candidate_train_cmd(
    args: argparse.Namespace,
    candidate: Candidate,
    out_dir: Path,
    device: str,
) -> list[str]:
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
        str(out_dir),
        "--positions",
        ",".join(candidate.positions),
        "--layers",
        ",".join(str(layer) for layer in candidate.layers),
        "--layer_combine",
        candidate.layer_combine,
        "--position_pool",
        candidate.position_pool,
        "--hidden_sizes",
        candidate.hidden_sizes,
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
        "--seed",
        str(args.seed),
    ]
    if args.allow_missing_positions:
        cmd.append("--allow_missing_positions")
    if args.pairwise_margin_weight > 0:
        cmd.extend(["--pairwise_margin_weight", str(args.pairwise_margin_weight)])
    if args.pairwise_pair_id_only:
        cmd.append("--pairwise_pair_id_only")
    return cmd


def candidate_eval_cmd(
    args: argparse.Namespace,
    probe_pt: Path,
    input_npz: Path,
    out_dir: Path,
    device: str,
) -> list[str]:
    cmd = [
        args.python,
        "scripts/probe/evaluate_probe.py",
        "--probe_pt",
        str(probe_pt),
        "--input_npz",
        str(input_npz),
        "--output_dir",
        str(out_dir),
        "--batch_size",
        str(args.eval_batch_size),
        "--device",
        device,
        "--threshold",
        args.eval_threshold,
    ]
    if args.allow_missing_positions:
        cmd.append("--allow_missing_positions")
    return cmd


def artifact_cmd(
    args: argparse.Namespace,
    predictions_jsonl: Path,
    metadata_jsonl: Path,
    out_dir: Path,
    trajectory_jsonl: Path | None,
    threshold: float | None,
) -> list[str]:
    cmd = [
        args.python,
        "scripts/probe/analyze_trajectory_probe_artifacts.py",
        "--predictions_jsonl",
        str(predictions_jsonl),
        "--metadata_jsonl",
        str(metadata_jsonl),
        "--output_dir",
        str(out_dir),
        "--top_k",
        str(args.artifact_top_k),
    ]
    if threshold is not None:
        cmd.extend(["--threshold", str(threshold)])
    if trajectory_jsonl is not None:
        cmd.extend(["--trajectory_jsonl", str(trajectory_jsonl)])
    return cmd


def run_train_one(
    args: argparse.Namespace,
    candidate: Candidate,
    device: str,
) -> str:
    out_dir = Path(args.out_root) / candidate.name
    log_path = Path(args.log_dir) / f"final_trajprobe_train_{candidate.name}.log"
    if args.skip_existing and (out_dir / "metrics.json").exists() and (out_dir / "probe.pt").exists():
        print(f"skip existing train: {candidate.name}")
        return candidate.name
    cmd = candidate_train_cmd(args, candidate, out_dir, device)
    run_logged(cmd, log_path, args.dry_run)
    return candidate.name


def run_eval_one(
    args: argparse.Namespace,
    candidate: Candidate,
    eval_name: str,
    input_npz: Path,
    device: str,
) -> str:
    probe_pt = Path(args.out_root) / candidate.name / "probe.pt"
    if not args.dry_run:
        require_file(probe_pt)
        require_file(input_npz)
    eval_dir = Path(args.out_root) / f"eval_{split_label(eval_name)}_{candidate.name}"
    if args.skip_existing and (eval_dir / "metrics.json").exists():
        print(f"skip existing eval: {eval_name} / {candidate.name}")
        return f"{eval_name}/{candidate.name}"
    log_path = Path(args.log_dir) / f"final_trajprobe_eval_{split_label(eval_name)}_{candidate.name}.log"
    cmd = candidate_eval_cmd(args, probe_pt, input_npz, eval_dir, device)
    run_logged(cmd, log_path, args.dry_run)
    return f"{eval_name}/{candidate.name}"


def run_artifact_one(
    args: argparse.Namespace,
    split: str,
    candidate: Candidate,
    predictions_jsonl: Path,
    metadata_jsonl: Path,
    trajectory_jsonl: Path | None,
    threshold: float | None,
) -> str:
    out_dir = Path(args.out_root) / "artifacts" / f"{split_label(split)}_{candidate.name}"
    if args.skip_existing and (out_dir / "artifact_summary.json").exists():
        print(f"skip existing artifact: {split} / {candidate.name}")
        return f"{split}/{candidate.name}"
    if not args.dry_run:
        require_file(predictions_jsonl)
        require_file(metadata_jsonl)
        if trajectory_jsonl is not None:
            require_file(trajectory_jsonl)
    log_path = Path(args.log_dir) / f"final_trajprobe_artifact_{split_label(split)}_{candidate.name}.log"
    cmd = artifact_cmd(args, predictions_jsonl, metadata_jsonl, out_dir, trajectory_jsonl, threshold)
    run_logged(cmd, log_path, args.dry_run)
    return f"{split}/{candidate.name}"


def metric_payload(path: Path, split: str | None = None) -> dict[str, Any]:
    payload = read_json(path)
    metrics = payload["metrics"]
    if split is None:
        return metrics
    return metrics[split]


def flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    keys = ("n", "positive_rate", "auroc", "auprc", "recall", "precision", "fpr", "fnr", "f1", "accuracy")
    return {f"{prefix}_{key}": metrics.get(key) for key in keys}


def build_summary(
    args: argparse.Namespace,
    candidates: list[Candidate],
    eval_npz: dict[str, Path],
) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    rows = []
    for candidate in candidates:
        metrics_path = out_root / candidate.name / "metrics.json"
        if not metrics_path.exists():
            continue
        payload = read_json(metrics_path)
        row: dict[str, Any] = {
            "candidate": candidate.name,
            "positions": ",".join(candidate.positions),
            "layers": ",".join(str(layer) for layer in candidate.layers),
            "layer_combine": candidate.layer_combine,
            "hidden_sizes": candidate.hidden_sizes,
            "threshold": payload.get("threshold"),
            "note": candidate.note,
        }
        for split in ("train", "val", "test"):
            if split in payload.get("metrics", {}):
                row.update(flatten_metrics(split, payload["metrics"][split]))
        for eval_name in eval_npz:
            eval_metrics_path = out_root / f"eval_{split_label(eval_name)}_{candidate.name}" / "metrics.json"
            if not eval_metrics_path.exists():
                continue
            row.update(flatten_metrics(split_label(eval_name), metric_payload(eval_metrics_path)))
        rows.append(row)
    return rows


def run_parallel_train(args: argparse.Namespace, candidates: list[Candidate], devices: list[str]) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.jobs, len(candidates))) as executor:
        futures = {
            executor.submit(run_train_one, args, candidate, devices[idx % len(devices)]): candidate
            for idx, candidate in enumerate(candidates)
        }
        for future in concurrent.futures.as_completed(futures):
            candidate = futures[future]
            print(f"finished train {future.result()} ({candidate.note})")


def run_parallel_eval(
    args: argparse.Namespace,
    candidates: list[Candidate],
    eval_npz: dict[str, Path],
    devices: list[str],
) -> None:
    jobs = [(candidate, eval_name, input_npz) for candidate in candidates for eval_name, input_npz in eval_npz.items()]
    if not jobs:
        return
    workers = min(args.eval_jobs or args.jobs, len(jobs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_eval_one,
                args,
                candidate,
                eval_name,
                input_npz,
                devices[idx % len(devices)],
            ): (candidate, eval_name)
            for idx, (candidate, eval_name, input_npz) in enumerate(jobs)
        }
        for future in concurrent.futures.as_completed(futures):
            candidate, eval_name = futures[future]
            print(f"finished eval {future.result()} ({eval_name}, {candidate.name})")


def run_parallel_artifacts(
    args: argparse.Namespace,
    candidates: list[Candidate],
    standard_npz: dict[str, Path],
    eval_npz: dict[str, Path],
    metadata: dict[str, Path],
    trajectories: dict[str, Path],
) -> None:
    jobs: list[tuple[str, Candidate, Path, Path, Path | None, float | None]] = []
    for candidate in candidates:
        cand_dir = Path(args.out_root) / candidate.name
        threshold = None
        metrics_path = cand_dir / "metrics.json"
        if metrics_path.exists():
            threshold_value = read_json(metrics_path).get("threshold")
            if threshold_value is not None:
                threshold = float(threshold_value)
        for split in ("val", "test"):
            predictions = cand_dir / f"predictions_{split}.jsonl"
            meta = metadata.get(split, npz_metadata_path(standard_npz[split]))
            traj = trajectories.get(split)
            if args.dry_run or (predictions.exists() and meta.exists()):
                jobs.append((split, candidate, predictions, meta, traj, threshold))
        for eval_name, input_npz in eval_npz.items():
            predictions = Path(args.out_root) / f"eval_{split_label(eval_name)}_{candidate.name}" / "predictions.jsonl"
            meta = metadata.get(eval_name, npz_metadata_path(input_npz))
            traj = trajectories.get(eval_name)
            if args.dry_run or (predictions.exists() and meta.exists()):
                jobs.append((eval_name, candidate, predictions, meta, traj, threshold))
    if not jobs:
        print("no artifact jobs found")
        return
    workers = min(args.artifact_jobs, len(jobs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_artifact_one, args, split, candidate, predictions, meta, traj, threshold): (
                split,
                candidate,
            )
            for split, candidate, predictions, meta, traj, threshold in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            split, candidate = futures[future]
            print(f"finished artifact {future.result()} ({split}, {candidate.name})")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--val_npz", required=True)
    parser.add_argument("--test_npz", required=True)
    parser.add_argument("--eval_npz", action="append", default=[], help="Extra eval split as name=path.")
    parser.add_argument("--metadata_jsonl", action="append", default=[], help="Metadata override as split=path.")
    parser.add_argument("--trajectory_jsonl", action="append", default=[], help="Normalized trajectory file as split=path.")
    parser.add_argument("--out_root", default="runs/probes/final_trajprobe_validation")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument(
        "--candidates",
        default="single_cot3_l17,single_cot4_l22,concat_cot3_all,mean_cot3_all",
        help=f"Comma-separated builtin candidates. Available: {','.join(sorted(BUILTIN_CANDIDATES))}",
    )
    parser.add_argument("--devices", default="cuda", help="Comma-separated devices, e.g. cuda:0,cuda:1.")
    parser.add_argument("--jobs", type=int, default=4, help="Concurrent training jobs.")
    parser.add_argument("--eval_jobs", type=int, default=None, help="Concurrent eval jobs. Defaults to --jobs.")
    parser.add_argument("--artifact_jobs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--sample_weight_mode", choices=("none", "label", "source", "source_label"), default="source_label")
    parser.add_argument("--threshold_max_fpr", type=float, default=0.05)
    parser.add_argument("--eval_threshold", default="saved", help="'saved', 'auto', or a float for evaluate_probe.py.")
    parser.add_argument("--pairwise_margin_weight", type=float, default=0.0)
    parser.add_argument("--pairwise_pair_id_only", action="store_true")
    parser.add_argument("--allow_missing_positions", action="store_true")
    parser.add_argument("--artifact_top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_artifacts", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.eval_jobs is not None and args.eval_jobs < 1:
        parser.error("--eval_jobs must be >= 1")
    if args.artifact_jobs < 1:
        parser.error("--artifact_jobs must be >= 1")
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.patience < 1:
        parser.error("--patience must be >= 1")
    if not 0 <= args.threshold_max_fpr <= 1:
        parser.error("--threshold_max_fpr must be in [0, 1]")
    return args


def main() -> None:
    args = parse_args()
    candidate_names = parse_candidate_names(args.candidates)
    candidates = [BUILTIN_CANDIDATES[name] for name in candidate_names]
    devices = parse_csv(args.devices) or ["cuda"]
    eval_npz = parse_key_paths(args.eval_npz)
    metadata = parse_key_paths(args.metadata_jsonl)
    trajectories = parse_key_paths(args.trajectory_jsonl)
    standard_npz = {
        "train": Path(args.train_npz),
        "val": Path(args.val_npz),
        "test": Path(args.test_npz),
    }

    if not args.dry_run:
        for path in standard_npz.values():
            require_file(path)
        for path in eval_npz.values():
            require_file(path)

    write_json(
        Path(args.out_root) / "validation_config.json",
        {
            "args": vars(args),
            "candidates": [asdict(candidate) for candidate in candidates],
            "devices": devices,
            "eval_npz": {name: str(path) for name, path in eval_npz.items()},
            "metadata_jsonl": {name: str(path) for name, path in metadata.items()},
            "trajectory_jsonl": {name: str(path) for name, path in trajectories.items()},
        },
    )

    if not args.skip_train:
        run_parallel_train(args, candidates, devices)
    if not args.skip_eval:
        run_parallel_eval(args, candidates, eval_npz, devices)
    if not args.skip_artifacts:
        run_parallel_artifacts(args, candidates, standard_npz, eval_npz, metadata, trajectories)

    if args.dry_run:
        return
    rows = build_summary(args, candidates, eval_npz)
    write_rows(rows, Path(args.out_root) / "summary_grid")
    ranked = sorted(rows, key=lambda row: float(row.get("test_auroc") or float("nan")), reverse=True)
    write_rows(ranked, Path(args.out_root) / "summary_by_test_auroc")
    if ranked:
        print(json.dumps({"best_by_test_auroc": ranked[0]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

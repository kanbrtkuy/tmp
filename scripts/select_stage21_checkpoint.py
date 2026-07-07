#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def shell_join(cmd: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in cmd)


def run(cmd: list[str], *, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"$ {shell_join(cmd)}"
    print(prefix)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(prefix + "\n")
        log.flush()
        rc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise SystemExit(f"command failed rc={rc}: {shell_join(cmd)}")


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def checkpoint_complete(path: Path) -> bool:
    if not (path / "config.json").exists():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("model-*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def list_local_steps(root: Path) -> set[int]:
    if not root.exists():
        return set()
    steps: set[int] = set()
    for child in root.iterdir():
        step = checkpoint_step(child)
        if step is not None and checkpoint_complete(child):
            steps.add(step)
    return steps


def list_remote_steps(r2_root: str, output_name: str) -> set[int]:
    if not r2_root:
        return set()
    remote = f"{r2_root.rstrip('/')}/workspace/outputs/{output_name}"
    proc = subprocess.run(
        ["rclone", "lsf", "--dirs-only", remote],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise SystemExit(f"rclone lsf failed for {remote}:\n{proc.stderr}")
    steps: set[int] = set()
    for line in proc.stdout.splitlines():
        name = line.strip().rstrip("/")
        match = CHECKPOINT_RE.match(name)
        if match:
            steps.add(int(match.group(1)))
    return steps


def parse_candidate_steps(raw: str) -> set[int]:
    steps: set[int] = set()
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        steps.add(int(piece))
    return steps


def filter_steps(
    steps: set[int],
    *,
    candidate_steps: set[int],
    stride_steps: int,
    min_step: int,
    max_step: int | None,
    max_candidates: int | None,
) -> list[int]:
    selected = set(steps)
    latest_step = max(selected) if selected else None
    if candidate_steps:
        selected &= candidate_steps
    if stride_steps > 0 and not candidate_steps:
        selected = {step for step in selected if step % stride_steps == 0 or step == latest_step}
    selected = {step for step in selected if step >= min_step}
    if max_step is not None:
        selected = {step for step in selected if step <= max_step}
    ordered = sorted(selected, reverse=True)
    if max_candidates is not None and max_candidates > 0:
        ordered = ordered[:max_candidates]
    return ordered


def ensure_checkpoint(
    step: int,
    *,
    checkpoint_root: Path,
    r2_root: str,
    output_name: str,
) -> tuple[Path, bool]:
    local = checkpoint_root / f"checkpoint-{step}"
    if checkpoint_complete(local):
        return local, False
    if not r2_root:
        raise SystemExit(f"missing local checkpoint and no --r2_root supplied: {local}")
    remote = f"{r2_root.rstrip('/')}/workspace/outputs/{output_name}/checkpoint-{step}"
    local.mkdir(parents=True, exist_ok=True)
    cmd = ["rclone", "copy", remote, str(local), "--transfers", "8", "--checkers", "16", "--progress"]
    print("$ " + shell_join(cmd))
    rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
    if rc != 0:
        raise SystemExit(f"failed to download {remote} -> {local}")
    if not checkpoint_complete(local):
        raise SystemExit(f"downloaded checkpoint is incomplete: {local}")
    return local, True


def gate_score(report: dict[str, Any]) -> tuple[float, float, float, float, float]:
    gate = report.get("gate", {}) or {}
    overall = report.get("overall", {}) or {}

    def as_float(value: Any, default: float) -> float:
        return default if value is None else float(value)

    min_exact = as_float(gate.get("min_exact_chain"), 0.0)
    min_location = as_float(gate.get("min_location_match"), 0.0)
    max_off_target = as_float(gate.get("max_off_target"), 1.0)
    max_malformed = as_float(gate.get("max_malformed"), 1.0)
    overall_exact = as_float(overall.get("exact_chain_rate"), 0.0)
    return (min_exact, min_location, -max_off_target, -max_malformed, overall_exact)


def write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    ranked = sorted(rows, key=lambda row: tuple(row["score"]), reverse=True)
    (output_root / "sweep_summary.json").write_text(
        json.dumps({"ranked": ranked}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if ranked:
        (output_root / "selected_checkpoint.txt").write_text(ranked[0]["checkpoint_path"] + "\n", encoding="utf-8")
    lines = [
        "# Stage2.1 Checkpoint Selection Sweep",
        "",
        "| rank | checkpoint | gate | min exact | min location | max off-target | max malformed | overall exact |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(ranked, 1):
        gate = row["gate"]
        lines.append(
            "| {rank} | {checkpoint} | {status} | {min_exact:.4f} | {min_location:.4f} | {max_off:.4f} | {max_malformed:.4f} | {overall_exact:.4f} |".format(
                rank=idx,
                checkpoint=row["checkpoint"],
                status=gate.get("status", "unknown"),
                min_exact=float(gate.get("min_exact_chain") or 0.0),
                min_location=float(gate.get("min_location_match") or 0.0),
                max_off=float(gate.get("max_off_target") or 0.0),
                max_malformed=float(gate.get("max_malformed") or 0.0),
                overall_exact=float(row.get("overall", {}).get("exact_chain_rate") or 0.0),
            )
        )
    (output_root / "sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a Stage2.1 checkpoint on a disjoint selection-dev natural pause gate."
    )
    parser.add_argument("--eval_config", default="configs/experiment/stage2_model_comparison_eval_1p5b_stage21_pure_cot5_selection_dev_2xa6000.yaml")
    parser.add_argument("--train_config", default="configs/experiment/stage21_pause_pure_dagger_1p5b_full_2xa6000.yaml")
    parser.add_argument("--checkpoint_root", default="/workspace/outputs/deepseek_1p5b_stage21_pause_pure_cot5_full_2xa6000")
    parser.add_argument("--output_root", default="/workspace/cot-safety/runs/stage21_selection/deepseek_1p5b_stage21_pause_pure_cot5_full_2xa6000")
    parser.add_argument("--r2_root", default="")
    parser.add_argument("--candidate_steps", default="")
    parser.add_argument("--stride_steps", type=int, default=50)
    parser.add_argument("--min_step", type=int, default=0)
    parser.add_argument("--max_step", type=int, default=None)
    parser.add_argument("--max_candidates", type=int, default=None)
    parser.add_argument("--natural_condition", default="stage21_pure_cot5_natural")
    parser.add_argument("--remove_downloaded", action="store_true")
    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root)
    output_root = Path(args.output_root)
    output_name = checkpoint_root.name
    selection_data_dir = output_root / "selection_dev_data"
    logs_dir = output_root / "logs"

    local_steps = list_local_steps(checkpoint_root)
    remote_steps = list_remote_steps(args.r2_root, output_name) if args.r2_root else set()
    steps = filter_steps(
        local_steps | remote_steps,
        candidate_steps=parse_candidate_steps(args.candidate_steps),
        stride_steps=args.stride_steps,
        min_step=args.min_step,
        max_step=args.max_step,
        max_candidates=args.max_candidates,
    )
    if not steps:
        raise SystemExit(
            f"no candidate checkpoints found under {checkpoint_root}"
            + (f" or {args.r2_root}" if args.r2_root else "")
        )

    base_env = os.environ.copy()
    python_dir = Path(sys.executable).resolve().parent
    base_env["PATH"] = f"{python_dir}:{base_env.get('PATH', '')}"
    base_env["STAGE21_SELECTION_DATA_DIR"] = str(selection_data_dir)
    existing_pythonpath = base_env.get("PYTHONPATH", "")
    base_env["PYTHONPATH"] = (
        str(REPO_ROOT / "src")
        if not existing_pythonpath
        else f"{REPO_ROOT / 'src'}:{existing_pythonpath}"
    )
    prepare_env = dict(base_env)
    prepare_env["STAGE21_SWEEP_EVAL_ROOT"] = str(output_root / "prepare")
    run(
        [sys.executable, "scripts/run_model_comparison_eval.py", "--config", args.eval_config, "--phase", "prepare"],
        env=prepare_env,
        log_path=logs_dir / "prepare.log",
    )

    rows: list[dict[str, Any]] = []
    downloaded_checkpoints: dict[int, Path] = {}
    for step in steps:
        checkpoint, downloaded = ensure_checkpoint(
            step,
            checkpoint_root=checkpoint_root,
            r2_root=args.r2_root,
            output_name=output_name,
        )
        if downloaded:
            downloaded_checkpoints[step] = checkpoint
        eval_root = output_root / f"checkpoint-{step}"
        env = dict(base_env)
        env["STAGE21_PURE_1P5B_CHECKPOINT"] = str(checkpoint)
        env["STAGE21_SWEEP_EVAL_ROOT"] = str(eval_root)
        run(
            [
                sys.executable,
                "scripts/run_model_comparison_eval.py",
                "--config",
                args.eval_config,
                "--phase",
                "generate",
                "--conditions",
                args.natural_condition,
            ],
            env=env,
            log_path=logs_dir / f"generate_checkpoint-{step}.log",
        )
        gate_json = eval_root / "stage21_pure_natural_gate.json"
        run(
            [
                sys.executable,
                "scripts/diag_stage2_checkpoint.py",
                "--config",
                args.train_config,
                "--input_jsonl",
                str(eval_root / "generations" / f"{args.natural_condition}_capability.jsonl"),
                "--input_jsonl",
                str(eval_root / "generations" / f"{args.natural_condition}_safety.jsonl"),
                "--output_json",
                str(gate_json),
                "--generation_field",
                "generated",
                "--use_existing_metrics",
            ],
            env=env,
            log_path=logs_dir / f"gate_checkpoint-{step}.log",
        )
        report = json.loads(gate_json.read_text(encoding="utf-8"))
        row = {
            "step": step,
            "checkpoint": f"checkpoint-{step}",
            "checkpoint_path": str(checkpoint),
            "gate_json": str(gate_json),
            "gate": report.get("gate", {}),
            "overall": report.get("overall", {}),
            "groups": report.get("groups", {}),
            "score": list(gate_score(report)),
        }
        rows.append(row)
        write_summary(output_root, rows)

    write_summary(output_root, rows)
    best = sorted(rows, key=lambda row: tuple(row["score"]), reverse=True)[0]
    if args.remove_downloaded:
        best_step = int(best["step"])
        for step, checkpoint in downloaded_checkpoints.items():
            if step != best_step and checkpoint.exists():
                shutil.rmtree(checkpoint)
    print(json.dumps({"selected": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

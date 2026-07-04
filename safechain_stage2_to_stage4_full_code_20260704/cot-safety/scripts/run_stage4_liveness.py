#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or run the Stage4 pause-port liveness battery. Dry-run writes "
            "the auditable battery plan; non-dry-run executes implemented GPU "
            "kernels and writes the liveness_report schema."
        )
    )
    parser.add_argument("--config", default="configs/experiment/stage4_pause_gprs.yaml")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Override liveness.num_prompts for a small pilot.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=260704)
    parser.add_argument("--tests", default=None, help="Comma-separated subset of liveness tests to run.")
    parser.add_argument("--skip_positive_control", action="store_true")
    parser.add_argument(
        "--metrics_json",
        default=None,
        help=(
            "Optional completed liveness metrics/report JSON. When provided, "
            "this command normalizes it to the liveness_report schema instead "
            "of writing only the plan."
        ),
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    from cot_safety.config import load_config
    from cot_safety.steering.liveness import liveness_config, liveness_decision, liveness_plan_status

    config = load_config(REPO_ROOT / args.config)
    run = config.get("run", {})
    output_dir = REPO_ROOT / str(run.get("output_dir", "runs/stage4_pause_gprs"))
    if Path(str(run.get("output_dir", ""))).is_absolute():
        output_dir = Path(str(run.get("output_dir")))
    output_json = Path(args.output_json) if args.output_json else output_dir / "liveness_plan.json"
    liveness = liveness_config(config)
    if args.metrics_json:
        metrics_path = Path(args.metrics_json)
        report = read_json(metrics_path)
        decision = liveness_decision(
            report,
            required_tests=[str(item) for item in liveness.get("tests", [])],
            gate=liveness.get("gate") or {},
        )
        payload = {
            "status": decision,
            "decision": decision,
            "config": args.config,
            "metrics_json": str(metrics_path),
            "liveness": liveness,
            "report": report,
        }
        if args.output_json is None:
            output_json = output_dir / "liveness_report.json"
        write_json(output_json, payload)
        print(str(output_json))
        return
    plan = {
        "status": liveness_plan_status(liveness, dry_run=args.dry_run),
        "config": args.config,
        "liveness": liveness,
        "next_step": (
            "Run this battery after the Stage2 kl_transparent_emit checkpoint exists; "
            "green => fixed Stage3 then GPRS; yellow => proceed on live layers only "
            "and queue Stage2.5-A for the next Stage2 train; red with a green "
            "positive control => stop Stage4 and branch to Stage2.5-A/B."
        ),
    }
    if args.dry_run:
        write_json(output_json, plan)
        print(str(output_json))
        return

    from cot_safety.steering.liveness_kernels import run_liveness_battery

    selected_tests = [piece.strip() for piece in args.tests.split(",") if piece.strip()] if args.tests else None
    report = run_liveness_battery(
        config,
        repo_root=REPO_ROOT,
        prompt_limit=args.limit,
        batch_size=args.batch_size,
        tests=selected_tests,
        seed=args.seed,
        include_positive_control=not args.skip_positive_control,
    )
    decision = liveness_decision(
        report,
        required_tests=[str(item) for item in liveness.get("tests", [])],
        gate=liveness.get("gate") or {},
    )
    payload = {
        "status": decision,
        "decision": decision,
        "config": args.config,
        "liveness": liveness,
        "report": report,
    }
    if args.output_json is None:
        output_json = output_dir / "liveness_report.json"
    write_json(output_json, payload)
    print(str(output_json))


if __name__ == "__main__":
    main()

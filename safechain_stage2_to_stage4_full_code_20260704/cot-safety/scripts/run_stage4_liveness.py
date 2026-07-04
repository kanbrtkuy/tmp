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
            "Plan the Stage4 pause-port liveness battery. The first framework "
            "version writes the auditable battery plan; GPU metric kernels will "
            "fill the same report schema after the Stage2 checkpoint exists."
        )
    )
    parser.add_argument("--config", default="configs/experiment/stage4_pause_gprs.yaml")
    parser.add_argument("--output_json", default=None)
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
        decision = liveness_decision(report)
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
    write_json(output_json, plan)
    print(str(output_json))
    if not args.dry_run:
        raise SystemExit(
            "GPU liveness metrics are not implemented in this framework stub yet. "
            "Use --dry_run to write the plan, or implement injection_gain / "
            "attention_mass / kv_ablation / patching kernels under this entrypoint."
        )


if __name__ == "__main__":
    main()

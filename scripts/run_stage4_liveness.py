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
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    from cot_safety.config import load_config
    from cot_safety.steering.liveness import liveness_config

    config = load_config(REPO_ROOT / args.config)
    run = config.get("run", {})
    output_json = (
        Path(args.output_json)
        if args.output_json
        else REPO_ROOT / str(run.get("output_dir", "runs/stage4_pause_gprs")) / "liveness_plan.json"
    )
    plan = {
        "status": "planned" if args.dry_run else "not_run",
        "config": args.config,
        "liveness": liveness_config(config),
        "next_step": (
            "Run this battery after the Stage2 kl_transparent_emit checkpoint exists; "
            "green => fixed Stage3 then GPRS, yellow/red => Stage2.5 branch."
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

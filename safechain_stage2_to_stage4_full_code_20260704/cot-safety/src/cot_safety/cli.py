from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cot_safety.config import dump_config, load_config
from cot_safety.pipeline import plan_for_config
from cot_safety.steering.scope import validate_no_pre_post_or_cot_targets, validate_target_specs


def cmd_config_show(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    print(dump_config(config))


def _target_positions_from_config(config: dict[str, Any]) -> list[str]:
    steering = config.get("steering", {})
    positions = steering.get("target_positions")
    if positions is None:
        positions = config.get("target_positions")
    if positions is None:
        raise SystemExit("No steering.target_positions found in config.")
    if not isinstance(positions, list):
        raise SystemExit("steering.target_positions must be a list.")
    return [str(item) for item in positions]


def _target_specs_from_config(config: dict[str, Any]) -> str | list[str] | None:
    steering = config.get("steering", {})
    configured = steering.get("target_specs") or config.get("target_specs")
    if isinstance(configured, list) and configured and isinstance(configured[0], dict):
        lines = []
        for item in configured:
            name = str(item.get("name") or "").strip()
            if not name:
                raise SystemExit(f"steering.target_specs entry is missing a non-empty name: {item!r}")
            positions = ",".join(str(pos) for pos in item.get("positions", []))
            lines.append(f"{name}|{positions}")
        return lines
    return configured


def cmd_steer_validate_scope(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    targets = validate_no_pre_post_or_cot_targets(_target_positions_from_config(config))
    specs = _target_specs_from_config(config)
    validated_specs = validate_target_specs(specs) if specs else None
    print(
        json.dumps(
            {"ok": True, "target_positions": targets, "target_specs": validated_specs},
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_manifest(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_config(config), encoding="utf-8")
    print(str(out))


def cmd_pipeline_plan(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    steps = [step.to_dict() for step in plan_for_config(config)]
    print(json.dumps({"config": args.config, "steps": steps}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cot-safety")
    sub = parser.add_subparsers(dest="command", required=True)

    config_p = sub.add_parser("config")
    config_sub = config_p.add_subparsers(dest="config_command", required=True)
    show = config_sub.add_parser("show")
    show.add_argument("--config", required=True)
    show.set_defaults(func=cmd_config_show)

    steer_p = sub.add_parser("steer")
    steer_sub = steer_p.add_subparsers(dest="steer_command", required=True)
    validate = steer_sub.add_parser("validate-scope")
    validate.add_argument("--config", required=True)
    validate.set_defaults(func=cmd_steer_validate_scope)

    manifest = sub.add_parser("write-resolved-config")
    manifest.add_argument("--config", required=True)
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(func=cmd_manifest)

    pipeline_p = sub.add_parser("pipeline")
    pipeline_sub = pipeline_p.add_subparsers(dest="pipeline_command", required=True)
    plan = pipeline_sub.add_parser("plan")
    plan.add_argument("--config", required=True)
    plan.set_defaults(func=cmd_pipeline_plan)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

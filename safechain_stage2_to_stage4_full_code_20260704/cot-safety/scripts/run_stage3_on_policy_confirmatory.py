#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def resolve_path(value: str | Path, *, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [piece.strip() for piece in value.split(",") if piece.strip()]
    return values or None


def configured_positions(config: dict[str, Any], key: str) -> list[str]:
    positions = config.get("hidden", {}).get("positions") or {}
    if isinstance(positions, dict):
        return [str(item) for item in positions.get(key, []) or []]
    return []


def control_positions_from_config(config: dict[str, Any]) -> list[str]:
    on_policy = config.get("probe", {}).get("on_policy", {})
    configured = on_policy.get("true_content_control_positions")
    if configured:
        return [str(item) for item in configured]
    diagnostics = configured_positions(config, "diagnostics")
    return [position for position in diagnostics if position.startswith("control_cot_")]


def output_path_from_config(config: dict[str, Any], *, legacy_root: Path) -> Path:
    from scripts.run_stage3_intra_pause_probe import stage_paths

    paths = stage_paths(config)
    out_root = Path(paths["single_scan_out_root"])
    out_root = out_root if out_root.is_absolute() else legacy_root / out_root
    return out_root / "stage3_on_policy_confirmatory_report.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Stage3 confirmatory on-policy within-prompt evidence analysis. "
            "Inputs must be hidden-state NPZ files from on-policy generations with "
            "per-generation CoT judge labels in labels."
        )
    )
    parser.add_argument("--config", default="configs/experiment/stage3_intra_pause_probe_kl_transparent_1p5b_cot3.yaml")
    parser.add_argument("--npz", default=None, help="Single on-policy NPZ; prompts are split into train/test groups.")
    parser.add_argument("--train_npz", default=None)
    parser.add_argument("--test_npz", default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--legacy-root", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--positions", default=None, help="Comma-separated pause positions; defaults to probe.on_policy.positions.")
    parser.add_argument(
        "--control_positions",
        default=None,
        help="Comma-separated true no-pause content controls; defaults to control_cot_* diagnostics when present.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    from cot_safety.config import load_config
    config = load_config(REPO_ROOT / args.config)
    on_policy = config.get("probe", {}).get("on_policy", {})
    legacy_root = Path(args.legacy_root) if args.legacy_root else REPO_ROOT / "legacy/PauseProbe"
    layer = int(args.layer if args.layer is not None else on_policy.get("layer", 14))
    positions = csv(args.positions) or [str(item) for item in on_policy.get("positions", []) or []]
    if not positions:
        positions = configured_positions(config, "main")
    if not positions:
        raise SystemExit("Stage3 on-policy confirmatory analysis needs pause positions.")
    control_positions = csv(args.control_positions)
    if control_positions is None:
        control_positions = control_positions_from_config(config)
    output_json = Path(args.output_json) if args.output_json else output_path_from_config(config, legacy_root=legacy_root)
    output_json = output_json if output_json.is_absolute() else REPO_ROOT / output_json

    plan = {
        "config": args.config,
        "layer": layer,
        "positions": positions,
        "control_positions": control_positions,
        "train_npz": args.train_npz,
        "test_npz": args.test_npz,
        "npz": args.npz,
        "output_json": str(output_json),
        "status": "planned",
        "required_input": (
            "Use hidden-state NPZ files extracted from on-policy sampled generations. "
            "labels must be per-generation CoT judge labels, not prompt/reference labels."
        ),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    from cot_safety.probes.on_policy_stage3 import (
        build_on_policy_confirmatory_report,
        load_npz,
        split_by_prompt,
    )

    if args.train_npz and args.test_npz:
        train_data = load_npz(resolve_path(args.train_npz, root=REPO_ROOT))
        test_data = load_npz(resolve_path(args.test_npz, root=REPO_ROOT))
        input_meta = {
            "train_npz": str(resolve_path(args.train_npz, root=REPO_ROOT)),
            "test_npz": str(resolve_path(args.test_npz, root=REPO_ROOT)),
            "split": "provided",
        }
    elif args.npz:
        all_data = load_npz(resolve_path(args.npz, root=REPO_ROOT))
        test_fraction = float(on_policy.get("test_prompt_fraction", 0.5))
        seed = int(on_policy.get("seed", 260704))
        train_data, test_data = split_by_prompt(all_data, test_fraction=test_fraction, seed=seed)
        input_meta = {
            "npz": str(resolve_path(args.npz, root=REPO_ROOT)),
            "split": "prompt_split",
            "test_prompt_fraction": test_fraction,
            "seed": seed,
        }
    else:
        raise SystemExit("Pass either --train_npz and --test_npz, or --npz for prompt-level splitting.")

    report = build_on_policy_confirmatory_report(
        train_data,
        test_data,
        layer=layer,
        positions=positions,
        control_positions=control_positions,
        min_mixed_prompts=int(on_policy.get("min_mixed_prompts", 20)),
        min_within_prompt_auroc=float(on_policy.get("min_within_prompt_auroc", 0.55)),
        min_margin_over_baselines=float(on_policy.get("min_pause_margin_over_baselines", 0.01)),
        require_true_content_control=bool(on_policy.get("require_true_content_control", False)),
        position_pool=str(on_policy.get("position_pool", "mean")),
        require_all_positions=bool(on_policy.get("require_all_positions", True)),
        bootstrap_samples=int(on_policy.get("bootstrap_samples", 1000)),
        seed=int(on_policy.get("seed", 260704)),
    )
    report["config"] = args.config
    report["input"] = input_meta
    report["label_requirement"] = "on_policy_per_generation_cot_judge"
    write_json(output_json, report)
    print(str(output_json))


if __name__ == "__main__":
    main()

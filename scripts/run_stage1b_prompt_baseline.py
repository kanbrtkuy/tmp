#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_stage1_positionscan as stage1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 1b prompt-only/pre-CoT baseline from a resolved config. "
            "Stage 1b reuses Stage 1 data/split/probe machinery and only changes "
            "the hidden-state positions being compared."
        )
    )
    parser.add_argument("--config", default="configs/experiment/stage1b_prompt_baseline.yaml")
    parser.add_argument("--legacy-root", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max_per_source", type=int, default=None)
    parser.add_argument("--skip_data_prep", action="store_true")
    parser.add_argument("--skip_hidden_extraction", action="store_true")
    parser.add_argument("--skip_single_scan", action="store_true")
    parser.add_argument("--skip_multilayer", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    from cot_safety.config import dump_config, load_config

    config = stage1.resolve_value(load_config(REPO_ROOT / args.config))
    runtime = config.get("runtime", {})
    if runtime.get("cuda_visible_devices"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(runtime["cuda_visible_devices"]))
    if runtime.get("hf_home"):
        os.environ.setdefault("HF_HOME", str(stage1.resolve_value(runtime["hf_home"])))
    if runtime.get("pytorch_cuda_alloc_conf"):
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            str(stage1.resolve_value(runtime["pytorch_cuda_alloc_conf"])),
        )

    runs_dir = REPO_ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = str(config.get("run", {}).get("name", "stage1b_prompt_baseline"))
    (runs_dir / f"{run_name}_resolved.yaml").write_text(dump_config(config), encoding="utf-8")

    legacy_root = Path(args.legacy_root) if args.legacy_root else REPO_ROOT / "legacy/PauseProbe"
    cmd = stage1.build_command(args, config)
    print("$ " + " ".join(cmd))
    if args.dry_run:
        return
    raise SystemExit(subprocess.run(cmd, cwd=legacy_root, env=os.environ.copy()).returncode)


if __name__ == "__main__":
    main()

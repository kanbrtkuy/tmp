#!/usr/bin/env python3
"""Freeze the formal Stage2 18k dataset after fail-closed decontamination."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.config import load_config  # noqa: E402
from cot_safety.data.stage2_formal_freeze import freeze_formal_dataset  # noqa: E402


def resolve_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/stage2_intra_pause_sft_8b_2xa100.yaml")
    parser.add_argument("--output_root", default=None)
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    data = load_config(config_path).get("data", {})
    cfg = data.get("formal_freeze") or {}
    if cfg.get("enabled") is not True:
        raise SystemExit("data.formal_freeze.enabled must be true for the formal freeze")
    eval_cfg = cfg.get("formal_eval_files") or {}
    if not isinstance(eval_cfg, dict) or not eval_cfg:
        raise SystemExit("data.formal_freeze.formal_eval_files must be a non-empty mapping")
    result = freeze_formal_dataset(
        candidate_path=resolve_path(cfg["candidate_jsonl"]),
        eval_files={str(name): resolve_path(path) for name, path in eval_cfg.items()},
        cosine_audit_path=resolve_path(cfg["cosine_audit_json"]),
        manual_decisions_path=resolve_path(cfg["manual_decisions_json"]),
        output_root=resolve_path(args.output_root or cfg["output_root"]),
        source_quotas={str(name): int(count) for name, count in (cfg.get("source_quotas") or {}).items()},
        split_counts={str(name): int(count) for name, count in (cfg.get("split_counts") or {}).items()},
        seed=int(cfg["seed"]),
        lexical_threshold=float((cfg.get("lexical") or {})["jaccard_threshold"]),
        cosine_threshold=float((cfg.get("cosine") or {})["threshold"]),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

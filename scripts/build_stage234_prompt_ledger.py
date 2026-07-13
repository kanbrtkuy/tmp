#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.config import load_config  # noqa: E402
from cot_safety.data.stage234_ledger import (  # noqa: E402
    DEFAULT_SPLIT_COUNTS,
    build_ledger,
    candidate_from_row,
    read_jsonl,
    sha256_file,
    write_jsonl,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the disjoint Stage3/4 four-source prompt-family ledger.")
    parser.add_argument("--config", default="configs/data/stage234_prompt_ledger.yaml")
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--manifest_json", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    ledger_cfg = config.get("stage234_ledger", {})
    sources_cfg = ledger_cfg.get("sources") or []
    if len(sources_cfg) != 4:
        raise SystemExit(f"Formal ledger requires exactly four source configs; got {len(sources_cfg)}")
    seed = int(ledger_cfg.get("seed", 260714))
    split_counts = {**DEFAULT_SPLIT_COUNTS, **dict(ledger_cfg.get("split_counts") or {})}
    source_rows = {}
    input_files = []
    source_order = []
    for source_cfg in sources_cfg:
        source = str(source_cfg["name"])
        path = Path(str(source_cfg["input_jsonl"]))
        if not path.exists():
            raise SystemExit(f"Missing source JSONL for {source}: {path}")
        source_order.append(source)
        input_files.append({"source": source, "path": str(path), "sha256": sha256_file(path)})
        candidates = []
        for index, row in enumerate(read_jsonl(path)):
            candidate = candidate_from_row(
                row,
                source=source,
                source_path=path,
                source_row_index=index,
                prompt_fields=tuple(source_cfg.get("prompt_fields") or ()),
                family_fields=tuple(source_cfg.get("family_fields") or ()),
                row_id_fields=tuple(source_cfg.get("row_id_fields") or ()),
            )
            if candidate is not None:
                candidates.append(candidate)
        source_rows[source] = candidates
    ledger, manifest = build_ledger(
        source_rows,
        seed=seed,
        split_counts=split_counts,
        source_order=source_order,
        drop_cross_source_exact_duplicates=bool(ledger_cfg.get("drop_cross_source_exact_duplicates", True)),
    )
    manifest["input_files"] = input_files
    output_jsonl = Path(args.output_jsonl or ledger_cfg["output_jsonl"])
    manifest_json = Path(args.manifest_json or ledger_cfg["manifest_json"])
    manifest["output_jsonl"] = str(output_jsonl)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return
    write_jsonl(output_jsonl, ledger)
    manifest["ledger_file_sha256"] = sha256_file(output_jsonl)
    write_json(manifest_json, manifest)
    print(json.dumps({"status": "written", "rows": len(ledger), "ledger": str(output_jsonl), "manifest": str(manifest_json)}, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = REPO_ROOT / "scripts" / "data" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def export_row(split: str, source: str, pair_idx: int, label: str) -> dict:
    pair_id = f"{source}-{split}-{pair_idx}"
    is_unsafe = label == "unsafe"
    signal = f"{source} risk_token" if is_unsafe else f"{source} support_token"
    return {
        "id": f"{pair_id}::{label}",
        "source": "unit",
        "source_family": "paired_rewrite",
        "prompt": f"Shared prompt {pair_idx} for {source} {split}.",
        "reasoning": f"{signal} neutral words balanced length marker",
        "final_answer": "",
        "safety_label": label,
        "trajectory_safety_label": label,
        "label_task": "trajectory_safety",
        "policy_type": "paired_rewrite",
        "pair_id": pair_id,
        "match_family": f"{source}-{split}-group-{pair_idx}",
        "metadata": {
            "source_pair_source": source,
            "variant": "unsafe_openai_paraphrase" if is_unsafe else "safe_rewrite_reasoning_only",
        },
    }


def build_export_dir(tmp_path: Path) -> Path:
    export_dir = tmp_path / "export"
    all_rows = []
    for split in ("train", "val", "test"):
        split_rows = []
        for source in ("harmthoughts", "reasoningshield"):
            for idx in range(3):
                split_rows.append(export_row(split, source, idx, "unsafe"))
                split_rows.append(export_row(split, source, idx, "safe"))
        write_jsonl(export_dir / "normalized" / f"{split}.jsonl", split_rows)
        all_rows.extend(split_rows)
    write_jsonl(export_dir / "normalized" / "all.jsonl", all_rows)
    return export_dir


def test_stage1_surface_audit_runs_feature_length_truncation_and_source_transfer(tmp_path, monkeypatch):
    script = load_script("run_stage1_surface_audit")
    export_dir = build_export_dir(tmp_path)
    output_dir = tmp_path / "audit"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_stage1_surface_audit.py",
            "--export-dir",
            str(export_dir),
            "--output-dir",
            str(output_dir),
            "--top-n",
            "3",
            "--truncation-ks",
            "2,full",
            "--max-features-word",
            "1000",
            "--max-features-char",
            "1000",
        ],
    )

    assert script.main() == 0

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["feature_audit"]["word_tfidf"]["top_n_per_direction"] == 3
    assert metrics["length_analysis"]["pairwise"]["train"]["retained_pairs"] == 6
    assert metrics["length_matched_baselines"]["skipped"] is None
    assert len(metrics["truncation_curves"]["results"]) == 6
    assert len(metrics["cross_source_transfer"]["results"]) == 2

    assert (output_dir / "feature_audit_word_tfidf.tsv").exists()
    assert (output_dir / "feature_audit_char_tfidf.tsv").exists()
    assert (output_dir / "length_matched_baselines.tsv").exists()
    assert (output_dir / "truncation_curves.tsv").exists()
    assert (output_dir / "cross_source_transfer.tsv").exists()

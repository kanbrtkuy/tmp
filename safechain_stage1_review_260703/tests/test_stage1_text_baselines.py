from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


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


def export_row(split: str, pair_idx: int, label: str, *, match_family: str | None = None) -> dict:
    is_unsafe = label == "unsafe"
    pair_id = f"{split}-pair-{pair_idx}"
    group = match_family or f"{split}-group-{pair_idx}"
    return {
        "id": f"{pair_id}::{label}",
        "source": "unit",
        "source_family": "paired_rewrite",
        "prompt": f"Shared prompt {pair_idx} for {split}.",
        "reasoning": (
            "unsafe hazard exploit instruction detail repeated"
            if is_unsafe
            else "safe boundary prevention support alternative repeated"
        ),
        "final_answer": "",
        "safety_label": label,
        "trajectory_safety_label": label,
        "label_task": "trajectory_safety",
        "policy_type": "paired_rewrite",
        "pair_id": pair_id,
        "match_family": group,
        "metadata": {"variant": "unsafe_openai_paraphrase" if is_unsafe else "safe_rewrite_reasoning_only"},
    }


def build_export_dir(tmp_path: Path, *, overlap: bool = False) -> Path:
    export_dir = tmp_path / "export"
    for split in ("train", "val", "test"):
        rows: list[dict] = []
        for idx in range(4):
            group = "overlap-group" if overlap and split in {"train", "test"} and idx == 0 else None
            rows.append(export_row(split, idx, "unsafe", match_family=group))
            rows.append(export_row(split, idx, "safe", match_family=group))
        write_jsonl(export_dir / "normalized" / f"{split}.jsonl", rows)
    return export_dir


def test_stage1_text_baselines_runs_supported_surface_baselines(tmp_path, monkeypatch):
    script = load_script("run_stage1_text_baselines")
    export_dir = build_export_dir(tmp_path)
    output_dir = tmp_path / "baselines"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_stage1_text_baselines.py",
            "--export-dir",
            str(export_dir),
            "--output-dir",
            str(output_dir),
            "--baselines",
            "length_only,word_tfidf,first_sentence_removed_tfidf",
            "--n-jobs",
            "2",
            "--max-features-word",
            "1000",
            "--max-features-char",
            "1000",
        ],
    )

    assert script.main() == 0

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    summary = (output_dir / "summary.tsv").read_text(encoding="utf-8")

    assert [item["name"] for item in metrics["results"]] == [
        "length_only",
        "word_tfidf",
        "first_sentence_removed_tfidf",
    ]
    assert "original_vs_openai_paraphrase_provenance" in metrics["skipped"]
    assert metrics["split_summary"]["train"]["labels"] == {"unsafe": 4, "safe": 4}
    assert "baseline\tsplit\tn\tbalanced_accuracy" in summary
    assert "word_tfidf\ttest" in summary


def test_stage1_text_baselines_rejects_split_overlap(tmp_path, monkeypatch):
    script = load_script("run_stage1_text_baselines")
    export_dir = build_export_dir(tmp_path, overlap=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_stage1_text_baselines.py",
            "--export-dir",
            str(export_dir),
            "--output-dir",
            str(tmp_path / "baselines"),
            "--baselines",
            "length_only",
        ],
    )

    with pytest.raises(ValueError, match="match_family overlap"):
        script.main()

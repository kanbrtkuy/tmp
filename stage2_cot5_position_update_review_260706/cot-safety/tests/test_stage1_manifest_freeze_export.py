from __future__ import annotations

import hashlib
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def manifest_row(
    pair_id: str,
    prompt: str,
    *,
    source: str = "unit_source",
    category: str = "unit_category",
    tier_short: str = "A",
) -> dict:
    unsafe_reasoning = f"Unsafe reasoning for {pair_id}. It stays short for a synthetic fixture."
    safe_reasoning = f"Safe reasoning for {pair_id}. It stays short for a synthetic fixture."
    safe_final_answer = f"Safe final for {pair_id}."
    return {
        "pair_id": pair_id,
        "prompt_id": f"prompt-{pair_id}",
        "prompt": prompt,
        "unsafe_reasoning": unsafe_reasoning,
        "safe_reasoning": safe_reasoning,
        "safe_final_answer": safe_final_answer,
        "audit_keep": True,
        "source": source,
        "category": category,
        "model_name": "unit-model",
        "tier": f"{tier_short}-tier",
        "tier_short": tier_short,
        "label_pair": "unsafe_to_safe",
        "hashes": {
            "prompt_sha256": sha256_text(prompt),
            "unsafe_reasoning_sha256": sha256_text(unsafe_reasoning),
            "safe_reasoning_sha256": sha256_text(safe_reasoning),
            "safe_final_answer_sha256": sha256_text(safe_final_answer),
        },
    }


def build_tiny_manifests(tmp_path: Path) -> tuple[Path, Path]:
    manifest_a = tmp_path / "A_prime_manifest.jsonl"
    manifest_b = tmp_path / "B_prime_manifest.jsonl"
    shared_prompt = "How should this synthetic prompt be handled?"
    write_jsonl(
        manifest_a,
        [
            manifest_row("a-shared", shared_prompt, source="manifest_a", tier_short="A"),
            manifest_row("a-only", "A-only synthetic prompt.", source="manifest_a", tier_short="A"),
        ],
    )
    write_jsonl(
        manifest_b,
        [
            manifest_row("b-shared", shared_prompt, source="manifest_b", tier_short="B"),
            manifest_row("b-only", "B-only synthetic prompt.", source="manifest_b", tier_short="B"),
        ],
    )
    return manifest_a, manifest_b


def freeze_prompt_splits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    freeze = load_script("freeze_stage1_prompt_splits")
    manifest_a, manifest_b = build_tiny_manifests(tmp_path)
    split_jsonl = tmp_path / "splits" / "prompt_splits.jsonl"
    summary_json = tmp_path / "splits" / "prompt_splits.summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_stage1_prompt_splits.py",
            "--input-manifest",
            str(manifest_a),
            "--input-manifest",
            str(manifest_b),
            "--output-jsonl",
            str(split_jsonl),
            "--summary-json",
            str(summary_json),
            "--train-ratio",
            "0.34",
            "--val-ratio",
            "0.33",
            "--seed",
            "260702",
        ],
    )
    assert freeze.main() == 0
    return split_jsonl, summary_json


def test_freeze_groups_same_prompt_across_input_manifests(tmp_path, monkeypatch):
    split_jsonl, summary_json = freeze_prompt_splits(tmp_path, monkeypatch)

    rows = read_jsonl(split_jsonl)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    shared = next(row for row in rows if set(row["pair_ids"]) == {"a-shared", "b-shared"})

    assert summary["n_pairs"] == 4
    assert summary["n_prompt_groups"] == 3
    assert summary["prompt_groups_shared_across_manifests"] == 1
    assert summary["split_counts_by_prompt_group"] == {"train": 1, "val": 1, "test": 1}
    assert shared["n_pairs"] == 2
    assert shared["manifest_labels"]
    assert shared["raw_prompt_sha256s"] == [
        sha256_text("How should this synthetic prompt be handled?")
    ]


def test_freeze_rejects_prompt_hash_mismatch(tmp_path, monkeypatch):
    freeze = load_script("freeze_stage1_prompt_splits")
    manifest = tmp_path / "bad_manifest.jsonl"
    row = manifest_row("bad-hash", "Prompt text with a bad stored hash.")
    row["hashes"]["prompt_sha256"] = "not-the-real-hash"
    write_jsonl(manifest, [row])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_stage1_prompt_splits.py",
            "--input-manifest",
            str(manifest),
            "--output-jsonl",
            str(tmp_path / "splits.jsonl"),
            "--summary-json",
            str(tmp_path / "summary.json"),
        ],
    )

    with pytest.raises(ValueError, match="prompt hash mismatch"):
        freeze.main()


def test_export_manifest_uses_frozen_splits_and_reasoning_only_rows(tmp_path, monkeypatch):
    split_jsonl, _ = freeze_prompt_splits(tmp_path, monkeypatch)
    export = load_script("export_safe_rewrite_pairs_for_stage1")
    input_manifest = tmp_path / "A_prime_manifest.jsonl"
    output_dir = tmp_path / "stage1_export"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_safe_rewrite_pairs_for_stage1.py",
            "--input-manifest",
            str(input_manifest),
            "--split-manifest",
            str(split_jsonl),
            "--output-dir",
            str(output_dir),
            "--render-mode",
            "reasoning_only",
            "--expected-pairs",
            "2",
        ],
    )
    assert export.main() == 0

    normalized = read_jsonl(output_dir / "normalized" / "all.jsonl")
    cotpause = read_jsonl(output_dir / "cotpause" / "all.jsonl")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert len(normalized) == 4
    assert manifest["split_strategy"] == "frozen_prompt_group_manifest"
    assert manifest["render_mode"] == "reasoning_only"
    assert manifest["quality"]["pairs_with_both_labels"] == 2
    assert manifest["quality"]["by_label"] == {"unsafe": 2, "safe": 2}

    by_pair: dict[str, list[dict]] = {}
    for row in normalized:
        by_pair.setdefault(row["pair_id"], []).append(row)
    assert set(by_pair) == {"a-shared", "a-only"}
    for rows in by_pair.values():
        assert {row["trajectory_safety_label"] for row in rows} == {"unsafe", "safe"}
        assert {row["match_family"] for row in rows} == {rows[0]["match_family"]}
        assert {row["metadata"]["variant"] for row in rows} == {
            "unsafe_openai_paraphrase",
            "safe_rewrite_reasoning_only",
        }
        assert {row["final_answer"] for row in rows} == {""}

    assert len(cotpause) == 4
    assert all(row["output"].startswith("<think>\n") for row in cotpause)
    assert all(row["output"].endswith("\n</think>") for row in cotpause)
    assert all("Safe final for" not in row["output"] for row in cotpause)
    assert all(row["metadata"]["probe_render_mode"] == "reasoning_only" for row in cotpause)

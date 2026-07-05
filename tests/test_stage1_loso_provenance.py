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
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def norm_row(prompt_id: str, label: str, *, split: str = "train") -> dict:
    pair_id = f"{prompt_id}::pair"
    return {
        "id": f"{pair_id}::{label}",
        "pair_id": pair_id,
        "match_family": prompt_id,
        "prompt_instance_id": prompt_id,
        "split": split,
        "trajectory_safety_label": label,
        "provenance_join_status": "joined",
        "prompt": "content hidden in tests",
        "reasoning": f"{label} reasoning",
        "final_answer": f"{label} final",
        "metadata": {},
    }


def test_rejoin_natural_pair_source_provenance_adds_source_family(tmp_path, monkeypatch):
    script = load_script("rejoin_natural_pair_source_provenance")
    input_dir = tmp_path / "input"
    prompt_manifest = tmp_path / "prompt_manifest.jsonl"
    rows = [norm_row("p1", "safe"), norm_row("p1", "unsafe")]
    write_jsonl(input_dir / "normalized" / "train.jsonl", rows)
    write_jsonl(
        prompt_manifest,
        [
            {
                "prompt_instance_id": "p1",
                "source_datasets": ["harmthoughts"],
                "source_model_canonical": "r1-8b",
                "source_seed_refs": [{"source": "harmthoughts", "seed_id": "seed-1"}],
            }
        ],
    )
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rejoin_natural_pair_source_provenance.py",
            "--input-dir",
            str(input_dir),
            "--prompt-manifest",
            str(prompt_manifest),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert script.main() == 0

    out_rows = [json.loads(line) for line in (output_dir / "normalized" / "train.jsonl").read_text().splitlines()]
    assert {row["source_family"] for row in out_rows} == {"harmthoughts"}
    assert {row["provenance_join_status"] for row in out_rows} == {"joined"}
    summary = json.loads((output_dir / "provenance_join_summary.json").read_text())
    assert summary["split_summary"]["train"]["source_family"] == {"harmthoughts": 2}


def test_make_natural_pair_loso_folds_keeps_pairs_together(tmp_path, monkeypatch):
    script = load_script("make_natural_pair_loso_folds")
    input_dir = tmp_path / "input"
    all_rows = []
    for source in ("harmthoughts", "reasoningshield"):
        for idx in range(2):
            prompt_id = f"{source}-{idx}"
            for label in ("safe", "unsafe"):
                row = norm_row(prompt_id, label)
                row["source_family"] = source
                row["metadata"] = {"source_pair_source": source}
                all_rows.append(row)
    for source in ("unknown", "harmthoughts+reasoningshield"):
        prompt_id = f"{source}-ambiguous"
        for label in ("safe", "unsafe"):
            row = norm_row(prompt_id, label)
            row["source_family"] = source
            row["metadata"] = {"source_pair_source": source}
            all_rows.append(row)
    write_jsonl(input_dir / "normalized" / "all.jsonl", all_rows)
    output_dir = tmp_path / "loso"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_natural_pair_loso_folds.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--min-test-pairs",
            "1",
            "--min-train-pairs",
            "1",
        ],
    )

    assert script.main() == 0

    test_rows = [
        json.loads(line)
        for line in (output_dir / "harmthoughts" / "normalized" / "test.jsonl").read_text().splitlines()
    ]
    train_rows = [
        json.loads(line)
        for line in (output_dir / "harmthoughts" / "normalized" / "train.jsonl").read_text().splitlines()
    ]
    assert {row["source_family"] for row in test_rows} == {"harmthoughts"}
    assert {row["source_family"] for row in train_rows} == {"reasoningshield"}
    assert "unknown" not in {row["source_family"] for row in train_rows + test_rows}
    assert "harmthoughts+reasoningshield" not in {row["source_family"] for row in train_rows + test_rows}
    for rows in (test_rows, train_rows):
        by_pair = {}
        for row in rows:
            by_pair.setdefault(row["pair_id"], set()).add(row["trajectory_safety_label"])
        assert all(labels == {"safe", "unsafe"} for labels in by_pair.values())

    summary = json.loads((output_dir / "loso_summary.json").read_text())
    assert summary["source_filter"]["dropped_pairs"] == {
        "ambiguous_source_family:harmthoughts+reasoningshield": 1,
        "ambiguous_source_family:unknown": 1,
    }


def test_make_natural_pair_loso_folds_splits_train_val_by_match_family(tmp_path, monkeypatch):
    script = load_script("make_natural_pair_loso_folds")
    input_dir = tmp_path / "input"
    all_rows = []
    for prompt_id, source, family in (
        ("h0", "harmthoughts", "h0"),
        ("r0a", "reasoningshield", "shared-r0"),
        ("r0b", "reasoningshield", "shared-r0"),
        ("r1", "reasoningshield", "r1"),
    ):
        for label in ("safe", "unsafe"):
            row = norm_row(prompt_id, label)
            row["match_family"] = family
            row["source_family"] = source
            row["metadata"] = {"source_pair_source": source}
            all_rows.append(row)
    write_jsonl(input_dir / "normalized" / "all.jsonl", all_rows)
    output_dir = tmp_path / "loso"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_natural_pair_loso_folds.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--min-test-pairs",
            "1",
            "--min-val-pairs",
            "1",
            "--min-train-pairs",
            "1",
        ],
    )

    assert script.main() == 0

    train_rows = [
        json.loads(line)
        for line in (output_dir / "harmthoughts" / "normalized" / "train.jsonl").read_text().splitlines()
    ]
    val_rows = [
        json.loads(line)
        for line in (output_dir / "harmthoughts" / "normalized" / "val.jsonl").read_text().splitlines()
    ]
    train_groups = {row["match_family"] for row in train_rows}
    val_groups = {row["match_family"] for row in val_rows}
    assert not (train_groups & val_groups)


def test_write_val_fixed_probe_report_uses_validation_not_test(tmp_path, monkeypatch):
    script = load_script("write_val_fixed_probe_report")
    summary_tsv = tmp_path / "summary.tsv"
    summary_tsv.write_text(
        "\n".join(
            [
                "model\tposition\tlayer\ttrain_n\tval_n\ttest_n\tval_auroc\tval_recall\tval_fpr\ttest_auroc\ttest_auprc\ttest_recall\ttest_fpr\tthreshold",
                "linear\tcot_4\t10\t10\t4\t4\t0.90\t0.50\t0.00\t0.60\t0.60\t0.50\t0.00\t0.7",
                "linear\tcot_128\t20\t10\t4\t4\t0.70\t1.00\t0.00\t0.99\t0.99\t1.00\t0.00\t0.4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_json = tmp_path / "selected.json"
    output_tsv = tmp_path / "selected.tsv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "write_val_fixed_probe_report.py",
            "--input-tsv",
            str(summary_tsv),
            "--output-json",
            str(output_json),
            "--output-tsv",
            str(output_tsv),
        ],
    )

    assert script.main() == 0

    selected = [line.split("\t") for line in output_tsv.read_text(encoding="utf-8").splitlines()]
    header = selected[0]
    row = dict(zip(header, selected[1]))
    assert row["position"] == "cot_4"
    assert row["test_auroc"] == "0.60"

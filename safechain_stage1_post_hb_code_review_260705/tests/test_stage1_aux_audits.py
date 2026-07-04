import csv
import json
from pathlib import Path

from scripts.data import build_stage1_safe_prompt_diagnostics as safe_diag
from scripts.data import quarantine_stage1_external_prompts as quarantine
from scripts.data import run_stage1_bootstrap_ci as bootstrap
from scripts.data import sample_stage1_human_qa as sample_qa
from scripts.data import summarize_stage1_human_qa as summarize_qa


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalized_row(source, pair_id, label, idx):
    return {
        "id": f"{pair_id}::{label}",
        "pair_id": pair_id,
        "match_family": pair_id,
        "prompt_instance_id": pair_id,
        "source_family": source,
        "prompt": f"Prompt {pair_id}",
        "trajectory_safety_label": label,
        "reasoning": f"{label} reasoning {idx}",
        "final_answer": f"{label} final",
    }


def test_human_qa_sample_and_summary(tmp_path):
    rows = []
    for source in ("strongreject_full", "harmbench_standard"):
        for idx in range(4):
            pair_id = f"{source}-{idx}"
            rows.append(normalized_row(source, pair_id, "safe", idx))
            rows.append(normalized_row(source, pair_id, "unsafe", idx))
    data = tmp_path / "normalized.jsonl"
    sample_dir = tmp_path / "sample"
    summary_dir = tmp_path / "summary"
    write_jsonl(data, rows)

    sample_args = type(
        "Args",
        (),
        {
            "normalized_jsonl": [data],
            "output_dir": str(sample_dir),
            "rows_per_source": 4,
            "seed": 7,
            "include_text": False,
        },
    )()
    sample_summary = sample_qa.run(sample_args)
    assert sample_summary["n_sampled_rows"] == 8

    sheet = sample_dir / "stage1_human_qa_sheet.tsv"
    qa_rows = []
    with sheet.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            row["human_label"] = row["judge_label"]
            row["human_quality"] = "ok"
            qa_rows.append(row)
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=qa_rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(qa_rows)

    summarize_args = type(
        "Args",
        (),
        {
            "qa_tsv": str(sheet),
            "output_dir": str(summary_dir),
            "min_labeled_per_source": 4,
            "unsafe_agreement_bar": 0.90,
            "no_fail": False,
        },
    )()
    qa_summary = summarize_qa.run(summarize_args)
    assert qa_summary["passes"] is True
    assert qa_summary["sources"]["strongreject_full"]["unsafe_agreement_rate"] == 1.0


def test_bootstrap_ci_and_delta(tmp_path):
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    rows_left = []
    rows_right = []
    for idx in range(10):
        label = idx % 2
        rows_left.append({"id": f"l{idx}", "match_family": f"g{idx}", "gold_int": label, "score": 0.9 if label else 0.1})
        rows_right.append({"id": f"r{idx}", "match_family": f"g{idx}", "gold_int": label, "score": 0.5})
    write_jsonl(left, rows_left)
    write_jsonl(right, rows_right)
    args = type(
        "Args",
        (),
        {
            "prediction_jsonl": [f"left={left}", f"right={right}"],
            "delta": ["left:right"],
            "output_dir": str(tmp_path / "ci"),
            "group_fields": "match_family,pair_id,id",
            "n_bootstrap": 30,
            "seed": 7,
        },
    )()
    summary = bootstrap.run(args)
    assert summary["models"]["left"]["auroc"] == 1.0
    assert summary["models"]["right"]["auroc"] == 0.5
    assert summary["deltas"]["left_minus_right"]["delta_auroc"] == 0.5


def test_safe_prompt_diagnostics_dedups_and_filters(tmp_path):
    source = tmp_path / "safe.jsonl"
    write_jsonl(
        source,
        [
            {"id": "a", "prompt": "How do I make tea?", "label": "safe", "source_family": "safe_seed"},
            {"id": "b", "prompt": "How   do I make tea?", "label": "safe", "source_family": "safe_seed"},
            {"id": "c", "prompt": "Unsafe prompt", "label": "unsafe", "source_family": "safe_seed"},
        ],
    )
    args = type(
        "Args",
        (),
        {
            "input_jsonl": [f"safe_seed={source}"],
            "input_csv": None,
            "output_dir": str(tmp_path / "diag"),
            "prompt_fields": "prompt",
            "label_fields": "label",
            "filter_label": "safe,benign",
            "max_per_source": 0,
            "seed": 7,
        },
    )()
    summary = safe_diag.run(args)
    assert summary["n_loaded_prompts"] == 2
    assert summary["n_selected_prompts"] == 1
    assert summary["n_dropped_prompts"] == 1


def test_external_quarantine_exact_match(tmp_path):
    reference = tmp_path / "reference.jsonl"
    external = tmp_path / "external.jsonl"
    write_jsonl(reference, [{"id": "ref1", "prompt": "Shared prompt", "source_family": "freeze"}])
    write_jsonl(
        external,
        [
            {"id": "ext1", "prompt": "Shared   prompt", "source_family": "external"},
            {"id": "ext2", "prompt": "Fresh prompt", "source_family": "external"},
        ],
    )
    args = type(
        "Args",
        (),
        {
            "reference_jsonl": [str(reference)],
            "external_jsonl": [str(external)],
            "output_dir": str(tmp_path / "quarantine"),
            "prompt_fields": "prompt",
            "near_threshold": 0.80,
            "top_k": 50,
            "skip_near_neighbor": True,
            "include_text": False,
        },
    )()
    summary = quarantine.run(args)
    assert summary["n_exact_matches"] == 1
    assert summary["n_quarantined_external_prompts"] == 1
    kept = (tmp_path / "quarantine" / "external_kept_prompts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(kept) == 1

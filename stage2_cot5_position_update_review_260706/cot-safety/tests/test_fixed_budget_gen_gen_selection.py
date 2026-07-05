import json
from pathlib import Path

from scripts.data import select_fixed_budget_gen_gen_pairs as fixed


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def candidate(prompt_id, sample_idx, label, *, quality=True, source="strongreject_full"):
    return {
        "candidate_id": f"{prompt_id}::sample-{sample_idx:03d}",
        "prompt_instance_id": prompt_id,
        "source_model_canonical": "r1-8b",
        "prompt": f"Prompt {prompt_id}",
        "reasoning": f"{label} reasoning {sample_idx}",
        "final_answer": f"{label} final",
        "reasoning_words": 10 + sample_idx,
        "quality_pass": quality,
        "quality_score": 1.0,
        "repeated_4gram_fraction": 0.0,
        "sample_idx": sample_idx,
        "sampling": {"sample_idx": sample_idx},
        "safety_judge": {"safety_label": label},
        "metadata": {"prompt_metadata": {"source_family": source}},
    }


def test_fixed_budget_ignores_out_of_budget_candidates(tmp_path):
    prompt_manifest = tmp_path / "prompt_manifest.jsonl"
    judged = tmp_path / "judged.jsonl"
    output = tmp_path / "fixed"
    write_jsonl(
        prompt_manifest,
        [
            {
                "prompt_instance_id": "p1",
                "source_model_canonical": "r1-8b",
                "prompt": "Prompt p1",
                "metadata": {"source_family": "strongreject_full"},
            },
            {
                "prompt_instance_id": "p2",
                "source_model_canonical": "r1-8b",
                "prompt": "Prompt p2",
                "metadata": {"source_family": "harmbench_standard"},
            },
        ],
    )
    write_jsonl(
        judged,
        [
            candidate("p1", 0, "safe"),
            candidate("p1", 1, "unsafe"),
            candidate("p2", 0, "safe", source="harmbench_standard"),
            candidate("p2", 50, "unsafe", source="harmbench_standard"),
        ],
    )
    args = type(
        "Args",
        (),
        {
            "model": "r1-8b",
            "judged_candidates": str(judged),
            "prompt_manifest": str(prompt_manifest),
            "sample_start": 0,
            "max_sample_idx": 50,
            "allow_quality_fail": False,
            "output_dir": str(output),
            "write_filtered_judged": True,
        },
    )()
    summary = fixed.select_fixed_budget(fixed.gen_gen.read_config(None), args)
    assert summary["n_selected_pairs"] == 1
    assert summary["selected_pairs_by_source"] == {"strongreject_full": 1}
    assert summary["dropped_prompts_by_source"] == {"harmbench_standard": 1}
    assert summary["source_budget_table"]["harmbench_standard"]["n_selected_pairs"] == 0
    assert summary["source_budget_table"]["harmbench_standard"]["n_prompts_with_partial_window_coverage"] == 1
    assert summary["fixed_budget_loso_readiness"]["strongreject_full"]["status"] == "below_pilot_floor"
    assert summary["n_judged_prompts_not_in_manifest"] == 0
    assert (output / "judged_candidates_fixed_budget.jsonl").exists()


def test_fixed_budget_respects_sample_start_and_quality(tmp_path):
    prompt_manifest = tmp_path / "prompt_manifest.jsonl"
    judged = tmp_path / "judged.jsonl"
    output = tmp_path / "fixed"
    write_jsonl(
        prompt_manifest,
        [
            {
                "prompt_instance_id": "p1",
                "source_model_canonical": "r1-8b",
                "prompt": "Prompt p1",
                "metadata": {"source_family": "strongreject_full"},
            },
        ],
    )
    write_jsonl(
        judged,
        [
            candidate("p1", 0, "safe"),
            candidate("p1", 5, "safe", quality=False),
            candidate("p1", 6, "unsafe"),
            candidate("not-in-manifest", 5, "safe"),
        ],
    )
    args = type(
        "Args",
        (),
        {
            "model": "r1-8b",
            "judged_candidates": str(judged),
            "prompt_manifest": str(prompt_manifest),
            "sample_start": 5,
            "max_sample_idx": 7,
            "allow_quality_fail": False,
            "output_dir": str(output),
            "write_filtered_judged": False,
        },
    )()
    summary = fixed.select_fixed_budget(fixed.gen_gen.read_config(None), args)
    assert summary["n_selected_pairs"] == 0
    assert summary["n_judged_prompts_not_in_manifest"] == 1
    assert summary["n_judged_rows_not_in_manifest"] == 1
    assert summary["source_budget_table"]["strongreject_full"]["n_prompts_with_full_window_coverage"] == 1

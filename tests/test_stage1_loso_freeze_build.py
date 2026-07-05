import json
from pathlib import Path

from scripts.data import build_stage1_loso_freeze as loso


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def pair(pair_id: str, source: str):
    return {
        "pair_id": pair_id,
        "prompt_instance_id": pair_id,
        "source_family": source,
        "prompt": f"Prompt {pair_id}",
        "safe_reasoning": "safe reasoning",
        "safe_final_answer": "safe final",
        "unsafe_reasoning": "unsafe reasoning",
        "unsafe_final_answer": "unsafe final",
        "metadata": {"source_family": source},
    }


def test_loso_freeze_builds_four_holdouts_and_excludes_hb_from_trainval(tmp_path):
    path = tmp_path / "pairs.jsonl"
    output = tmp_path / "freeze"
    sources = [
        "reasoningshield",
        "strongreject_full",
        "wildjailbreak_vanilla_harmful",
        "harmbench_standard",
    ]
    write_jsonl(
        path,
        [pair(f"{source}-{idx}", source) for source in sources for idx in range(2)],
    )

    args = type(
        "Args",
        (),
        {
            "input_jsonl": [str(path)],
            "output_dir": str(output),
            "registered_sources": ",".join(sources),
            "holdout_sources": ",".join(sources),
            "seed": 123,
            "val_frac": 0.10,
            "wjb_trainval_cap": 700,
            "force": False,
        },
    )()
    summary = loso.build_freeze(args)

    assert summary["n_keep_pairs"] == 8
    assert summary["keep_pairs_by_source"]["harmbench_standard"] == 2
    assert set(summary["folds"]) == set(sources)

    hb_fold = summary["folds"]["harmbench_standard"]
    assert hb_fold["splits"]["test"]["sources"] == {"harmbench_standard": 2}
    assert "harmbench_standard" not in hb_fold["trainval_sources"]

    rs_fold = summary["folds"]["reasoningshield"]
    assert "harmbench_standard" not in rs_fold["trainval_sources"]
    train_rows = [
        json.loads(line)
        for line in (output / "folds" / "reasoningshield" / "normalized" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["source_family"] for row in train_rows} <= {"strongreject_full", "wildjailbreak_vanilla_harmful"}
    assert (output / "stage1_loso_freeze_summary.json").exists()


def test_loso_freeze_drops_incomplete_pairs(tmp_path):
    path = tmp_path / "normalized.jsonl"
    output = tmp_path / "freeze"
    write_jsonl(
        path,
        [
            {
                "id": "p1::safe",
                "pair_id": "p1",
                "prompt_instance_id": "p1",
                "source_family": "strongreject_full",
                "prompt": "Prompt",
                "trajectory_safety_label": "safe",
                "reasoning": "safe",
            }
        ],
    )
    args = type(
        "Args",
        (),
        {
            "input_jsonl": [str(path)],
            "output_dir": str(output),
            "registered_sources": "strongreject_full,harmbench_standard",
            "holdout_sources": "strongreject_full",
            "seed": 123,
            "val_frac": 0.10,
            "wjb_trainval_cap": 0,
            "force": False,
        },
    )()
    summary = loso.build_freeze(args)
    assert summary["n_keep_pairs"] == 0
    assert summary["n_dropped_pairs"] == 1


def test_loso_freeze_drops_pairs_over_word_caps(tmp_path):
    path = tmp_path / "pairs.jsonl"
    output = tmp_path / "freeze"
    row = pair("wjb-long", "wildjailbreak_vanilla_harmful")
    row["unsafe_reasoning"] = " ".join(["too_long"] * 6)
    write_jsonl(path, [row])

    args = type(
        "Args",
        (),
        {
            "input_jsonl": [str(path)],
            "output_dir": str(output),
            "registered_sources": "wildjailbreak_vanilla_harmful,harmbench_standard",
            "holdout_sources": "wildjailbreak_vanilla_harmful",
            "seed": 123,
            "val_frac": 0.10,
            "wjb_trainval_cap": 0,
            "max_prompt_words": 0,
            "max_reasoning_words": 5,
            "max_final_words": 0,
            "force": False,
        },
    )()
    summary = loso.build_freeze(args)
    assert summary["n_keep_pairs"] == 0
    assert summary["n_dropped_pairs"] == 1
    assert summary["drop_reason_counts"] == {"reasoning_words_gt_cap": 1}
    dropped = [
        json.loads(line)
        for line in (output / "dropped_pairs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert dropped[0]["drop_reasons"] == ["reasoning_words_gt_cap"]
    assert dropped[0]["violations"][0]["label"] == "unsafe"

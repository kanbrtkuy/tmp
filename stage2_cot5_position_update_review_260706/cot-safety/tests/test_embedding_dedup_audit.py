import json
from pathlib import Path

from scripts.data import audit_stage1_embedding_dedup as emb


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def pair(pair_id, source, prompt):
    return {
        "pair_id": pair_id,
        "prompt_instance_id": pair_id,
        "source_family": source,
        "prompt": prompt,
        "safe_reasoning": "safe text " * 8,
        "unsafe_reasoning": "unsafe text " * 8,
    }


def test_embedding_dedup_tfidf_outputs_histogram_and_top_neighbors(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    output = tmp_path / "audit"
    write_jsonl(
        pairs,
        [
            pair("a", "strongreject_full", "Explain how to make a harmful thing"),
            pair("b", "harmbench_standard", "Explain how to make a harmful thing safely"),
            pair("c", "wildjailbreak_vanilla_harmful", "Write a benign poem about rain"),
            pair("d", "harmthoughts+reasoningshield", "Ambiguous source should be excluded"),
        ],
    )
    args = type(
        "Args",
        (),
        {
            "input_jsonl": [str(pairs)],
            "output_dir": str(output),
            "embedding_mode": "tfidf",
            "embedding_model": "unused",
            "batch_size": 8,
            "local_files_only": True,
            "allow_tfidf_fallback": True,
            "chunk_size": 2,
            "threshold": 0.5,
            "near_band_low": 0.0,
            "near_band_high": 1.0,
            "bin_width": 0.1,
            "top_k": 2,
            "include_text": False,
            "max_text_chars": 100,
        },
    )()
    summary = emb.run_audit(args)
    assert summary["n_pairs_loaded"] == 3
    assert summary["n_rejected_during_load"] == 1
    assert summary["n_cross_source_pairwise_comparisons"] == 3
    assert summary["cross_source_near_band"]["count"] == 3
    assert summary["embedding"]["mode"] in {"tfidf-char-wb-3-5", "hashed-char-tfidf-3-5"}
    top_rows = [json.loads(line) for line in (output / "top_cross_source_neighbors.jsonl").read_text().splitlines()]
    assert 1 <= len(top_rows) <= 2
    assert "a_prompt" not in top_rows[0]
    assert (output / "cross_source_similarity_histogram.tsv").exists()


def test_embedding_dedup_fails_without_tfidf_fallback_when_sklearn_missing(monkeypatch, tmp_path):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ModuleNotFoundError("sklearn intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    try:
        emb.embed_prompts(
            ["a", "b"],
            mode="tfidf",
            model_name="unused",
            batch_size=8,
            local_files_only=True,
            allow_tfidf_fallback=False,
        )
    except ModuleNotFoundError as exc:
        assert "sklearn" in str(exc)
    else:
        raise AssertionError("expected sklearn import failure without fallback")

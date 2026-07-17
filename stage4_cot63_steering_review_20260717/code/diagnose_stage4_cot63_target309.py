#!/usr/bin/env python3
"""Diagnostic Stage-4 improvements on the cot63 target-resolved subset.

This is intentionally outside the formal Stage-4 pass/fail protocol.  It is a
small diagnostic runner for the 309 cot63 calibration rows whose A1 generation
resolved the three pause targets correctly.

Supported modes:

* score
    Replay each A1 through pause_2 and score mean(pause hidden) by projection
    onto the frozen Stage-3 unsafe-minus-safe direction.
* compose-existing
    No generation.  Use the projection score as a gate and compose existing A2
    outputs with A1 for gate-off rows.  This estimates whether collateral can
    be reduced by score gating.
* generate-shard
    Generate new diagnostic A2 outputs for either unconditional fine-alpha or
    score-gated fine-alpha variants.  Each shard writes shard JSONL files.
* merge
    Merge shard JSONLs into the generations.jsonl layout expected by
    scripts/run_stage4_judge.py.
* analyze
    Compose judged diagnostic outputs with A1 gate-off rows and summarize
    unsafe / unsafe-or-degenerate rates.

The script uses the same HF minimal-prefix counterfactual primitive as the
formal Stage-4 runner, but writes diagnostic rows with a separate schema and
does not alter formal calibration artifacts.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_safety.steering.stage4_generation import (  # noqa: E402
    CounterKey,
    SamplingSpec,
    Stage4GenerationError,
    TargetPlan,
    _continue_from_outputs,
    _position_ids,
    canonical_json,
    content_sha256,
    counterfactual_generate_batch,
    get_decoder_layers,
    left_pad_sequences,
    sha256_text,
)
from cot_safety.steering.gprs import gprs_forward_hook  # noqa: E402


SCHEMA_VERSION = "safechain.stage4.cot63_target309_diagnostic.v1"
DEFAULT_RUN_ROOT = Path("/workspace/cot-safety/runs/stage4_formal_cot63_ckpt500_4xa100")
DEFAULT_STAGE3_DIR = Path(
    "/workspace/cot-safety/runs/stage3_formal_cot63_ckpt500_4xa100/analysis_formal_offset63_v2"
)
DEFAULT_CHECKPOINT = Path(
    "/dev/shm/cot-safety-hot/outputs/"
    "deepseek_8b_intra_pause_cot63_trusted_cot_18k_full_4xa100/checkpoint-500"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/workspace/cot-safety/runs/stage4_cot63_target309_diagnostic_v1"
)
DEFAULT_SAFE_CENTROID = DEFAULT_STAGE3_DIR / "safe_centroid_replay_gprs.pt"
DEFAULT_HIDDEN_DIR = Path(
    "/workspace/cot-safety/runs/stage3_formal_cot63_ckpt500_4xa100/hidden_formal_offset63_v2"
)


def _load_stage4_hf_module() -> Any:
    path = REPO_ROOT / "scripts" / "run_stage4_formal_generation_hf.py"
    spec = importlib.util.spec_from_file_location("stage4_formal_generation_hf", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot_load_stage4_hf_module:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
    tmp.replace(path)


def alpha_label(alpha: float) -> str:
    return f"{float(alpha):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def stem_cell(cell_id: str) -> str:
    return re.sub(r"::A[0-9][A-Za-z_]*::alpha_[0-9.]+$", "", str(cell_id))


def parse_csv_floats(value: str) -> list[float]:
    return [float(piece) for piece in str(value).split(",") if piece.strip()]


def generation_files(run_root: Path) -> list[Path]:
    return sorted(run_root.glob("harmful.calibration.full_sft.shard_*.jsonl"))


def judge_files(run_root: Path) -> list[Path]:
    return sorted((run_root / "judges" / "wildguard_calibration").glob("*.jsonl"))


def load_formal_rows(run_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    generations: dict[str, dict[str, Any]] = {}
    for path in generation_files(run_root):
        for row in read_jsonl(path):
            cell_id = str(row.get("cell_id") or row.get("id") or "")
            if cell_id:
                generations[cell_id] = row
    judges: dict[str, dict[str, Any]] = {}
    for path in judge_files(run_root):
        for row in read_jsonl(path):
            cell_id = str(row.get("cell_id") or row.get("id") or "")
            if cell_id:
                judges[cell_id] = row
    return generations, judges


def merge_judge(row: Mapping[str, Any], judges: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    merged = dict(row)
    judge = judges.get(str(row.get("cell_id") or row.get("id") or "")) or {}
    for key in (
        "judge",
        "judge_label",
        "binary_safety_label",
        "judge_valid",
        "persistent_judge_failure",
        "failure_binding",
    ):
        if key in judge:
            merged[key] = judge[key]
    return merged


def load_target_resolved_a1(
    run_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    generations, judges = load_formal_rows(run_root)
    baseline: dict[str, dict[str, Any]] = {}
    for row in generations.values():
        if (
            str(row.get("arm") or "") == "A1"
            and row.get("target_resolved") is True
            and str(row.get("generation_status") or "") == "complete"
        ):
            baseline[stem_cell(str(row["cell_id"]))] = row
    return baseline, generations, judges


def label(row: Mapping[str, Any]) -> str:
    for key in ("judge_label", "binary_safety_label", "normalized_label", "safety_label"):
        if row.get(key) is not None:
            return str(row.get(key)).strip().lower()
    return "unknown"


def degenerate(row: Mapping[str, Any]) -> bool:
    return (
        bool(row.get("broken"))
        or bool(row.get("length_truncated"))
        or str(row.get("generation_status") or "complete") != "complete"
    )


def unsafe_or_deg_unknown(row: Mapping[str, Any]) -> bool:
    lab = label(row)
    return lab == "unsafe" or lab not in {"safe", "unsafe"} or degenerate(row)


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = Counter(label(row) for row in rows)
    n = len(rows)
    return {
        "n": n,
        "labels": dict(sorted(labels.items())),
        "unsafe_rate": labels.get("unsafe", 0) / n if n else None,
        "safe_rate": labels.get("safe", 0) / n if n else None,
        "unknown_rate": labels.get("unknown", 0) / n if n else None,
        "degeneration_rate": sum(degenerate(row) for row in rows) / n if n else None,
        "unsafe_or_deg_unknown_rate": sum(unsafe_or_deg_unknown(row) for row in rows) / n if n else None,
    }


def target_plan_from_payload(payload: Mapping[str, Any]) -> TargetPlan:
    return TargetPlan(
        positions={str(k): int(v) for k, v in dict(payload.get("positions") or {}).items()},
        token_ids={str(k): int(v) for k, v in dict(payload.get("token_ids") or {}).items()},
        output_offsets={str(k): int(v) for k, v in dict(payload.get("output_offsets") or {}).items()},
        structural_valid=bool(payload.get("structural_valid")),
        missing=tuple(str(item) for item in (payload.get("missing") or [])),
        info=dict(payload.get("info") or {}),
    )


def counter_key_from_a1(row: Mapping[str, Any]) -> CounterKey:
    payload = dict(row.get("counter_random_key") or {})
    return CounterKey(
        run_id=str(payload.get("run_id") or ""),
        prompt_id=str(payload.get("prompt_id") or row.get("prompt_id") or ""),
        rollout_seed=int(payload.get("rollout_seed")),
    )


def load_direction(stage3_dir: Path) -> tuple[Any, int, dict[str, Any]]:
    import torch

    path = stage3_dir / "unsafe_minus_safe_direction.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("direction") is None:
        raise Stage4GenerationError(f"missing_direction:{path}")
    metadata = dict(payload.get("metadata") or payload)
    layer = int(metadata.get("layer", payload.get("layer", -1)))
    if layer <= 0:
        raise Stage4GenerationError("direction_layer_missing")
    direction = payload["direction"].detach().float().reshape(-1)
    norm = direction.norm()
    if not bool(torch.isfinite(norm)) or float(norm.item()) <= 0:
        raise Stage4GenerationError("invalid_direction_norm")
    return direction / norm, layer, metadata


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_safe_centroid(path: Path, *, expected_layer: int) -> tuple[Any, dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("safe_centroid") is None:
        raise Stage4GenerationError(f"missing_safe_centroid:{path}")
    layer = int(payload.get("layer", -1))
    if layer != int(expected_layer):
        raise Stage4GenerationError(f"safe_centroid_layer_mismatch:{layer}!={expected_layer}")
    centroid = payload["safe_centroid"].detach().float().reshape(-1)
    if centroid.numel() <= 0:
        raise Stage4GenerationError("empty_safe_centroid")
    return centroid, dict(payload)


def build_safe_centroid(args: argparse.Namespace) -> None:
    """Build a training-only safe centroid from Stage3 pause hidden states."""

    import numpy as np
    import torch

    _direction, layer, direction_meta = load_direction(Path(args.stage3_dir))
    hidden_dir = Path(args.hidden_dir)
    files = sorted(hidden_dir.glob("stage3_train.all.shard_*_of_*.part_*.npz"))
    if not files:
        raise SystemExit(f"no_stage3_train_hidden_parts:{hidden_dir}")
    total: np.ndarray | None = None
    n_seen = 0
    n_valid = 0
    n_safe = 0
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            states = np.asarray(data["pause_states"], dtype=np.float32)
            labels = np.asarray(data["labels"], dtype=np.int64)
            valid = np.asarray(data["formal_valid_mask"], dtype=bool)
            layer_ids = [int(item) for item in data["layer_ids"].tolist()]
        if int(layer) not in layer_ids:
            raise SystemExit(f"layer_missing_in_hidden:{layer}:{path}")
        layer_idx = layer_ids.index(int(layer))
        keep = (labels == 0) & valid
        n_seen += int(labels.shape[0])
        n_valid += int(valid.sum())
        if not bool(keep.any()):
            continue
        block = states[keep, layer_idx, :].astype(np.float64)
        subtotal = block.sum(axis=0)
        total = subtotal if total is None else total + subtotal
        n_safe += int(block.shape[0])
    if total is None or n_safe <= 0:
        raise SystemExit("no_valid_safe_rows_for_centroid")
    centroid = (total / float(n_safe)).astype(np.float32)
    output = Path(args.safe_centroid)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "safe_centroid": torch.as_tensor(centroid),
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "safe_centroid_for_gprs_replay_diagnostic",
        "training_only": True,
        "source_hidden_dir": str(hidden_dir),
        "layer": int(layer),
        "positions": ["pause_0", "pause_1", "pause_2"],
        "pooling": "raw_mean",
        "n_seen": int(n_seen),
        "n_formal_valid": int(n_valid),
        "n_safe_used": int(n_safe),
        "direction_metadata": direction_meta,
    }
    torch.save(payload, output)
    write_json(
        output.with_suffix(".manifest.json"),
        {
            **{key: value for key, value in payload.items() if key != "safe_centroid"},
            "path": str(output),
            "sha256": sha256_file(output),
        },
    )
    print(
        json.dumps(
            {"status": "done", "safe_centroid": str(output), "n_safe_used": n_safe},
            sort_keys=True,
        )
    )


def load_model(checkpoint: Path, tokenizer_path: Path, *, device: str) -> tuple[Any, Any, Any]:
    stage4_hf = _load_stage4_hf_module()
    model, tokenizer, torch_device, _fingerprint = stage4_hf._model_load(
        str(checkpoint), str(tokenizer_path), device=device, dtype_name="bfloat16"
    )
    return model, tokenizer, torch_device


def score_rows(args: argparse.Namespace) -> None:
    import torch

    baseline, _generations, judges = load_target_resolved_a1(Path(args.run_root))
    direction, layer, direction_meta = load_direction(Path(args.stage3_dir))
    model, tokenizer, device = load_model(Path(args.checkpoint), Path(args.tokenizer), device=args.device)
    rows = sorted(baseline.values(), key=lambda row: str(row["cell_id"]))
    scores: list[dict[str, Any]] = []
    unit = direction.to(device=device, dtype=torch.float32)

    for start in range(0, len(rows), int(args.batch_size)):
        batch = rows[start : start + int(args.batch_size)]
        prefixes: list[list[int]] = []
        absolute_positions: list[list[int]] = []
        for row in batch:
            plan = target_plan_from_payload(row["a1_target_plan"])
            pause_positions = [int(plan.positions[name]) for name in ("pause_0", "pause_1", "pause_2")]
            last = max(pause_positions)
            output_stop = last - len(row["prompt_token_ids"]) + 1
            prefixes.append(
                [int(item) for item in row["prompt_token_ids"]]
                + [int(item) for item in row["output_token_ids"][:output_stop]]
            )
            absolute_positions.append(pause_positions)
        input_ids, attention_mask, lengths = left_pad_sequences(
            prefixes, pad_token_id=int(tokenizer.pad_token_id), device=device
        )
        width = int(input_ids.shape[1])
        padded_positions: list[list[int]] = []
        for positions, length in zip(absolute_positions, lengths):
            left_pad = width - int(length)
            padded_positions.append([left_pad + int(pos) for pos in positions])
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=_position_ids(attention_mask),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden = outputs.hidden_states[int(layer)].detach().float()
        for local, row in enumerate(batch):
            selected = hidden[local, torch.tensor(padded_positions[local], device=hidden.device)]
            pooled = selected.mean(dim=0)
            score = float(torch.dot(pooled, unit).detach().cpu().item())
            merged = merge_judge(row, judges)
            scores.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "cell_id": str(row["cell_id"]),
                    "stem": stem_cell(str(row["cell_id"])),
                    "source": str(row.get("source") or ""),
                    "prompt_id": str(row.get("prompt_id") or ""),
                    "draw_index": int(row.get("draw_index", -1)),
                    "score": score,
                    "label": label(merged),
                    "degenerate": degenerate(merged),
                    "unsafe_or_deg_unknown": unsafe_or_deg_unknown(merged),
                    "layer": int(layer),
                    "positions": ["pause_0", "pause_1", "pause_2"],
                }
            )
        print(json.dumps({"mode": "score", "processed": min(start + len(batch), len(rows)), "total": len(rows)}))

    ordered = sorted(scores, key=lambda item: float(item["score"]), reverse=True)
    total = len(ordered)
    for rank, item in enumerate(ordered, start=1):
        item["rank_desc"] = rank
        item["top_fraction"] = rank / total if total else None
    output = Path(args.scores_jsonl)
    write_jsonl(output, ordered)
    labels = [1 if item["label"] == "unsafe" else 0 for item in ordered if item["label"] in {"safe", "unsafe"}]
    values = [float(item["score"]) for item in ordered if item["label"] in {"safe", "unsafe"}]
    auc = binary_auc(values, labels) if len(set(labels)) == 2 else None
    write_json(
        output.with_suffix(".manifest.json"),
        {
            "schema_version": SCHEMA_VERSION,
            "mode": "score",
            "n": len(scores),
            "direction_layer": int(layer),
            "direction_metadata": direction_meta,
            "unsafe_score_auc": auc,
            "score_mean_by_label": {
                lab: sum(float(item["score"]) for item in scores if item["label"] == lab)
                / max(1, sum(1 for item in scores if item["label"] == lab))
                for lab in sorted({str(item["label"]) for item in scores})
            },
        },
    )
    print(json.dumps({"status": "done", "scores_jsonl": str(output), "unsafe_score_auc": auc}, sort_keys=True))


def binary_auc(values: Sequence[float], labels: Sequence[int]) -> float:
    pairs = sorted(zip(values, labels), key=lambda item: item[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = 0.0
    i = 0
    rank = 1
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        rank_sum += avg_rank * sum(label for _value, label in pairs[i:j])
        rank += j - i
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load_scores(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["stem"]): row for row in read_jsonl(path)}


def selected_stems(scores: Mapping[str, Mapping[str, Any]], gate_fraction: float) -> set[str]:
    total = len(scores)
    keep = max(1, int(math.ceil(float(gate_fraction) * total)))
    ranked = sorted(scores.items(), key=lambda item: float(item[1]["score"]), reverse=True)
    return {stem for stem, _row in ranked[:keep]}


def compose_existing(args: argparse.Namespace) -> None:
    baseline, generations, judges = load_target_resolved_a1(Path(args.run_root))
    scores = load_scores(Path(args.scores_jsonl))
    base_merged = {stem: merge_judge(row, judges) for stem, row in baseline.items()}
    existing_by: dict[tuple[float, str], dict[str, Any]] = {}
    for row in generations.values():
        if str(row.get("arm") or "") == "A2":
            existing_by[(float(row.get("alpha")), stem_cell(str(row["cell_id"])))] = row
    base_summary = summarize_rows(list(base_merged.values()))
    summaries = []
    for alpha in parse_csv_floats(args.alphas):
        for gate_fraction in parse_csv_floats(args.gate_fracs):
            gated = selected_stems(scores, gate_fraction)
            composed = []
            for stem, base_row in baseline.items():
                if stem in gated and (alpha, stem) in existing_by:
                    composed.append(merge_judge(existing_by[(alpha, stem)], judges))
                else:
                    composed.append(base_merged[stem])
            summary = summarize_rows(composed)
            summary.update(
                {
                    "method": "score_gate_existing_a2",
                    "alpha": alpha,
                    "gate_fraction": gate_fraction,
                    "n_gated": len(gated),
                    "delta_unsafe_pp": 100.0 * (summary["unsafe_rate"] - base_summary["unsafe_rate"]),
                    "delta_unsafe_or_deg_unknown_pp": 100.0
                    * (summary["unsafe_or_deg_unknown_rate"] - base_summary["unsafe_or_deg_unknown_rate"]),
                }
            )
            summaries.append(summary)
    out = Path(args.output_root) / "score_gate_existing_a2_summary.json"
    write_json(out, {"baseline": base_summary, "summaries": summaries})
    print(json.dumps({"status": "done", "output": str(out), "baseline": base_summary, "summaries": summaries}, ensure_ascii=False, indent=2))


def diagnostic_path(output_root: Path, *, method: str, gate_fraction: float | None, alpha: float) -> Path:
    if method == "score_gated":
        mode = f"mode_projection_top_{int(round(float(gate_fraction or 0.0) * 100)):02d}pct"
        target = "pause_all3_score_gated"
    elif method == "unconditional":
        mode = "mode_unconditional_fine"
        target = "pause_all3_unconditional"
    elif method == "gprs_replay":
        mode = "mode_gprs_projection_replay"
        target = "pause_all3_gprs_replay"
    else:
        raise ValueError(f"unknown_method:{method}")
    return (
        output_root
        / "condition_full_sft"
        / "direction_unsafe_minus_safe"
        / "cot63_target_resolved309"
        / target
        / mode
        / "seed_260713"
        / f"alpha_{alpha_label(alpha)}"
    )


def make_diag_row(
    *,
    base_row: Mapping[str, Any],
    generated_ids: Sequence[int],
    generated_text: str,
    finish_reason: str,
    audit: Mapping[str, Any],
    alpha: float,
    rho: float,
    method: str,
    gate_fraction: float | None,
    score_payload: Mapping[str, Any],
) -> dict[str, Any]:
    stem = stem_cell(str(base_row["cell_id"]))
    if method == "unconditional":
        method_tag = method
    elif method == "gprs_replay":
        method_tag = "gprs_replay"
    else:
        method_tag = f"score_gated_top_{int(round(float(gate_fraction or 0.0) * 100)):02d}pct"
    cell_id = f"{stem}::A2diag_{method_tag}::alpha_{float(alpha):.5f}"
    gate_score = score_payload.get("score")
    gate_rank = score_payload.get("rank_desc")
    row = {
        "schema_version": SCHEMA_VERSION,
        "id": cell_id,
        "cell_id": cell_id,
        "base_a1_cell_id": str(base_row["cell_id"]),
        "stem": stem,
        "source": str(base_row.get("source") or ""),
        "prompt_id": str(base_row.get("prompt_id") or ""),
        "family_id": str(base_row.get("family_id") or ""),
        "draw_index": int(base_row.get("draw_index", -1)),
        "prompt": str(base_row.get("prompt") or ""),
        "method": method,
        "gate_fraction": None if gate_fraction is None else float(gate_fraction),
        "gate_score": None if gate_score is None else float(gate_score),
        "gate_rank_desc": None if gate_rank is None else int(gate_rank),
        "alpha": float(alpha),
        "rho": float(rho),
        "arm": "A2diag",
        "generated": True,
        "generation_status": "complete",
        "finish_reason": str(finish_reason),
        "output_token_ids": [int(item) for item in generated_ids],
        "generated_text": str(generated_text),
        "generated": str(generated_text),
        "generated_for_judge": str(generated_text),
        "generated_content_sha256": content_sha256(
            base_row["prompt_token_ids"], generated_ids
        ),
        "generated_text_sha256": sha256_text(str(generated_text)),
        "generated_for_judge_sha256": sha256_text(str(generated_text)),
        "target_resolved": True,
        "intervention_audit": dict(audit),
    }
    row["row_integrity_sha256"] = sha256_text(canonical_json({k: v for k, v in row.items() if k != "row_integrity_sha256"}))
    return row


def counterfactual_gprs_generate_batch(
    model: Any,
    *,
    prompt_token_ids: Sequence[Sequence[int]],
    a1_output_token_ids: Sequence[Sequence[int]],
    target_plans: Sequence[TargetPlan],
    target_names: Sequence[str],
    direction: Any,
    safe_centroid: Any,
    hidden_state_index: int,
    alpha: float,
    norm_cap: float,
    strength_mode: str,
    counter_keys: Sequence[CounterKey],
    sampling: SamplingSpec,
    pad_token_id: int,
    eos_token_ids: int | Sequence[int] | None,
    device: Any,
) -> tuple[list[list[int]], list[str], list[dict[str, Any]]]:
    """A1-replay counterfactual with the legacy GPRS/projection update rule."""

    import torch

    batch_size = len(prompt_token_ids)
    if not (
        batch_size
        == len(a1_output_token_ids)
        == len(target_plans)
        == len(counter_keys)
    ):
        raise Stage4GenerationError("gprs_replay_batch_length_mismatch")
    names = tuple(str(item) for item in target_names)
    if len(names) != 3:
        raise Stage4GenerationError(f"exactly_three_target_names_required:{names}")
    prefixes: list[list[int]] = []
    initial_outputs: list[list[int]] = []
    absolute_positions_by_row: list[list[int]] = []
    for prompt_ids, output_ids, plan in zip(prompt_token_ids, a1_output_token_ids, target_plans):
        if not plan.structural_valid:
            raise Stage4GenerationError(f"a1_target_plan_not_structurally_valid:{plan.missing}")
        positions, _target_ids = plan.for_names(names)
        last_position = max(positions)
        output_stop = int(last_position) - len(prompt_ids) + 1
        if output_stop <= 0 or output_stop > len(output_ids):
            raise Stage4GenerationError(
                f"invalid_teacher_replay_boundary:{output_stop}:output_len={len(output_ids)}"
            )
        prefixes.append([int(item) for item in prompt_ids] + [int(item) for item in output_ids[:output_stop]])
        initial_outputs.append([int(item) for item in output_ids[:output_stop]])
        absolute_positions_by_row.append([int(item) for item in positions])

    input_ids, attention_mask, prefix_lengths = left_pad_sequences(
        prefixes, pad_token_id=int(pad_token_id), device=device
    )
    width = int(input_ids.shape[1])
    target_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    padded_positions_by_row: list[list[int]] = []
    for row_index, (absolute_positions, prefix_len) in enumerate(zip(absolute_positions_by_row, prefix_lengths)):
        left_pad = width - int(prefix_len)
        padded_positions = [left_pad + int(position) for position in absolute_positions]
        for position in padded_positions:
            target_mask[row_index, position] = True
        padded_positions_by_row.append(padded_positions)

    with gprs_forward_hook(
        get_decoder_layers(model),
        layer=int(hidden_state_index),
        target_mask=target_mask,
        direction=direction,
        safe_centroid=safe_centroid,
        strength=float(alpha),
        norm_cap=float(norm_cap),
        strength_mode=str(strength_mode),
    ) as hook_stats:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=_position_ids(attention_mask),
                use_cache=True,
                return_dict=True,
            )
        if int(hook_stats.get("num_applied_calls", 0)) != 1:
            raise Stage4GenerationError(
                f"gprs_replay_hook_application_count:{hook_stats.get('num_applied_calls')}"
            )
        generated, finish = _continue_from_outputs(
            model,
            initial_outputs=outputs,
            attention_mask=attention_mask,
            initial_output_ids=initial_outputs,
            counter_keys=counter_keys,
            sampling=sampling,
            eos_token_ids=eos_token_ids,
        )
    audits: list[dict[str, Any]] = []
    for row_index in range(batch_size):
        audits.append(
            {
                "algorithm": "gprs_projection_replay",
                "hidden_state_index": int(hidden_state_index),
                "target_names": list(names),
                "target_positions_absolute": absolute_positions_by_row[row_index],
                "target_positions_padded": padded_positions_by_row[row_index],
                "teacher_replay_output_tokens": len(initial_outputs[row_index]),
                "strength": float(alpha),
                "norm_cap": float(norm_cap),
                "strength_mode": str(strength_mode),
                "num_target_tokens": int(hook_stats["per_row_target_tokens"][row_index]),
                "actual_relative_norms": [
                    float(item)
                    for item in hook_stats["per_row_applied_relative_norms"][row_index]
                ],
                "actual_delta_norms": [
                    float(item)
                    for item in hook_stats["per_row_applied_delta_norms"][row_index]
                ],
                "pre_update_hidden_norms": [
                    float(item)
                    for item in hook_stats["per_row_applied_hidden_norms"][row_index]
                ],
                "hook_timing": {
                    "registered_before_prefix_forward": True,
                    "applied_on_full_prefix": True,
                    "cache_returned_after_application": True,
                    "hidden_state_index": int(hidden_state_index),
                    "num_hook_calls": int(hook_stats.get("num_hook_calls", 0)),
                    "num_applied_calls": int(hook_stats.get("num_applied_calls", 0)),
                    "shape_mismatches": list(hook_stats.get("shape_mismatches") or []),
                },
            }
        )
    return generated, finish, audits


def existing_ids(path: Path) -> set[str]:
    return {str(row.get("cell_id") or row.get("id") or "") for row in read_jsonl(path)}


def generate_shard(args: argparse.Namespace) -> None:
    import torch

    if not 0 <= int(args.shard_index) < int(args.num_shards):
        raise SystemExit("shard_index must be in [0, num_shards)")
    baseline, _generations, _judges = load_target_resolved_a1(Path(args.run_root))
    scores = load_scores(Path(args.scores_jsonl))
    direction, layer, _meta = load_direction(Path(args.stage3_dir))
    model, tokenizer, device = load_model(Path(args.checkpoint), Path(args.tokenizer), device=args.device)
    if tokenizer.eos_token_id is None:
        raise Stage4GenerationError("tokenizer_missing_eos")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = int(tokenizer.eos_token_id)
    unit = direction.to(device=device, dtype=torch.float32)
    methods = [piece.strip() for piece in str(args.methods).split(",") if piece.strip()]
    safe_centroid = None
    if "gprs_replay" in methods:
        safe_centroid, _centroid_meta = load_safe_centroid(
            Path(args.safe_centroid), expected_layer=int(layer)
        )
        safe_centroid = safe_centroid.to(device=device, dtype=torch.float32)
    sampling = SamplingSpec(temperature=0.6, top_p=0.95, max_new_tokens=2048)
    norm_cap = 0.10
    rows_sorted = sorted(baseline.items(), key=lambda item: item[0])
    rows_sorted = [item for idx, item in enumerate(rows_sorted) if idx % int(args.num_shards) == int(args.shard_index)]

    tasks: list[tuple[str, float | None, float, list[tuple[str, dict[str, Any]]]]] = []
    alphas = parse_csv_floats(args.alphas)
    if "unconditional" in methods:
        for alpha in alphas:
            tasks.append(("unconditional", None, alpha, rows_sorted))
    if "score_gated" in methods:
        for gate_fraction in parse_csv_floats(args.gate_fracs):
            gated = selected_stems(scores, gate_fraction)
            selected = [(stem, row) for stem, row in rows_sorted if stem in gated]
            for alpha in alphas:
                tasks.append(("score_gated", gate_fraction, alpha, selected))
    if "gprs_replay" in methods:
        for alpha in alphas:
            tasks.append(("gprs_replay", None, alpha, rows_sorted))

    for method, gate_fraction, alpha, task_rows in tasks:
        out_dir = diagnostic_path(Path(args.output_root), method=method, gate_fraction=gate_fraction, alpha=alpha)
        shard_path = out_dir / f"generations.shard_{int(args.shard_index):02d}_of_{int(args.num_shards):02d}.jsonl"
        done = existing_ids(shard_path)
        pending = [(stem, row) for stem, row in task_rows if f"{stem}::A2diag_" not in "".join(done)]
        generated_count = 0
        for start in range(0, len(pending), int(args.batch_size)):
            batch = pending[start : start + int(args.batch_size)]
            if not batch:
                continue
            prompt_ids = [[int(item) for item in row["prompt_token_ids"]] for _stem, row in batch]
            output_ids = [[int(item) for item in row["output_token_ids"]] for _stem, row in batch]
            plans = [target_plan_from_payload(row["a1_target_plan"]) for _stem, row in batch]
            keys = [counter_key_from_a1(row) for _stem, row in batch]
            rho = float(alpha) * norm_cap
            if method == "gprs_replay":
                if safe_centroid is None:
                    raise Stage4GenerationError("missing_safe_centroid_for_gprs_replay")
                gen_ids, finishes, audits = counterfactual_gprs_generate_batch(
                    model,
                    prompt_token_ids=prompt_ids,
                    a1_output_token_ids=output_ids,
                    target_plans=plans,
                    target_names=("pause_0", "pause_1", "pause_2"),
                    direction=unit,
                    safe_centroid=safe_centroid,
                    hidden_state_index=int(layer),
                    alpha=float(alpha),
                    norm_cap=norm_cap,
                    strength_mode=str(args.strength_mode),
                    counter_keys=keys,
                    sampling=sampling,
                    pad_token_id=int(tokenizer.pad_token_id),
                    eos_token_ids=tokenizer.eos_token_id,
                    device=device,
                )
            else:
                gen_ids, finishes, audits = counterfactual_generate_batch(
                    model,
                    prompt_token_ids=prompt_ids,
                    a1_output_token_ids=output_ids,
                    target_plans=plans,
                    target_names=("pause_0", "pause_1", "pause_2"),
                    unit_direction=unit,
                    hidden_state_index=int(layer),
                    rho=rho,
                    counter_keys=keys,
                    sampling=sampling,
                    pad_token_id=int(tokenizer.pad_token_id),
                    eos_token_ids=tokenizer.eos_token_id,
                    device=device,
                )
            out_rows = []
            for local, (stem, row) in enumerate(batch):
                text = tokenizer.decode(gen_ids[local], skip_special_tokens=False)
                out_rows.append(
                    make_diag_row(
                        base_row=row,
                        generated_ids=gen_ids[local],
                        generated_text=text,
                        finish_reason=finishes[local],
                        audit=audits[local],
                        alpha=alpha,
                        rho=rho,
                        method=method,
                        gate_fraction=gate_fraction,
                        score_payload=scores.get(stem, {}),
                    )
                )
            append_jsonl(shard_path, out_rows)
            generated_count += len(out_rows)
            print(
                json.dumps(
                    {
                        "mode": "generate-shard",
                        "method": method,
                        "gate_fraction": gate_fraction,
                        "alpha": alpha,
                        "shard": args.shard_index,
                        "written_this_task": generated_count,
                        "task_total": len(task_rows),
                    },
                    sort_keys=True,
                )
            )
        write_json(
            shard_path.with_suffix(".done.json"),
            {
                "schema_version": SCHEMA_VERSION,
                "method": method,
                "gate_fraction": gate_fraction,
                "alpha": alpha,
                "shard_index": int(args.shard_index),
                "num_shards": int(args.num_shards),
                "rows": len(read_jsonl(shard_path)),
            },
        )


def merge_shards(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    merged = 0
    for alpha_dir in sorted(root.glob("condition_*/direction_*/*/*/mode_*/*/alpha_*")):
        shards = sorted(alpha_dir.glob("generations.shard_*_of_*.jsonl"))
        if not shards:
            continue
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for shard in shards:
            for row in read_jsonl(shard):
                cell_id = str(row.get("cell_id") or row.get("id") or "")
                if cell_id in seen:
                    raise SystemExit(f"duplicate_diagnostic_cell:{cell_id}")
                seen.add(cell_id)
                rows.append(row)
        rows = sorted(rows, key=lambda row: str(row.get("cell_id") or row.get("id") or ""))
        write_jsonl(alpha_dir / "generations.jsonl", rows)
        write_json(
            alpha_dir / "generations.manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "rows": len(rows),
                "shards": [str(path) for path in shards],
            },
        )
        merged += 1
    print(json.dumps({"status": "done", "merged_generation_files": merged}, sort_keys=True))


def parse_diag_path(path: Path, root: Path) -> dict[str, Any]:
    rel = path.relative_to(root)
    parts = rel.parts
    # condition/direction/dataset/target/mode/seed/alpha/generations.jsonl
    if len(parts) < 8:
        raise ValueError(f"bad_diag_path:{path}")
    target = parts[3]
    mode = parts[4]
    alpha = float(parts[6].replace("alpha_", "").replace("p", "."))
    if "unconditional" in target:
        method = "unconditional"
    elif "gprs_replay" in target:
        method = "gprs_replay"
    else:
        method = "score_gated"
    gate_fraction = None
    if method == "score_gated":
        match = re.search(r"top_(\d+)pct", mode)
        if match:
            gate_fraction = int(match.group(1)) / 100.0
    return {"method": method, "gate_fraction": gate_fraction, "alpha": alpha}


def load_diag_judges(gen_path: Path) -> dict[str, dict[str, Any]]:
    norm = gen_path.parent / "open_judges_normalized.jsonl"
    if not norm.exists():
        return {}
    return {str(row.get("cell_id") or row.get("id") or ""): row for row in read_jsonl(norm)}


def analyze(args: argparse.Namespace) -> None:
    baseline, _generations, judges = load_target_resolved_a1(Path(args.run_root))
    base_merged = {stem: merge_judge(row, judges) for stem, row in baseline.items()}
    base_summary = summarize_rows(list(base_merged.values()))
    summaries = []
    root = Path(args.output_root)
    for gen_path in sorted(root.glob("condition_*/direction_*/*/*/mode_*/*/alpha_*/generations.jsonl")):
        meta = parse_diag_path(gen_path, root)
        rows = {str(row["stem"]): row for row in read_jsonl(gen_path)}
        diag_judges = load_diag_judges(gen_path)
        composed = []
        generated_merged = []
        for stem, base_row in base_merged.items():
            if stem in rows:
                row = merge_judge(rows[stem], diag_judges)
                generated_merged.append(row)
                composed.append(row)
            else:
                composed.append(base_row)
        summary = summarize_rows(composed)
        generated_summary = summarize_rows(generated_merged) if generated_merged else {}
        summary.update(
            {
                **meta,
                "n_generated": len(generated_merged),
                "generated_only": generated_summary,
                "delta_unsafe_pp": 100.0 * (summary["unsafe_rate"] - base_summary["unsafe_rate"]),
                "delta_unsafe_or_deg_unknown_pp": 100.0
                * (summary["unsafe_or_deg_unknown_rate"] - base_summary["unsafe_or_deg_unknown_rate"]),
            }
        )
        summaries.append(summary)
    out = root / "diagnostic_analysis_summary.json"
    write_json(out, {"baseline": base_summary, "summaries": summaries})
    print(json.dumps({"status": "done", "output": str(out), "baseline": base_summary, "summaries": summaries}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("score", "build-centroid", "compose-existing", "generate-shard", "merge", "analyze"))
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--stage3-dir", default=str(DEFAULT_STAGE3_DIR))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--scores-jsonl", default=None)
    parser.add_argument("--safe-centroid", default=str(DEFAULT_SAFE_CENTROID))
    parser.add_argument("--hidden-dir", default=str(DEFAULT_HIDDEN_DIR))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--alphas", default="0.03,0.05,0.07,0.10,0.13")
    parser.add_argument("--gate-fracs", default="0.30,0.40,0.50,0.60")
    parser.add_argument("--methods", default="score_gated")
    parser.add_argument("--strength-mode", choices=("projection", "matched_relative"), default="projection")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.checkpoint
    if args.scores_jsonl is None:
        args.scores_jsonl = str(Path(args.output_root) / "projection_scores.jsonl")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "score":
        score_rows(args)
    elif args.mode == "build-centroid":
        build_safe_centroid(args)
    elif args.mode == "compose-existing":
        compose_existing(args)
    elif args.mode == "generate-shard":
        generate_shard(args)
    elif args.mode == "merge":
        merge_shards(args)
    elif args.mode == "analyze":
        analyze(args)
    else:
        raise SystemExit(f"unknown_mode:{args.mode}")


if __name__ == "__main__":
    main()

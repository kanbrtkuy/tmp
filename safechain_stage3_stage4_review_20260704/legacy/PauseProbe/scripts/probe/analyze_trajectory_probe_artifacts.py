#!/usr/bin/env python3
"""Analyze source, format, and length artifacts in trajectory-probe outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TEXT_KEYS = {
    "prompt": ("prompt", "input", "instruction", "question", "prompt_key"),
    "reasoning": ("reasoning", "model_thinking", "thoughts", "rationale", "analysis"),
    "final": ("final_answer", "model_response", "response", "answer"),
    "output": ("output", "generated", "generated_for_judge"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for row in rows:
            f.write("\t".join(fmt(row.get(key)) for key in keys) + "\n")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def word_len(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    return float(values[lo] * (hi - pos) + values[hi] * (pos - lo))


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return math.nan
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return math.nan
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return float(cov / math.sqrt(vx * vy))


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(order):
        end = idx + 1
        while end < len(order) and values[order[end]] == values[order[idx]]:
            end += 1
        avg = (idx + end - 1) / 2 + 1
        for pos in range(idx, end):
            ranks[order[pos]] = avg
        idx = end
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return math.nan
    return pearson(rank(xs), rank(ys))


def binary_curve_metrics(labels: list[int], scores: list[float]) -> dict[str, float]:
    if len(set(labels)) < 2:
        return {"auroc": math.nan, "auprc": math.nan}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError:
        return {"auroc": math.nan, "auprc": math.nan}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def first_present(*rows: dict[str, Any], keys: tuple[str, ...]) -> str:
    for row in rows:
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                return clean_text(value)
    return ""


def nested_value(row: dict[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def parse_output_text(output: str) -> tuple[str, str]:
    if not output:
        return "", ""
    lower = output.lower()
    start = lower.find("<think>")
    end = lower.find("</think>")
    if start >= 0 and end > start:
        reasoning = output[start + len("<think>") : end].strip()
        final = output[end + len("</think>") :].strip()
        return reasoning, final
    return "", output.strip()


def read_optional_jsonl(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = read_jsonl(path)
    return {clean_text(row.get("id")): row for row in rows if clean_text(row.get("id"))}


def merge_text_fields(meta: dict[str, Any], traj: dict[str, Any], pred: dict[str, Any]) -> dict[str, str]:
    meta_extra = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
    prompt = first_present(traj, meta, pred, meta_extra, keys=TEXT_KEYS["prompt"])
    reasoning = first_present(traj, meta, meta_extra, keys=TEXT_KEYS["reasoning"])
    final = first_present(traj, meta, meta_extra, keys=TEXT_KEYS["final"])
    output = first_present(traj, meta, pred, meta_extra, keys=TEXT_KEYS["output"])
    if output and (not reasoning or not final):
        parsed_reasoning, parsed_final = parse_output_text(output)
        reasoning = reasoning or parsed_reasoning
        final = final or parsed_final
    return {"prompt": prompt, "reasoning": reasoning, "final": final, "output": output}


def length_bucket(value: int) -> str:
    if value <= 0:
        return "000_empty"
    if value <= 32:
        return "032_short"
    if value <= 128:
        return "128_medium"
    if value <= 512:
        return "512_long"
    if value <= 1024:
        return "1024_very_long"
    return "gt1024_extreme"


def summarize_group(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["unsafe_score"]) for row in rows]
    preds = [int(score >= threshold) for score in scores]
    tp = sum(1 for label, pred in zip(labels, preds) if label == 1 and pred == 1)
    tn = sum(1 for label, pred in zip(labels, preds) if label == 0 and pred == 0)
    fp = sum(1 for label, pred in zip(labels, preds) if label == 0 and pred == 1)
    fn = sum(1 for label, pred in zip(labels, preds) if label == 1 and pred == 0)
    safe = max(1, sum(1 for label in labels if label == 0))
    unsafe = max(1, sum(1 for label in labels if label == 1))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    total_words = [float(row["total_word_len"]) for row in rows]
    reasoning_words = [float(row["reasoning_word_len"]) for row in rows]
    prompt_words = [float(row["prompt_word_len"]) for row in rows]
    curve = binary_curve_metrics(labels, scores)
    return {
        "n": len(rows),
        "safe_count": sum(1 for label in labels if label == 0),
        "unsafe_count": sum(1 for label in labels if label == 1),
        "positive_rate": sum(labels) / len(labels) if labels else math.nan,
        "mean_unsafe_score": sum(scores) / len(scores) if scores else math.nan,
        "score_p50": percentile(scores, 0.5),
        "score_p90": percentile(scores, 0.9),
        "predicted_positive_rate": sum(preds) / len(preds) if preds else math.nan,
        "accuracy": (tp + tn) / max(1, len(rows)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / safe,
        "fnr": fn / unsafe,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "auroc": curve["auroc"],
        "auprc": curve["auprc"],
        "prompt_word_p50": percentile(prompt_words, 0.5),
        "prompt_word_p90": percentile(prompt_words, 0.9),
        "reasoning_word_p50": percentile(reasoning_words, 0.5),
        "reasoning_word_p90": percentile(reasoning_words, 0.9),
        "total_word_p50": percentile(total_words, 0.5),
        "total_word_p90": percentile(total_words, 0.9),
        "pearson_score_prompt_words": pearson(scores, prompt_words),
        "spearman_score_prompt_words": spearman(scores, prompt_words),
        "pearson_score_reasoning_words": pearson(scores, reasoning_words),
        "spearman_score_reasoning_words": spearman(scores, reasoning_words),
        "pearson_score_total_words": pearson(scores, total_words),
        "spearman_score_total_words": spearman(scores, total_words),
    }


def redact(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def infer_threshold(preds: list[dict[str, Any]], explicit: float | None) -> float:
    if explicit is not None:
        return explicit
    thresholds = sorted({float(row.get("threshold", 0.5)) for row in preds if "threshold" in row})
    return thresholds[0] if len(thresholds) == 1 else 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_jsonl", required=True)
    parser.add_argument("--metadata_jsonl", required=True)
    parser.add_argument(
        "--trajectory_jsonl",
        default=None,
        help="Optional normalized trajectory JSONL with prompt/reasoning/final_answer fields, keyed by id.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--prompt_excerpt_chars", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds = read_jsonl(Path(args.predictions_jsonl))
    metadata = read_optional_jsonl(Path(args.metadata_jsonl))
    trajectories = read_optional_jsonl(Path(args.trajectory_jsonl)) if args.trajectory_jsonl else {}
    threshold = infer_threshold(preds, args.threshold)

    enriched = []
    missing_metadata = 0
    missing_trajectory = 0
    for pred in preds:
        example_id = clean_text(pred.get("example_id"))
        meta = metadata.get(example_id, {})
        traj = trajectories.get(example_id, {})
        if not meta:
            missing_metadata += 1
        if args.trajectory_jsonl and not traj:
            missing_trajectory += 1
        text = merge_text_fields(meta, traj, pred)
        parse_info = meta.get("parse_info") if isinstance(meta.get("parse_info"), dict) else {}
        row = dict(pred)
        row.update(
            {
                "example_id": example_id,
                "source": first_present(traj, meta, pred, keys=("source",)),
                "source_family": first_present(traj, meta, pred, keys=("source_family",)),
                "risk_type": first_present(traj, meta, pred, keys=("risk_type",)),
                "policy_type": first_present(traj, meta, pred, keys=("policy_type",)),
                "label_name": first_present(meta, traj, keys=("label_name", "trajectory_safety_label", "safety_label")),
                "label_task": first_present(traj, meta, keys=("label_task",)),
                "prompt_source": first_present(traj, keys=("prompt_source",)),
                "trajectory_source": first_present(traj, keys=("trajectory_source",)),
                "label_source": first_present(traj, keys=("label_source",)),
                "parse_status": clean_text(parse_info.get("parse_status") or nested_value(meta, "metadata", "parse_status")),
                "reasoning_token_len": int(parse_info.get("reasoning_token_len") or 0),
                "prompt_word_len": word_len(text["prompt"]),
                "reasoning_word_len": word_len(text["reasoning"]),
                "final_word_len": word_len(text["final"]),
                "output_word_len": word_len(text["output"]),
                "prompt_char_len": len(text["prompt"]),
                "reasoning_char_len": len(text["reasoning"]),
                "final_char_len": len(text["final"]),
                "output_char_len": len(text["output"]),
            }
        )
        row["total_word_len"] = row["prompt_word_len"] + row["reasoning_word_len"] + row["final_word_len"]
        row["total_char_len"] = row["prompt_char_len"] + row["reasoning_char_len"] + row["final_char_len"]
        row["predicted_label"] = int(float(row["unsafe_score"]) >= threshold)
        row["prompt_excerpt"] = redact(text["prompt"], args.prompt_excerpt_chars)
        row["reasoning_excerpt"] = redact(text["reasoning"], args.prompt_excerpt_chars)
        enriched.append(row)

    group_specs = {
        "by_source": "source",
        "by_source_family": "source_family",
        "by_risk_type": "risk_type",
        "by_policy_type": "policy_type",
        "by_label": "label",
        "by_label_name": "label_name",
        "by_label_task": "label_task",
        "by_parse_status": "parse_status",
        "by_prompt_source": "prompt_source",
        "by_trajectory_source": "trajectory_source",
        "by_label_source": "label_source",
        "by_prompt_length_bucket": "prompt_length_bucket",
        "by_reasoning_length_bucket": "reasoning_length_bucket",
        "by_total_length_bucket": "total_length_bucket",
    }
    for row in enriched:
        row["prompt_length_bucket"] = length_bucket(int(row["prompt_word_len"]))
        row["reasoning_length_bucket"] = length_bucket(int(row["reasoning_word_len"]))
        row["total_length_bucket"] = length_bucket(int(row["total_word_len"]))

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for group_name, field in group_specs.items():
        bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in enriched:
            bucket[clean_text(row.get(field)) or "<missing>"].append(row)
        grouped[group_name] = bucket

    summary: dict[str, Any] = {
        "predictions_jsonl": args.predictions_jsonl,
        "metadata_jsonl": args.metadata_jsonl,
        "trajectory_jsonl": args.trajectory_jsonl,
        "threshold": threshold,
        "rows": len(enriched),
        "missing_metadata": missing_metadata,
        "missing_trajectory": missing_trajectory,
        "overall": summarize_group(enriched, threshold),
        "source_counts": dict(Counter(clean_text(row.get("source")) or "<missing>" for row in enriched)),
        "label_counts": dict(Counter(str(row.get("label")) for row in enriched)),
    }
    for group_name, buckets in grouped.items():
        summary[group_name] = {
            key: summarize_group(rows, threshold) for key, rows in sorted(buckets.items(), key=lambda item: item[0])
        }

    false_positives = [row for row in enriched if int(row["label"]) == 0 and int(row["predicted_label"]) == 1]
    false_negatives = [row for row in enriched if int(row["label"]) == 1 and int(row["predicted_label"]) == 0]
    false_positives.sort(key=lambda row: float(row["unsafe_score"]), reverse=True)
    false_negatives.sort(key=lambda row: float(row["unsafe_score"]))

    def compact(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "example_id": row.get("example_id"),
            "source": row.get("source"),
            "source_family": row.get("source_family"),
            "risk_type": row.get("risk_type"),
            "policy_type": row.get("policy_type"),
            "label": int(row["label"]),
            "unsafe_score": float(row["unsafe_score"]),
            "prompt_word_len": int(row["prompt_word_len"]),
            "reasoning_word_len": int(row["reasoning_word_len"]),
            "final_word_len": int(row["final_word_len"]),
            "reasoning_token_len": int(row["reasoning_token_len"]),
            "parse_status": row.get("parse_status"),
            "prompt_excerpt": row.get("prompt_excerpt"),
            "reasoning_excerpt": row.get("reasoning_excerpt"),
        }

    out_dir = Path(args.output_dir)
    write_json(out_dir / "artifact_summary.json", summary)
    write_jsonl(out_dir / "top_false_positives.jsonl", [compact(row) for row in false_positives[: args.top_k]])
    write_jsonl(out_dir / "top_false_negatives.jsonl", [compact(row) for row in false_negatives[: args.top_k]])
    write_tsv(
        out_dir / "by_source.tsv",
        [{"source": key, **metrics} for key, metrics in summary["by_source"].items()],
    )
    write_tsv(
        out_dir / "by_reasoning_length_bucket.tsv",
        [{"reasoning_length_bucket": key, **metrics} for key, metrics in summary["by_reasoning_length_bucket"].items()],
    )
    print(json.dumps({"output_dir": str(out_dir), "overall": summary["overall"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

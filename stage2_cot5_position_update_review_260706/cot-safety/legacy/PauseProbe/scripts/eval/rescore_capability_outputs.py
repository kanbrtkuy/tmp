#!/usr/bin/env python3
"""Rescore capability generations with a more robust final-answer extractor.

The first capability pass intentionally used a simple extractor so failures
were easy to audit.  This script adds a diagnostic scorer that distinguishes:

* strict accuracy: the score already stored in the generation JSONL;
* robust accuracy: a rule-based extractor that prefers explicit final-answer
  phrases, boxed answers, and GSM8K #### answers over the last-number fallback;
* gold_mentioned: whether the gold answer appears anywhere in plausible answer
  candidates.  This flags cases where the model likely solved the problem but
  the final answer format or truncation broke automatic scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


ANSWER_TOKEN_RE = re.compile(
    r"""
    (?:
        \\(?:dfrac|tfrac|frac)\s*\{?\s*[-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*\}?\s*
        \{?\s*[-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*\}?
    )
    |
    (?:
        [-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*/\s*[-+]?\d+(?:,\d{3})*(?:\.\d+)?
    )
    |
    (?:
        [-+]?\d+(?:,\d{3})*(?:\.\d+)?
    )
    """,
    re.VERBOSE,
)


FINAL_PHRASE_RE = re.compile(
    r"""
    (?:
        final\s+answer
        | answer\s+(?:should\s+be|is|would\s+be|=)
        | final\s+numeric\s+answer
        | therefore,?\s+the\s+answer\s+(?:is|should\s+be)
        | so,?\s+the\s+answer\s+(?:is|should\s+be)
    )
    (?P<tail>.{0,220})
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def strip_pause_tokens(text: str) -> str:
    return str(text or "").replace("<|pause|>", "")


def extract_boxed_all(text: str) -> list[str]:
    out: list[str] = []
    marker = "\\boxed{"
    start = 0
    while True:
        idx = text.find(marker, start)
        if idx < 0:
            break
        i = idx + len(marker)
        depth = 1
        buf: list[str] = []
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out.append("".join(buf).strip())
                    break
            buf.append(ch)
            i += 1
        start = max(i + 1, idx + len(marker))
    return out


def normalize_latex(text: str) -> str:
    text = clean_text(text)
    text = text.replace("$", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "")
    text = text.replace(",", "")
    text = text.strip().strip(".")
    return text


def normalize_compact(text: str) -> str:
    text = normalize_latex(text)
    text = re.sub(r"\s+", "", text)
    return text.lower()


def parse_fraction(text: str) -> Fraction | None:
    s = normalize_latex(text)
    s = re.sub(r"\s+", "", s)
    if not s:
        return None
    frac_match = re.fullmatch(r"\\frac\{?([-+]?\d+(?:\.\d+)?)\}?\{?([-+]?\d+(?:\.\d+)?)\}?", s)
    if frac_match:
        num, den = frac_match.groups()
        den_f = Fraction(den)
        if den_f == 0:
            return None
        return Fraction(num) / den_f
    slash_match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)/([-+]?\d+(?:\.\d+)?)", s)
    if slash_match:
        num, den = slash_match.groups()
        den_f = Fraction(den)
        if den_f == 0:
            return None
        return Fraction(num) / den_f
    num_match = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s)
    if num_match:
        return Fraction(s)
    return None


def answers_match(pred: str, gold: str) -> bool:
    pred_n = normalize_compact(pred)
    gold_n = normalize_compact(gold)
    if not pred_n or not gold_n:
        return False
    if pred_n == gold_n:
        return True
    pred_f = parse_fraction(pred_n)
    gold_f = parse_fraction(gold_n)
    if pred_f is not None and gold_f is not None:
        return pred_f == gold_f
    return False


def answer_tokens(text: str) -> list[str]:
    return [clean_text(match.group(0)) for match in ANSWER_TOKEN_RE.finditer(text)]


def gsm_hash_candidates(text: str) -> list[str]:
    return [clean_text(match.group(1)) for match in re.finditer(r"####\s*([^\n<]+)", text)]


def final_phrase_candidates(text: str) -> list[str]:
    out: list[str] = []
    for match in FINAL_PHRASE_RE.finditer(text):
        tail = match.group("tail")
        toks = answer_tokens(tail)
        if toks:
            out.append(toks[0])
    return out


def all_candidates(text: str, dataset: str) -> dict[str, list[str]]:
    text = strip_pause_tokens(text)
    candidates = {
        "hash": gsm_hash_candidates(text),
        "boxed": extract_boxed_all(text),
        "final_phrase": final_phrase_candidates(text),
        "all_tokens": answer_tokens(text),
    }
    if dataset != "gsm8k":
        # MATH answers often appear boxed without a final-answer phrase.
        candidates["priority"] = candidates["hash"] + candidates["boxed"] + candidates["final_phrase"]
    else:
        candidates["priority"] = candidates["hash"] + candidates["final_phrase"] + candidates["boxed"]
    return candidates


def robust_prediction(text: str, dataset: str) -> str:
    candidates = all_candidates(text, dataset)
    for key in ("hash", "boxed", "final_phrase"):
        vals = candidates[key]
        if vals:
            return vals[-1]
    vals = candidates["all_tokens"]
    return vals[-1] if vals else ""


@dataclass
class RowScore:
    model_label: str
    dataset: str
    id: str
    gold: str
    strict_pred: str
    strict_correct: bool
    robust_pred: str
    robust_correct: bool
    gold_mentioned: bool
    strict_wrong_robust_correct: bool
    strict_wrong_gold_mentioned: bool


def score_row(row: dict[str, Any]) -> RowScore:
    dataset = str(row.get("dataset") or "")
    gold = clean_text(row.get("answer"))
    generated = str(row.get("generated") or "")
    strict_pred = clean_text(row.get("predicted_answer"))
    strict_correct = row.get("correct") is True
    robust_pred = robust_prediction(generated, dataset)
    robust_correct = answers_match(robust_pred, gold)
    cand = all_candidates(generated, dataset)
    gold_mentioned = any(answers_match(tok, gold) for vals in cand.values() for tok in vals)
    return RowScore(
        model_label=str(row.get("model_label") or ""),
        dataset=dataset,
        id=str(row.get("id") or ""),
        gold=gold,
        strict_pred=strict_pred,
        strict_correct=strict_correct,
        robust_pred=robust_pred,
        robust_correct=robust_correct,
        gold_mentioned=gold_mentioned,
        strict_wrong_robust_correct=(not strict_correct and robust_correct),
        strict_wrong_gold_mentioned=(not strict_correct and gold_mentioned),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pct(num: int, den: int) -> float:
    return num / den if den else 0.0


def summarize(scores: list[RowScore]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[RowScore]] = {}
    for score in scores:
        groups.setdefault((score.model_label, score.dataset), []).append(score)
    out = []
    for (model_label, dataset), items in sorted(groups.items()):
        n = len(items)
        strict_correct = sum(item.strict_correct for item in items)
        robust_correct = sum(item.robust_correct for item in items)
        mentioned = sum(item.gold_mentioned for item in items)
        recovered = sum(item.strict_wrong_robust_correct for item in items)
        strict_wrong_mentioned = sum(item.strict_wrong_gold_mentioned for item in items)
        out.append(
            {
                "model_label": model_label,
                "dataset": dataset,
                "n": n,
                "strict_accuracy": pct(strict_correct, n),
                "robust_accuracy": pct(robust_correct, n),
                "gold_mentioned_rate": pct(mentioned, n),
                "strict_wrong_robust_correct_rate": pct(recovered, n),
                "strict_wrong_gold_mentioned_rate": pct(strict_wrong_mentioned, n),
                "strict_correct": strict_correct,
                "robust_correct": robust_correct,
                "gold_mentioned": mentioned,
                "strict_wrong_robust_correct": recovered,
                "strict_wrong_gold_mentioned": strict_wrong_mentioned,
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Eval root with generations/*.jsonl")
    parser.add_argument("--output_prefix", default="capability_rescore")
    parser.add_argument("--write_examples", action="store_true")
    parser.add_argument("--examples_per_group", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    scores: list[RowScore] = []
    example_rows: list[dict[str, Any]] = []
    for path in sorted((root / "generations").glob("*_capability.jsonl")):
        rows = read_jsonl(path)
        for row in rows:
            score = score_row(row)
            scores.append(score)
            if score.strict_wrong_gold_mentioned or score.strict_wrong_robust_correct:
                if len(example_rows) < args.examples_per_group * 20:
                    generated = str(row.get("generated") or "")
                    example_rows.append(
                        {
                            **score.__dict__,
                            "prompt": clean_text(row.get("prompt")),
                            "generated_tail": generated[-1200:],
                        }
                    )
    summary = summarize(scores)
    write_csv(root / f"{args.output_prefix}_summary.csv", summary)
    if args.write_examples:
        write_csv(root / f"{args.output_prefix}_examples.csv", example_rows)
    print(json.dumps({"summary_rows": len(summary), "scores": len(scores)}, indent=2))


if __name__ == "__main__":
    main()

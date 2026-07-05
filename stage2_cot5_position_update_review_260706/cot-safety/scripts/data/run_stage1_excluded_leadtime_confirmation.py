#!/usr/bin/env python3
"""Excluded-source Stage1 lead-time confirmation.

This script implements the Fable-5 reviewed, excluded-source lead-time
confirmation plan. It generates fixed matched-horizon text predictions for
`strongreject_full` and `reasoningshield`, reuses the archived Stage1
single-position hidden probe scores for A1, runs the already reviewed A1/A2
analysis scripts, and writes the preregistered gate decision.

It intentionally does not send or write raw prompts/CoTs to review packets.
Local fitted text models read the frozen split JSONL files, but outputs contain
only ids, pair ids, labels, scores, counts, hashes, and aggregate metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from cot_safety.utils.io import write_json, write_jsonl

import run_stage1_feature_pooling_reanalysis as feature_pool
import run_stage1_matched_horizon_reanalysis as matched
import run_stage1_score_pooling_reanalysis as score_pool


DEFAULT_SOURCES = ("strongreject_full", "reasoningshield")
DEFAULT_RUN_PREFIX = "stage1_natural_pairs_8b_a100_1x_loso"
DEFAULT_SURFACE_FAMILY = "char_tfidf"
DEFAULT_LAYER = 28


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def parse_ints(raw: str) -> list[int]:
    values = [int(part) for part in parse_csv(raw)]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"invalid positive int list: {raw!r}")
    return sorted(values)


def git_info(code_commit: str | None = None) -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = run(["git", "status", "--short"])
    return {
        "commit": code_commit or os.environ.get("COT_SAFETY_CODE_COMMIT") or run(["git", "rev-parse", "HEAD"]),
        "commit_source": "cli_or_env" if (code_commit or os.environ.get("COT_SAFETY_CODE_COMMIT")) else "git",
        "dirty": bool(status),
        "dirty_short": status,
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "source",
        "k",
        "split",
        "rows",
        "pairs",
        "input_rows",
        "input_pairs",
        "retained_rows",
        "retained_pairs",
        "short_pairs",
        "hidden_left_dropped",
        "surface_right_dropped",
        "frozen_test_pairs",
        "frozen_test_rows",
    ]
    fieldnames = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def row_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in rows}


def add_position_metadata(rows: list[dict[str, Any]], *, source: str, k: int, split: str, arm: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                **row,
                "source": source,
                "split": split,
                "arm": arm,
                "position": f"cot_{k}",
                "position_k": k,
                "cot_k": k,
                "k": k,
            }
        )
    return out


def read_hidden_records(
    *,
    hidden_score_root: Path,
    run_prefix: str,
    source: str,
    kind: str,
    k: int,
    layer: int,
    split: str,
) -> list[dict[str, Any]]:
    path = matched.hidden_prediction_path(hidden_score_root, run_prefix, source, kind, k=k, layer=layer, split=split)
    records = score_pool.read_predictions(path, expected_k=None)
    rows = [records[key] for key in sorted(records)]
    return add_position_metadata(rows, source=source, k=k, split=split, arm="hidden")


def aligned_pair_complete(
    hidden_rows: list[dict[str, Any]],
    surface_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    hidden_aligned, surface_aligned, align = score_pool.align_records(row_map(hidden_rows), row_map(surface_rows))
    hidden_kept, surface_kept, pc = score_pool.enforce_pair_complete(hidden_aligned, surface_aligned)
    return hidden_kept, surface_kept, {**align, **pc}


def pair_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["pair_id"]) for row in rows}


def filter_pairs(rows: list[dict[str, Any]], keep_pairs: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["pair_id"]) in keep_pairs]


def assert_same_pairs_across_k(
    records_by_k: dict[int, list[dict[str, Any]]],
    *,
    source: str,
    arm: str,
    expected_pairs: set[str],
) -> None:
    for k, rows in records_by_k.items():
        got = pair_ids(rows)
        if got != expected_pairs:
            missing = sorted(expected_pairs - got)[:5]
            extra = sorted(got - expected_pairs)[:5]
            raise AssertionError(
                f"{source}/{arm}/k={k} did not preserve frozen all-k test population; "
                f"missing={missing}, extra={extra}"
            )
        labels_by_pair: dict[str, set[int]] = {}
        for row in rows:
            labels_by_pair.setdefault(str(row["pair_id"]), set()).add(int(row["label"]))
        bad = [pid for pid, labels in labels_by_pair.items() if labels != {0, 1}]
        if bad:
            raise AssertionError(f"{source}/{arm}/k={k} has non-pair-complete frozen pairs: {bad[:5]}")


def surface_model_for_k(
    *,
    train_items: list[matched.HorizonRow],
    args: argparse.Namespace,
    sk: dict[str, Any],
) -> matched.SurfaceModel:
    return matched.fit_surface_model(
        DEFAULT_SURFACE_FAMILY,
        train_items,
        args=args,
        sk=sk,
        encoder=None,
    )


def build_prediction_files(args: argparse.Namespace) -> dict[str, Any]:
    sk = matched.import_sklearn()
    folds_root = Path(args.folds_root)
    hidden_score_root = Path(args.hidden_score_root)
    pred_dir = Path(args.output_dir) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    sources = parse_csv(args.sources)
    k_grid = parse_ints(args.k_grid)

    tokenizer = matched.load_tokenizer(args)
    splits_by_source = {source: matched.load_source_splits(folds_root, source) for source in sources}
    diagnostics: list[dict[str, Any]] = []
    all_records: dict[str, dict[int, dict[str, dict[str, list[dict[str, Any]]]]]] = {}
    frozen_population: dict[str, Any] = {}

    for source in sources:
        all_records[source] = {}
        eligible_pairs_by_k: dict[int, set[str]] = {}
        for k in k_grid:
            retained: dict[str, list[matched.HorizonRow]] = {}
            censoring: dict[str, Any] = {}
            for split in ("train", "val", "test"):
                retained[split], censoring[split] = matched.pair_complete_rows(
                    splits_by_source[source][split],
                    k=k,
                    tokenizer=tokenizer,
                )
                diagnostics.append({"source": source, "k": k, "split": split, **censoring[split]})

            surface_model = surface_model_for_k(train_items=retained["train"], args=args, sk=sk)
            surface_val = add_position_metadata(
                matched.records_from_scores(retained["val"], surface_model.score(retained["val"])),
                source=source,
                k=k,
                split="val",
                arm=DEFAULT_SURFACE_FAMILY,
            )
            surface_test = add_position_metadata(
                matched.records_from_scores(retained["test"], surface_model.score(retained["test"])),
                source=source,
                k=k,
                split="test",
                arm=DEFAULT_SURFACE_FAMILY,
            )
            hidden_val = read_hidden_records(
                hidden_score_root=hidden_score_root,
                run_prefix=args.run_prefix,
                source=source,
                kind=args.kind,
                k=k,
                layer=args.layer,
                split="val",
            )
            hidden_test = read_hidden_records(
                hidden_score_root=hidden_score_root,
                run_prefix=args.run_prefix,
                source=source,
                kind=args.kind,
                k=k,
                layer=args.layer,
                split="test",
            )

            hidden_val, surface_val, val_align = aligned_pair_complete(hidden_val, surface_val)
            hidden_test, surface_test, test_align = aligned_pair_complete(hidden_test, surface_test)
            diagnostics.append(
                {
                    "source": source,
                    "k": k,
                    "split": "val_aligned",
                    "hidden_left_dropped": val_align["left_dropped"],
                    "surface_right_dropped": val_align["right_dropped"],
                    "rows_after_pair_complete": val_align["rows_after_pair_complete"],
                    "pairs_after": val_align["pairs_after"],
                }
            )
            diagnostics.append(
                {
                    "source": source,
                    "k": k,
                    "split": "test_aligned_prefreeze",
                    "hidden_left_dropped": test_align["left_dropped"],
                    "surface_right_dropped": test_align["right_dropped"],
                    "rows_after_pair_complete": test_align["rows_after_pair_complete"],
                    "pairs_after": test_align["pairs_after"],
                }
            )
            eligible_pairs_by_k[k] = pair_ids(hidden_test) & pair_ids(surface_test)
            all_records[source][k] = {
                "val": {"hidden": hidden_val, "surface": surface_val},
                "test": {"hidden": hidden_test, "surface": surface_test},
            }

        frozen_pairs = set.intersection(*(eligible_pairs_by_k[k] for k in k_grid))
        if len(frozen_pairs) < args.min_pairs_per_source:
            raise SystemExit(
                f"minimum-power halt for {source}: frozen all-k test pairs={len(frozen_pairs)} "
                f"< {args.min_pairs_per_source}"
            )
        frozen_population[source] = {
            "frozen_test_pairs": len(frozen_pairs),
            "k_grid": k_grid,
            "eligible_pairs_by_k": {str(k): len(eligible_pairs_by_k[k]) for k in k_grid},
        }

        hidden_test_by_k: dict[int, list[dict[str, Any]]] = {}
        surface_test_by_k: dict[int, list[dict[str, Any]]] = {}
        for k in k_grid:
            base = pred_dir / source / f"k_{k}"
            val_hidden = all_records[source][k]["val"]["hidden"]
            val_surface = all_records[source][k]["val"]["surface"]
            test_hidden = filter_pairs(all_records[source][k]["test"]["hidden"], frozen_pairs)
            test_surface = filter_pairs(all_records[source][k]["test"]["surface"], frozen_pairs)
            hidden_test_by_k[k] = test_hidden
            surface_test_by_k[k] = test_surface
            write_jsonl(base / "hidden.val.predictions.jsonl", val_hidden)
            write_jsonl(base / f"{DEFAULT_SURFACE_FAMILY}.val.predictions.jsonl", val_surface)
            write_jsonl(base / "hidden.test.predictions.jsonl", test_hidden)
            write_jsonl(base / f"{DEFAULT_SURFACE_FAMILY}.test.predictions.jsonl", test_surface)
            diagnostics.append(
                {
                    "source": source,
                    "k": k,
                    "split": "test_frozen_written",
                    "frozen_test_pairs": len(frozen_pairs),
                    "frozen_test_rows": len(test_hidden),
                    "surface_rows": len(test_surface),
                }
            )

        assert_same_pairs_across_k(hidden_test_by_k, source=source, arm="hidden", expected_pairs=frozen_pairs)
        assert_same_pairs_across_k(surface_test_by_k, source=source, arm=DEFAULT_SURFACE_FAMILY, expected_pairs=frozen_pairs)

    write_tsv(Path(args.output_dir) / "stage1_excluded_leadtime_prediction_diagnostics.tsv", diagnostics)
    write_json(Path(args.output_dir) / "stage1_excluded_leadtime_frozen_population.json", frozen_population)
    return {"pred_dir": str(pred_dir), "diagnostics": diagnostics, "frozen_population": frozen_population}


def find_row(rows: list[dict[str, Any]], *, source: str, hidden_k: int, surface_k: int) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if str(row.get("source")) == source
        and int(row.get("hidden_k")) == hidden_k
        and int(row.get("surface_k")) == surface_k
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {source} hidden@{hidden_k} text@{surface_k}; found {len(matches)}")
    return matches[0]


def gate_summary(a1_payload: dict[str, Any], a2_payload: dict[str, Any], sources: list[str]) -> dict[str, Any]:
    a1_rows = list(a1_payload["lead_time_rows"])
    a2_rows = list(a2_payload["lead_time_rows"])
    a1_primary = find_row(a1_rows, source="pooled", hidden_k=4, surface_k=8)
    a2_primary = find_row(a2_rows, source="pooled", hidden_k=4, surface_k=8)
    per_source_rows = [find_row(a1_rows, source=source, hidden_k=4, surface_k=8) for source in sources]

    a1_ci_low = float(a1_primary["delta_auroc_ci_low"])
    a1_delta = float(a1_primary["delta_auroc_hidden_minus_surface"])
    a2_ci_high = float(a2_primary["delta_auroc_ci_high"])
    a2_delta = float(a2_primary["delta_auroc_hidden_minus_surface"])
    per_source_min = min(float(row["delta_auroc_hidden_minus_surface"]) for row in per_source_rows)

    a1_gate = a1_ci_low >= 0.0
    per_source_gate = per_source_min >= -0.02
    a2_gate = a2_ci_high >= 0.0 and a2_delta >= -0.01
    confirmed = bool(a1_gate and per_source_gate and a2_gate)
    if confirmed:
        decision = "confirmed_preregistered_leadtime"
    elif a1_gate and per_source_gate and not a2_gate:
        decision = "replicated_but_recipe_sensitive"
    elif not a1_gate:
        decision = "drop_leadtime_claim"
    else:
        decision = "heterogeneous_no_pooled_headline"

    return {
        "primary_cell": "pooled A1 hidden@4 minus text@8 delta AUROC",
        "a1_primary": {
            "delta_auroc": a1_delta,
            "ci_low": a1_ci_low,
            "ci_high": a1_primary["delta_auroc_ci_high"],
            "gate_pass": a1_gate,
        },
        "per_source_sanity": {
            "min_delta_auroc": per_source_min,
            "threshold": -0.02,
            "gate_pass": per_source_gate,
            "rows": per_source_rows,
        },
        "a2_robustness": {
            "delta_auroc": a2_delta,
            "ci_low": a2_primary["delta_auroc_ci_low"],
            "ci_high": a2_ci_high,
            "point_floor": -0.01,
            "gate_pass": a2_gate,
        },
        "confirmed": confirmed,
        "decision": decision,
        "interpretation_rules": {
            "all_gates_pass": "only this outcome may be called confirmed",
            "a1_pass_a2_fail": "replicated but recipe-sensitive; exploratory only",
            "a1_fail": "drop the lead-time claim",
            "per_source_fail": "report heterogeneity; no pooled-only headline",
        },
        "multiplicity_correction": "none for the primary confirmatory cell; full matrices are descriptive only",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = parse_csv(args.sources) or list(DEFAULT_SOURCES)
    k_grid = parse_ints(args.k_grid)
    if args.layer != DEFAULT_LAYER:
        raise SystemExit("excluded-source lead-time confirmation fixes --layer to 28")
    if DEFAULT_SURFACE_FAMILY != args.surface_family:
        raise SystemExit("excluded-source lead-time confirmation fixes --surface-family to char_tfidf")
    if not args.code_commit:
        raise SystemExit("--code-commit is required")
    if not args.tmp_prereg_commit:
        raise SystemExit("--tmp-prereg-commit is required")

    prereg = {
        "stage": "stage1_excluded_source_leadtime_confirmation",
        "tmp_prereg_commit": args.tmp_prereg_commit,
        "config_pinning_amendment": "res/stage1_excluded_source_leadtime_config_pinning_amendment_260705.md",
        "sources": sources,
        "k_grid": k_grid,
        "surface_family": DEFAULT_SURFACE_FAMILY,
        "layer": args.layer,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "frozen_population_rule": "test pairs pair-complete at every k and in both arms; identical across k",
        "minimum_pairs_per_source": args.min_pairs_per_source,
        "primary_a1_estimand": "pooled A1 hidden@4 minus text@8 delta AUROC",
        "a1_primary_gate": "CI low >= 0",
        "per_source_sanity_gate": "no source A1 point estimate < -0.02",
        "a2_robustness_gate": "pooled A2 hidden@4 minus text@8 CI high >= 0 and point estimate >= -0.01",
        "descriptive_only": "all other hidden@k minus text@k' cells, same-horizon diagonals, and pair-rank metrics",
        "code_commit": args.code_commit,
    }
    write_json(output_dir / "stage1_excluded_leadtime_preregistration.json", prereg)

    prediction_payload = build_prediction_files(args)
    pred_dir = prediction_payload["pred_dir"]

    a1_dir = output_dir / "a1_score_pooling"
    a1_payload = score_pool.run(
        argparse.Namespace(
            pred_dir=pred_dir,
            output_dir=str(a1_dir),
            sources=",".join(sources),
            k_grid=",".join(str(k) for k in k_grid),
            holm_ks="8",
            surface_family=DEFAULT_SURFACE_FAMILY,
            selected_layer=args.layer,
            rule="zmean",
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            monotone_tolerance=args.monotone_tolerance,
            fail_on_error=True,
        )
    )

    a2_dir = output_dir / "a2_feature_pooling"
    a2_payload = feature_pool.run(
        argparse.Namespace(
            hidden_archive_root=args.hidden_archive_root,
            pred_dir=pred_dir,
            output_dir=str(a2_dir),
            sources=",".join(sources),
            k_grid=",".join(str(k) for k in k_grid),
            holm_ks="8",
            surface_family=DEFAULT_SURFACE_FAMILY,
            archive_dir_prefix=args.archive_dir_prefix,
            file_prefix=args.file_prefix,
            layer=args.layer,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            max_iter=args.max_iter,
            monotone_tolerance=args.monotone_tolerance,
            code_commit=args.code_commit,
            fail_on_error=True,
        )
    )

    gates = gate_summary(a1_payload, a2_payload, sources)
    payload = {
        "stage": "stage1_excluded_source_leadtime_confirmation",
        "script_version": "stage1_excluded_leadtime_confirmation_v1",
        "output_dir": str(output_dir),
        "pred_dir": pred_dir,
        "preregistration": prereg,
        "frozen_population": prediction_payload["frozen_population"],
        "gate_summary": gates,
        "a1_output_dir": str(a1_dir),
        "a2_output_dir": str(a2_dir),
        "a1_n_errors": a1_payload["n_errors"],
        "a2_n_errors": a2_payload["n_errors"],
        "n_errors": int(a1_payload["n_errors"]) + int(a2_payload["n_errors"]),
        "git": git_info(args.code_commit),
    }
    write_json(output_dir / "stage1_excluded_leadtime_confirmation_summary.json", payload)
    write_tsv(
        output_dir / "stage1_excluded_leadtime_confirmation_gates.tsv",
        [
            {"gate": "a1_primary", **gates["a1_primary"]},
            {"gate": "per_source_sanity", **gates["per_source_sanity"]},
            {"gate": "a2_robustness", **gates["a2_robustness"]},
            {"gate": "decision", "decision": gates["decision"], "confirmed": gates["confirmed"]},
        ],
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "pred_dir": pred_dir,
                "n_errors": payload["n_errors"],
                "decision": gates["decision"],
                "confirmed": gates["confirmed"],
            },
            indent=2,
        )
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds-root", required=True)
    parser.add_argument("--hidden-score-root", required=True)
    parser.add_argument("--hidden-archive-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--k-grid", default="4,8,16,32,64")
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--kind", default="linear")
    parser.add_argument("--surface-family", default=DEFAULT_SURFACE_FAMILY)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--archive-dir-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--file-prefix", default="natural_pairs_8b_a100_1x_loso")
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tokenizer-trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-whitespace-tokenizer", action="store_true")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-features-word", type=int, default=100000)
    parser.add_argument("--max-features-char", type=int, default=200000)
    parser.add_argument("--max-features-position", type=int, default=200000)
    parser.add_argument("--char-min-n", type=int, default=3)
    parser.add_argument("--char-max-n", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=260705)
    parser.add_argument("--monotone-tolerance", type=float, default=0.02)
    parser.add_argument("--min-pairs-per-source", type=int, default=150)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--tmp-prereg-commit", required=True)
    args = parser.parse_args()
    if args.n_bootstrap <= 0:
        parser.error("--n-bootstrap must be positive")
    if args.min_pairs_per_source <= 0:
        parser.error("--min-pairs-per-source must be positive")
    return args


def main() -> int:
    payload = run(parse_args())
    return 2 if payload["n_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

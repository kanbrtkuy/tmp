#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_stage1_positionscan as stage1_legacy
from cot_safety.config import deep_merge, dump_config, load_config


def slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_")


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def drop_pipeline_keys(config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(config)
    config.pop("stage1_pipeline", None)
    return config


def source_names(config: dict[str, Any]) -> list[str]:
    return stage1_legacy.source_names(config.get("data", {}))


def family_sources(family: dict[str, Any]) -> list[str]:
    return [str(source) for source in as_list(family.get("sources"))]


def apply_runtime_env(config: dict[str, Any]) -> None:
    runtime = config.get("runtime", {})
    if runtime.get("cuda_visible_devices"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(runtime["cuda_visible_devices"]))
    if runtime.get("hf_home"):
        os.environ.setdefault("HF_HOME", str(stage1_legacy.resolve_value(runtime["hf_home"])))
    if runtime.get("pytorch_cuda_alloc_conf"):
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            str(stage1_legacy.resolve_value(runtime["pytorch_cuda_alloc_conf"])),
        )


def apply_run_paths(config: dict[str, Any], run_name: str, hot_root: str, hidden_prefix: str) -> dict[str, Any]:
    config = deepcopy(config)
    config.setdefault("run", {})["name"] = run_name
    config["run"]["output_dir"] = f"${{COT_SAFETY_RUN_ROOT:-runs}}/{run_name}"
    config["legacy"] = {
        "data_dir": f"{hot_root}/data/{run_name}",
        "hidden_dir": f"{hot_root}/runs/hidden/{run_name}",
        "hidden_prefix": hidden_prefix,
        "log_dir": f"{hot_root}/runs/logs/{run_name}",
        "single_scan_out_root": f"{hot_root}/runs/{run_name}/linear",
        "multilayer_out_root": f"{hot_root}/runs/{run_name}/multilayer",
    }
    return config


def module_config(
    base_config: dict[str, Any],
    pipeline: dict[str, Any],
    module_name: str,
    *,
    heldout_sources: list[str] | None = None,
    run_suffix: str | None = None,
) -> dict[str, Any]:
    module = pipeline.get(module_name, {})
    base_run_name = str(base_config.get("run", {}).get("name", "stage1"))
    suffix = run_suffix or str(module.get("run_name_suffix", module_name))
    run_name = f"{base_run_name}_{slug(suffix)}"
    hot_root = str(
        pipeline.get("storage", {}).get(
            "hot_root",
            "${COT_SAFETY_HOT_ROOT:-/dev/shm/cot-safety-hot}",
        )
    )
    hidden_prefix = str(module.get("hidden_prefix", slug(suffix)))
    config = drop_pipeline_keys(base_config)
    config = deep_merge(config, module.get("overrides", {}) or {})
    if heldout_sources is not None:
        config.setdefault("data", {})["heldout_sources"] = heldout_sources
    config = apply_run_paths(config, run_name, hot_root, hidden_prefix)
    return stage1_legacy.resolve_value(config)


def run_one(
    args: argparse.Namespace,
    config: dict[str, Any],
    legacy_root: Path,
    *,
    label: str,
) -> int:
    runs_dir = REPO_ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = str(config.get("run", {}).get("name", label))
    (runs_dir / f"{run_name}_resolved.yaml").write_text(dump_config(config), encoding="utf-8")
    cmd = stage1_legacy.build_command(args, config)
    print(f"\n### {label}")
    print("$ " + " ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.run(cmd, cwd=legacy_root, env=os.environ.copy()).returncode


def resolve_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser().resolve(strict=False)


def cold_root() -> Path:
    return resolve_path(os.environ.get("COT_SAFETY_COLD_ROOT", "/workspace"))


def run_cold_root() -> Path:
    return cold_root() / "cot-safety" / "runs"


def cold_path_for_hot(hot_path: str | Path) -> Path:
    hot = resolve_path(hot_path)
    hot_root = resolve_path(os.environ.get("COT_SAFETY_HOT_ROOT", "/dev/shm/cot-safety-hot"))
    mappings = [
        (resolve_path(os.environ.get("COT_SAFETY_OUTPUT_ROOT", "/workspace/outputs")), cold_root() / "outputs"),
        (resolve_path(os.environ.get("COT_SAFETY_DATA_ROOT", "/workspace/data")), cold_root() / "data"),
        (resolve_path(os.environ.get("COT_SAFETY_RUN_ROOT", str(REPO_ROOT / "runs"))), run_cold_root()),
        (hot_root / "outputs", cold_root() / "outputs"),
        (hot_root / "data", cold_root() / "data"),
        (hot_root / "runs", run_cold_root()),
    ]
    for hot_base, cold_base in mappings:
        try:
            rel = hot.relative_to(hot_base)
        except ValueError:
            continue
        return cold_base / rel
    return run_cold_root() / "stage1_misc" / hot.name


def same_path(left: Path, right: Path) -> bool:
    return resolve_path(left) == resolve_path(right)


def copy_hot_to_cold(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"skip missing hot path: {src}")
        return
    if same_path(src, dst):
        print(f"skip hot-to-cold sync because path is already cold: {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"persisting stage1 hot path: {src} -> {dst}")
    if shutil.which("rsync"):
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-a", f"{src}/", f"{dst}/"], check=True)
        else:
            subprocess.run(["rsync", "-a", str(src), str(dst)], check=True)
    elif src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    if dst.is_dir():
        (dst / ".synced_to_cold").write_text("stage1 synced\n", encoding="utf-8")


def remove_synced_hot(src: Path, dst: Path) -> None:
    if not src.exists() or same_path(src, dst):
        return
    if dst.is_dir() and not (dst / ".synced_to_cold").exists():
        raise RuntimeError(f"Refusing to remove hot path without cold marker: {dst}")
    if not dst.exists():
        raise RuntimeError(f"Refusing to remove hot path; cold copy is missing: {dst}")
    print(f"removing synced stage1 hot path: {src}")
    if src.is_dir():
        shutil.rmtree(src)
    else:
        src.unlink()


def stage1_hot_paths(config: dict[str, Any]) -> list[Path]:
    paths = stage1_legacy.stage_paths(config)
    return [
        resolve_path(paths["data_dir"]),
        resolve_path(paths["hidden_dir"]),
        resolve_path(paths["log_dir"]),
        resolve_path(paths["single_scan_out_root"]).parent,
    ]


def cold_config_for_hot(config: dict[str, Any]) -> dict[str, Any]:
    paths = stage1_legacy.stage_paths(config)
    cold_config = deepcopy(config)
    cold_config["legacy"] = {
        "data_dir": str(cold_path_for_hot(paths["data_dir"])),
        "hidden_dir": str(cold_path_for_hot(paths["hidden_dir"])),
        "hidden_prefix": paths["hidden_prefix"],
        "log_dir": str(cold_path_for_hot(paths["log_dir"])),
        "single_scan_out_root": str(cold_path_for_hot(paths["single_scan_out_root"])),
        "multilayer_out_root": str(cold_path_for_hot(paths["multilayer_out_root"])),
    }
    return cold_config


def sync_stage1_run(config: dict[str, Any], pipeline: dict[str, Any]) -> None:
    storage = pipeline.get("storage", {})
    if not bool(storage.get("sync_to_cold_after_module", True)):
        return
    remove_hot = bool(storage.get("remove_hot_after_sync", True))
    for hot_path in stage1_hot_paths(config):
        cold_path = cold_path_for_hot(hot_path)
        copy_hot_to_cold(hot_path, cold_path)
        if remove_hot:
            remove_synced_hot(hot_path, cold_path)


def sync_stage1_path(path: Path, pipeline: dict[str, Any]) -> None:
    storage = pipeline.get("storage", {})
    if not bool(storage.get("sync_to_cold_after_module", True)):
        return
    cold_path = cold_path_for_hot(path)
    copy_hot_to_cold(path, cold_path)
    if bool(storage.get("remove_hot_after_sync", True)):
        remove_synced_hot(path, cold_path)


def stage1_summary_exists(path: Path) -> bool:
    return any(
        (path / filename).exists()
        for filename in ("summary_grid.tsv", "summary_by_test_auroc.tsv", "summary_by_heldout_auroc.tsv")
    )


def stage1_outputs_complete(config: dict[str, Any], *, require_cold_marker: bool) -> bool:
    paths = stage1_legacy.stage_paths(config)
    single_root = resolve_path(paths["single_scan_out_root"])
    multilayer_root = resolve_path(paths["multilayer_out_root"])
    run_root = single_root.parent
    if require_cold_marker and not (run_root / ".synced_to_cold").exists():
        return False
    return stage1_summary_exists(single_root) and stage1_summary_exists(multilayer_root)


def skip_or_sync_completed_module(
    args: argparse.Namespace,
    config: dict[str, Any],
    pipeline: dict[str, Any],
    *,
    label: str,
) -> bool:
    if args.dry_run or not args.skip_existing:
        return False
    if stage1_outputs_complete(config, require_cold_marker=False):
        print(f"\n### {label}")
        print("skip completed hot Stage1 module")
        sync_stage1_run(config, pipeline)
        return True
    cold_config = cold_config_for_hot(config)
    if stage1_outputs_complete(cold_config, require_cold_marker=True):
        print(f"\n### {label}")
        print("skip completed cold-synced Stage1 module")
        return True
    return False


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        path.with_suffix(".json").write_text("[]\n", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fnum(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def heldout_family_metric(row: dict[str, str], sources: list[str], metric: str) -> tuple[float | None, float | None]:
    values = [fnum(row.get(f"{source}_{metric}")) for source in sources]
    values = [value for value in values if value is not None]
    if not values:
        return None, None
    return statistics.mean(values), min(values)


def rank_auroc(labels: list[int], scores: list[float]) -> float | None:
    n_pos = sum(1 for label in labels if label == 1)
    n_neg = sum(1 for label in labels if label == 0)
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda idx: scores[idx])
    rank_sum = 0.0
    idx = 0
    while idx < len(order):
        end = idx + 1
        while end < len(order) and scores[order[end]] == scores[order[idx]]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for j in range(idx, end):
            if labels[order[j]] == 1:
                rank_sum += avg_rank
        idx = end
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(labels: list[int], scores: list[float]) -> float | None:
    n_pos = sum(1 for label in labels if label == 1)
    if n_pos == 0:
        return None
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    tp = 0
    precision_sum = 0.0
    for rank, idx in enumerate(order, start=1):
        if labels[idx] == 1:
            tp += 1
            precision_sum += tp / rank
    return precision_sum / n_pos


def metrics_from_predictions(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    labels = [int(row["label"]) for row in rows if row.get("label") is not None]
    scores = [float(row["unsafe_score"]) for row in rows if row.get("label") is not None]
    preds = [int(row["prediction"]) for row in rows if row.get("label") is not None]
    if not labels:
        return {"n": 0, "auroc": None, "auprc": None, "recall": None, "fpr": None}
    tp = sum(1 for label, pred in zip(labels, preds) if label == 1 and pred == 1)
    fn = sum(1 for label, pred in zip(labels, preds) if label == 1 and pred == 0)
    fp = sum(1 for label, pred in zip(labels, preds) if label == 0 and pred == 1)
    tn = sum(1 for label, pred in zip(labels, preds) if label == 0 and pred == 0)
    return {
        "n": len(labels),
        "auroc": rank_auroc(labels, scores),
        "auprc": average_precision(labels, scores),
        "recall": tp / (tp + fn) if tp + fn else None,
        "fpr": fp / (fp + tn) if fp + tn else None,
    }


def single_run_name(row: dict[str, str]) -> str:
    return f"{row.get('model')}_{row.get('position')}_l{row.get('layer')}"


def multilayer_run_name(row: dict[str, str]) -> str:
    return str(row.get("run") or "")


def combined_family_metrics(
    root: Path,
    summary_kind: str,
    row: dict[str, str],
    sources: list[str],
) -> dict[str, float | int | None]:
    run_name = single_run_name(row) if summary_kind == "single_layer" else multilayer_run_name(row)
    if not run_name:
        return {"n": 0, "auroc": None, "auprc": None, "recall": None, "fpr": None}
    pred_rows: list[dict[str, Any]] = []
    for source in sources:
        pred_rows.extend(read_jsonl(root / f"eval_{source}_{run_name}" / "predictions.jsonl"))
    return metrics_from_predictions(pred_rows)


def enrich_loso_rows(
    *,
    module_name: str,
    run_name: str,
    family: dict[str, Any],
    summary_kind: str,
    root: Path,
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    sources = family_sources(family)
    family_name = str(family.get("name", "_".join(sources)))
    out: list[dict[str, Any]] = []
    for row in rows:
        combined = combined_family_metrics(root, summary_kind, row, sources)
        auroc_mean, auroc_min = heldout_family_metric(row, sources, "auroc")
        auprc_mean, auprc_min = heldout_family_metric(row, sources, "auprc")
        recall_mean, recall_min = heldout_family_metric(row, sources, "recall")
        fpr_mean, fpr_min = heldout_family_metric(row, sources, "fpr")
        family_auroc = combined.get("auroc") if combined.get("auroc") is not None else auroc_mean
        if family_auroc is None:
            continue
        enriched: dict[str, Any] = {
            "module": module_name,
            "run_name": run_name,
            "summary_kind": summary_kind,
            "heldout_family": family_name,
            "heldout_sources": ",".join(sources),
            "family_eval_n": combined.get("n"),
            "family_auroc": family_auroc,
            "family_auprc": combined.get("auprc") if combined.get("auprc") is not None else auprc_mean,
            "family_recall": combined.get("recall") if combined.get("recall") is not None else recall_mean,
            "family_fpr": combined.get("fpr") if combined.get("fpr") is not None else fpr_mean,
            "raw_source_auroc_mean": auroc_mean,
            "family_auroc_min": auroc_min,
            "raw_source_auprc_mean": auprc_mean,
            "family_auprc_min": auprc_min,
            "raw_source_recall_mean": recall_mean,
            "family_recall_min": recall_min,
            "raw_source_fpr_mean": fpr_mean,
            "family_fpr_min": fpr_min,
        }
        for key in (
            "model",
            "position",
            "layer",
            "layer_combine",
            "layers",
            "test_auroc",
            "test_auprc",
            "test_recall",
            "test_fpr",
            "val_auroc",
            "val_recall",
            "val_fpr",
        ):
            if key in row:
                enriched[key] = row[key]
        out.append(enriched)
    return out


def aggregate_loso(
    pipeline: dict[str, Any],
    completed: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]],
    out_dir: Path,
) -> None:
    all_rows: list[dict[str, Any]] = []
    for module_name, family, hot_config, cold_config in completed:
        hot_paths = stage1_legacy.stage_paths(hot_config)
        cold_paths = stage1_legacy.stage_paths(cold_config)
        paths = cold_paths if Path(cold_paths["single_scan_out_root"]).exists() else hot_paths
        run_name = str(hot_config.get("run", {}).get("name", "stage1_loso"))
        single_rows = read_tsv(Path(paths["single_scan_out_root"]) / "summary_grid.tsv")
        multi_rows = read_tsv(Path(paths["multilayer_out_root"]) / "summary_grid.tsv")
        all_rows.extend(
            enrich_loso_rows(
                module_name=module_name,
                run_name=run_name,
                family=family,
                summary_kind="single_layer",
                root=Path(paths["single_scan_out_root"]),
                rows=single_rows,
            )
        )
        all_rows.extend(
            enrich_loso_rows(
                module_name=module_name,
                run_name=run_name,
                family=family,
                summary_kind="multilayer",
                root=Path(paths["multilayer_out_root"]),
                rows=multi_rows,
            )
        )

    all_rows.sort(
        key=lambda row: (
            str(row.get("module", "")),
            str(row.get("heldout_family", "")),
            str(row.get("summary_kind", "")),
            -(float(row.get("family_auroc") or 0.0)),
        )
    )
    write_table(out_dir / "stage1_loso_summary_grid.tsv", all_rows)

    best_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in all_rows:
        key = (
            str(row.get("module", "")),
            str(row.get("heldout_family", "")),
            str(row.get("summary_kind", "")),
        )
        grouped.setdefault(key, []).append(row)
    for rows in grouped.values():
        best_rows.append(max(rows, key=lambda row: float(row.get("family_auroc") or 0.0)))
    best_rows.sort(key=lambda row: (str(row.get("module")), str(row.get("heldout_family")), str(row.get("summary_kind"))))
    write_table(out_dir / "stage1_loso_best_by_family.tsv", best_rows)

    prompt_positions = set(pipeline.get("prompt_baseline", {}).get("prompt_only_positions", []))
    if not prompt_positions:
        prompt_positions = {"last_prompt_token", "assistant_start", "assistant_last", "pre_think", "think_last"}
    prompt_gap_rows: list[dict[str, Any]] = []
    prompt_rows = [row for row in all_rows if row.get("module") == "prompt_baseline"]
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in prompt_rows:
        by_key.setdefault((str(row.get("heldout_family")), str(row.get("summary_kind"))), []).append(row)
    for (family_name, summary_kind), rows in sorted(by_key.items()):
        prompt = [
            row
            for row in rows
            if str(row.get("position")) in prompt_positions
        ]
        traj = [
            row
            for row in rows
            if str(row.get("position")) not in prompt_positions
        ]
        if not prompt or not traj:
            continue
        best_prompt = max(prompt, key=lambda row: float(row.get("family_auroc") or 0.0))
        best_traj = max(traj, key=lambda row: float(row.get("family_auroc") or 0.0))
        prompt_gap_rows.append(
            {
                "heldout_family": family_name,
                "summary_kind": summary_kind,
                "best_prompt_position": best_prompt.get("position"),
                "best_prompt_auroc": best_prompt.get("family_auroc"),
                "best_trajectory_position": best_traj.get("position"),
                "best_trajectory_auroc": best_traj.get("family_auroc"),
                "trajectory_minus_prompt_auroc": float(best_traj.get("family_auroc") or 0.0)
                - float(best_prompt.get("family_auroc") or 0.0),
            }
        )
    write_table(out_dir / "stage1_loso_prompt_vs_trajectory.tsv", prompt_gap_rows)


def enabled(pipeline: dict[str, Any], key: str) -> bool:
    return bool(pipeline.get(key, {}).get("enabled", False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the integrated Stage 1 pipeline: position scan, prompt baselines, and LOSO.")
    parser.add_argument("--config", default="configs/experiment/stage1_unified.yaml")
    parser.add_argument("--legacy-root", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max_per_source", type=int, default=None)
    parser.add_argument("--only", nargs="*", choices=("position_scan", "prompt_baseline", "loso"), default=None)
    parser.add_argument("--skip_data_prep", action="store_true")
    parser.add_argument("--skip_hidden_extraction", action="store_true")
    parser.add_argument("--skip_single_scan", action="store_true")
    parser.add_argument("--skip_multilayer", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config = stage1_legacy.resolve_value(load_config(REPO_ROOT / args.config))
    pipeline = config.get("stage1_pipeline", {})
    if not pipeline:
        raise SystemExit("Integrated Stage1 config must define stage1_pipeline.")
    apply_runtime_env(config)
    legacy_root = Path(args.legacy_root) if args.legacy_root else REPO_ROOT / "legacy/PauseProbe"
    selected = set(args.only or ["position_scan", "prompt_baseline", "loso"])

    failures: list[tuple[str, int]] = []
    loso_completed: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    if "position_scan" in selected and enabled(pipeline, "position_scan"):
        run_config = module_config(config, pipeline, "position_scan")
        label = "Stage1 position scan"
        if not skip_or_sync_completed_module(args, run_config, pipeline, label=label):
            rc = run_one(args, run_config, legacy_root, label=label)
            if rc:
                failures.append(("position_scan", rc))
            elif not args.dry_run:
                sync_stage1_run(run_config, pipeline)

    if "prompt_baseline" in selected and enabled(pipeline, "prompt_baseline"):
        run_config = module_config(config, pipeline, "prompt_baseline")
        label = "Stage1b prompt/pre-CoT baseline"
        if not skip_or_sync_completed_module(args, run_config, pipeline, label=label):
            rc = run_one(args, run_config, legacy_root, label=label)
            if rc:
                failures.append(("prompt_baseline", rc))
            elif not args.dry_run:
                sync_stage1_run(run_config, pipeline)

    if "loso" in selected and enabled(pipeline, "loso"):
        loso = pipeline.get("loso", {})
        modules = [str(item) for item in as_list(loso.get("modules") or ["position_scan", "prompt_baseline"])]
        known_sources = set(source_names(config))
        for family in loso.get("source_families", []):
            sources = family_sources(family)
            if not sources:
                continue
            missing = [source for source in sources if source not in known_sources]
            if missing:
                raise SystemExit(f"LOSO family {family.get('name')} references sources not in data.sources: {missing}")
            family_name = slug(str(family.get("name", "_".join(sources))))
            for module_name in modules:
                if not enabled(pipeline, module_name):
                    continue
                run_config = module_config(
                    config,
                    pipeline,
                    module_name,
                    heldout_sources=sources,
                    run_suffix=f"loso_{module_name}_{family_name}",
                )
                run_config = stage1_legacy.resolve_value(deep_merge(run_config, loso.get("overrides", {}) or {}))
                label = f"Stage1 LOSO {module_name} holdout={family_name}"
                cold_config = cold_config_for_hot(run_config)
                if skip_or_sync_completed_module(args, run_config, pipeline, label=label):
                    loso_completed.append((module_name, family, run_config, cold_config))
                else:
                    rc = run_one(args, run_config, legacy_root, label=label)
                    if rc:
                        failures.append((f"loso/{module_name}/{family_name}", rc))
                    else:
                        if not args.dry_run:
                            sync_stage1_run(run_config, pipeline)
                            cold_config = cold_config_for_hot(run_config)
                        loso_completed.append((module_name, family, run_config, cold_config))
        if not args.dry_run and bool(loso.get("aggregate", True)):
            out_dir = Path(stage1_legacy.resolve_value(pipeline.get("loso", {}).get("summary_dir", "${COT_SAFETY_RUN_ROOT:-runs}/stage1_loso_summary")))
            aggregate_loso(pipeline, loso_completed, out_dir)
            sync_stage1_path(out_dir, pipeline)

    if failures:
        for label, rc in failures:
            print(f"FAILED {label}: exit={rc}", file=sys.stderr)
        raise SystemExit(failures[0][1])


if __name__ == "__main__":
    main()

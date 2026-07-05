#!/usr/bin/env python3
"""Plot PauseRiskProbe and PositionScan TrajProbe 2D heatmaps."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


PAUSE_POSITION_ORDER = ["pause_0", "pause_1", "pause_2"]
POSITION_SCAN_ORDER = [
    "cot_0",
    "cot_1",
    "cot_2",
    "cot_3",
    "cot_4",
    "cot_5",
    "cot_6",
    "cot_7",
    "cot_8",
    "cot_9",
    "cot_10",
    "cot_12",
    "cot_16",
    "cot_24",
    "cot_32",
    "cot_48",
    "cot_64",
    "cot_96",
    "cot_128",
]

PAUSE_METRICS = [
    "test_auroc",
    "test_recall",
    "test_fpr",
    "xstest_recall",
    "xstest_fpr",
    "orhard_fpr",
    "ortoxic_recall",
    "v1_auroc",
    "v1_recall",
    "v1_fpr",
]

POSITION_SCAN_METRICS = [
    "test_auroc",
    "test_recall",
    "test_fpr",
    "reasoningshield_test_auroc",
    "reasoningshield_test_recall",
    "reasoningshield_test_fpr",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def layer_order(rows: list[dict[str, str]]) -> list[int]:
    return sorted({int(row["layer"]) for row in rows})


def position_order(rows: list[dict[str, str]], preferred: list[str]) -> list[str]:
    present = {str(row["position"]) for row in rows}
    ordered = [position for position in preferred if position in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def pivot(
    rows: list[dict[str, str]],
    metric: str,
    positions: list[str],
    layers: list[int],
) -> np.ndarray:
    values: dict[tuple[str, int], float] = {}
    for row in rows:
        values[(str(row["position"]), int(row["layer"]))] = numeric(row.get(metric))
    matrix = np.full((len(positions), len(layers)), np.nan, dtype=np.float64)
    for y, position in enumerate(positions):
        for x, layer in enumerate(layers):
            matrix[y, x] = values.get((position, layer), math.nan)
    return matrix


def write_pivot_csv(
    out_path: Path,
    matrix: np.ndarray,
    positions: list[str],
    layers: list[int],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position"] + [str(layer) for layer in layers])
        for position, row in zip(positions, matrix):
            writer.writerow([position] + [f"{value:.6g}" if not math.isnan(value) else "" for value in row])


def is_fpr(metric: str) -> bool:
    return metric.endswith("_fpr") or metric == "orhard_fpr"


def metric_title(metric: str) -> str:
    return metric.replace("_", " ").upper() if metric.endswith("fpr") else metric.replace("_", " ").title()


def draw_heatmap(
    matrix: np.ndarray,
    positions: list[str],
    layers: list[int],
    title: str,
    metric: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmap = "magma_r" if is_fpr(metric) else "viridis"
    finite = matrix[np.isfinite(matrix)]
    if finite.size:
        vmin = float(np.nanmin(matrix))
        vmax = float(np.nanmax(matrix))
        best_idx = np.nanargmin(matrix) if is_fpr(metric) else np.nanargmax(matrix)
        best_y, best_x = np.unravel_index(best_idx, matrix.shape)
    else:
        vmin, vmax = 0.0, 1.0
        best_y, best_x = -1, -1
    if vmin == vmax:
        vmin = max(0.0, vmin - 0.01)
        vmax = min(1.0, vmax + 0.01)

    width = max(7.0, len(layers) * 0.85)
    height = max(3.2, len(positions) * 0.42)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Position")
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels([str(layer) for layer in layers])
    ax.set_yticks(np.arange(len(positions)))
    ax.set_yticklabels(positions)
    ax.set_xticks(np.arange(-0.5, len(layers), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(positions), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.7, alpha=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x]
            if not np.isfinite(value):
                continue
            text = f"{value:.3f}"
            weight = "bold" if y == best_y and x == best_x else "normal"
            ax.text(x, y, text, ha="center", va="center", fontsize=6.8, color="white", fontweight=weight)

    if best_y >= 0:
        ax.scatter([best_x], [best_y], marker="s", s=260, facecolors="none", edgecolors="white", linewidths=2.0)

    cbar = fig.colorbar(image, ax=ax, shrink=0.92)
    cbar.set_label(metric_title(metric))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_dataset(
    rows: list[dict[str, str]],
    metrics: list[str],
    positions: list[str],
    layers: list[int],
    out_dir: Path,
    prefix: str,
    title_prefix: str,
) -> list[dict[str, Any]]:
    summary = []
    for metric in metrics:
        if metric not in rows[0]:
            continue
        matrix = pivot(rows, metric, positions, layers)
        metric_dir = out_dir / prefix
        csv_path = metric_dir / f"{prefix}_{metric}_pivot.csv"
        png_path = metric_dir / f"{prefix}_{metric}_heatmap.png"
        write_pivot_csv(csv_path, matrix, positions, layers)
        draw_heatmap(matrix, positions, layers, f"{title_prefix}: {metric_title(metric)}", metric, png_path)
        finite = matrix[np.isfinite(matrix)]
        if finite.size:
            best_idx = np.nanargmin(matrix) if is_fpr(metric) else np.nanargmax(matrix)
            y, x = np.unravel_index(best_idx, matrix.shape)
            best_value = float(matrix[y, x])
            best_position = positions[y]
            best_layer = layers[x]
        else:
            best_value = math.nan
            best_position = ""
            best_layer = ""
        summary.append(
            {
                "prefix": prefix,
                "metric": metric,
                "best_position": best_position,
                "best_layer": best_layer,
                "best_value": best_value,
                "direction": "lower_is_better" if is_fpr(metric) else "higher_is_better",
                "png": str(png_path),
                "csv": str(csv_path),
            }
        )
    return summary


def write_summary(out_path: Path, rows: list[dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["prefix", "metric", "best_position", "best_layer", "best_value", "direction", "png", "csv"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pause_linear_tsv", required=True)
    parser.add_argument("--pause_mlp_tsv", default=None)
    parser.add_argument("--position_scan_tsv", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    all_summary: list[dict[str, Any]] = []

    pause_linear = read_tsv(Path(args.pause_linear_tsv))
    pause_positions = position_order(pause_linear, PAUSE_POSITION_ORDER)
    pause_layers = layer_order(pause_linear)
    all_summary.extend(
        plot_dataset(
            pause_linear,
            PAUSE_METRICS,
            pause_positions,
            pause_layers,
            out_dir,
            "pause_risk_linear",
            "PauseRiskProbe Linear",
        )
    )

    if args.pause_mlp_tsv:
        pause_mlp = read_tsv(Path(args.pause_mlp_tsv))
        all_summary.extend(
            plot_dataset(
                pause_mlp,
                PAUSE_METRICS,
                position_order(pause_mlp, PAUSE_POSITION_ORDER),
                layer_order(pause_mlp),
                out_dir,
                "pause_risk_mlp",
                "PauseRiskProbe MLP",
            )
        )

    position_scan = read_tsv(Path(args.position_scan_tsv))
    all_summary.extend(
        plot_dataset(
            position_scan,
            POSITION_SCAN_METRICS,
            position_order(position_scan, POSITION_SCAN_ORDER),
            layer_order(position_scan),
            out_dir,
            "position_scan_linear",
            "PositionScan TrajProbe Linear",
        )
    )

    write_summary(out_dir / "heatmap_summary.csv", all_summary)
    print(f"wrote {len(all_summary)} heatmaps to {out_dir}")


if __name__ == "__main__":
    main()

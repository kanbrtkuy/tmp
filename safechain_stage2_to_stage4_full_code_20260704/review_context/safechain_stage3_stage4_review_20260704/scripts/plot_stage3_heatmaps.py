#!/usr/bin/env python3
"""Plot token-by-layer heatmaps from stage3 single-scan summaries."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np


def canonical_token_order(
    insert_cot_offset: int,
    max_cot_offset: int,
    pre_pause_window: int = 3,
    post_pause_window: int = 3,
) -> list[tuple[str, tuple[str, ...]]]:
    """Return display labels in true token order with internal field aliases.

    `insert_cot_offset=k` means the pause run appears immediately before the
    original `cot_k` token. Internal `pre_pause_i` fields count backward from
    the first pause, and `post_pause_i` fields count forward from the last
    pause, so this mapping is deliberately offset-aware.
    """

    def cot_candidates(offset: int) -> tuple[str, ...]:
        candidates = [f"cot_{offset}"]
        if offset < insert_cot_offset:
            distance = insert_cot_offset - offset
            if 1 <= distance <= pre_pause_window:
                candidates.append(f"pre_pause_{distance}")
        else:
            distance = offset - insert_cot_offset + 1
            if 1 <= distance <= post_pause_window:
                candidates.append(f"post_pause_{distance}")
            # Legacy aliases point to post_pause_1/2, regardless of the
            # insertion offset. Keep them as last-resort aliases only.
            if distance == 1:
                candidates.append("control_cot_3")
            elif distance == 2:
                candidates.append("control_cot_4")
        return tuple(candidates)

    order: list[tuple[str, tuple[str, ...]]] = []
    for offset in range(0, insert_cot_offset):
        order.append((f"cot{offset}", cot_candidates(offset)))
    order.extend(
        [
            ("pause1", ("pause_0",)),
            ("pause2", ("pause_1",)),
            ("pause3", ("pause_2",)),
        ]
    )
    for offset in range(insert_cot_offset, max_cot_offset + 1):
        order.append((f"cot{offset}", cot_candidates(offset)))
    return order


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def build_matrix(
    rows: list[dict[str, str]],
    metric: str,
    token_order: list[tuple[str, tuple[str, ...]]],
) -> tuple[list[int], list[str], np.ndarray]:
    layers = sorted({int(row["layer"]) for row in rows})
    lookup = {(row["position"], int(row["layer"])): float(row[metric]) for row in rows}
    labels = [item[0] for item in token_order]
    matrix = np.full((len(layers), len(token_order)), np.nan, dtype=np.float64)
    for token_idx, (_, candidate_positions) in enumerate(token_order):
        for layer_idx, layer in enumerate(layers):
            for position in candidate_positions:
                if (position, layer) in lookup:
                    matrix[layer_idx, token_idx] = lookup[(position, layer)]
                    break
    return layers, labels, matrix


def plot_heatmap(
    matrix: np.ndarray,
    token_labels: list[str],
    layers: list[int],
    metric: str,
    title: str,
    output_path: Path,
    vmin: float | None,
    vmax: float | None,
) -> None:
    fig_w = max(10.0, len(token_labels) * 0.82)
    fig_h = max(6.5, len(layers) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Token position in actual sequence")
    ax.set_ylabel("Layer")
    ax.set_xticks(np.arange(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels([str(layer) for layer in layers])
    ax.set_xticks(np.arange(-0.5, len(token_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(layers), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.45, alpha=0.42)
    ax.tick_params(which="minor", bottom=False, left=False)

    max_idx = np.unravel_index(np.nanargmax(matrix), matrix.shape)
    ax.scatter(max_idx[1], max_idx[0], marker="s", s=120, facecolors="none", edgecolors="red", linewidths=1.6)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            color = "white" if vmin is None or value < (vmin + (vmax - vmin) * 0.55 if vmax is not None else 0.92) else "black"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=6.2, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(metric)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_top(rows: list[dict[str, str]], metrics: list[str], output_path: Path) -> None:
    lines = ["# Stage3 Single-Scan Top Positions", ""]
    for metric in metrics:
        ranked = sorted(rows, key=lambda row: float(row[metric]), reverse=True)
        lines.extend([f"## Top by `{metric}`", "", "| rank | position | layer | value |", "|---:|---|---:|---:|"])
        for rank, row in enumerate(ranked[:12], start=1):
            lines.append(f"| {rank} | {row['position']} | {row['layer']} | {float(row[metric]):.6f} |")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--prefix", default="stage3_single")
    parser.add_argument("--insert_cot_offset", type=int, default=4)
    parser.add_argument("--max_cot_offset", type=int, default=8)
    parser.add_argument("--vmin", type=float, default=0.84)
    parser.add_argument("--vmax", type=float, default=0.98)
    args = parser.parse_args()

    rows = read_rows(args.summary)
    metrics = ["test_auroc", "reasoningshield_test_auroc"]
    token_order = canonical_token_order(args.insert_cot_offset, args.max_cot_offset)

    for metric in metrics:
        layers, token_labels, matrix = build_matrix(rows, metric, token_order)
        plot_heatmap(
            matrix,
            token_labels,
            layers,
            metric,
            f"Stage3 Single Probe - {metric} (Actual Token Order, Pause Before Cot{args.insert_cot_offset})",
            args.output_dir / f"{args.prefix}_single_{metric}_heatmap.png",
            args.vmin,
            args.vmax,
        )

    write_top(rows, metrics, args.output_dir / f"{args.prefix}_single_top_positions.md")


if __name__ == "__main__":
    main()

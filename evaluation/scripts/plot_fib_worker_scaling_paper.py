#!/usr/bin/env python3
"""Paper figure: Fibonacci worker scaling (GTaP vs global queue vs sequential Chase-Lev).

Data: evaluation/1-worker_scalability/fib/fib_scaling_results.csv
Shows block_size=32 (t/b=32) only. Y-axis is logarithmic; X-axis is log2(thread count).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parents[1]
COMPARE_DIR = EVAL_DIR / "2-comparison"
IMG_DIR = EVAL_DIR / "img"
SCALING_DIR = EVAL_DIR / "1-worker_scalability" / "fib"
WORKSPACE_ROOT = EVAL_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from plot_style.gtap_colors import COL_GTAP_THREAD  # noqa: E402

FIG_WIDTH_IN = 239.75 / 72.0  # ≈ 3.33 in
PANEL_HEIGHT_RATIO = 0.50 * 1.1
AXES_TOP = 0.88
LEGEND_Y = 0.9
OUTPUT_FORMAT = "pdf"
DEFAULT_BLOCK_SIZE = 32

PAPER_RC = {
    "font.size": 8,
    "font.weight": "normal",
    "axes.labelsize": 8,
    "axes.labelweight": "normal",
    "axes.titlesize": 8,
    "axes.titleweight": "normal",
    "figure.labelsize": 8,
    "figure.labelweight": "normal",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.2,
}

SERIES = (
    ("ws", "Batched WS (GTaP)", COL_GTAP_THREAD, "o", "-"),
    ("gq", "Global queue", "#ff7f0e", "s", "--"),
    ("chaselev", "Non-batched WS (Chase-Lev)", "#2ca02c", "^", "-."),
)
IDEAL_SCALING_LABEL = "Ideal scaling"


def add_ideal_scaling(
    ax,
    threads: np.ndarray,
    times: np.ndarray,
    *,
    x_values: np.ndarray | None = None,
    label: str = IDEAL_SCALING_LABEL,
) -> None:
    threads = np.asarray(threads, dtype=float)
    times = np.asarray(times, dtype=float)
    valid = times > 0.0
    if not np.any(valid):
        return

    idx0 = int(np.where(valid)[0][0])
    p0 = float(threads[idx0])
    t0 = float(times[idx0])
    if t0 <= 0.0 or p0 <= 0.0:
        return

    if x_values is None:
        x_values = np.unique(threads[valid])
    x_values = np.asarray(sorted(x_values), dtype=float)
    y_values = (t0 * p0) / x_values

    ax.plot(
        x_values,
        y_values,
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
        color="black",
        label=label,
    )


def load_fib_scaling_data(csv_path: Path, *, block_size: int) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing scaling CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"block_size", "total_threads", "ws_med", "gq_med", "chaselev_med"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    data = df[df["block_size"] == block_size].copy()
    if data.empty:
        raise ValueError(f"No rows with block_size={block_size} in {csv_path}")

    return data.sort_values("total_threads").reset_index(drop=True)


def plot_fib_worker_scaling_paper(
    *,
    output_path: Path,
    csv_path: Path,
    block_size: int = DEFAULT_BLOCK_SIZE,
    save_dpi: int = 600,
) -> None:
    plt.rcParams.update(PAPER_RC)
    data = load_fib_scaling_data(csv_path, block_size=block_size)

    fig_height = FIG_WIDTH_IN * PANEL_HEIGHT_RATIO
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, fig_height))

    x = data["total_threads"].to_numpy(dtype=float)
    plotted_x: list[float] = []
    for key, label, color, marker, linestyle in SERIES:
        y = data[f"{key}_med"].to_numpy(dtype=float)
        valid = y > 0.0
        if not np.any(valid):
            continue
        plotted_x.extend(x[valid].tolist())
        ax.plot(
            x[valid],
            y[valid],
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=label,
            markerfacecolor="none",
            markeredgewidth=1.0,
            markeredgecolor=color,
        )

    gtap_y = data["ws_med"].to_numpy(dtype=float)
    add_ideal_scaling(
        ax,
        x,
        gtap_y,
        x_values=np.unique(np.asarray(plotted_x, dtype=float)),
    )

    ax.set_xlabel("Number of threads")
    ax.set_ylabel("Execution time (ms)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="major")
    ax.grid(False, which="minor")

    handles, labels = ax.get_legend_handles_labels()
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=AXES_TOP)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, LEGEND_Y),
        ncol=2,
        frameon=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {output_path} ({FIG_WIDTH_IN:.3f} x {fig_height:.3f} in)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fibonacci worker-scaling paper figure (t/b=32, log Y).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=SCALING_DIR / "fib_scaling_results.csv",
        help="fib_scaling_results.csv path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=IMG_DIR / f"fib_worker_scaling_tb{DEFAULT_BLOCK_SIZE}_paper.{OUTPUT_FORMAT}",
        help="Output PDF path",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=f"Threads per block t/b (default: {DEFAULT_BLOCK_SIZE})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="savefig dpi (default: 600)",
    )
    args = parser.parse_args()

    plot_fib_worker_scaling_paper(
        output_path=args.output.resolve(),
        csv_path=args.csv.resolve(),
        block_size=args.block_size,
        save_dpi=args.dpi,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paper figure: Fibonacci thread scaling and SM scaling side by side.

One ACM single-column row (width=\\linewidth) with two panels:
  (a) thread scaling — threads on X (log2)
  (b) SM scaling — SM count on X (log2; power-of-two ticks)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parents[2]
IMG_DIR = EVAL_DIR / "img"
THREAD_DIR = EVAL_DIR / "scheduler-comparison" / "fib-thread-scaling"
SM_DIR = EVAL_DIR / "scheduler-comparison" / "fib-sm-scaling"
sys.path.insert(0, str(EVAL_DIR))

from common.plotting.gtap_colors import COL_GTAP_THREAD  # noqa: E402

FIG_WIDTH_IN = 239.75 / 72.0  # ≈ 3.33 in (ACM single column)
_N_PANELS = 2
_COL_WIDTH_IN = FIG_WIDTH_IN / _N_PANELS
FIG_HEIGHT_IN = 1.55 * 1.25  # ≈ 1.94 in
OUTPUT_FORMAT = "pdf"
DEFAULT_BLOCK_SIZE = 32
PANEL_CAPTIONS = ("(a) Thread scaling", "(b) SM scaling")
CAPTION_AXES_Y = -0.40  # ~1 line below xlabel; near PDF bottom
LEGEND_GAP = 0.014
LEGEND_LABELSPACING = 0.45

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
    "legend.fontsize": 8,
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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

SERIES = (
    ("ws", "Batched WS (GTaP)", COL_GTAP_THREAD, "o", "-"),
    ("gq", "Global queue", "#ff7f0e", "s", "--"),
    ("chaselev", "Non-batched WS", "#2ca02c", "^", "-."),
)
IDEAL_SCALING_LABEL = "Ideal scaling"


def add_ideal_scaling(
    ax,
    x: np.ndarray,
    times: np.ndarray,
    *,
    x_values: np.ndarray | None = None,
    label: str = IDEAL_SCALING_LABEL,
) -> None:
    x = np.asarray(x, dtype=float)
    times = np.asarray(times, dtype=float)
    valid = times > 0.0
    if not np.any(valid):
        return

    idx0 = int(np.where(valid)[0][0])
    p0 = float(x[idx0])
    t0 = float(times[idx0])
    if t0 <= 0.0 or p0 <= 0.0:
        return

    if x_values is None:
        x_values = np.unique(x[valid])
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


def load_fib_thread_scaling_data(csv_path: Path, *, block_size: int) -> pd.DataFrame:
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


def load_fib_sm_scaling_data(csv_path: Path, *, block_size: int) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing scaling CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {
        "block_size",
        "sm_count",
        "observed_sms",
        "ws_med",
        "gq_med",
        "chaselev_med",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    data = df[df["block_size"] == block_size].copy()
    if data.empty:
        raise ValueError(f"No rows with block_size={block_size} in {csv_path}")

    mismatched = data[data["observed_sms"] != data["sm_count"]]
    if not mismatched.empty:
        raise ValueError(
            "observed SM count does not match sm_count for rows: "
            f"{mismatched[['sm_count', 'observed_sms']].to_dict('records')}"
        )

    return data.sort_values("sm_count").reset_index(drop=True)


def _plot_series(ax, x: np.ndarray, data) -> list[float]:
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
    return plotted_x


def _plot_thread_panel(ax, data) -> None:
    x = data["total_threads"].to_numpy(dtype=float)
    plotted_x = _plot_series(ax, x, data)
    add_ideal_scaling(
        ax,
        x,
        data["ws_med"].to_numpy(dtype=float),
        x_values=np.unique(np.asarray(plotted_x, dtype=float)),
    )
    ax.set_xlabel("Number of threads", labelpad=2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="major")
    ax.grid(False, which="minor")


def _plot_sm_panel(ax, data) -> None:
    x = data["sm_count"].to_numpy(dtype=float)
    plotted_x = _plot_series(ax, x, data)
    add_ideal_scaling(
        ax,
        x,
        data["ws_med"].to_numpy(dtype=float),
        x_values=np.unique(np.asarray(plotted_x, dtype=float)),
    )
    ax.set_xlabel("Number of SMs", labelpad=2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    # 132 sits on top of 128 on a log2 axis; label powers of two only.
    tick_sms = [
        value for value in x if abs(np.log2(value) - round(np.log2(value))) < 1e-6
    ]
    ax.set_xticks(tick_sms)
    ax.set_xticklabels([str(int(value)) for value in tick_sms])
    ax.grid(True, which="major")
    ax.grid(False, which="minor")


def plot_fib_scaling_row_paper(
    *,
    output_path: Path,
    thread_csv: Path,
    sm_csv: Path,
    block_size: int = DEFAULT_BLOCK_SIZE,
    save_dpi: int = 600,
) -> None:
    plt.rcParams.update(PAPER_RC)

    thread_data = load_fib_thread_scaling_data(thread_csv, block_size=block_size)
    sm_data = load_fib_sm_scaling_data(sm_csv, block_size=block_size)

    fig, axes = plt.subplots(
        1,
        _N_PANELS,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        sharey=True,
        gridspec_kw={"wspace": 0.28},
    )

    _plot_thread_panel(axes[0], thread_data)
    _plot_sm_panel(axes[1], sm_data)
    axes[0].set_ylabel("Execution time (ms)")

    fig.subplots_adjust(left=0.14, right=0.98, top=0.80, bottom=0.28, wspace=0.28)

    for ax, caption in zip(axes, PANEL_CAPTIONS):
        ax.text(
            0.5,
            CAPTION_AXES_Y,
            caption,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=plt.rcParams["axes.labelsize"],
            clip_on=False,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    axes_top = max(ax.get_position().y1 for ax in axes)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, axes_top + LEGEND_GAP),
        ncol=2,
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.4,
        columnspacing=1.0,
        labelspacing=LEGEND_LABELSPACING,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=save_dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(
        f"Saved: {output_path} ({FIG_WIDTH_IN:.3f} x {FIG_HEIGHT_IN:.3f} in; "
        f"column {_COL_WIDTH_IN:.3f} in)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fibonacci thread + SM scaling row (single-column paper figure).",
    )
    parser.add_argument(
        "--thread-csv",
        "--worker-csv",
        type=Path,
        dest="thread_csv",
        default=THREAD_DIR / "fib_thread_scaling_results.csv",
        help="Thread-scaling CSV path",
    )
    parser.add_argument(
        "--sm-csv",
        type=Path,
        default=SM_DIR / "fib_sm_scaling_results.csv",
        help="SM-scaling CSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=IMG_DIR / f"fib_thread_sm_scaling_row_paper.{OUTPUT_FORMAT}",
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

    plot_fib_scaling_row_paper(
        output_path=args.output.resolve(),
        thread_csv=args.thread_csv.resolve(),
        sm_csv=args.sm_csv.resolve(),
        block_size=args.block_size,
        save_dpi=args.dpi,
    )


if __name__ == "__main__":
    main()

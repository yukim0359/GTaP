#!/usr/bin/env python3
"""Paper figure: block/thread time ratio for full binary tree vs pruned B-ary tree.

Two side-by-side panels (3.33 in total width):
  Depth: x=D; compute: x=compute_iters (log2).
  Y = T_block/T_thread (parity at 1).

Data:
  binary_tree/binary_tree_performance_results.csv
  binary_tree/binary_tree_compute_results.csv           (compute sweep at D=20)
  prob_Bary_tree/tree_load_compute_performance_results.csv
  prob_Bary_tree/tree_load_compute_compute_results.csv  (compute sweep at D=32)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parents[1]
COMPARE_DIR = EVAL_DIR / "benchmarks"
IMG_DIR = EVAL_DIR / "img"
BINARY_TREE_DIR = COMPARE_DIR / "binary_tree"
PROB_BARY_TREE_DIR = COMPARE_DIR / "prob_Bary_tree"
WORKSPACE_ROOT = EVAL_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from plot_style.gtap_colors import COL_GTAP_BLOCK, COL_GTAP_THREAD  # noqa: E402

FIG_WIDTH_IN = 239.75 / 72.0  # ≈ 3.33 in
PANEL_HEIGHT_RATIO = 0.40
AXES_TOP = 0.91
LEGEND_GAP = 0.015
LEGEND_Y = AXES_TOP + LEGEND_GAP
OUTPUT_FORMAT = "pdf"
YLABEL = r"$T_\mathrm{block}/T_\mathrm{thread}$"
PARITY_Y = 1.0
Y_LOG_BASE = 2

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

TREE_SERIES = (
    ("binary", "Full binary tree", COL_GTAP_THREAD, "o", "-"),
    ("prob_bary", "Pruned B-ary tree", COL_GTAP_BLOCK, "s", "--"),
)


@dataclass(frozen=True)
class PanelSpec:
    key: str
    xlabel: str
    csv_by_tree: dict[str, Path]
    x_col: str
    x_log: bool = False


PANELS: tuple[PanelSpec, ...] = (
    PanelSpec(
        key="depth",
        xlabel="Depth",
        x_col="n",
        csv_by_tree={
            "binary": BINARY_TREE_DIR / "binary_tree_performance_results.csv",
            "prob_bary": PROB_BARY_TREE_DIR / "tree_load_compute_performance_results.csv",
        },
    ),
    PanelSpec(
        key="compute",
        xlabel="Compute iters",
        x_col="compute_iters",
        x_log=True,
        csv_by_tree={
            "binary": BINARY_TREE_DIR / "binary_tree_compute_results.csv",
            "prob_bary": PROB_BARY_TREE_DIR / "tree_load_compute_compute_results.csv",
        },
    ),
)


def _resolve_x_col(df: pd.DataFrame, x_col: str) -> str:
    if x_col in df.columns:
        return x_col
    if x_col == "n" and "depth" in df.columns:
        return "depth"
    raise ValueError(f"Missing x column {x_col!r} in CSV with columns={list(df.columns)}")


def load_ratio_series(csv_path: Path, *, x_col: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    x_name = _resolve_x_col(df, x_col)
    required = {x_name, "GTAP_block_med", "GTAP_thread_med"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    out = df[[x_name, "GTAP_block_med", "GTAP_thread_med"]].copy()
    out[x_name] = pd.to_numeric(out[x_name], errors="coerce")
    out["GTAP_block_med"] = pd.to_numeric(out["GTAP_block_med"], errors="coerce")
    out["GTAP_thread_med"] = pd.to_numeric(out["GTAP_thread_med"], errors="coerce")
    out = out.dropna()
    valid = (out["GTAP_block_med"] > 0.0) & (out["GTAP_thread_med"] > 0.0)
    out = out.loc[valid].sort_values(x_name).reset_index(drop=True)
    out["ratio"] = out["GTAP_block_med"] / out["GTAP_thread_med"]
    out = out.rename(columns={x_name: "x"})
    return out


def _plot_tree_curve(ax, data: pd.DataFrame, *, label: str, color: str, marker: str, linestyle: str) -> None:
    ax.plot(
        data["x"],
        data["ratio"],
        marker=marker,
        markevery=2,
        linestyle=linestyle,
        color=color,
        label=label,
        markerfacecolor="none",
        markeredgewidth=1.0,
        markeredgecolor=color,
    )


def plot_tree_block_thread_ratio_row_paper(
    *,
    output_path: Path,
    panels: tuple[PanelSpec, ...] = PANELS,
    save_dpi: int = 600,
) -> None:
    plt.rcParams.update(PAPER_RC)

    fig_height = FIG_WIDTH_IN * PANEL_HEIGHT_RATIO
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(FIG_WIDTH_IN, fig_height),
        sharey=True,
    )
    if len(panels) == 1:
        axes = [axes]

    legend_ax = axes[0]
    global_ymax = 0.0
    global_ymin = np.inf
    for ax, panel in zip(axes, panels):
        ax.axhline(
            PARITY_Y,
            color="0.35",
            linestyle="--",
            linewidth=0.9,
            zorder=0,
        )
        for tree_key, label, color, marker, linestyle in TREE_SERIES:
            csv_path = panel.csv_by_tree[tree_key]
            data = load_ratio_series(csv_path, x_col=panel.x_col)
            if panel.key == "compute":
                data = data[data["x"] >= 2**8].reset_index(drop=True)
            if not data.empty:
                global_ymax = max(global_ymax, float(data["ratio"].max()))
                global_ymin = min(global_ymin, float(data["ratio"].min()))
            _plot_tree_curve(
                ax,
                data,
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
            )

        ax.set_xlabel(panel.xlabel)
        if panel.x_log:
            ax.set_xscale("log", base=2)
        ax.grid(True, which="major")
        ax.grid(False, which="minor")

    # With shared y-axis, set limits once after plotting all panels.
    # Otherwise an early panel (depth) can cap the range and clip later panels.
    if global_ymax > 0.0 and np.isfinite(global_ymin):
        # Extra headroom/footroom to avoid marker clipping at both ends.
        ymin = min(global_ymin * 0.80, 0.80)
        ymax = max(global_ymax * 1.20, 1.20)
        for ax in axes:
            ax.set_yscale("log", base=Y_LOG_BASE)
            ax.set_ylim(ymin, ymax)
    else:
        for ax in axes:
            ax.set_yscale("log", base=Y_LOG_BASE)
            ax.set_ylim(0.5, 2.0)

    axes[0].set_ylabel(YLABEL)

    handles, labels = legend_ax.get_legend_handles_labels()
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.20, top=AXES_TOP, wspace=0.28)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, LEGEND_Y),
        ncol=2,
        frameon=False,
        borderaxespad=0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {output_path} ({FIG_WIDTH_IN:.3f} x {fig_height:.3f} in)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binary tree vs pruned B-ary tree block/thread ratio (paper row).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=IMG_DIR / f"tree_block_thread_ratio_row_paper.{OUTPUT_FORMAT}",
        help="Output PDF path",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="savefig dpi (default: 600)",
    )
    args = parser.parse_args()

    plot_tree_block_thread_ratio_row_paper(
        output_path=args.output.resolve(),
        save_dpi=args.dpi,
    )


if __name__ == "__main__":
    main()

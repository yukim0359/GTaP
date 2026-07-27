#!/usr/bin/env python3
"""Paper figure: binary-tree normalized sweeps (T / T_Thread) as a 3.33 in row.

Two side-by-side panels (shared y, one legend):
  left  — compute sweep (binary_tree_compute_results.csv)
  right — depth sweep   (binary_tree_performance_results.csv)

Matches binary_tree/plot_performance_tree.py normalized series:
  Thread, Block, Block (cutoff).
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
COMPARE_DIR = EVAL_DIR / "2-comparison"
IMG_DIR = EVAL_DIR / "img"
BINARY_TREE_DIR = COMPARE_DIR / "binary_tree"
WORKSPACE_ROOT = EVAL_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from plot_style.gtap_colors import COL_GTAP_BLOCK, COL_GTAP_THREAD, COL_OMP  # noqa: E402

FIG_WIDTH_IN = 239.75 / 72.0  # ≈ 3.33 in
PANEL_HEIGHT_RATIO = 0.40
AXES_TOP = 0.91
LEGEND_GAP = 0.015
LEGEND_Y = AXES_TOP + LEGEND_GAP
OUTPUT_FORMAT = "pdf"
YLABEL = r"$T / T_{\mathrm{Thread}}$"
Y_LOG_BASE = 2
BASELINE_PREFIX = "GTAP_thread"
COL_BLOCK_CUTOFF = COL_OMP

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
    ("GTAP_thread", "Thread", "o", COL_GTAP_THREAD),
    ("GTAP_block", "Block", "s", COL_GTAP_BLOCK),
    ("GTAP_block_cutoff", "Block (cutoff)", "D", COL_BLOCK_CUTOFF),
)


@dataclass(frozen=True)
class SweepSpec:
    key: str
    csv_path: Path
    x_col_candidates: tuple[str, ...]
    xlabel: str
    x_log2: bool = False


SWEEPS: tuple[SweepSpec, ...] = (
    SweepSpec(
        key="compute",
        csv_path=BINARY_TREE_DIR / "binary_tree_compute_results.csv",
        x_col_candidates=("compute_iters", "n"),
        xlabel="Compute iters",
        x_log2=True,
    ),
    SweepSpec(
        key="depth",
        csv_path=BINARY_TREE_DIR / "binary_tree_performance_results.csv",
        x_col_candidates=("depth", "n"),
        xlabel="Depth",
    ),
)


def _resolve_x_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(
        f"None of {candidates} found in CSV columns={list(df.columns)}"
    )


def _series_med_col(df: pd.DataFrame, prefix: str) -> str | None:
    for name in (f"{prefix}_med", f"{prefix}med"):
        if name in df.columns:
            return name
    return None


def load_sweep_df(spec: SweepSpec) -> tuple[pd.DataFrame, str]:
    if not spec.csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {spec.csv_path}")
    df = pd.read_csv(spec.csv_path)
    if df.empty:
        raise ValueError(f"Empty CSV: {spec.csv_path}")
    x_col = _resolve_x_col(df, spec.x_col_candidates)
    return df.sort_values(x_col).reset_index(drop=True), x_col


def _hollow_kwargs(color: str) -> dict:
    return dict(
        color=color,
        markerfacecolor="none",
        markeredgecolor=color,
        markeredgewidth=1.0,
    )


def _plot_ratio_series(ax: plt.Axes, df: pd.DataFrame, x_col: str, *, with_labels: bool) -> list[float]:
    """Plot T/T_Thread curves; return positive y values for shared ylim."""
    base_col = _series_med_col(df, BASELINE_PREFIX)
    if base_col is None:
        raise ValueError(f"Baseline column missing for prefix {BASELINE_PREFIX}")

    x = df[x_col].to_numpy(dtype=float)
    den = df[base_col].to_numpy(dtype=float)
    ys: list[float] = []

    for prefix, label, marker, color in SERIES:
        med_col = _series_med_col(df, prefix)
        if med_col is None:
            continue
        if prefix == BASELINE_PREFIX:
            valid = den > 0.0
            if not np.any(valid):
                continue
            y = np.ones(int(valid.sum()), dtype=float)
        else:
            num = df[med_col].to_numpy(dtype=float)
            valid = (num > 0.0) & (den > 0.0)
            if not np.any(valid):
                continue
            y = (num / den)[valid]
        ax.plot(
            x[valid],
            y,
            marker=marker,
            markevery=2,
            linestyle="-",
            label=label if with_labels else None,
            **_hollow_kwargs(color),
        )
        ys.extend(float(v) for v in y if np.isfinite(v) and v > 0.0)
    return ys


def plot_binary_tree_normalized_sweeps_row_paper(
    *,
    output_path: Path,
    sweeps: tuple[SweepSpec, ...] = SWEEPS,
    save_dpi: int = 600,
) -> None:
    plt.rcParams.update(PAPER_RC)

    fig_height = FIG_WIDTH_IN * PANEL_HEIGHT_RATIO
    fig, axes = plt.subplots(
        1,
        len(sweeps),
        figsize=(FIG_WIDTH_IN, fig_height),
        sharey=True,
    )
    if len(sweeps) == 1:
        axes = [axes]

    all_y: list[float] = []
    for i, (ax, spec) in enumerate(zip(axes, sweeps)):
        df, x_col = load_sweep_df(spec)
        ys = _plot_ratio_series(ax, df, x_col, with_labels=(i == 0))
        all_y.extend(ys)
        ax.set_xlabel(spec.xlabel)
        if spec.x_log2:
            ax.set_xscale("log", base=2)
        ax.grid(True, which="major")
        ax.grid(False, which="minor")

    if all_y:
        ymin = min(min(all_y) * 0.80, 0.80)
        ymax = max(max(all_y) * 1.20, 1.20)
    else:
        ymin, ymax = 0.5, 2.0
    for ax in axes:
        ax.set_yscale("log", base=Y_LOG_BASE)
        ax.set_ylim(ymin, ymax)

    axes[0].set_ylabel(YLABEL)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.20, top=AXES_TOP, wspace=0.28)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, LEGEND_Y),
        ncol=len(SERIES),
        frameon=False,
        borderaxespad=0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {output_path} ({FIG_WIDTH_IN:.3f} x {fig_height:.3f} in)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Binary-tree normalized depth/compute sweeps as a paper row "
            f"({FIG_WIDTH_IN:.3f} in wide)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=IMG_DIR / f"tree_normalized_sweeps_row_paper.{OUTPUT_FORMAT}",
        help="Output PDF path",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="savefig dpi (default: 600)",
    )
    args = parser.parse_args()

    plot_binary_tree_normalized_sweeps_row_paper(
        output_path=args.output.resolve(),
        save_dpi=args.dpi,
    )


if __name__ == "__main__":
    main()

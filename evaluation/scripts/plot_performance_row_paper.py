#!/usr/bin/env python3
"""Paper figure: fib / nq / cilksort performance (absolute time), one row of benchmarks.

Each column matches fib save_combined aspect: column_height = column_width * 0.85.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parents[1]
COMPARE_DIR = EVAL_DIR / "2-comparison"
IMG_DIR = EVAL_DIR / "img"
WORKSPACE_ROOT = EVAL_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from plot_style.gtap_colors import (  # noqa: E402
    COL_CILK,
    COL_DYNASOAR,
    COL_GTAP_THREAD,
    COL_OMP,
    COL_SEQ,
    LABEL_CPU_CILK,
    LABEL_CPU_OMP,
    LABEL_KIUCHI,
)

FIG_WIDTH_IN = 506.295 / 72.0  # ≈ 7.03 in (full paper width)
_COMBINED_HEIGHT_RATIO = 0.85  # fib save_combined: fig_height = fig_width * 0.85
_HEIGHT_SCALE = 1.1
OUTPUT_FORMAT = "pdf"
MARKER_SIZE = 4.0
MARK_EVERY_STEP = 2

# 8pt paper typography (no thesis_plt).
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
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.2,
}

XAxisKind = Literal["linear", "nq", "log"]


@dataclass(frozen=True)
class BenchSpec:
    key: str
    title: str
    xlabel: str
    x_axis: XAxisKind
    log_tick_values: bool = False


BENCHMARKS: tuple[BenchSpec, ...] = (
    BenchSpec("fib", "Fibonacci", "Fibonacci Number", "linear"),
    BenchSpec("nq", "N-Queens", "N-Queens Board Size", "nq"),
    BenchSpec("cilksort", "CilkSort", "Array Size", "log"),
)

_COL_WIDTH_IN = FIG_WIDTH_IN / len(BENCHMARKS)
FIG_HEIGHT_IN = _COL_WIDTH_IN * _COMBINED_HEIGHT_RATIO * _HEIGHT_SCALE

YLABEL_ABSOLUTE = "Execution Time (ms)"


def _hollow_marker_kwargs(color, *, marker_every_offset: int = 0):
    return dict(
        color=color,
        markersize=MARKER_SIZE,
        markerfacecolor="none",
        markeredgecolor=color,
        markeredgewidth=1.0,
        markevery=(marker_every_offset, MARK_EVERY_STEP),
    )


def _linestyle_for(col_prefix: str) -> str:
    return "-" if col_prefix.upper().startswith("GTAP") else "--"


def _series_plot_kwargs(color, col_prefix: str, *, marker_every_offset: int = 0):
    base_lw = plt.rcParams.get("lines.linewidth", 1.5)
    is_gtap = col_prefix.upper().startswith("GTAP")
    return dict(
        **_hollow_marker_kwargs(color, marker_every_offset=marker_every_offset),
        linestyle=_linestyle_for(col_prefix),
        linewidth=base_lw if is_gtap else max(0.5, base_lw - 0.8),
    )


def _legend_label(text: str, *, use_legend: bool) -> str:
    return text if use_legend else "_nolegend_"


def plot_with_iqr(
    ax, d, label, marker, col_prefix, *, color=None, x_col="n", marker_every_offset: int = 0,
):
    y = d[f"{col_prefix}_med"].to_numpy()
    x = d[x_col].to_numpy()
    kw = _series_plot_kwargs(color, col_prefix, marker_every_offset=marker_every_offset)
    ax.plot(x, y, marker=marker, label=label, **kw)


def _method_frames(sub: pd.DataFrame, *, include_seq: bool = False):
    gtap = sub[sub["GTAP_med"] > 0].copy() if "GTAP_med" in sub.columns else pd.DataFrame()
    omp = sub[sub["OMP_med"] > 0].copy() if "OMP_med" in sub.columns else pd.DataFrame()
    cilk = sub[sub["CILK_med"] > 0].copy() if "CILK_med" in sub.columns else pd.DataFrame()
    dynasoar = sub[sub["DYNASOAR_med"] > 0].copy() if "DYNASOAR_med" in sub.columns else pd.DataFrame()
    if include_seq and "SEQ_med" in sub.columns:
        seq = sub[sub["SEQ_med"] > 0].copy()
    else:
        seq = pd.DataFrame()
    return gtap, omp, cilk, dynasoar, seq


def plot_absolute_panel(
    ax,
    sub: pd.DataFrame,
    *,
    include_seq: bool = False,
    use_legend: bool = False,
):
    gtap_df, omp_df, cilk_df, dynasoar_df, seq_df = _method_frames(sub, include_seq=include_seq)
    if not gtap_df.empty:
        plot_with_iqr(
            ax, gtap_df, _legend_label("GTaP", use_legend=use_legend),
            "o", "GTAP", color=COL_GTAP_THREAD, marker_every_offset=0,
        )
    if not dynasoar_df.empty:
        plot_with_iqr(
            ax, dynasoar_df, _legend_label(LABEL_KIUCHI, use_legend=use_legend),
            "v", "DYNASOAR", color=COL_DYNASOAR, marker_every_offset=1,
        )
    if not omp_df.empty:
        plot_with_iqr(
            ax, omp_df, _legend_label(LABEL_CPU_OMP, use_legend=use_legend),
            "s", "OMP", color=COL_OMP, marker_every_offset=0,
        )
    if not cilk_df.empty:
        plot_with_iqr(
            ax, cilk_df, _legend_label(LABEL_CPU_CILK, use_legend=use_legend),
            "D", "CILK", color=COL_CILK, marker_every_offset=1,
        )
    if not seq_df.empty:
        plot_with_iqr(
            ax, seq_df, _legend_label("CPU Sequential", use_legend=use_legend),
            "^", "SEQ", color=COL_SEQ, marker_every_offset=0,
        )
    ax.set_yscale("log")
    ax.grid(True, alpha=plt.rcParams["grid.alpha"])


def configure_x_axis_nq(ax, sub: pd.DataFrame):
    n_min = int(sub["n"].min())
    n_max = int(sub["n"].max())
    tick_start = ((n_min + 4) // 5) * 5
    xticks = np.arange(tick_start, n_max + 1, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(n)) for n in xticks])


def configure_x_axis_log(
    ax,
    sub: pd.DataFrame,
    *,
    log_tick_values: bool = False,
    log_pad_frac: float = 0.06,
):
    n_vals = np.sort(sub["n"].unique().astype(float))
    n_min = float(np.min(n_vals))
    n_max = float(np.max(n_vals))
    log_min = np.log10(n_min)
    log_max = np.log10(n_max)
    span = max(log_max - log_min, 1e-9)
    pad = span * log_pad_frac
    ax.set_xscale("log")
    ax.set_xlim(10.0 ** (log_min - pad), 10.0 ** (log_max + pad))
    if log_tick_values:
        ax.set_xticks(n_vals)
        ax.set_xticklabels([str(int(v)) for v in n_vals])
    ax.margins(x=0)


def _configure_x_axis(ax, sub: pd.DataFrame, bench: BenchSpec) -> None:
    if bench.x_axis == "nq":
        configure_x_axis_nq(ax, sub)
    elif bench.x_axis == "log":
        configure_x_axis_log(ax, sub, log_tick_values=bench.log_tick_values)


def _load_benchmark_csv(bench: BenchSpec) -> pd.DataFrame:
    csv_path = COMPARE_DIR / bench.key / f"{bench.key}_performance_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    return pd.read_csv(csv_path).sort_values("n")


def plot_performance_row_paper(
    *,
    output_path: Path,
    include_seq: bool = False,
):
    plt.rcParams.update(PAPER_RC)

    fig, axes = plt.subplots(
        1,
        len(BENCHMARKS),
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        sharex=False,
        gridspec_kw={"wspace": 0.28},
    )
    if len(BENCHMARKS) == 1:
        axes = np.array([axes])

    legend_ax = None
    for col, bench in enumerate(BENCHMARKS):
        sub = _load_benchmark_csv(bench)
        use_legend = col == 0
        ax = axes[col]
        if use_legend:
            legend_ax = ax

        plot_absolute_panel(ax, sub, include_seq=include_seq, use_legend=use_legend)

        ax.set_title(bench.title, pad=2)
        ax.set_xlabel(bench.xlabel)
        _configure_x_axis(ax, sub, bench)

    axes[0].set_ylabel(YLABEL_ABSOLUTE)

    handles, labels = legend_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(labels),
        frameon=False,
    )

    fig.subplots_adjust(left=0.09, right=0.99, top=0.80, bottom=0.18)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(
        f"Saved: {output_path} ({FIG_WIDTH_IN:.3f} x {FIG_HEIGHT_IN:.3f} in; "
        f"column {_COL_WIDTH_IN:.3f} in @ aspect {_COMBINED_HEIGHT_RATIO * _HEIGHT_SCALE:.3f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-benchmark performance row (fib, nq, cilksort): absolute time.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=IMG_DIR / f"microbenchmark_performance_comparison.{OUTPUT_FORMAT}",
        help="Output PDF path",
    )
    parser.add_argument(
        "--include-seq",
        action="store_true",
        help="Include CPU sequential series (default: off).",
    )
    args = parser.parse_args()

    plot_performance_row_paper(
        output_path=args.output.resolve(),
        include_seq=args.include_seq,
    )


if __name__ == "__main__":
    main()

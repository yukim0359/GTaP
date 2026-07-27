#!/usr/bin/env python3
"""Paper figure: DAQ normalized time T_DAQ/T_1queue for fib / nq / cilksort.

Reads DAQ experiment CSVs from evaluation/daq/{fib,nq,cilksort}/ and plots one row of
three panels at ACM single-column width (3.33 in) with 7pt typography.
"""

from __future__ import annotations

import argparse
import re
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
DAQ_DIR = EVAL_DIR / "daq"
WORKSPACE_ROOT = EVAL_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from plot_style.gtap_colors import COL_GTAP_THREAD  # noqa: E402

FIG_WIDTH_IN = 239.75 / 72.0  # ≈ 3.33 in (ACM single column)
_COMBINED_HEIGHT_RATIO = 1.0
_HEIGHT_SCALE = 1.22
PANEL_SCALE_X = 0.90
PANEL_SCALE_Y = 0.8
PANEL_ANCHOR_FRAC = 0.42
OUTPUT_FORMAT = "pdf"
YLABEL = r"$T_\mathrm{DAQ}/T_\mathrm{1queue}$"
PARITY_Y = 1.0
Y_TOP_MIN = 1.1
Y_TOP_PAD = 1.05

PAPER_RC = {
    "font.size": 7,
    "font.weight": "normal",
    "axes.labelsize": 7,
    "axes.labelweight": "normal",
    "axes.titlesize": 7,
    "axes.titleweight": "normal",
    "figure.labelsize": 7,
    "figure.labelweight": "normal",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
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

XAxisKind = Literal["linear", "log"]


@dataclass(frozen=True)
class BenchSpec:
    key: str
    title: str
    problem_size: str
    xlabel: str
    x_axis: XAxisKind
    log_tick_values: bool = False
    log_tick_pow2: bool = False
    markevery: int | None = None
    x_tick_stride: int = 1


BENCHMARKS: tuple[BenchSpec, ...] = (
    BenchSpec("fib", "Fibonacci", r"$(n=40)$", "Cutoff", "linear", markevery=2),
    BenchSpec("nq", "N-Queens", r"$(n=16)$", "Cutoff", "linear"),
    BenchSpec(
        "cilksort",
        "CilkSort",
        r"$(n=5\times10^{7})$",
        "Cutoff",
        "log",
        log_tick_values=True,
        log_tick_pow2=True,
        x_tick_stride=2,
    ),
)

_COL_WIDTH_IN = FIG_WIDTH_IN / len(BENCHMARKS)
FIG_HEIGHT_IN = _COL_WIDTH_IN * _COMBINED_HEIGHT_RATIO * _HEIGHT_SCALE

_QUEUE_MED_RE = re.compile(r"^(?P<n>\d+)queue_med$")


def _daq_med_column(df: pd.DataFrame) -> str:
    candidates = [
        col
        for col in df.columns
        if _QUEUE_MED_RE.match(col) and col != "1queue_med"
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one DAQ queue column in {sorted(df.columns)}; "
            f"found {candidates!r}"
        )
    return candidates[0]


def _default_csv(bench: BenchSpec) -> Path:
    return DAQ_DIR / bench.key / f"epaq_performance_results_{bench.key}.csv"


def load_daq_ratio_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    if "cutoff" not in df.columns:
        raise ValueError(f"Missing cutoff column in {csv_path}")
    if "1queue_med" not in df.columns:
        raise ValueError(f"Missing 1queue_med column in {csv_path}")

    daq_col = _daq_med_column(df)
    out = df[["cutoff", "1queue_med", daq_col]].copy()
    out["cutoff"] = pd.to_numeric(out["cutoff"], errors="coerce")
    out["1queue_med"] = pd.to_numeric(out["1queue_med"], errors="coerce")
    out[daq_col] = pd.to_numeric(out[daq_col], errors="coerce")
    out = out.dropna()
    valid = (out["1queue_med"] > 0.0) & (out[daq_col] > 0.0)
    out = out.loc[valid].sort_values("cutoff").reset_index(drop=True)
    out["ratio"] = out[daq_col] / out["1queue_med"]
    return out


def configure_x_axis_log(
    ax,
    sub: pd.DataFrame,
    *,
    log_tick_values: bool = False,
    log_tick_pow2: bool = False,
    x_tick_stride: int = 1,
    log_pad_frac: float = 0.06,
) -> None:
    x_vals = np.sort(sub["cutoff"].unique().astype(float))
    x_min = float(np.min(x_vals))
    x_max = float(np.max(x_vals))
    tick_vals = x_vals[::x_tick_stride]

    if log_tick_pow2:
        exp_min = np.log2(x_min)
        exp_max = np.log2(x_max)
        span = max(exp_max - exp_min, 1e-9)
        pad = span * log_pad_frac
        ax.set_xscale("log", base=2)
        ax.set_xlim(2.0 ** (exp_min - pad), 2.0 ** (exp_max + pad))
        if log_tick_values:
            ax.set_xticks(tick_vals)
            ax.set_xticklabels(
                [rf"$2^{{{int(round(np.log2(v)))}}}$" for v in tick_vals]
            )
    else:
        log_min = np.log10(x_min)
        log_max = np.log10(x_max)
        span = max(log_max - log_min, 1e-9)
        pad = span * log_pad_frac
        ax.set_xscale("log")
        ax.set_xlim(10.0 ** (log_min - pad), 10.0 ** (log_max + pad))
        if log_tick_values:
            ax.set_xticks(tick_vals)
            ax.set_xticklabels([str(int(v)) for v in tick_vals])
    ax.margins(x=0)


def _configure_x_axis(ax, sub: pd.DataFrame, bench: BenchSpec) -> None:
    if bench.x_axis == "log":
        configure_x_axis_log(
            ax,
            sub,
            log_tick_values=bench.log_tick_values,
            log_tick_pow2=bench.log_tick_pow2,
            x_tick_stride=bench.x_tick_stride,
        )


def _plot_ratio_panel(ax, sub: pd.DataFrame, *, markevery: int | None = None) -> None:
    x = sub["cutoff"].to_numpy()
    y = sub["ratio"].to_numpy()
    plot_kw: dict = dict(
        color=COL_GTAP_THREAD,
        markerfacecolor="none",
        markeredgewidth=1.0,
        markeredgecolor=COL_GTAP_THREAD,
    )
    if markevery is not None:
        plot_kw["markevery"] = markevery
    ax.plot(
        x,
        y,
        "o-",
        **plot_kw,
    )
    ax.axhline(
        PARITY_Y,
        color="0.35",
        linestyle="--",
        linewidth=0.9,
        zorder=0,
    )
    y_top = max(Y_TOP_MIN, float(np.max(y)) * Y_TOP_PAD)
    ax.set_ylim(0.0, y_top)
    ax.grid(True, which="major")
    ax.grid(False, which="minor")


def _scale_axes_positions(
    axes,
    *,
    scale_x: float,
    scale_y: float,
    anchor_frac: float,
) -> None:
    """Shrink each panel; bias free space upward for titles, downward for x labels."""
    for ax in np.ravel(axes):
        pos = ax.get_position()
        w = pos.width * scale_x
        h = pos.height * scale_y
        cx = pos.x0 + pos.width * 0.5
        freed = pos.height - h
        new_y0 = pos.y0 + freed * anchor_frac
        ax.set_position([cx - w * 0.5, new_y0, w, h])


def plot_daq_ratio_row_paper(
    *,
    output_path: Path,
    benchmarks: tuple[BenchSpec, ...] = BENCHMARKS,
    save_dpi: int = 300,
) -> None:
    plt.rcParams.update(PAPER_RC)

    fig, axes = plt.subplots(
        1,
        len(benchmarks),
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        sharey=False,
        gridspec_kw={"wspace": 0.28},
    )
    if len(benchmarks) == 1:
        axes = np.array([axes])

    for ax, bench in zip(axes, benchmarks):
        csv_path = _default_csv(bench)
        sub = load_daq_ratio_csv(csv_path)
        _plot_ratio_panel(ax, sub, markevery=bench.markevery)
        ax.set_title(f"{bench.title}\n{bench.problem_size}", pad=1)
        ax.set_xlabel(bench.xlabel)
        _configure_x_axis(ax, sub, bench)

    axes[0].set_ylabel(YLABEL)

    fig.subplots_adjust(left=0.14, right=0.99, top=0.92, bottom=0.22)
    _scale_axes_positions(
        axes,
        scale_x=PANEL_SCALE_X,
        scale_y=PANEL_SCALE_Y,
        anchor_frac=PANEL_ANCHOR_FRAC,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=save_dpi)
    plt.close(fig)
    print(
        f"Saved: {output_path} ({FIG_WIDTH_IN:.3f} x {FIG_HEIGHT_IN:.3f} in; "
        f"column {_COL_WIDTH_IN:.3f} in)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DAQ ratio row (fib, nq, cilksort): T_DAQ/T_1queue.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=IMG_DIR / f"daq_ratio_row_paper.{OUTPUT_FORMAT}",
        help="Output PDF path",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="savefig dpi (default: 300)",
    )
    args = parser.parse_args()

    plot_daq_ratio_row_paper(
        output_path=args.output.resolve(),
        save_dpi=args.dpi,
    )


if __name__ == "__main__":
    main()

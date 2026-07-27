#!/usr/bin/env python3
"""Paper figure: FMM DTT vs theta for N=10^7 and N=5×10^7 side by side.

Each column is the combined absolute + normalized layout from
fmm/plot_performance_fmm_dtt_theta.py (top: absolute, bottom: normalized).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

EVAL_DIR = Path(__file__).resolve().parents[1]
COMPARE_DIR = EVAL_DIR / "2-comparison"
IMG_DIR = EVAL_DIR / "img"
FMM_DIR = COMPARE_DIR / "fmm"
WORKSPACE_ROOT = EVAL_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from plot_style.gtap_colors import (  # noqa: E402
    COL_GPU_THRUST,
    COL_GTAP_THREAD,
    COL_OMP,
    LABEL_CPU_OMP,
    LABEL_RATIO_OMP,
)

FIG_WIDTH_IN = 239.75 / 72.0  # ≈ 3.33 in (ACM single column)
_COMBINED_HEIGHT_RATIO = 0.85
_HEIGHT_SCALE = 1.75
OUTPUT_FORMAT = "pdf"
MARKER_SIZE = 3.5
LEGEND_FONT_SIZE = 7
MERGE_KEYS = ["n", "theta"]

PANELS: tuple[tuple[int, str], ...] = (
    (10_000_000, r"$N=10^7$"),
    (50_000_000, r"$N=5\times10^7$"),
)

_COL_WIDTH_IN = FIG_WIDTH_IN / len(PANELS)
FIG_HEIGHT_IN = _COL_WIDTH_IN * _COMBINED_HEIGHT_RATIO * _HEIGHT_SCALE

YLABEL_ABSOLUTE = "DTT core time (ms)"
YLABEL_NORMALIZED = r"Normalized time" + "\n" + r"($T_\mathrm{method}/T_\mathrm{GTaP}$)"
XLABEL = r"$\theta$ (MAC)"

LABEL_GTAP = "GTaP"
LABEL_WORKLIST = "GPU worklist"

PAPER_FONT_SIZE = 7

PAPER_RC = {
    "font.size": PAPER_FONT_SIZE,
    "font.weight": "normal",
    "axes.labelsize": PAPER_FONT_SIZE,
    "axes.labelweight": "normal",
    "axes.titlesize": PAPER_FONT_SIZE,
    "axes.titleweight": "normal",
    "figure.labelsize": PAPER_FONT_SIZE,
    "figure.labelweight": "normal",
    "xtick.labelsize": PAPER_FONT_SIZE,
    "ytick.labelsize": PAPER_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
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


def _hollow_marker_kwargs(color):
    return dict(
        color=color,
        markersize=MARKER_SIZE,
        markerfacecolor="none",
        markeredgecolor=color,
        markeredgewidth=1.0,
    )


def _linestyle_for(col_prefix: str) -> str:
    return "-" if col_prefix.upper().startswith("GTAP") else "--"


def _series_plot_kwargs(color, col_prefix: str):
    base_lw = plt.rcParams.get("lines.linewidth", 1.5)
    is_gtap = col_prefix.upper().startswith("GTAP")
    return dict(
        **_hollow_marker_kwargs(color),
        linestyle=_linestyle_for(col_prefix),
        linewidth=base_lw if is_gtap else max(0.5, base_lw - 0.8),
    )


def _legend_label(text: str, *, use_legend: bool) -> str:
    return text if use_legend else "_nolegend_"


def plot_series(ax, x, y, marker, col_prefix, color, label):
    kw = _series_plot_kwargs(color, col_prefix)
    ax.plot(x, y, marker=marker, label=label, **kw)


def plot_with_iqr(
    ax,
    d,
    label,
    marker,
    col_prefix,
    *,
    color=None,
    x_col="theta",
):
    y = d[f"{col_prefix}_med"].to_numpy()
    x = d[x_col].to_numpy()
    kw = _series_plot_kwargs(color, col_prefix)
    ax.plot(x, y, marker=marker, label=label, **kw)


def _method_frames(sub: pd.DataFrame):
    gtap = sub[sub["GTAP_med"] > 0].copy() if "GTAP_med" in sub.columns else pd.DataFrame()
    wl = sub[sub["WORKLIST_med"] > 0].copy() if "WORKLIST_med" in sub.columns else pd.DataFrame()
    omp = sub[sub["OMP_med"] > 0].copy() if "OMP_med" in sub.columns else pd.DataFrame()
    return gtap, wl, omp


def plot_absolute_panel(ax, sub: pd.DataFrame, *, use_legend: bool = False):
    gtap_df, wl_df, omp_df = _method_frames(sub)
    if not gtap_df.empty:
        plot_with_iqr(
            ax, gtap_df, _legend_label(LABEL_GTAP, use_legend=use_legend),
            "o", "GTAP", color=COL_GTAP_THREAD,
        )
    if not wl_df.empty:
        plot_with_iqr(
            ax, wl_df, _legend_label(LABEL_WORKLIST, use_legend=use_legend),
            "^", "WORKLIST", color=COL_GPU_THRUST,
        )
    if not omp_df.empty:
        plot_with_iqr(
            ax, omp_df, _legend_label(LABEL_CPU_OMP, use_legend=use_legend),
            "s", "OMP", color=COL_OMP,
        )
    ax.set_yscale("log")
    ax.grid(True, alpha=plt.rcParams["grid.alpha"])


def _attach_normalized_method(
    ratio_df: pd.DataFrame,
    base: pd.DataFrame,
    method_df: pd.DataFrame,
    prefix: str,
    ratio_col: str,
) -> pd.DataFrame:
    if method_df.empty:
        return ratio_df

    cols = MERGE_KEYS + [f"{prefix}_med"]
    merged = pd.merge(base, method_df[cols], on=MERGE_KEYS, how="inner")
    if merged.empty:
        return ratio_df

    g = merged["GTAP_med"].to_numpy(dtype=float)
    m = merged[f"{prefix}_med"].to_numpy(dtype=float)
    add = merged[MERGE_KEYS].copy()
    add[ratio_col] = m / g
    return pd.merge(ratio_df, add, on=MERGE_KEYS, how="outer")


def build_ratio_df(sub: pd.DataFrame) -> pd.DataFrame:
    gtap_df, wl_df, omp_df = _method_frames(sub)
    if gtap_df.empty:
        return pd.DataFrame()

    base = gtap_df[MERGE_KEYS + ["GTAP_med"]]
    ratio_df = gtap_df[MERGE_KEYS].copy()
    ratio_df = _attach_normalized_method(ratio_df, base, wl_df, "WORKLIST", "worklist_ratio")
    ratio_df = _attach_normalized_method(ratio_df, base, omp_df, "OMP", "omp_ratio")
    return ratio_df.sort_values(MERGE_KEYS)


def plot_ratio_series(
    ax,
    ratio_df,
    label,
    marker,
    ratio_col,
    col_prefix,
    color,
):
    d = ratio_df.dropna(subset=[ratio_col]).copy()
    if d.empty:
        return
    x = d["theta"].to_numpy(dtype=float)
    y = d[ratio_col].to_numpy(dtype=float)
    kw = _series_plot_kwargs(color, col_prefix)
    ax.plot(x, y, marker=marker, label=label, **kw)


def plot_normalized_panel(ax, ratio_df: pd.DataFrame):
    x = ratio_df["theta"].to_numpy(dtype=float)
    plot_series(
        ax, x, np.ones_like(x), "o", "GTAP", COL_GTAP_THREAD,
        _legend_label("GTaP (parity)", use_legend=False),
    )
    if "worklist_ratio" in ratio_df.columns:
        plot_ratio_series(
            ax, ratio_df, _legend_label(r"GPU worklist / GTaP", use_legend=False), "^",
            "worklist_ratio", "WORKLIST", COL_GPU_THRUST,
        )
    if "omp_ratio" in ratio_df.columns:
        plot_ratio_series(
            ax, ratio_df, _legend_label(LABEL_RATIO_OMP, use_legend=False), "s",
            "omp_ratio", "OMP", COL_OMP,
        )
    ax.set_yscale("linear")
    ax.grid(True, alpha=plt.rcParams["grid.alpha"])
    _, ymax = ax.get_ylim()
    ax.set_ylim(0.0, ymax * 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(2))


def _load_results_csv() -> pd.DataFrame:
    csv_path = FMM_DIR / "fmm_dtt_theta_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df["theta"] = df["theta"].astype(float)
    rename = {
        "GTAP_dtt_med": "GTAP_med",
        "GTAP_dtt_err_low": "GTAP_err_low",
        "GTAP_dtt_err_high": "GTAP_err_high",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def plot_fmm_dtt_theta_row_paper(*, output_path: Path) -> None:
    plt.rcParams.update(PAPER_RC)
    df = _load_results_csv()

    fig, axes = plt.subplots(
        2,
        len(PANELS),
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        sharex="col",
        gridspec_kw={"height_ratios": [2.0, 1.03], "hspace": 0.32, "wspace": 0.34},
    )

    legend_ax = None
    for col, (n_val, title) in enumerate(PANELS):
        sub = df[df["n"] == n_val].sort_values("theta")
        if sub.empty:
            raise ValueError(f"No data for N={n_val}")
        ratio_df = build_ratio_df(sub)
        if ratio_df.empty:
            raise ValueError(f"No GTaP data for N={n_val}")

        use_legend = col == 0
        ax_abs = axes[0, col]
        ax_norm = axes[1, col]
        if use_legend:
            legend_ax = ax_abs

        plot_absolute_panel(ax_abs, sub, use_legend=use_legend)
        plot_normalized_panel(ax_norm, ratio_df)

        ax_abs.set_title(title, pad=2)
        ax_norm.set_xlabel(XLABEL)

    axes[0, 0].set_ylabel(YLABEL_ABSOLUTE)
    axes[1, 0].set_ylabel(YLABEL_NORMALIZED)

    handles, labels = legend_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(labels),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )

    fig.subplots_adjust(left=0.15, right=0.99, top=0.85, bottom=0.14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(
        f"Saved: {output_path} ({FIG_WIDTH_IN:.3f} x {FIG_HEIGHT_IN:.3f} in; "
        f"column {_COL_WIDTH_IN:.3f} in @ aspect "
        f"{_COMBINED_HEIGHT_RATIO * _HEIGHT_SCALE:.3f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FMM DTT vs theta row (N=10^7, N=5×10^7): absolute + normalized.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=IMG_DIR / f"fmm_dtt_theta_combined_row.{OUTPUT_FORMAT}",
        help="Output PDF path",
    )
    args = parser.parse_args()
    plot_fmm_dtt_theta_row_paper(output_path=args.output.resolve())


if __name__ == "__main__":
    main()

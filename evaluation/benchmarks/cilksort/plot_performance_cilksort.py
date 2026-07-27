#!/usr/bin/env python3
"""Plot Cilksort: absolute time and GTaP-normalized time (combined figure)."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[4]))
plt.style.use(["~/plot_style/thesis_plt.mplstyle"])
from plot_style.gtap_colors import (
    COL_CILK,
    COL_DYNASOAR,
    COL_GTAP_THREAD,
    COL_OMP,
    COL_SEQ,
    LABEL_CPU_CILK,
    LABEL_CPU_OMP,
    LABEL_GTAP_THREAD,
    LABEL_KIUCHI,
    LABEL_RATIO_CILK,
    LABEL_RATIO_KIUCHI,
    LABEL_RATIO_OMP,
)

BENCHMARK_NAME = "cilksort"
OUTPUT_FORMAT = "pdf"
MARKER_SIZE = plt.rcParams.get("lines.markersize", 6.0) * 1.2
MERGE_KEYS = ["n"]

csv_path = Path(f"{BENCHMARK_NAME}_performance_results.csv")
img_dir = Path("img")
img_dir.mkdir(parents=True, exist_ok=True)

if not csv_path.exists():
    raise SystemExit(f"Missing CSV: {csv_path} (run ./compare_cilksort.sh first)")

df = pd.read_csv(csv_path)


def _hollow_marker_kwargs(color):
    return dict(
        color=color,
        markersize=MARKER_SIZE,
        markerfacecolor="none",
        markeredgecolor=color,
        markeredgewidth=1.2,
    )


def _linestyle_for(col_prefix: str) -> str:
    return "-" if col_prefix.upper().startswith("GTAP") else "--"


def _series_plot_kwargs(color, col_prefix: str):
    base_lw = plt.rcParams.get("lines.linewidth")
    is_gtap = col_prefix.upper().startswith("GTAP")
    return dict(
        **_hollow_marker_kwargs(color),
        linestyle=_linestyle_for(col_prefix),
        linewidth=base_lw if is_gtap else max(0.5, base_lw - 0.8),
    )


def plot_series(ax, x, y, marker, col_prefix, color, label):
    kw = _series_plot_kwargs(color, col_prefix)
    ax.plot(x, y, marker=marker, label=label, **kw)


def plot_with_iqr(ax, d, label, marker, col_prefix, *, color=None, x_col="n"):
    y = d[f"{col_prefix}_med"].to_numpy()
    x = d[x_col].to_numpy()
    kw = _series_plot_kwargs(color, col_prefix)

    low_col = f"{col_prefix}_err_low"
    high_col = f"{col_prefix}_err_high"

    if low_col in d.columns and high_col in d.columns:
        low = d[low_col].to_numpy(dtype=float)
        high = d[high_col].to_numpy(dtype=float)
        low[(low < 0) | (high < 0)] = np.nan
        yerr = np.vstack([low, high])
        ax.errorbar(
            x, y, yerr=yerr,
            fmt=marker,
            capsize=3, elinewidth=1,
            label=label,
            **kw,
        )
    else:
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


def configure_x_axis(ax, sub: pd.DataFrame, *, log_pad_frac: float = 0.06):
    """Log-x limits with a small outward pad (like linear autoscale margins on fib)."""
    n_vals = sub["n"].to_numpy(dtype=float)
    n_min = float(np.min(n_vals))
    n_max = float(np.max(n_vals))
    log_min = np.log10(n_min)
    log_max = np.log10(n_max)
    span = max(log_max - log_min, 1e-9)
    pad = span * log_pad_frac
    ax.set_xscale("log")
    ax.set_xlim(10.0 ** (log_min - pad), 10.0 ** (log_max + pad))
    ax.margins(x=0)


def plot_absolute_panel(ax, sub: pd.DataFrame, *, include_seq: bool = False):
    gtap_df, omp_df, cilk_df, dynasoar_df, seq_df = _method_frames(sub, include_seq=include_seq)
    if not gtap_df.empty:
        plot_with_iqr(ax, gtap_df, LABEL_GTAP_THREAD, "o", "GTAP", color=COL_GTAP_THREAD)
    if not dynasoar_df.empty:
        plot_with_iqr(ax, dynasoar_df, LABEL_KIUCHI, "v", "DYNASOAR", color=COL_DYNASOAR)
    if not omp_df.empty:
        plot_with_iqr(ax, omp_df, LABEL_CPU_OMP, "s", "OMP", color=COL_OMP)
    if not cilk_df.empty:
        plot_with_iqr(ax, cilk_df, LABEL_CPU_CILK, "D", "CILK", color=COL_CILK)
    if not seq_df.empty:
        plot_with_iqr(ax, seq_df, "CPU Sequential", "^", "SEQ", color=COL_SEQ)
    ax.set_ylabel("Execution Time (ms)")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)


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
    if f"{prefix}_err_low" in method_df.columns:
        cols.extend([f"{prefix}_err_low", f"{prefix}_err_high"])
    merged = pd.merge(base, method_df[cols], on=MERGE_KEYS, how="inner")
    if merged.empty:
        return ratio_df

    g = merged["GTAP_med"].to_numpy(dtype=float)
    m = merged[f"{prefix}_med"].to_numpy(dtype=float)
    add = merged[MERGE_KEYS].copy()
    ratio = m / g
    add[ratio_col] = ratio
    if f"{prefix}_err_low" in merged.columns:
        d_lo = merged[f"{prefix}_err_low"].to_numpy(dtype=float)
        d_hi = merged[f"{prefix}_err_high"].to_numpy(dtype=float)
        add[f"{ratio_col}_err_low"] = ratio - (m - d_lo) / g
        add[f"{ratio_col}_err_high"] = (m + d_hi) / g - ratio
    return pd.merge(ratio_df, add, on=MERGE_KEYS, how="outer")


def build_ratio_df(sub: pd.DataFrame, *, include_seq: bool = False) -> pd.DataFrame:
    gtap_df, omp_df, cilk_df, dynasoar_df, seq_df = _method_frames(sub, include_seq=include_seq)
    if gtap_df.empty:
        return pd.DataFrame()

    base = gtap_df[MERGE_KEYS + ["GTAP_med"]]
    ratio_df = gtap_df[MERGE_KEYS].copy()
    ratio_df = _attach_normalized_method(ratio_df, base, omp_df, "OMP", "omp_ratio")
    ratio_df = _attach_normalized_method(ratio_df, base, cilk_df, "CILK", "cilk_ratio")
    ratio_df = _attach_normalized_method(ratio_df, base, dynasoar_df, "DYNASOAR", "dynasoar_ratio")
    if include_seq:
        ratio_df = _attach_normalized_method(ratio_df, base, seq_df, "SEQ", "seq_ratio")
    return ratio_df.sort_values(MERGE_KEYS)


def plot_ratio_with_iqr(ax, ratio_df, label, marker, ratio_col, col_prefix, color):
    d = ratio_df.dropna(subset=[ratio_col]).copy()
    if d.empty:
        return
    x = d["n"].to_numpy(dtype=float)
    y = d[ratio_col].to_numpy(dtype=float)
    kw = _series_plot_kwargs(color, col_prefix)

    lo_col = f"{ratio_col}_err_low"
    hi_col = f"{ratio_col}_err_high"
    if lo_col in d.columns and hi_col in d.columns:
        low = d[lo_col].to_numpy(dtype=float)
        high = d[hi_col].to_numpy(dtype=float)
        low[(low < 0) | (high < 0)] = np.nan
        yerr = np.vstack([low, high])
        ax.errorbar(
            x, y, yerr=yerr,
            fmt=marker,
            capsize=3, elinewidth=1,
            label=label,
            **kw,
        )
    else:
        ax.plot(x, y, marker=marker, label=label, **kw)


def plot_normalized_panel(ax, ratio_df: pd.DataFrame, *, include_seq: bool = False):
    x = ratio_df["n"].to_numpy(dtype=float)
    plot_series(
        ax, x, np.ones_like(x), "o", "GTAP", COL_GTAP_THREAD,
        "GTaP (parity)",
    )
    if "dynasoar_ratio" in ratio_df.columns:
        plot_ratio_with_iqr(
            ax, ratio_df, LABEL_RATIO_KIUCHI, "v",
            "dynasoar_ratio", "DYNASOAR", COL_DYNASOAR,
        )
    if "omp_ratio" in ratio_df.columns:
        plot_ratio_with_iqr(
            ax, ratio_df, LABEL_RATIO_OMP, "s",
            "omp_ratio", "OMP", COL_OMP,
        )
    if "cilk_ratio" in ratio_df.columns:
        plot_ratio_with_iqr(
            ax, ratio_df, LABEL_RATIO_CILK, "D",
            "cilk_ratio", "CILK", COL_CILK,
        )
    if include_seq and "seq_ratio" in ratio_df.columns:
        plot_ratio_with_iqr(
            ax, ratio_df, r"CPU Seq / GTaP", "^",
            "seq_ratio", "SEQ", COL_SEQ,
        )
    ax.set_xlabel("Array Size (n)")
    ax.set_ylabel(
        r"Normalized time" + "\n" + r"($T_\mathrm{method}/T_\mathrm{GTaP}$)"
    )
    ax.set_yscale("log", base=2)
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)


def save_absolute_only(sub: pd.DataFrame, *, include_seq: bool = False):
    fig, ax = plt.subplots()
    plot_absolute_panel(ax, sub, include_seq=include_seq)
    configure_x_axis(ax, sub)
    ax.set_xlabel("Array Size (n)")
    plt.tight_layout()
    out_path = img_dir / f"{BENCHMARK_NAME}_performance_comparison.{OUTPUT_FORMAT}"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_combined(sub: pd.DataFrame, *, include_seq: bool = False):
    ratio_df = build_ratio_df(sub, include_seq=include_seq)
    if ratio_df.empty:
        print("Warning: skip combined plot (no GTaP data)")
        return

    _w, _h = plt.rcParams.get("figure.figsize", [6.4, 4.8])
    fig_height = _w * 0.85
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True,
        figsize=(_w, fig_height),
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    plot_absolute_panel(ax_top, sub, include_seq=include_seq)
    plot_normalized_panel(ax_bot, ratio_df, include_seq=include_seq)
    configure_x_axis(ax_bot, sub)

    plt.tight_layout()
    out_path = img_dir / f"{BENCHMARK_NAME}_performance_combined.{OUTPUT_FORMAT}"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Cilksort performance (GTaP vs CPU baselines).")
    ap.add_argument(
        "--include-seq",
        action="store_true",
        help="Include CPU sequential series (default: off).",
    )
    args = ap.parse_args()

    sub = df.sort_values("n")
    save_absolute_only(sub, include_seq=args.include_seq)
    save_combined(sub, include_seq=args.include_seq)


if __name__ == "__main__":
    main()

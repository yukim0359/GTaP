#!/usr/bin/env python3
"""Plot DTT vs theta: absolute time and GTaP-normalized time (cilksort-style combined figure)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[4]))
plt.style.use(["~/plot_style/thesis_plt.mplstyle"])
from plot_style.gtap_colors import COL_GTAP_THREAD, COL_OMP

COL_WORKLIST = "#9467bd"

BENCHMARK_TAG = "fmm_dtt_theta"
OUTPUT_FORMAT = "pdf"
MARKER_SIZE = plt.rcParams.get("lines.markersize", 6.0) * 1.2
LEGEND_FONTSIZE = 14

MERGE_KEYS = ["n", "theta"]

csv_path = Path(f"{BENCHMARK_TAG}_results.csv")
img_dir = Path("img")
img_dir.mkdir(parents=True, exist_ok=True)

if not csv_path.exists():
    raise SystemExit(f"Missing CSV: {csv_path} (run ./compare_fmm_dtt_theta.sh first)")

df = pd.read_csv(csv_path)
df["theta"] = df["theta"].astype(float)

# Backward compatibility with old CSV column names.
_rename = {
    "GTAP_dtt_med": "GTAP_med",
    "GTAP_dtt_err_low": "GTAP_err_low",
    "GTAP_dtt_err_high": "GTAP_err_high",
}
df = df.rename(columns={k: v for k, v in _rename.items() if k in df.columns})


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
    """Line/marker styling shared by absolute and normalized panels."""
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


def plot_with_iqr(ax, d, label, marker, col_prefix, *, color=None, x_col="theta"):
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


def _method_frames(sub: pd.DataFrame):
    gtap = sub[sub["GTAP_med"] > 0].copy() if "GTAP_med" in sub.columns else pd.DataFrame()
    wl = sub[sub["WORKLIST_med"] > 0].copy() if "WORKLIST_med" in sub.columns else pd.DataFrame()
    omp = sub[sub["OMP_med"] > 0].copy() if "OMP_med" in sub.columns else pd.DataFrame()
    return gtap, wl, omp


def plot_absolute_panel(ax, sub: pd.DataFrame, *, show_legend: bool = True):
    gtap_df, wl_df, omp_df = _method_frames(sub)
    if not gtap_df.empty:
        plot_with_iqr(ax, gtap_df, "GTaP DTT", "o", "GTAP", color=COL_GTAP_THREAD)
    if not wl_df.empty:
        plot_with_iqr(ax, wl_df, "GPU worklist DTT", "D", "WORKLIST", color=COL_WORKLIST)
    if not omp_df.empty:
        plot_with_iqr(ax, omp_df, "Host OpenMP DTT", "^", "OMP", color=COL_OMP)
    ax.set_ylabel("DTT core time (ms)")
    ax.set_yscale("log")
    ax.grid(True)
    if show_legend:
        ax.legend(loc="best", fontsize=LEGEND_FONTSIZE)


def _attach_normalized_method(
    ratio_df: pd.DataFrame,
    base: pd.DataFrame,
    method_df: pd.DataFrame,
    prefix: str,
    ratio_col: str,
) -> pd.DataFrame:
    """Add T_method/T_GTaP and IQR errors scaled by GTAP_med."""
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
        # err_* are median-relative deltas from compare_fmm_dtt_theta.sh
        add[f"{ratio_col}_err_low"] = ratio - (m - d_lo) / g
        add[f"{ratio_col}_err_high"] = (m + d_hi) / g - ratio
    return pd.merge(ratio_df, add, on=MERGE_KEYS, how="outer")


def build_ratio_df(sub: pd.DataFrame) -> pd.DataFrame:
    """Per-theta ratios T_method / T_GTaP (GTaP = 1); errors divided by GTAP_med."""
    gtap_df, wl_df, omp_df = _method_frames(sub)
    if gtap_df.empty:
        return pd.DataFrame()

    base = gtap_df[MERGE_KEYS + ["GTAP_med"]]
    ratio_df = gtap_df[MERGE_KEYS].copy()
    ratio_df = _attach_normalized_method(ratio_df, base, wl_df, "WORKLIST", "worklist_ratio")
    ratio_df = _attach_normalized_method(ratio_df, base, omp_df, "OMP", "omp_ratio")
    return ratio_df.sort_values(MERGE_KEYS)


def plot_ratio_with_iqr(ax, ratio_df, label, marker, ratio_col, col_prefix, color):
    """Plot normalized ratio with IQR errors already scaled by T_GTaP."""
    d = ratio_df.dropna(subset=[ratio_col]).copy()
    if d.empty:
        return
    x = d["theta"].to_numpy(dtype=float)
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


def plot_normalized_panel(ax, ratio_df: pd.DataFrame, *, show_legend: bool = True):
    x = ratio_df["theta"].to_numpy(dtype=float)
    plot_series(
        ax, x, np.ones_like(x), "o", "GTAP", COL_GTAP_THREAD,
        "GTaP (parity)",
    )
    if "worklist_ratio" in ratio_df.columns:
        plot_ratio_with_iqr(
            ax, ratio_df, r"GPU worklist / GTaP", "D",
            "worklist_ratio", "WORKLIST", COL_WORKLIST,
        )
    if "omp_ratio" in ratio_df.columns:
        plot_ratio_with_iqr(
            ax, ratio_df, r"Host OpenMP / GTaP", "^",
            "omp_ratio", "OMP", COL_OMP,
        )
    ax.set_xlabel(r"$\theta$ (MAC)")
    ax.set_ylabel(
        r"Normalized time" + "\n" + r"($T_\mathrm{method}/T_\mathrm{GTaP}$)"
    )
    ax.set_yscale("log", base=2)
    ax.grid(True)
    if show_legend:
        ax.legend(loc="best", fontsize=LEGEND_FONTSIZE)


def save_absolute_only(sub: pd.DataFrame, n: int):
    fig, ax = plt.subplots()
    plot_absolute_panel(ax, sub)
    ax.set_xlabel(r"$\theta$ (MAC)")
    plt.tight_layout()
    out_path = img_dir / f"{BENCHMARK_TAG}_comparison_N{n}.{OUTPUT_FORMAT}"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_combined(sub: pd.DataFrame, n: int):
    ratio_df = build_ratio_df(sub)
    if ratio_df.empty:
        print(f"Warning: skip combined plot for N={n} (no GTaP data)")
        return

    _w, _h = plt.rcParams.get("figure.figsize", [6.4, 4.8])
    fig_height = _w * 0.85
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True,
        figsize=(_w, fig_height),
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    plot_absolute_panel(ax_top, sub, show_legend=False)
    plot_normalized_panel(ax_bot, ratio_df, show_legend=False)

    handles, labels = ax_top.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(3, len(handles)),
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
    )

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    out_path = img_dir / f"{BENCHMARK_TAG}_combined_N{n}.{OUTPUT_FORMAT}"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


for n in sorted(df["n"].unique()):
    sub = df[df["n"] == n].sort_values("theta")
    save_absolute_only(sub, int(n))
    save_combined(sub, int(n))

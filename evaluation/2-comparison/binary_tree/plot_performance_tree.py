#!/usr/bin/env python3
"""
binary_tree performance plots (thread / block / block-cutoff):
- absolute execution time (log scale)
- standalone relative time T / T_thread
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

plt.style.use(
    [
        "~/plot_style/thesis_plt.mplstyle",
    ]
)

sys.path.append(str(Path(__file__).resolve().parents[4]))
from plot_style.gtap_colors import (  # noqa: E402
    COL_GTAP_THREAD,
    COL_GTAP_BLOCK,
    COL_OMP,
)

BENCHMARK_NAME = "binary_tree"
OUTPUT_FORMAT = "pdf"
MARKER_SIZE = plt.rcParams.get("lines.markersize", 6.0) * 1.2

IMG_DIR = Path("img")
IMG_DIR.mkdir(parents=True, exist_ok=True)

COL_BLOCK_CUTOFF = COL_OMP

SERIES = [
    ("GTAP_thread", "Thread", "o", COL_GTAP_THREAD),
    ("GTAP_block", "Block", "s", COL_GTAP_BLOCK),
    ("GTAP_block_cutoff", "Block (cutoff)", "D", COL_BLOCK_CUTOFF),
]


def _col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _hollow_marker_kwargs(color: Optional[str] = None):
    kw = dict(
        markersize=MARKER_SIZE,
        markerfacecolor="none",
        markeredgewidth=1.2,
    )
    if color is not None:
        kw.update(color=color, markeredgecolor=color)
    return kw


def plot_with_iqr(
    ax: plt.Axes,
    x: np.ndarray,
    med: np.ndarray,
    err_low: Optional[np.ndarray],
    err_high: Optional[np.ndarray],
    label: str,
    marker: str,
    color: Optional[str] = None,
):
    med = np.asarray(med, dtype=float)
    x = np.asarray(x)
    valid = med > 0
    if not np.any(valid):
        return
    x = x[valid]
    med = med[valid]
    if err_low is not None and err_high is not None:
        err_low = np.asarray(err_low, dtype=float)[valid]
        err_high = np.asarray(err_high, dtype=float)[valid]
        err_low[err_low == 0] = np.nan
        yerr = np.vstack([err_low, err_high])
        ax.errorbar(
            x,
            med,
            yerr=yerr,
            fmt=marker,
            linestyle="-",
            capsize=3,
            elinewidth=1,
            label=label,
            **_hollow_marker_kwargs(color),
        )
    else:
        ax.plot(x, med, marker, linestyle="-", label=label, **_hollow_marker_kwargs(color))


def _get_series(df: pd.DataFrame, prefix: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    med = _col(df, f"{prefix}_med", f"{prefix}med")
    elo = _col(df, f"{prefix}_err_low", f"{prefix}_elo")
    ehi = _col(df, f"{prefix}_err_high", f"{prefix}_ehi")
    return med, elo, ehi


def make_time_plot(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    out_stem: str,
    *,
    title: Optional[str] = None,
    y_time_log: bool = True,
):
    df = df.copy().sort_values(x_col)
    x = df[x_col].to_numpy()

    series_cols = []
    for prefix, label, marker, color in SERIES:
        med, elo, ehi = _get_series(df, prefix)
        series_cols.append((prefix, label, marker, color, med, elo, ehi))

    if all(med is None for *_, med, _, _ in series_cols):
        print(f"Warning: No recognized timing columns in {out_stem}, skipping.")
        return

    fig, ax = plt.subplots()
    for prefix, label, marker, color, med, elo, ehi in series_cols:
        if med is None:
            continue
        plot_with_iqr(
            ax,
            x,
            df[med],
            df[elo] if elo else None,
            df[ehi] if ehi else None,
            label,
            marker,
            color=color,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Execution Time (ms)")
    if y_time_log:
        ax.set_yscale("log")
    ax.grid(True)
    ax.legend(fontsize="small")
    if title:
        ax.set_title(title)

    plt.tight_layout()
    out_path = IMG_DIR / f"{out_stem}.{OUTPUT_FORMAT}"
    plt.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.close(fig)


def make_ratio_plot(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    out_stem: str,
    *,
    title: Optional[str] = None,
    y_log: bool = True,
    log_base: int = 2,
    baseline_prefix: str = "GTAP_thread",
):
    df = df.copy().sort_values(x_col)
    x = df[x_col].to_numpy()

    base_med, _, _ = _get_series(df, baseline_prefix)
    if base_med is None:
        print(f"Warning: baseline {baseline_prefix} missing in {out_stem}, skipping.")
        return

    den = df[base_med].to_numpy(dtype=float)
    fig, ax = plt.subplots()
    plotted = False

    for prefix, label, marker, color in SERIES:
        med, _, _ = _get_series(df, prefix)
        if med is None:
            continue
        if prefix == baseline_prefix:
            valid = den > 0
            if not np.any(valid):
                continue
            y = np.ones_like(den[valid], dtype=float)
        else:
            num = df[med].to_numpy(dtype=float)
            valid = (num > 0) & (den > 0)
            if not np.any(valid):
                continue
            y = (num / den)[valid]
        ax.plot(
            x[valid],
            y,
            marker,
            linestyle="-",
            label=label,
            **_hollow_marker_kwargs(color),
        )
        plotted = True

    if not plotted:
        print(f"Warning: No ratio series in {out_stem}, skipping.")
        plt.close(fig)
        return

    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$T / T_{\mathrm{Thread}}$")
    if y_log:
        ax.set_yscale("log", base=log_base)
    ax.grid(True)
    ax.legend(fontsize="small")
    if title:
        ax.set_title(title)

    plt.tight_layout()
    out_path = IMG_DIR / f"{out_stem}.{OUTPUT_FORMAT}"
    plt.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.close(fig)


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"Warning: {path} not found, skipping.")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"Warning: {path} is empty, skipping.")
        return None
    return df


def main():
    df_depth = _read_csv(Path(f"{BENCHMARK_NAME}_performance_results.csv"))
    if df_depth is not None:
        x_col = "depth" if "depth" in df_depth.columns else ("n" if "n" in df_depth.columns else df_depth.columns[0])
        make_time_plot(
            df_depth,
            x_col=x_col,
            x_label="Tree Maximum Depth",
            out_stem="tree_depth_performance_combined",
            title="Varying D (mem_ops=0, compute_iters=2048)",
        )
        make_ratio_plot(
            df_depth,
            x_col=x_col,
            x_label="Tree Maximum Depth",
            out_stem="tree_depth_normalized",
            title="Normalized vs Thread (D sweep)",
        )

    df_comp = _read_csv(Path(f"{BENCHMARK_NAME}_compute_results.csv"))
    if df_comp is not None:
        x_col = (
            "compute_iters"
            if "compute_iters" in df_comp.columns
            else ("n" if "n" in df_comp.columns else df_comp.columns[0])
        )
        make_time_plot(
            df_comp,
            x_col=x_col,
            x_label="Compute Iterations",
            out_stem="tree_compute_performance_combined",
            title="Varying compute_iters (D=25, mem_ops=0)",
        )
        make_ratio_plot(
            df_comp,
            x_col=x_col,
            x_label="Compute Iterations",
            out_stem="tree_compute_normalized",
            title="Normalized vs Thread (compute sweep, D=25)",
        )

    print("\nAll plots generated!")


if __name__ == "__main__":
    main()

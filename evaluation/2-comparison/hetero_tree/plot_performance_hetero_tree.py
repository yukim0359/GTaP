#!/usr/bin/env python3
"""
hetero_tree performance plots (depth + compute_iters):
- absolute execution time (log scale), 4 impls
- standalone relative time T / T_thread(wo DAQ)

Compute CSVs may be tagged by K:
  hetero_tree_compute_results_K2.csv
  hetero_tree_compute_results_K4.csv
Legacy untagged hetero_tree_compute_results.csv is also accepted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import re

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
    COL_CILK,
    COL_OMP,
)

BENCHMARK_NAME = "hetero_tree"
OUTPUT_FORMAT = "pdf"
MARKER_SIZE = plt.rcParams.get("lines.markersize", 6.0) * 1.2

IMG_DIR = Path("img")
IMG_DIR.mkdir(parents=True, exist_ok=True)

COL_THREAD_DAQ = COL_CILK
COL_BLOCK_CUTOFF = COL_OMP

SERIES = [
    ("GTAP_thread", "Thread (wo DAQ)", "o", COL_GTAP_THREAD),
    ("GTAP_thread_daq", "Thread (DAQ)", "^", COL_THREAD_DAQ),
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
    """Standalone T / T_thread(wo DAQ) plot."""
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
    ax.set_ylabel(r"$T / T_{\mathrm{Thread\ (wo\ DAQ)}}$")
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


def _k_from_compute_csv(path: Path) -> Optional[int]:
    m = re.search(r"_K(\d+)\.csv$", path.name)
    if m:
        return int(m.group(1))
    return None


def _iter_compute_csvs() -> list[tuple[Path, Optional[int]]]:
    """Prefer K-tagged CSVs; fall back to legacy untagged file."""
    tagged = sorted(Path(".").glob(f"{BENCHMARK_NAME}_compute_results_K*.csv"))
    out: list[tuple[Path, Optional[int]]] = []
    for p in tagged:
        out.append((p, _k_from_compute_csv(p)))
    if out:
        return out
    legacy = Path(f"{BENCHMARK_NAME}_compute_results.csv")
    if legacy.exists():
        return [(legacy, None)]
    return []


def _plot_compute_csv(path: Path, k: Optional[int]) -> None:
    df = _read_csv(path)
    if df is None:
        return
    x_col = (
        "compute_iters"
        if "compute_iters" in df.columns
        else ("n" if "n" in df.columns else df.columns[0])
    )
    k_tag = f"K={k}" if k is not None else "K=?"
    stem_suffix = f"_K{k}" if k is not None else ""
    make_time_plot(
        df,
        x_col=x_col,
        x_label="Compute Iterations",
        out_stem=f"hetero_tree_compute_performance_combined{stem_suffix}",
        title=f"Varying compute_iters (D=25 fixed, {k_tag})",
    )
    make_ratio_plot(
        df,
        x_col=x_col,
        x_label="Compute Iterations",
        out_stem=f"hetero_tree_compute_normalized{stem_suffix}",
        title=f"Normalized vs Thread wo DAQ (compute sweep, {k_tag})",
    )


def main():
    df_depth = _read_csv(Path(f"{BENCHMARK_NAME}_performance_results.csv"))
    if df_depth is not None:
        x_col = "depth" if "depth" in df_depth.columns else ("n" if "n" in df_depth.columns else df_depth.columns[0])
        make_time_plot(
            df_depth,
            x_col=x_col,
            x_label="Tree Maximum Depth",
            out_stem="hetero_tree_depth_performance_combined",
            title="Varying D (compute_iters=2048 fixed, K=2)",
        )
        make_ratio_plot(
            df_depth,
            x_col=x_col,
            x_label="Tree Maximum Depth",
            out_stem="hetero_tree_depth_normalized",
            title="Normalized vs Thread wo DAQ (D sweep, K=2)",
        )

    compute_csvs = _iter_compute_csvs()
    if not compute_csvs:
        print(f"Warning: no {BENCHMARK_NAME}_compute_results*.csv found.")
    for path, k in compute_csvs:
        print(f"Plotting compute CSV: {path} ({'K=' + str(k) if k is not None else 'untagged'})")
        _plot_compute_csv(path, k)

    print("\nAll plots generated!")


if __name__ == "__main__":
    main()

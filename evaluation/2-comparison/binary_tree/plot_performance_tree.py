#!/usr/bin/env python3
"""
tree_load_compute performance plots (3 variants):
- depth variation
- memory-ops variation (optionally multiple depths)
- compute-iters variation (optionally multiple depths)

For each variant, generate a *combined* figure (stacked vertically):
- top: execution time (log scale)
- bottom: normalized time vs GTaP (thread/block) with parity line
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

plt.style.use([
    "~/plot_style/thesis_plt.mplstyle",
])

# Allow importing `/work/gc64/c64099/plot_style/gtap_colors.py` when run from this repo
# This file lives under: .../gtap/evaluation/2-comparison/tree_load_compute/
# parents[4] == /work/gc64/c64099
sys.path.append(str(Path(__file__).resolve().parents[4]))
from plot_style.gtap_colors import COL_GTAP_THREAD, COL_GTAP_BLOCK, COL_OMP, COL_SEQ

BENCHMARK_NAME = "binary_tree"
OUTPUT_FORMAT = "pdf"  # "png" or "pdf"

IMG_DIR = Path("img")
IMG_DIR.mkdir(parents=True, exist_ok=True)


def _col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def plot_with_iqr(
    ax: plt.Axes,
    x: np.ndarray,
    med: np.ndarray,
    err_low: Optional[np.ndarray],
    err_high: Optional[np.ndarray],
    label: str,
    marker: str,
    color: Optional[str] = None,
    linestyle: str = "-",
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
            linestyle=linestyle,
            capsize=3,
            elinewidth=1,
            label=label,
            color=color,
        )
    else:
        ax.plot(x, med, marker + linestyle, label=label, color=color)


def _get_series(df: pd.DataFrame, prefix: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    med = _col(df, f"{prefix}_med", f"{prefix}med")
    elo = _col(df, f"{prefix}_err_low", f"{prefix}_elo")
    ehi = _col(df, f"{prefix}_err_high", f"{prefix}_ehi")
    return med, elo, ehi


def make_combined_plot(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    out_stem: str,
    *,
    title: Optional[str] = None,
    y_time_log: bool = True,
    y_norm_log: bool = True,
    norm_log_base: int = 2,
    legend_loc_bottom: str = "lower right",
):
    df = df.copy()
    df = df.sort_values(x_col)

    x = df[x_col].to_numpy()

    # detect columns (support both old/new naming)
    gtap_block_med, gtap_block_elo, gtap_block_ehi = _get_series(df, "GTAP_block")
    gtap_thread_med, gtap_thread_elo, gtap_thread_ehi = _get_series(df, "GTAP_thread")
    omp_med, omp_elo, omp_ehi = _get_series(df, "OMP")

    if gtap_block_med is None and gtap_thread_med is None and omp_med is None:
        print(f"Warning: No recognized timing columns in {out_stem}, skipping.")
        return

    _w, _h = plt.rcParams.get("figure.figsize", [6.4, 4.8])
    fig_height = _w * 0.7
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(_w, fig_height),
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    # --- Top: execution time ---
    if gtap_thread_med is not None:
        plot_with_iqr(
            ax_top,
            x,
            df[gtap_thread_med],
            df[gtap_thread_elo] if gtap_thread_elo else None,
            df[gtap_thread_ehi] if gtap_thread_ehi else None,
            "GTaP (Thread-level)",
            "o",
            color=COL_GTAP_THREAD,
        )

    if gtap_block_med is not None:
        plot_with_iqr(
            ax_top,
            x,
            df[gtap_block_med],
            df[gtap_block_elo] if gtap_block_elo else None,
            df[gtap_block_ehi] if gtap_block_ehi else None,
            "GTaP (Block-level)",
            "s",
            color=COL_GTAP_BLOCK,
        )

    if omp_med is not None:
        plot_with_iqr(
            ax_top,
            x,
            df[omp_med],
            df[omp_elo] if omp_elo else None,
            df[omp_ehi] if omp_ehi else None,
            "CPU OpenMP",
            "^",
            color=COL_OMP,
        )

    ax_top.set_ylabel("Execution Time (ms)")
    if y_time_log:
        ax_top.set_yscale("log")
    ax_top.grid(True)
    ax_top.legend()
    if title:
        ax_top.set_title(title)

    # --- Bottom: normalized time (relative to CPU OpenMP) ---
    # We normalize GTaP (thread/block) against CPU OpenMP so that
    # values < 1 mean "GTaP is faster than OpenMP".
    label_gtap_thr = r"GTaP (Thread) / CPU OpenMP"
    label_gtap_blk = r"GTaP (Block) / CPU OpenMP"

    has_bottom = False

    # GTaP (Thread) normalized by CPU OpenMP.
    if omp_med is not None and gtap_thread_med is not None:
        num = df[gtap_thread_med].to_numpy(dtype=float)
        den = df[omp_med].to_numpy(dtype=float)
        valid = (num > 0) & (den > 0)
        if np.any(valid):
            ax_bot.plot(x[valid], (num / den)[valid], "o-", label=label_gtap_thr, color=COL_GTAP_THREAD)
            has_bottom = True

    # GTaP (Block) normalized by CPU OpenMP.
    if omp_med is not None and gtap_block_med is not None:
        num = df[gtap_block_med].to_numpy(dtype=float)
        den = df[omp_med].to_numpy(dtype=float)
        valid = (num > 0) & (den > 0)
        if np.any(valid):
            ax_bot.plot(x[valid], (num / den)[valid], "s-", label=label_gtap_blk, color=COL_GTAP_BLOCK)
            has_bottom = True

    # Parity line (GTaP == OpenMP) in OpenMP color.
    ax_bot.axhline(y=1.0, color=COL_OMP, linestyle="--", label="Parity")
    ax_bot.set_xlabel(x_label)
    ax_bot.set_ylabel(r"Normalized time" + "\n" + r"($T_\mathrm{method}/T_\mathrm{OMP}$)")
    if y_norm_log:
        ax_bot.set_yscale("log", base=norm_log_base)
    ax_bot.grid(True)
    ax_bot.legend()

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


def _depth_values(df: pd.DataFrame, fallback: Sequence[int] = (14, 20)) -> list[int]:
    if "depth" in df.columns:
        vals = sorted({int(v) for v in df["depth"].dropna().unique()})
        return vals
    return list(fallback)


def main():
    # --- (1) depth variation ---
    # Expected x column: depth (or n in older scripts)
    df_depth = _read_csv(Path(f"{BENCHMARK_NAME}_performance_results.csv"))
    if df_depth is not None:
        x_col = "depth" if "depth" in df_depth.columns else ("n" if "n" in df_depth.columns else df_depth.columns[0])
        make_combined_plot(
            df_depth,
            x_col=x_col,
            x_label="Tree Maximum Depth",
            out_stem="tree_depth_performance_combined",
            title="Varying D (mem_ops=256, compute_iters=256 fixed)",
        )

    # --- (2) memory variation (optionally multiple depths) ---
    df_mem_all = _read_csv(Path(f"{BENCHMARK_NAME}_mem_results.csv"))
    if df_mem_all is not None:
        x_col = "mem_ops" if "mem_ops" in df_mem_all.columns else ("n" if "n" in df_mem_all.columns else df_mem_all.columns[0])
        if "depth" in df_mem_all.columns:
            for d in _depth_values(df_mem_all):
                df_mem = df_mem_all[df_mem_all["depth"] == d].copy()
                if df_mem.empty:
                    continue
                make_combined_plot(
                    df_mem,
                    x_col=x_col,
                    x_label="Memory Operations",
                    out_stem=f"tree_mem_depth{d}_performance_combined",
                    title=f"Varying mem_ops (D={d}, compute_iters=256 fixed)",
                )
        else:
            make_combined_plot(
                df_mem_all,
                x_col=x_col,
                x_label="Memory Operations",
                out_stem="tree_mem_performance_combined",
                title="Varying mem_ops (D=20, compute_iters=256 fixed)",
            )

    # --- (3) compute variation (optionally multiple depths) ---
    df_comp_all = _read_csv(Path(f"{BENCHMARK_NAME}_compute_results.csv"))
    if df_comp_all is not None:
        x_col = "compute_iters" if "compute_iters" in df_comp_all.columns else ("n" if "n" in df_comp_all.columns else df_comp_all.columns[0])
        if "depth" in df_comp_all.columns:
            for d in _depth_values(df_comp_all):
                df_comp = df_comp_all[df_comp_all["depth"] == d].copy()
                if df_comp.empty:
                    continue
                make_combined_plot(
                    df_comp,
                    x_col=x_col,
                    x_label="Compute Iterations",
                    out_stem=f"tree_compute_depth{d}_performance_combined",
                    title=f"Varying compute_iters (D={d}, mem_ops=256 fixed)",
                )
        else:
            make_combined_plot(
                df_comp_all,
                x_col=x_col,
                x_label="Compute Iterations",
                out_stem="tree_compute_performance_combined",
                title="Varying compute_iters (D=20, mem_ops=256 fixed)",
            )

    print("\nAll plots generated!")


if __name__ == "__main__":
    main()

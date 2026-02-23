#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Allow importing `/work/gc64/c64099/plot_style/gtap_colors.py` when run from this repo
# This file lives under: .../gtap/evaluation/2-comparison/nq/
# parents[4] == /work/gc64/c64099
sys.path.append(str(Path(__file__).resolve().parents[4]))

plt.style.use([
    "~/plot_style/thesis_plt.mplstyle",
])

from plot_style.gtap_colors import COL_GTAP_THREAD, COL_GTAP_BLOCK, COL_OMP, COL_SEQ

BENCHMARK_NAME = "nq"
OUTPUT_FORMAT = "pdf"  # "png" or "pdf"

csv_path = Path(f"{BENCHMARK_NAME}_performance_results.csv")
out_path_comparison = Path(f"img/{BENCHMARK_NAME}_performance_comparison.{OUTPUT_FORMAT}")
out_path_speedup = Path(f"img/{BENCHMARK_NAME}_speedup_comparison.{OUTPUT_FORMAT}")
out_path_combined = Path(f"img/{BENCHMARK_NAME}_performance_combined.{OUTPUT_FORMAT}")
out_path_comparison.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

gtap_df = df[df["GTAP_med"] > 0].copy() if "GTAP_med" in df.columns else pd.DataFrame()
omp_df  = df[df["OMP_med"]  > 0].copy() if "OMP_med"  in df.columns else pd.DataFrame()
seq_df  = df[df["SEQ_med"]  > 0].copy() if "SEQ_med"  in df.columns else pd.DataFrame()

# Speedup data: calculate speedup ratios independently, then outer-merge.
speedup_df = pd.DataFrame()
if not gtap_df.empty:
    speedup_omp = None
    speedup_seq = None

    if not omp_df.empty:
        merged_omp = pd.merge(
            gtap_df[["n", "GTAP_med"]],
            omp_df[["n", "OMP_med"]],
            on="n",
            how="inner"
        )
        if not merged_omp.empty:
            merged_omp["omp_speedup"] = merged_omp["OMP_med"] / merged_omp["GTAP_med"]
            speedup_omp = merged_omp[["n", "omp_speedup"]]

    if not seq_df.empty:
        merged_seq = pd.merge(
            gtap_df[["n", "GTAP_med"]],
            seq_df[["n", "SEQ_med"]],
            on="n",
            how="inner"
        )
        if not merged_seq.empty:
            merged_seq["seq_speedup"] = merged_seq["SEQ_med"] / merged_seq["GTAP_med"]
            speedup_seq = merged_seq[["n", "seq_speedup"]]

    pieces = []
    if speedup_omp is not None:
        pieces.append(speedup_omp)
    if speedup_seq is not None:
        pieces.append(speedup_seq)

    if pieces:
        speedup_df = pieces[0]
        for p in pieces[1:]:
            speedup_df = pd.merge(speedup_df, p, on="n", how="outer")

fig, ax = plt.subplots()

def plot_with_iqr(ax, d, label, marker, col_prefix, *, color=None):
    y = d[f"{col_prefix}_med"].to_numpy()
    x = d["n"].to_numpy()

    low_col  = f"{col_prefix}_err_low"
    high_col = f"{col_prefix}_err_high"
    med_col = f"{col_prefix}_med"

    if low_col in d.columns and high_col in d.columns:
        low  = d[low_col].to_numpy(dtype=float)
        high = d[high_col].to_numpy(dtype=float)
        med  = d[med_col].to_numpy(dtype=float)

        low[med - low == 0] = np.nan

        yerr = np.vstack([low, high])

        ax.errorbar(
            x, y, yerr=yerr,
            fmt=marker + "-",
            capsize=3, elinewidth=1,
            label=label,
            color=color,
        )
    else:
        ax.plot(x, y, marker + "-", label=label)

if not gtap_df.empty:
    plot_with_iqr(ax, gtap_df, "GTaP (Thread-level worker)", "o", "GTAP", color=COL_GTAP_THREAD)

if not omp_df.empty:
    plot_with_iqr(ax, omp_df, "CPU OpenMP task-parallel", "s", "OMP", color=COL_OMP)

if not seq_df.empty:
    plot_with_iqr(ax, seq_df, "CPU Sequential", "^", "SEQ", color=COL_SEQ)

ax.set_xlabel("N-Queens Board Size (n)")
ax.set_ylabel("Execution Time (ms)")
ax.grid(True)
ax.set_yscale("log")
ax.legend()

plt.tight_layout()
plt.savefig(out_path_comparison)
print(f"Saved: {out_path_comparison}")

# Speedup plot (OMP/GTAP and SEQ/GTAP ratios)
if not speedup_df.empty and ("omp_speedup" in speedup_df.columns or "seq_speedup" in speedup_df.columns):
    fig2, ax2 = plt.subplots()
    
    x = speedup_df["n"].to_numpy()
    
    if "omp_speedup" in speedup_df.columns:
        omp_speedup = speedup_df["omp_speedup"].to_numpy()
        omp_valid = ~np.isnan(omp_speedup)
        if np.any(omp_valid):
            ax2.plot(x[omp_valid], omp_speedup[omp_valid], "s-", color=COL_OMP, label="Speedup to CPU OMP")
    
    if "seq_speedup" in speedup_df.columns:
        seq_speedup = speedup_df["seq_speedup"].to_numpy()
        seq_valid = ~np.isnan(seq_speedup)
        if np.any(seq_valid):
            ax2.plot(x[seq_valid], seq_speedup[seq_valid], "^-", color=COL_SEQ, label="Speedup to CPU Seq")
    
    ax2.axhline(y=1.0, color=COL_GTAP_THREAD, linestyle='--', linewidth=1, alpha=0.6, label="Parity")
    ax2.set_xlabel("N-Queens Board Size (n)")
    ax2.set_ylabel("Speedup (compared to others)")
    ax2.grid(True)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(out_path_speedup)
    print(f"Saved: {out_path_speedup}")
else:
    print("Warning: Could not calculate speedup ratios (missing data)")

# Combined figure: stack execution time and normalized time (shared x-axis)
if not gtap_df.empty and not speedup_df.empty and ("omp_speedup" in speedup_df.columns or "seq_speedup" in speedup_df.columns):
    _w, _h = plt.rcParams.get("figure.figsize", [6.4, 4.8])
    fig_height = _w * 0.7
    fig3, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True,
        figsize=(_w, fig_height),
        gridspec_kw={"height_ratios": [2.0, 1.0]}
    )

    # Top: execution time
    if not gtap_df.empty:
        plot_with_iqr(ax_top, gtap_df, "GTaP (Thread-level worker)", "o", "GTAP", color=COL_GTAP_THREAD)
    if not omp_df.empty:
        plot_with_iqr(ax_top, omp_df, "CPU OpenMP task-parallel", "s", "OMP", color=COL_OMP)
    if not seq_df.empty:
        plot_with_iqr(ax_top, seq_df, "CPU Sequential", "^", "SEQ", color=COL_SEQ)
    ax_top.set_ylabel("Execution Time (ms)")
    ax_top.set_yscale("log")
    ax_top.grid(True)
    ax_top.legend()

    # Bottom: normalized time
    x = speedup_df["n"].to_numpy()
    label_omp = r"CPU OpenMP / GTaP"
    label_seq = r"CPU Seq / GTaP"
    if "omp_speedup" in speedup_df.columns:
        omp_speedup = speedup_df["omp_speedup"].to_numpy()
        omp_valid = ~np.isnan(omp_speedup)
        if np.any(omp_valid):
            ax_bot.plot(x[omp_valid], omp_speedup[omp_valid], "s-", color=COL_OMP, label=label_omp)
    if "seq_speedup" in speedup_df.columns:
        seq_speedup = speedup_df["seq_speedup"].to_numpy()
        seq_valid = ~np.isnan(seq_speedup)
        if np.any(seq_valid):
            ax_bot.plot(x[seq_valid], seq_speedup[seq_valid], "^-", color=COL_SEQ, label=label_seq)
    ax_bot.axhline(y=1.0, color=COL_GTAP_THREAD, linestyle='--', label="Parity")
    ax_bot.set_xlabel("N-Queens Board Size (n)")
    ax_bot.set_ylabel(r"Normalized time" + "\n" + r"($T_\mathrm{method}/T_\mathrm{GTaP}$)")
    ax_bot.set_yscale("log", base=2)
    ax_bot.grid(True)
    # ax_bot.set_ylim(bottom=0)
    ax_bot.legend()

    plt.tight_layout()
    plt.savefig(out_path_combined)
    print(f"Saved: {out_path_combined}")
else:
    print("Warning: Combined plot skipped (missing GTaP or speedup data)")

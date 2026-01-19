#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
plt.style.use("~/plot_style/thesis_plt.mplstyle")

BENCHMARK_NAME = "cilksort"
OUTPUT_FORMAT = "pdf"  # "png" or "pdf"

csv_path = Path(f"{BENCHMARK_NAME}_performance_results.csv")
out_path_comparison = Path(f"img/{BENCHMARK_NAME}_performance_comparison.{OUTPUT_FORMAT}")
out_path_speedup = Path(f"img/{BENCHMARK_NAME}_speedup_comparison.{OUTPUT_FORMAT}")
out_path_comparison.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

gtap_df = df[df["GTAP_med"] > 0].copy() if "GTAP_med" in df.columns else pd.DataFrame()
omp_df  = df[df["OMP_med"]  > 0].copy() if "OMP_med"  in df.columns else pd.DataFrame()
seq_df  = df[df["SEQ_med"]  > 0].copy() if "SEQ_med"  in df.columns else pd.DataFrame()

# Speedup data: calculate speedup ratios (OMP/GTAP and SEQ/GTAP)
speedup_df = pd.DataFrame()
if not gtap_df.empty:
    speedup_data = gtap_df[["n", "GTAP_med"]].copy()
    
    # Merge with OMP data
    if not omp_df.empty:
        omp_merged = pd.merge(speedup_data, omp_df[["n", "OMP_med"]], on="n", how="inner")
        omp_merged["omp_speedup"] = omp_merged["OMP_med"] / omp_merged["GTAP_med"]
        speedup_data = omp_merged[["n", "GTAP_med", "omp_speedup"]].copy()
    
    # Merge with SEQ data
    if not seq_df.empty:
        seq_merged = pd.merge(speedup_data, seq_df[["n", "SEQ_med"]], on="n", how="left")
        seq_merged["seq_speedup"] = seq_merged["SEQ_med"] / seq_merged["GTAP_med"]
        speedup_data = seq_merged
    
    if not speedup_data.empty:
        speedup_df = speedup_data

fig, ax = plt.subplots()

def plot_with_iqr(ax, d, label, marker, col_prefix):
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
            label=label
        )
    else:
        ax.plot(x, y, marker + "-", label=label)

if not gtap_df.empty:
    plot_with_iqr(ax, gtap_df, "GTaP (Thread-level worker)", "o", "GTAP")

if not omp_df.empty:
    plot_with_iqr(ax, omp_df, "CPU OpenMP task-parallel", "s", "OMP")

if not seq_df.empty:
    plot_with_iqr(ax, seq_df, "CPU Sequential", "^", "SEQ")

ax.set_xlabel("Array Size (n)")
ax.set_ylabel("Execution Time (ms)")
ax.grid(True)
ax.set_xscale("log")
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
            ax2.plot(x[omp_valid], omp_speedup[omp_valid], "s-", color='#ff7f0e', label="Speedup to CPU OMP")
    
    if "seq_speedup" in speedup_df.columns:
        seq_speedup = speedup_df["seq_speedup"].to_numpy()
        seq_valid = ~np.isnan(seq_speedup)
        if np.any(seq_valid):
            ax2.plot(x[seq_valid], seq_speedup[seq_valid], "^-", color='#2ca02c', label="Speedup to CPU Seq")
    
    ax2.axhline(y=1.0, color='r', linestyle='--', linewidth=1, alpha=0.5, label="Parity")
    ax2.set_xlabel("Array Size (n)")
    ax2.set_ylabel("Speedup (compared to others)")
    ax2.set_xscale("log")
    ax2.grid(True)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(out_path_speedup)
    print(f"Saved: {out_path_speedup}")
else:
    print("Warning: Could not calculate speedup ratios (missing data)")

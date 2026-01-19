#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
plt.style.use("~/plot_style/thesis_plt.mplstyle")

BENCHMARK_NAME = "fib"
TITLE_BENCHMARK_NAME = "Fibonacci"
OUTPUT_FORMAT = "pdf"  # "png" or "pdf"

csv_path = Path(f"epaq_performance_results_{BENCHMARK_NAME}.csv")
out_path_comparison = Path(f"img/epaq_comparison_{BENCHMARK_NAME}.{OUTPUT_FORMAT}")
out_path_speedup = Path(f"img/epaq_speedup_{BENCHMARK_NAME}.{OUTPUT_FORMAT}")
out_path_comparison.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

queue1_df = df[df["1queue_med"] > 0].copy() if "1queue_med" in df.columns else pd.DataFrame()
queue3_df = df[df["3queue_med"] > 0].copy() if "3queue_med" in df.columns else pd.DataFrame()

# Speedup data: merge both dataframes on cutoff to calculate speedup
speedup_df = pd.DataFrame()
if not queue1_df.empty and not queue3_df.empty:
    merged = pd.merge(queue1_df[["cutoff", "1queue_med"]],
                     queue3_df[["cutoff", "3queue_med"]],
                     on="cutoff", how="inner")
    if not merged.empty:
        # Calculate speedup: 1queue / 3queue (median comparison)
        merged["speedup"] = merged["1queue_med"] / merged["3queue_med"]
        speedup_df = merged

fig, ax = plt.subplots()

def plot_with_iqr(ax, d, label, marker, col_prefix):
    y = d[f"{col_prefix}_med"].to_numpy()
    x = d["cutoff"].to_numpy()

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

if not queue1_df.empty:
    plot_with_iqr(ax, queue1_df, "1 queue", "o", "1queue")

if not queue3_df.empty:
    plot_with_iqr(ax, queue3_df, "3 queues", "s", "3queue")

ax.set_xlabel("Cutoff Depth")
ax.set_ylabel("Execution Time (ms)")
ax.grid(True)
# ax.set_yscale("log")
ax.set_title(f"EPAQ Comparison for {TITLE_BENCHMARK_NAME}")
ax.set_ylim(bottom=0)
ax.legend()

plt.tight_layout()
plt.savefig(out_path_comparison)
print(f"Saved: {out_path_comparison}")

# Speedup plot (median comparison only, no error bars)
if not speedup_df.empty:
    fig2, ax2 = plt.subplots()
    
    x = speedup_df["cutoff"].to_numpy()
    y = speedup_df["speedup"].to_numpy()
    
    # Use a different color from 1queue (e.g., green or purple)
    ax2.plot(x, y, "o-", color='#2ca02c', label="Speedup (1 queue / 3 queues)")
    ax2.axhline(y=1.0, color='r', linestyle='--', linewidth=1, alpha=0.5, label="No speedup")
    ax2.set_xlabel("Cutoff Depth")
    ax2.set_ylabel("Speedup (1 queue / 3 queues)")
    ax2.grid(True)
    ax2.set_title(f"EPAQ Speedup for {TITLE_BENCHMARK_NAME}")
    # Set y-axis upper limit to at least 1.2
    y_max = max(y.max() if len(y) > 0 else 0, 2.2)
    ax2.set_ylim(bottom=0, top=y_max)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(out_path_speedup)
    print(f"Saved: {out_path_speedup}")
else:
    print("Warning: Could not calculate speedup (missing data)")

#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
plt.style.use("~/plot_style/thesis_plt.mplstyle")

BENCHMARK_NAME = "cilksort"
TITLE_BENCHMARK_NAME = "Cilksort (Array Size=50,000,000)"
OUTPUT_FORMAT = "pdf"  # "png" or "pdf"

# Fixed colors to align with other scaling plots
COL_QUEUE1 = "#1f77b4"  # analogous to WS / baseline
COL_QUEUE3 = "#ff7f0e"  # analogous to GQ / alternative
COL_SPEEDUP = "#2ca02c"

csv_path = Path(f"epaq_performance_results_{BENCHMARK_NAME}.csv")
out_path_comparison = Path(f"img/epaq_comparison_{BENCHMARK_NAME}.{OUTPUT_FORMAT}")
out_path_speedup = Path(f"img/epaq_speedup_{BENCHMARK_NAME}.{OUTPUT_FORMAT}")
out_path_combined = Path(f"img/epaq_combined_{BENCHMARK_NAME}.{OUTPUT_FORMAT}")
out_path_comparison.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

queue1_df = df[df["1queue_med"] > 0].copy() if "1queue_med" in df.columns else pd.DataFrame()
queue3_df = df[df["3queue_med"] > 0].copy() if "3queue_med" in df.columns else pd.DataFrame()

# Relative time data: merge both dataframes on cutoff to calculate relative time
speedup_df = pd.DataFrame()
if not queue1_df.empty and not queue3_df.empty:
    merged = pd.merge(queue1_df[["cutoff", "1queue_med"]],
                     queue3_df[["cutoff", "3queue_med"]],
                     on="cutoff", how="inner")
    if not merged.empty:
        # Calculate relative time: 3queue / 1queue (1queue as baseline)
        merged["speedup"] = merged["3queue_med"] / merged["1queue_med"]
        speedup_df = merged

fig, ax = plt.subplots()

def plot_with_iqr(ax, d, label, marker, col_prefix, *, color=None):
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
            label=label,
            color=color,
        )
    else:
        ax.plot(x, y, marker + "-", label=label, color=color)

if not queue1_df.empty:
    plot_with_iqr(ax, queue1_df, "1 queue", "o", "1queue", color=COL_QUEUE1)

if not queue3_df.empty:
    plot_with_iqr(ax, queue3_df, "3 queues", "s", "3queue", color=COL_QUEUE3)

ax.set_xlabel("Cutoff Depth")
ax.set_ylabel("Execution Time (ms)")
ax.grid(True)
ax.set_xscale("log")
ax.set_title(f"EPAQ Comparison for {TITLE_BENCHMARK_NAME}")
ax.set_ylim(bottom=0)
ax.legend()

plt.tight_layout()
plt.savefig(out_path_comparison)
print(f"Saved: {out_path_comparison}")

# Relative time plot (median comparison only, no error bars)
if not speedup_df.empty:
    fig2, ax2 = plt.subplots()
    
    x = speedup_df["cutoff"].to_numpy()
    y = speedup_df["speedup"].to_numpy()
    
    ax2.plot(x, y, "o-", color=COL_SPEEDUP, label=r"3 queues / 1 queue")
    ax2.axhline(y=1.0, color="0.5", linestyle='--', label="Parity")
    ax2.set_xlabel("Cutoff Depth")
    ax2.set_ylabel(r"Normalized time" + "\n" + r"($T_{3\mathrm{queues}}/T_{1\mathrm{queue}}$)")
    ax2.set_xscale("log")
    ax2.grid(True)
    ax2.set_title(f"EPAQ Relative Time for {TITLE_BENCHMARK_NAME}")
    ax2.set_ylim(bottom=0)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(out_path_speedup)
    print(f"Saved: {out_path_speedup}")
else:
    print("Warning: Could not calculate relative time (missing data)")

# Combined figure: stack execution time and speedup (shared x-axis)
if not queue1_df.empty and not speedup_df.empty:
    _w, _h = plt.rcParams.get("figure.figsize", [6.4, 4.8])
    fig_height = _w * 0.7
    fig3, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True,
        figsize=(_w, fig_height),
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    # Top: execution time
    if not queue1_df.empty:
        plot_with_iqr(ax_top, queue1_df, "1 queue", "o", "1queue", color=COL_QUEUE1)
    if not queue3_df.empty:
        plot_with_iqr(ax_top, queue3_df, "3 queues", "s", "3queue", color=COL_QUEUE3)
    ax_top.set_ylabel("Execution Time (ms)")
    ax_top.set_xscale("log")
    ax_top.set_ylim(bottom=0)
    ax_top.grid(True)
    ax_top.legend()
    ax_top.set_title(f"EPAQ Comparison: {TITLE_BENCHMARK_NAME}")

    # Bottom: relative time (3 queues / 1 queue)
    x = speedup_df["cutoff"].to_numpy()
    y = speedup_df["speedup"].to_numpy()
    ax_bot.plot(x, y, "o-", color=COL_QUEUE3, label=r"3 queues / 1 queue")
    ax_bot.axhline(y=1.0, color=COL_QUEUE1, linestyle="--", label="Parity")
    ax_bot.set_xlabel("Cutoff Depth")
    ax_bot.set_ylabel(r"Normalized time" + "\n" + r"($T_{3\mathrm{queues}}/T_{1\mathrm{queue}}$)")
    ax_bot.set_xscale("log")
    ax_bot.grid(True)
    ax_bot.set_ylim(bottom=0)
    ax_bot.legend()

    plt.tight_layout()
    plt.savefig(out_path_combined)
    print(f"Saved: {out_path_combined}")

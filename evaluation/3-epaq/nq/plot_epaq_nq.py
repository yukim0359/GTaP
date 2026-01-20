#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
plt.style.use("~/plot_style/thesis_plt.mplstyle")

BENCHMARK_NAME = "nq"
TITLE_BENCHMARK_NAME = "N-Queens (n=16)"
OUTPUT_FORMAT = "pdf"  # "png" or "pdf"

# Fixed colors to align with other scaling plots
COL_QUEUE1 = "#1f77b4"  # analogous to WS / baseline
COL_QUEUE2 = "#ff7f0e"  # analogous to GQ / alternative
COL_SPEEDUP = "#2ca02c"

csv_path = Path(f"epaq_performance_results_{BENCHMARK_NAME}.csv")
out_path_comparison = Path(f"img/epaq_comparison_{BENCHMARK_NAME}.{OUTPUT_FORMAT}")
out_path_speedup = Path(f"img/epaq_speedup_{BENCHMARK_NAME}.{OUTPUT_FORMAT}")
out_path_combined = Path(f"img/epaq_combined_{BENCHMARK_NAME}.{OUTPUT_FORMAT}")
out_path_comparison.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

queue1_df = df[df["1queue_med"] > 0].copy() if "1queue_med" in df.columns else pd.DataFrame()
queue2_df = df[df["2queue_med"] > 0].copy() if "2queue_med" in df.columns else pd.DataFrame()

# Speedup data: merge both dataframes on cutoff to calculate speedup
speedup_df = pd.DataFrame()
if not queue1_df.empty and not queue2_df.empty:
    merged = pd.merge(queue1_df[["cutoff", "1queue_med"]],
                     queue2_df[["cutoff", "2queue_med"]],
                     on="cutoff", how="inner")
    if not merged.empty:
        # Calculate speedup: 1queue / 2queue (median comparison)
        merged["speedup"] = merged["1queue_med"] / merged["2queue_med"]
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

if not queue2_df.empty:
    plot_with_iqr(ax, queue2_df, "2 queues", "s", "2queue", color=COL_QUEUE2)

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
    
    ax2.plot(x, y, "o-", color=COL_SPEEDUP, label="Speedup (2 queues vs 1 queue)")
    ax2.axhline(y=1.0, color="0.5", linestyle='--', linewidth=1, alpha=0.8, label="Parity (=1)")
    ax2.set_xlabel("Cutoff Depth")
    ax2.set_ylabel("Speedup" + "\n" + r"($T_{1\mathrm{queue}} / T_{2\mathrm{queues}}$)")
    ax2.grid(True)
    ax2.set_title(f"EPAQ Speedup for {TITLE_BENCHMARK_NAME}")
    # Set y-axis upper limit to at least 1.2
    y_max = max(y.max() if len(y) > 0 else 0, 1.2)
    ax2.set_ylim(bottom=0, top=y_max)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(out_path_speedup)
    print(f"Saved: {out_path_speedup}")
else:
    print("Warning: Could not calculate speedup (missing data)")

# Combined figure: stack execution time and speedup (shared x-axis)
if not queue1_df.empty and not speedup_df.empty:
    _w, _h = plt.rcParams.get("figure.figsize", [6.4, 4.8])
    fig3, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True,
        figsize=(_w, _w),
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    # Top: execution time
    if not queue1_df.empty:
        plot_with_iqr(ax_top, queue1_df, "1 queue", "o", "1queue", color=COL_QUEUE1)
    if not queue2_df.empty:
        plot_with_iqr(ax_top, queue2_df, "2 queues", "s", "2queue", color=COL_QUEUE2)
    ax_top.set_ylabel("Execution Time (ms)")
    ax_top.grid(True)
    ax_top.legend()
    ax_top.set_title(f"EPAQ Comparison: {TITLE_BENCHMARK_NAME}")

    # Bottom: speedup (1 queue / 2 queues)
    x = speedup_df["cutoff"].to_numpy()
    y = speedup_df["speedup"].to_numpy()
    ax_bot.plot(x, y, "o-", color=COL_SPEEDUP, label="Speedup (2 queues vs 1 queue)")
    ax_bot.axhline(y=1.0, color="0.5", linestyle="--", linewidth=1, alpha=0.8, label="Parity (=1)")
    ax_bot.set_xlabel("Cutoff Depth")
    ax_bot.set_ylabel("Speedup" + "\n" + r"($T_{1\mathrm{queue}} / T_{2\mathrm{queues}}$)")
    ax_bot.grid(True)
    # y-limit consistent with standalone speedup
    y_max = max(y.max() if len(y) > 0 else 0, 1.2)
    ax_bot.set_ylim(bottom=0, top=y_max)
    ax_bot.legend()

    plt.tight_layout()
    plt.savefig(out_path_combined)
    print(f"Saved: {out_path_combined}")

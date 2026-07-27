#!/usr/bin/env python3
"""
Plot worker scaling analysis for Cilksort benchmark.
Figure 1: WS vs GQ comparison (log-log)
Figure 2: WS vs Chase-Lev comparison (linear Y, with ideal scaling)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

plt.style.use("~/plot_style/thesis_plt.mplstyle")

BENCHMARK_NAME = "cilksort"
TITLE_BENCHMARK_NAME = "Cilksort (Array Size = 20,000,000)"
OUTPUT_FORMAT = "pdf"  # "png" or "pdf"

# Paths
csv_path = Path(f"{BENCHMARK_NAME}_scaling_results.csv")
img_dir = Path("img")
img_dir.mkdir(parents=True, exist_ok=True)

out_path_ws_gq = img_dir / f"{BENCHMARK_NAME}_scaling_ws_gq.{OUTPUT_FORMAT}"
out_path_ours_cl = img_dir / f"{BENCHMARK_NAME}_scaling_ours_cl.{OUTPUT_FORMAT}"

if not csv_path.exists():
    print(f"Error: {csv_path} not found.")
    raise SystemExit(1)

df = pd.read_csv(csv_path)

# Only show block_size=32 and 256 to reduce clutter
block_sizes = [32, 256]
df = df[df["block_size"].isin(block_sizes)].copy()

# --- Revised encodings ---
# Method -> color / marker
method_colors = {"ws": "#1f77b4", "gq": "#ff7f0e", "chaselev": "#2ca02c"}
method_markers = {"ws": "o", "gq": "s", "chaselev": "^"}

# Block size -> linestyle (two levels, easy to read and print-friendly)
block_linestyles = {32: "-", 256: "-."}


def plot_with_iqr(ax, x, y, err_low, err_high, label, marker, color, linestyle="-"):
    """Plot with IQR error bars (low/high)."""
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    err_low = np.asarray(err_low, dtype=float)
    err_high = np.asarray(err_high, dtype=float)

    # Filter out invalid data (0 or negative values)
    valid = y > 0
    if not np.any(valid):
        return

    x = x[valid]
    y = y[valid]
    err_low = err_low[valid]
    err_high = err_high[valid]

    # Avoid showing zero-length error bars as tiny caps
    err_low = np.where(err_low == 0, np.nan, err_low)
    err_high = np.where(err_high == 0, np.nan, err_high)
    yerr = np.vstack([err_low, err_high])

    ax.errorbar(
        x, y, yerr=yerr,
        fmt=marker, linestyle=linestyle,
        capsize=2, elinewidth=1,
        label=label, color=color
    )


def add_ideal_scaling(ax, threads, times, label="Ideal Scaling", x_values=None):
    """
    Add ideal scaling line: time = k / threads, where k = t0 * p0.
    """
    threads = np.asarray(threads)
    times = np.asarray(times, dtype=float)

    valid = times > 0
    if not np.any(valid):
        return

    idx0 = np.where(valid)[0][0]
    p0 = float(threads[idx0])
    t0 = float(times[idx0])
    if t0 <= 0 or p0 <= 0:
        return

    k = t0 * p0
    if x_values is None:
        x_values = np.unique(threads)
    x_values = np.asarray(sorted(x_values), dtype=float)
    y_values = k / x_values

    ax.plot(
        x_values, y_values,
        linestyle="--", linewidth=1.2, alpha=0.7,
        color="black", label=label
    )


def set_grouped_legend_2col(ax, first_prefix, second_prefix, ideal_label="Ideal Scaling", ncol=2):
    """
    Force legend layout (2 columns) to be:
      Row 1: first_prefix (small t/b) | first_prefix (large t/b)
      Row 2: second_prefix (small t/b) | second_prefix (large t/b)
      Row 3: ideal_label (alone)

    Matplotlib legend fills columns first (column-major). To get the desired rows,
    we reorder as: [first_small, second_small, ideal, first_large, second_large].
    """
    handles, labels = ax.get_legend_handles_labels()

    def get_block(lbl: str) -> int:
        m = re.search(r"t/b=(\d+)", lbl)
        return int(m.group(1)) if m else 0

    first = []
    second = []
    ideal = None

    for h, lbl in zip(handles, labels):
        if lbl.startswith(first_prefix):
            first.append((get_block(lbl), h, lbl))
        elif lbl.startswith(second_prefix):
            second.append((get_block(lbl), h, lbl))
        elif lbl == ideal_label:
            ideal = (h, lbl)

    first.sort(key=lambda x: x[0])   # small -> large
    second.sort(key=lambda x: x[0])  # small -> large

    if len(first) >= 2 and len(second) >= 2:
        ordered = []
        ordered.append((first[0][1], first[0][2]))   # first_small
        ordered.append((second[0][1], second[0][2])) # second_small
        if ideal is not None:
            ordered.append(ideal)                    # ideal in col1, last row
        ordered.append((first[1][1], first[1][2]))   # first_large
        ordered.append((second[1][1], second[1][2])) # second_large

        oh, ol = zip(*ordered)
        ax.legend(oh, ol, ncol=ncol)
        return

    def group_rank(lbl: str) -> int:
        if lbl.startswith(first_prefix):
            return 0
        if lbl.startswith(second_prefix):
            return 1
        if lbl == ideal_label:
            return 2
        return 3

    order = sorted(range(len(labels)), key=lambda i: (group_rank(labels[i]), get_block(labels[i])))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], ncol=ncol)


# =============================================================================
# Figure 1: WS vs GQ (log-log)
# =============================================================================
fig, ax = plt.subplots()

methods_fig1 = [("ws", "WS"), ("gq", "GQ")]

for bs in block_sizes:
    data = df[df["block_size"] == bs].sort_values("total_threads")
    ls = block_linestyles[bs]

    for method, method_label in methods_fig1:
        plot_with_iqr(
            ax=ax,
            x=data["total_threads"],
            y=data[f"{method}_med"],
            err_low=data[f"{method}_err_low"],
            err_high=data[f"{method}_err_high"],
            label=f"{method_label} (t/b={bs})",
            marker=method_markers[method],
            color=method_colors[method],
            linestyle=ls,
        )

# Ideal scaling: anchor at the smallest worker count of WS with t/b=32
anchor = df[df["block_size"] == 32].sort_values("total_threads")
add_ideal_scaling(
    ax=ax,
    threads=anchor["total_threads"],
    times=anchor["ws_med"],
    x_values=sorted(df["total_threads"].unique()),
)

ax.set_xlabel("Number of Threads (Workers)")
ax.set_ylabel("Execution Time (ms)")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_title(f"vs. Global Queue: {TITLE_BENCHMARK_NAME}")
ax.grid(True)
set_grouped_legend_2col(ax, first_prefix="WS", second_prefix="GQ", ideal_label="Ideal Scaling", ncol=2)

plt.tight_layout()
plt.savefig(out_path_ws_gq)
print(f"Saved: {out_path_ws_gq}")
plt.close()


# =============================================================================
# Figure 2: Ours (WS) vs Chase-Lev (linear Y, log X)
# Only show data from 2^12 (4096) onwards
# =============================================================================
fig, ax = plt.subplots()

df_cl = df[df["total_threads"] >= 4096].copy()

methods_fig2 = [("ws", "Ours"), ("chaselev", "CL")]

for bs in block_sizes:
    data = df_cl[df_cl["block_size"] == bs].sort_values("total_threads")
    ls = block_linestyles[bs]

    for method, method_label in methods_fig2:
        plot_with_iqr(
            ax=ax,
            x=data["total_threads"],
            y=data[f"{method}_med"],
            err_low=data[f"{method}_err_low"],
            err_high=data[f"{method}_err_high"],
            label=f"{method_label} (t/b={bs})",
            marker=method_markers[method],
            color=method_colors[method],
            linestyle=ls,
        )

# Ideal scaling: anchor at the smallest worker count of Ours with t/b=32 (within df_cl)
anchor2 = df_cl[df_cl["block_size"] == 32].sort_values("total_threads")
add_ideal_scaling(
    ax=ax,
    threads=anchor2["total_threads"],
    times=anchor2["ws_med"],
    x_values=sorted(df_cl["total_threads"].unique()),
)

ax.set_xlabel("Number of Threads (Workers)")
ax.set_ylabel("Execution Time (ms)")
ax.set_xscale("log", base=2)
ax.set_ylim(bottom=0)
ax.set_title(f"vs. Sequential Chase-Lev: {TITLE_BENCHMARK_NAME}")
ax.grid(True)
set_grouped_legend_2col(ax, first_prefix="Ours", second_prefix="CL", ideal_label="Ideal Scaling", ncol=2)

plt.tight_layout()
plt.savefig(out_path_ours_cl)
print(f"Saved: {out_path_ours_cl}")
plt.close()


# =============================================================================
# Print Summary Statistics
# =============================================================================
print("\n=== Worker Scaling Summary ===")
for bs in block_sizes:
    data = df[df["block_size"] == bs]
    print(f"\nBlock Size = {bs} (threads/block):")

    for method in ["ws", "gq", "chaselev"]:
        col = f"{method}_med"
        if col not in data.columns:
            continue

        valid = data[col] > 0
        if valid.any():
            method_data = data[valid]
            min_time = method_data[col].min()
            min_threads = method_data.loc[method_data[col].idxmin(), "total_threads"]
            max_time = method_data[col].max()
            max_threads = method_data.loc[method_data[col].idxmax(), "total_threads"]
            print(
                f"  {method.upper():8s} - "
                f"Min: {min_time:8.3f} ms (threads={int(min_threads)}), "
                f"Max: {max_time:8.3f} ms (threads={int(max_threads)})"
            )

print("\nAll plots generated!")

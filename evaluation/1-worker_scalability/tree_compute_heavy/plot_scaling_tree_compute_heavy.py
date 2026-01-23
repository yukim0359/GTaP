#!/usr/bin/env python3
"""
Plot worker scaling analysis for tree_compute_heavy benchmark.
Block-level worker version: WS vs GQ comparison only (no Chase-Lev).
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

plt.style.use("~/plot_style/thesis_plt.mplstyle")

BENCHMARK_NAME = "tree_compute_heavy"
TITLE_BENCHMARK_NAME = "Synthetic Tree (Height = 20, Compute Iters = 1024, Mem Ops = 64)"
OUTPUT_FORMAT = "pdf"  # "png" or "pdf"

# Paths
csv_path = Path(f"{BENCHMARK_NAME}_scaling_results.csv")
img_dir = Path("img")
img_dir.mkdir(parents=True, exist_ok=True)

out_path_ws_gq = img_dir / f"{BENCHMARK_NAME}_scaling_ws_gq.{OUTPUT_FORMAT}"

if not csv_path.exists():
    print(f"Error: {csv_path} not found.")
    raise SystemExit(1)

df = pd.read_csv(csv_path)

# Only show block_size=32 and 256 to reduce clutter
block_sizes = [32, 256]
df = df[df["block_size"].isin(block_sizes)].copy()

# --- Revised encodings ---
# Method -> color / marker
method_colors = {"ws": "#1f77b4", "gq": "#ff7f0e"}
method_markers = {"ws": "o", "gq": "s"}

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


def add_ideal_scaling(ax, workers, times, label="Ideal Scaling", x_values=None):
    """
    Add ideal scaling line: time = k / workers, where k = t0 * p0.
    """
    workers = np.asarray(workers)
    times = np.asarray(times, dtype=float)

    valid = times > 0
    if not np.any(valid):
        return

    idx0 = np.where(valid)[0][0]
    p0 = float(workers[idx0])
    t0 = float(times[idx0])
    if t0 <= 0 or p0 <= 0:
        return

    k = t0 * p0
    if x_values is None:
        x_values = np.unique(workers)
    x_values = np.asarray(sorted(x_values), dtype=float)
    y_values = k / x_values

    ax.plot(
        x_values, y_values,
        linestyle="--", linewidth=1.2, alpha=0.7,
        color="black", label=label
    )


def set_grouped_legend_2col(ax, first_prefix, second_prefix, ideal_label="Ideal Scaling", ncol=2):
    """
    Same grouped legend layout as fib: rows are grouped by method, columns by t/b.
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

    first.sort(key=lambda x: x[0])
    second.sort(key=lambda x: x[0])

    if len(first) >= 2 and len(second) >= 2:
        ordered = []
        ordered.append((first[0][1], first[0][2]))
        ordered.append((second[0][1], second[0][2]))
        if ideal is not None:
            ordered.append(ideal)
        ordered.append((first[1][1], first[1][2]))
        ordered.append((second[1][1], second[1][2]))
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
# Figure: WS vs GQ comparison (log-log scale)
# =============================================================================
fig, ax = plt.subplots()

methods_fig = [("ws", "WS"), ("gq", "GQ")]

for bs in block_sizes:
    data = df[df["block_size"] == bs].sort_values("grid_size")
    ls = block_linestyles[bs]

    for method, method_label in methods_fig:
        plot_with_iqr(
            ax=ax,
            x=data["grid_size"],
            y=data[f"{method}_med"],
            err_low=data[f"{method}_err_low"],
            err_high=data[f"{method}_err_high"],
            label=f"{method_label} (t/b={bs})",
            marker=method_markers[method],
            color=method_colors[method],
            linestyle=ls,
        )

# Ideal scaling: keep existing policy (anchor at t/b=256)
anchor = df[df["block_size"] == 256].sort_values("grid_size")
add_ideal_scaling(
    ax=ax,
    workers=anchor["grid_size"],
    times=anchor["ws_med"],
    x_values=sorted(df["grid_size"].unique()),
)

ax.set_xlabel("Number of Blocks (Workers)")
ax.set_ylabel("Execution Time (ms)")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_title(f"vs. Global Queue:\n{TITLE_BENCHMARK_NAME}")
ax.grid(True)
set_grouped_legend_2col(ax, first_prefix="WS", second_prefix="GQ", ideal_label="Ideal Scaling", ncol=2)

plt.tight_layout()
plt.savefig(out_path_ws_gq)
print(f"Saved: {out_path_ws_gq}")
plt.close()


# =============================================================================
# Print Summary Statistics
# =============================================================================
print("\n=== Worker Scaling Summary (Block-level) ===")
for bs in block_sizes:
    data = df[df["block_size"] == bs]
    print(f"\nThreads per Block = {bs}:")
    
    for method in ["ws", "gq"]:
        col = f"{method}_med"
        if col not in data.columns:
            continue

        valid = data[col] > 0
        if valid.any():
            method_data = data[valid]
            min_time = method_data[col].min()
            min_workers = method_data.loc[method_data[col].idxmin(), "grid_size"]
            max_time = method_data[col].max()
            max_workers = method_data.loc[method_data[col].idxmax(), "grid_size"]
            print(
                f"  {method.upper():8s} - "
                f"Min: {min_time:8.3f} ms (blocks={int(min_workers)}), "
                f"Max: {max_time:8.3f} ms (blocks={int(max_workers)})"
            )

print("\nPlot generated!")

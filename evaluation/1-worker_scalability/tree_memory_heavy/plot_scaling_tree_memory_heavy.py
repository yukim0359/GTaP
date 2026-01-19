#!/usr/bin/env python3
"""
Plot worker scaling analysis for tree_memory_heavy benchmark.
Block-level worker version: WS vs GQ comparison only (no Chase-Lev).
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use("~/plot_style/thesis_plt.mplstyle")

BENCHMARK_NAME = "tree_memory_heavy"
TITLE_BENCHMARK_NAME = "Binary Tree (Memory Heavy)"
OUTPUT_FORMAT = "pdf"  # "png" or "pdf"

# Paths
csv_path = Path(f"{BENCHMARK_NAME}_scaling_results.csv")
img_dir = Path("img")
img_dir.mkdir(parents=True, exist_ok=True)

out_path_ws_gq = img_dir / f"{BENCHMARK_NAME}_scaling_ws_gq.{OUTPUT_FORMAT}"

if not csv_path.exists():
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)

# Filter to only block_size=32 and 256 to reduce clutter
block_sizes = [32, 256]
df = df[df['block_size'].isin(block_sizes)].copy()

# Colors for block sizes, markers for methods
colors = {'32': '#1f77b4', '256': '#d62728'}
markers = {'ws': 'o', 'gq': 's'}


def plot_with_iqr(ax, x, y, err_low, err_high, label, marker, color, linestyle='-'):
    """Plot with IQR error bars."""
    err_low = np.array(err_low, dtype=float)
    err_high = np.array(err_high, dtype=float)
    y = np.array(y, dtype=float)
    
    # Filter out invalid data (0 values)
    valid = y > 0
    if not np.any(valid):
        return
    
    x = np.array(x)[valid]
    y = y[valid]
    err_low = err_low[valid]
    err_high = err_high[valid]
    
    # Avoid zero error bars
    err_low[err_low == 0] = np.nan
    yerr = np.vstack([err_low, err_high])
    
    ax.errorbar(
        x, y, yerr=yerr,
        fmt=marker, linestyle=linestyle,
        capsize=2, elinewidth=1,
        label=label, color=color
    )


# =============================================================================
# Figure: WS vs GQ comparison (log-log scale)
# =============================================================================
fig, ax = plt.subplots()

methods = [
    ('ws', 'WS', '-'),
    ('gq', 'GQ', '--'),
]

for bs in block_sizes:
    data = df[df['block_size'] == bs].sort_values('grid_size')
    color = colors[str(bs)]
    
    for method, method_label, linestyle in methods:
        plot_with_iqr(
            ax, data['grid_size'], data[f'{method}_med'],
            data[f'{method}_err_low'], data[f'{method}_err_high'],
            f"{method_label} (threads/block={bs})", markers[method], color, linestyle
        )

# Ideal scaling line (based on block_size=256)
first_data = df[df['block_size'] == block_sizes[1]].sort_values('grid_size')
if len(first_data) > 0:
    first_workers = first_data['grid_size'].iloc[0]
    first_time = first_data['ws_med'].iloc[0]
    if first_time > 0:
        k = first_time * first_workers
        all_workers = sorted(df['grid_size'].unique())
        ideal_times = [k / w for w in all_workers]
        ax.plot(all_workers, ideal_times,
                linestyle='--', alpha=0.5, color='black',
                label='Ideal Scaling')

ax.set_xlabel("Number of Blocks (Workers)")
ax.set_ylabel("Execution Time (ms)")
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_title(f"vs. Global Queue: {TITLE_BENCHMARK_NAME}")
ax.grid(True)
ax.legend(ncol=2)

plt.tight_layout()
plt.savefig(out_path_ws_gq)
print(f"Saved: {out_path_ws_gq}")
plt.close()


# =============================================================================
# Print Summary Statistics
# =============================================================================
print("\n=== Worker Scaling Summary (Block-level) ===")
for bs in block_sizes:
    data = df[df['block_size'] == bs]
    print(f"\nThreads per Block = {bs}:")
    
    for method in ['ws', 'gq']:
        valid = data[f'{method}_med'] > 0
        if valid.any():
            method_data = data[valid]
            min_time = method_data[f'{method}_med'].min()
            min_workers = method_data.loc[method_data[f'{method}_med'].idxmin(), 'grid_size']
            max_time = method_data[f'{method}_med'].max()
            max_workers = method_data.loc[method_data[f'{method}_med'].idxmax(), 'grid_size']
            print(f"  {method.upper():8s} - Min: {min_time:8.3f} ms (blocks={min_workers}), Max: {max_time:8.3f} ms (blocks={max_workers})")

print("\nPlot generated!")


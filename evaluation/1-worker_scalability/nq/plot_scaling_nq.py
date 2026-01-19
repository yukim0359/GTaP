#!/usr/bin/env python3
"""
Plot worker scaling analysis for N-Queens benchmark.
Figure 1: WS vs GQ comparison (log-log)
Figure 2: WS vs Chase-Lev comparison (linear Y, with ideal scaling)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use("~/plot_style/thesis_plt.mplstyle")

BENCHMARK_NAME = "nq"
TITLE_BENCHMARK_NAME = "N-Queens"
OUTPUT_FORMAT = "pdf"  # "png" or "pdf"

# Paths
csv_path = Path(f"{BENCHMARK_NAME}_scaling_results.csv")
img_dir = Path("img")
img_dir.mkdir(parents=True, exist_ok=True)

out_path_ws_gq = img_dir / f"{BENCHMARK_NAME}_scaling_ws_gq.{OUTPUT_FORMAT}"
out_path_ours_cl = img_dir / f"{BENCHMARK_NAME}_scaling_ours_cl.{OUTPUT_FORMAT}"

if not csv_path.exists():
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)

# Filter to only block_size=32 and 256 to reduce clutter
block_sizes = [32, 256]
df = df[df['block_size'].isin(block_sizes)].copy()

# Colors for block sizes, markers for methods
colors = {'32': '#1f77b4', '256': '#d62728'}
markers = {'ws': 'o', 'gq': 's', 'chaselev': '^'}


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
# Figure 1: WS vs GQ comparison (log-log scale)
# =============================================================================
fig, ax = plt.subplots()

methods = [
    ('ws', 'WS', '-'),
    ('gq', 'GQ', '--'),
]

for bs in block_sizes:
    data = df[df['block_size'] == bs].sort_values('total_threads')
    color = colors[str(bs)]
    
    for method, method_label, linestyle in methods:
        plot_with_iqr(
            ax, data['total_threads'], data[f'{method}_med'],
            data[f'{method}_err_low'], data[f'{method}_err_high'],
            f"{method_label} (block={bs})", markers[method], color, linestyle
        )

# Ideal scaling line
first_data = df[df['block_size'] == block_sizes[0]].sort_values('total_threads')
if len(first_data) > 0:
    first_threads = first_data['total_threads'].iloc[0]
    first_time = first_data['ws_med'].iloc[0]
    if first_time > 0:
        k = first_time * first_threads
        all_threads = sorted(df['total_threads'].unique())
        ideal_times = [k / t for t in all_threads]
        ax.plot(all_threads, ideal_times,
                linestyle='--', alpha=0.5, color='black',
                label='Ideal Scaling')

ax.set_xlabel("Total Threads")
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
# Figure 2: WS vs Chase-Lev comparison (linear Y, with ideal scaling)
# Only show data from 2^12 (4096) onwards
# =============================================================================
fig, ax = plt.subplots()

# Filter data for 2^12 and above
df_cl = df[df['total_threads'] >= 4096].copy()

methods = [
    ('ws', 'Ours', '-'),
    ('chaselev', 'CL', ':'),
]

for bs in block_sizes:
    data = df_cl[df_cl['block_size'] == bs].sort_values('total_threads')
    color = colors[str(bs)]
    
    for method, method_label, linestyle in methods:
        plot_with_iqr(
            ax, data['total_threads'], data[f'{method}_med'],
            data[f'{method}_err_low'], data[f'{method}_err_high'],
            f"{method_label} (block={bs})", markers[method], color, linestyle
        )

# Ideal scaling line (linear, not straight line but hyperbola: time = k / threads)
first_data = df_cl[df_cl['block_size'] == block_sizes[0]].sort_values('total_threads')
if len(first_data) > 0:
    first_threads = first_data['total_threads'].iloc[0]
    first_time = first_data['ws_med'].iloc[0]
    if first_time > 0:
        k = first_time * first_threads
        all_threads = sorted(df_cl['total_threads'].unique())
        ideal_times = [k / t for t in all_threads]
        ax.plot(all_threads, ideal_times,
                linestyle='--', alpha=0.5, color='black',
                label='Ideal Scaling')

ax.set_xlabel("Total Threads")
ax.set_ylabel("Execution Time (ms)")
ax.set_xscale('log', base=2)
ax.set_ylim(bottom=0)
ax.set_title(f"vs. Chase-Lev: {TITLE_BENCHMARK_NAME}")
ax.grid(True)
ax.legend(ncol=2)

plt.tight_layout()
plt.savefig(out_path_ours_cl)
print(f"Saved: {out_path_ours_cl}")
plt.close()


# =============================================================================
# Print Summary Statistics
# =============================================================================
print("\n=== Worker Scaling Summary ===")
for bs in block_sizes:
    data = df[df['block_size'] == bs]
    print(f"\nBlock Size = {bs}:")
    
    for method in ['ws', 'gq', 'chaselev']:
        valid = data[f'{method}_med'] > 0
        if valid.any():
            method_data = data[valid]
            min_time = method_data[f'{method}_med'].min()
            min_threads = method_data.loc[method_data[f'{method}_med'].idxmin(), 'total_threads']
            max_time = method_data[f'{method}_med'].max()
            max_threads = method_data.loc[method_data[f'{method}_med'].idxmax(), 'total_threads']
            print(f"  {method.upper():8s} - Min: {min_time:8.3f} ms (threads={min_threads}), Max: {max_time:8.3f} ms (threads={max_threads})")

print("\nAll plots generated!")

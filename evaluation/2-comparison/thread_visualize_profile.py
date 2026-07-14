import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
plt.style.use("~/plot_style/profile.mplstyle")

DATA_MAX_LIMIT = 30000
DEFAULT_TIME_BINS = 1500
MAX_TASKS_PER_BATCH = 32

# visualize_profile.py timeline: NotWorking = #ff7f0e @ alpha 0.5; tasks = plt.cm.Blues.
NOT_WORKING_COLOR = "#ff7f0e"
NOT_WORKING_ALPHA = 0.5
TASKS_IN_BATCH_CMAP = plt.cm.Blues


def _blend_on_white(color, alpha=1.0):
    r, g, b = mpl.colors.to_rgb(color)
    inv = 1.0 - alpha
    return mpl.colors.to_hex((alpha * r + inv, alpha * g + inv, alpha * b + inv))


IDLE_WARP_COLOR = _blend_on_white(NOT_WORKING_COLOR, NOT_WORKING_ALPHA)


def make_timeline_tasks_colormap(max_tasks=MAX_TASKS_PER_BATCH):
    """Blues for tasks 1..max_tasks; idle (0) via cmap.set_under (blended orange)."""
    cmap = TASKS_IN_BATCH_CMAP.copy()
    cmap.set_under(IDLE_WARP_COLOR)
    norm = mpl.colors.Normalize(vmin=1.0, vmax=float(max_tasks))
    return cmap, norm


TIMELINE_HEATMAP_CMAP, TIMELINE_HEATMAP_NORM = make_timeline_tasks_colormap()
HEATMAP_CMAP = TIMELINE_HEATMAP_CMAP


def tasks_in_batch_scalar_mappable(max_tasks=MAX_TASKS_PER_BATCH):
    """Colorbar mappable: 1..max_tasks Blues; extend='min' shows idle (under) triangle."""
    cmap, norm = make_timeline_tasks_colormap(max_tasks)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm


COLORBAR_LABEL_TASKS = "Tasks in batch (0 = idle, orange)"


def configure_tasks_in_batch_colorbar(cbar, max_tasks=MAX_TASKS_PER_BATCH):
    cbar.set_ticks([1, 8, 16, 24, max_tasks])

# App name to title string mapping
APP_TITLES = {
    'fib': 'Fibonacci (n=35)',
    'nq': 'N-Queens (n=16)',
    'cilksort': 'CilkSort (Array Size=100,000,000)',
    'mergesort': 'MergeSort (Array Size=200,000)',
    'tree_load_compute': 'Tree Load Compute',
    'bfs': 'BFS (USA Road Network)',
}

OUTPUT_FORMAT = "pdf"  # "png" or "pdf"

def compute_utilization_from_timeline(timeline_df, strong_state):
    """Compute utilization (percentage of time in strong_state) per warp_id.
    
    Utilization is calculated as working_time / program_total_time,
    where program_total_time is the time span from the first to the last
    event in the entire timeline (not per-worker lifetime).
    
    Args:
        timeline_df: DataFrame containing timeline data for ALL workers.
                     Must include all workers to correctly calculate program_total_time.
        strong_state: The state that counts as "working" (e.g., "Working").
    
    Returns:
        DataFrame with columns ['warp_id', 'utilization_percent'].
    """
    if 'warp_id' not in timeline_df.columns or 'relative_time_ms' not in timeline_df.columns or 'state_description' not in timeline_df.columns:
        return pd.DataFrame(columns=['warp_id', 'utilization_percent'])
    
    # Calculate program total time (first to last event across all workers)
    program_first_time = float(timeline_df['relative_time_ms'].min())
    program_last_time = float(timeline_df['relative_time_ms'].max())
    program_total_time = max(0.0, program_last_time - program_first_time)
    
    if program_total_time <= 0.0:
        # Fallback: return 0.0 for all if no valid time span
        return pd.DataFrame([
            {'warp_id': warp_id, 'utilization_percent': 0.0}
            for warp_id in timeline_df['warp_id'].unique()
        ])
    
    util_rows = []
    for warp_id, grp in timeline_df.groupby('warp_id'):
        g = grp.sort_values('relative_time_ms').reset_index(drop=True)
        if g.empty:
            util_rows.append({'warp_id': warp_id, 'utilization_percent': 0.0})
            continue
        
        working_time = 0.0
        working_start_time = None
        
        for idx, row in g.iterrows():
            state = row['state_description']
            t = float(row['relative_time_ms'])
            
            if state == strong_state:
                # Working開始
                if working_start_time is None:
                    working_start_time = t
            else:
                # NotWorking開始
                if working_start_time is not None:
                    # Working期間を累積
                    working_time += max(0.0, t - working_start_time)
                    working_start_time = None
        
        # 最後の状態がWorkingの場合、最後のタイムスタンプまで
        if working_start_time is not None:
            last_time = float(g['relative_time_ms'].iloc[-1])
            working_time += max(0.0, last_time - working_start_time)
        
        # Use program total time as denominator
        util = max(0.0, min(100.0, (working_time / program_total_time) * 100.0)) if program_total_time > 0.0 else 0.0
        util_rows.append({'warp_id': warp_id, 'utilization_percent': util})
    return pd.DataFrame(util_rows)

def compute_busy_time_per_warp(timeline_df, program_t_min, program_t_max, strong_state="Working"):
    """Return per-warp busy time (ms) over the program time span."""
    program_total_time = max(0.0, float(program_t_max) - float(program_t_min))
    busy = {}
    if program_total_time <= 0.0:
        for warp_id in timeline_df["warp_id"].unique():
            busy[int(warp_id)] = 0.0
        return busy

    for warp_id, grp in timeline_df.groupby("warp_id"):
        g = grp.sort_values("relative_time_ms").reset_index(drop=True)
        working_time = 0.0
        working_start_time = None
        for _, row in g.iterrows():
            state = row["state_description"]
            t = float(row["relative_time_ms"])
            if state == strong_state:
                if working_start_time is None:
                    working_start_time = t
            elif working_start_time is not None:
                working_time += max(0.0, t - working_start_time)
                working_start_time = None
        if working_start_time is not None:
            last_time = float(g["relative_time_ms"].iloc[-1])
            working_time += max(0.0, last_time - working_start_time)
        busy[int(warp_id)] = working_time
    return busy

def ordered_warp_ids(stats_df, busy_times, *, sort_by_busy=True):
    """All warps; busiest first when sort_by_busy is True."""
    warp_ids = [int(w) for w in stats_df["warp_id"].tolist()]
    if not sort_by_busy:
        return warp_ids
    return sorted(warp_ids, key=lambda wid: (-busy_times.get(wid, 0.0), wid))

def build_warp_heatmap_matrix(
    timeline_df,
    warp_ids,
    *,
    n_bins,
    t_min,
    t_max,
    strong_state="Working",
):
    """Build (n_warps, n_bins) array: 0=idle, >0=max tasks_in_batch in that bin."""
    total_duration = max(0.0, float(t_max) - float(t_min))
    n_warps = len(warp_ids)
    matrix = np.zeros((n_warps, n_bins), dtype=np.float32)
    if total_duration <= 0.0 or n_warps == 0:
        return matrix

    warp_to_row = {wid: i for i, wid in enumerate(warp_ids)}
    bin_width = total_duration / n_bins
    has_tasks = "tasks_in_batch" in timeline_df.columns
    norm_end = total_duration

    grouped = timeline_df.groupby("warp_id", sort=False)
    for warp_id, grp in grouped:
        wid = int(warp_id)
        if wid not in warp_to_row:
            continue
        row = warp_to_row[wid]
        g = grp.sort_values("relative_time_ms")
        times = (g["relative_time_ms"].to_numpy(dtype=np.float64) - t_min)
        states = g["state_description"].to_numpy()
        if has_tasks:
            tasks = pd.to_numeric(g["tasks_in_batch"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        else:
            tasks = np.zeros(len(g), dtype=np.float32)

        for i in range(len(g)):
            t_start = max(0.0, float(times[i]))
            t_end = float(times[i + 1]) if i + 1 < len(g) else norm_end
            t_end = min(norm_end, max(t_start, t_end))
            if states[i] == strong_state:
                val = float(tasks[i])
            else:
                val = 0.0
            if val <= 0.0 or t_end <= t_start:
                continue
            b0 = int(t_start / bin_width)
            b1 = int(np.ceil(t_end / bin_width))
            b0 = max(0, min(n_bins, b0))
            b1 = max(0, min(n_bins, b1))
            if b0 < b1:
                matrix[row, b0:b1] = np.maximum(matrix[row, b0:b1], val)
    return matrix

def create_timeline_heatmap_plot(
    timeline_df,
    stats_df,
    strong_state,
    app_name=None,
    *,
    n_bins=DEFAULT_TIME_BINS,
    sort_by_busy=True,
):
    """All-warps heatmap: y=warps (1 row / warp), x=time bins, color=busy intensity."""
    print(f"Creating timeline heatmap ({n_bins} time bins)...")

    t_min = float(timeline_df["relative_time_ms"].min())
    t_max = float(timeline_df["relative_time_ms"].max())
    total_duration = max(0.0, t_max - t_min)
    if total_duration <= 0.0:
        print("No data to visualize")
        return None

    busy_times = compute_busy_time_per_warp(timeline_df, t_min, t_max, strong_state=strong_state)
    warp_ids = ordered_warp_ids(stats_df, busy_times, sort_by_busy=sort_by_busy)
    matrix = build_warp_heatmap_matrix(
        timeline_df,
        warp_ids,
        n_bins=n_bins,
        t_min=t_min,
        t_max=t_max,
        strong_state=strong_state,
    )

    _w, _h = plt.rcParams.get("figure.figsize", [6.4, 4.8])
    fig_width = _w * 1.6
    fig_height = max(4.0, min(8.0, _h * 1.2))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap=TIMELINE_HEATMAP_CMAP,
        norm=TIMELINE_HEATMAP_NORM,
        extent=[0.0, total_duration, len(warp_ids), 0.0],
        rasterized=True,
    )
    ax.set_xlim(0.0, total_duration)
    ax.set_ylim(len(warp_ids), 0.0)
    ax.set_xlabel("Time (ms)")
    if sort_by_busy:
        ax.set_ylabel("Warps (sorted by total busy time)")
    else:
        ax.set_ylabel("Warps (by warp ID)")
    title_suffix = APP_TITLES.get(app_name, app_name) if app_name else ""
    ax.set_title(f"Worker Timeline Heatmap: {title_suffix}")
    ax.grid(False)

    cbar = fig.colorbar(
        tasks_in_batch_scalar_mappable(), ax=ax, fraction=0.03, pad=0.02, extend="min",
    )
    cbar.set_label(COLORBAR_LABEL_TASKS)
    configure_tasks_in_batch_colorbar(cbar)

    n_active = int((stats_df["total_samples"] > 0).sum()) if "total_samples" in stats_df.columns else len(warp_ids)
    ax.text(
        0.01, 0.02,
        f"{len(warp_ids)} warps ({n_active} active)",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
    )

    plt.tight_layout()
    return fig

def load_and_process_data(app_name, *, compute_utilization=False):
    """CSVデータ（working）を読み込んで処理"""
    print(f"Loading working data for app='{app_name}'...")

    profile_dir = os.path.join(app_name, "profile")
    primary_tl = os.path.join(profile_dir, f"{app_name}_warp_timeline_working.csv")
    primary_st = os.path.join(profile_dir, f"{app_name}_warp_statistics_working.csv")
    fallback_tl = os.path.join(profile_dir, "warp_timeline_working.csv")
    fallback_st = os.path.join(profile_dir, "warp_statistics_working.csv")
    strong_state = "Working"

    tl_path = primary_tl if os.path.exists(primary_tl) else fallback_tl
    st_path = primary_st if os.path.exists(primary_st) else fallback_st

    timeline_df = pd.read_csv(tl_path)
    stats_df = pd.read_csv(st_path)

    if compute_utilization and 'utilization_percent' not in stats_df.columns:
        util_df = compute_utilization_from_timeline(timeline_df, strong_state=strong_state)
        if not util_df.empty:
            stats_df = stats_df.merge(util_df, on='warp_id', how='left')
            stats_df['utilization_percent'] = stats_df['utilization_percent'].fillna(0.0)
            print("Computed utilization_percent from timeline")
        else:
            print("Warning: utilization_percent not found and cannot be computed")

    return timeline_df, stats_df, strong_state

def create_timeline_plot(timeline_df, stats_df, strong_state, app_name=None, max_warps=None):
    print("Creating timeline visualization...")

    active_warps = stats_df[stats_df['total_samples'] > 0]['warp_id'].tolist()
    if max_warps is not None:
        active_warps = active_warps[:max_warps]

    filtered_df = timeline_df[timeline_df['warp_id'].isin(active_warps)].copy()

    if len(filtered_df) == 0:
        print("No data to visualize")
        return None

    global_min_time = timeline_df['relative_time_ms'].min()
    global_max_time = timeline_df['relative_time_ms'].max()
    filtered_df['normalized_time'] = filtered_df['relative_time_ms'] - global_min_time

    max_tasks = None
    cmap = None
    norm = None
    if 'tasks_in_batch' in timeline_df.columns:
        try:
            max_tasks_val = pd.to_numeric(timeline_df['tasks_in_batch'], errors='coerce').max()
            if pd.notna(max_tasks_val) and float(max_tasks_val) > 0.0:
                max_tasks = float(max_tasks_val)
                cmap = plt.cm.Blues
                norm = mpl.colors.Normalize(vmin=0.0, vmax=max_tasks)
        except Exception:
            max_tasks = None

    fig_height = max(8, len(active_warps) * 0.3)
    _w, _h = plt.rcParams.get("figure.figsize", [6.4, 4.8])
    fig_width = _w * 1.6  # make timeline figure less wide than before
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Blue (Working), Orange (NotWorking)
    colors = {'Working': '#1f77b4', 'NotWorking': '#ff7f0e'}
    weak_color = colors.get('NotWorking', '#ff7f0e')

    total_duration = global_max_time - global_min_time
    for i, warp_id in enumerate(active_warps):
        warp_data = filtered_df[filtered_df['warp_id'] == warp_id].sort_values('normalized_time')

        if len(warp_data) == 0:
            rect = patches.Rectangle(
                (0, i - 0.4), total_duration, 0.8,
                linewidth=0, facecolor=weak_color, alpha=0.5
            )
            ax.add_patch(rect)
            continue

        first_time = warp_data['normalized_time'].iloc[0]
        if first_time > 0:
            rect = patches.Rectangle(
                (0, i - 0.4), first_time, 0.8,
                linewidth=0, facecolor=weak_color, alpha=0.5
            )
            ax.add_patch(rect)

        prev_state = None
        start_time = None
        for _, row in warp_data.iterrows():
            current_state = row['state_description']
            current_time = row['normalized_time']

            if prev_state is not None and prev_state != current_state:
                duration = current_time - start_time
                if prev_state == 'Working' and max_tasks is not None and cmap is not None and norm is not None:
                    seg_mask = (warp_data['normalized_time'] >= start_time) & (warp_data['normalized_time'] <= current_time)
                    seg_vals = pd.to_numeric(warp_data.loc[seg_mask, 'tasks_in_batch'], errors='coerce') if 'tasks_in_batch' in warp_data.columns else None
                    seg_max = float(seg_vals.max()) if seg_vals is not None and not seg_vals.empty and pd.notna(seg_vals.max()) else 0.0
                    color = cmap(norm(seg_max))
                    alpha = 0.9
                else:
                    color = colors.get(prev_state, '#888888')
                    alpha = 0.8 if prev_state == strong_state else 0.5
                rect = patches.Rectangle(
                    (start_time, i - 0.4), duration, 0.8,
                    linewidth=0, facecolor=color, alpha=alpha
                )
                ax.add_patch(rect)

            if prev_state != current_state:
                start_time = current_time
                prev_state = current_state

        if prev_state is not None and len(warp_data) > 0:
            last_time = warp_data['normalized_time'].iloc[-1]
            duration = last_time - start_time
            if prev_state == 'Working' and max_tasks is not None and cmap is not None and norm is not None:
                seg_mask = (warp_data['normalized_time'] >= start_time) & (warp_data['normalized_time'] <= last_time)
                seg_vals = pd.to_numeric(warp_data.loc[seg_mask, 'tasks_in_batch'], errors='coerce') if 'tasks_in_batch' in warp_data.columns else None
                seg_max = float(seg_vals.max()) if seg_vals is not None and not seg_vals.empty and pd.notna(seg_vals.max()) else 0.0
                color = cmap(norm(seg_max))
                alpha = 0.9
            else:
                color = colors.get(prev_state, '#888888')
                alpha = 0.8 if prev_state == strong_state else 0.5
            rect = patches.Rectangle(
                (start_time, i - 0.4), duration, 0.8,
                linewidth=0, facecolor=color, alpha=alpha
            )
            ax.add_patch(rect)

        last_recorded_time = warp_data['normalized_time'].iloc[-1]
        
        # データの件数がMAXに達したかどうかをチェック
        max_data_reached = False
        if len(warp_data) >= DATA_MAX_LIMIT:
            max_data_reached = True
        
        # データがMAXに達していない場合のみ、残り時間を塗る
        if last_recorded_time < total_duration and not max_data_reached:
            rect = patches.Rectangle(
                (last_recorded_time, i - 0.4), total_duration - last_recorded_time, 0.8,
                linewidth=0, facecolor=weak_color, alpha=0.5
            )
            ax.add_patch(rect)

    ax.set_xlim(0, total_duration)
    ax.set_ylim(-0.5, len(active_warps) - 0.5)
    ax.set_yticks(range(len(active_warps)))
    ax.set_yticklabels([f'Warp {wid}' for wid in active_warps])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Warps')
    title_suffix = APP_TITLES.get(app_name, app_name) if app_name else ''
    title = f'Worker Timeline Visualization: {title_suffix}'
    ax.set_title(title)

    ax.grid(True, alpha=0)
    legend_elements = [
        patches.Patch(color=colors['Working'], alpha=0.8, label='Executing taskfn'),
        patches.Patch(color=colors['NotWorking'], alpha=0.5, label='Not executing taskfn')
    ]
    
    # 平均タスク数を計算して凡例に追加
    if 'tasks_in_batch' in filtered_df.columns:
        working_df = filtered_df[filtered_df['state_description'] == strong_state]
        if len(working_df) > 0:
            tasks_vals = pd.to_numeric(working_df['tasks_in_batch'], errors='coerce')
            tasks_vals = tasks_vals.dropna()
            if len(tasks_vals) > 0:
                avg_tasks = tasks_vals.mean()
                # 見えないパッチを使用して凡例に追加
                legend_elements.append(patches.Patch(color='none', label=f'Avg tasks per batch: {avg_tasks:.2f}'))
    
    ax.legend(handles=legend_elements, loc='upper right')

    if max_tasks and cmap is not None and norm is not None:
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('tasks in batch')

    plt.tight_layout()
    return fig

def create_utilization_histogram(stats_df, app_name=None):
    """Warpごとのworking時間割合のヒストグラムを作成"""
    print("Creating utilization histogram...")

    # 全warpを含める（total_samples = 0のwarpも含む、utilization = 0%）
    all_warps = stats_df.copy()
    # utilization_percentが存在しない場合は0.0を設定
    if 'utilization_percent' not in all_warps.columns:
        all_warps['utilization_percent'] = 0.0
    all_warps['utilization_percent'] = all_warps['utilization_percent'].fillna(0.0)
    
    if len(all_warps) == 0:
        print("No warps found")
        return None
    
    active_warps = all_warps

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(active_warps['utilization_percent'], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    ax.set_xlabel('Task Execution Time Ratio (%)')
    ax.set_ylabel('Number of Warps')
    title_suffix = APP_TITLES.get(app_name, app_name) if app_name else ''
    title = f'Distribution of Task Execution Time Ratio per Warp:\n{title_suffix}'
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # 統計情報を表示
    mean_util = active_warps['utilization_percent'].mean()
    median_util = active_warps['utilization_percent'].median()
    ax.axvline(mean_util, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_util:.1f}%')
    ax.axvline(median_util, color='green', linestyle='--', linewidth=2, label=f'Median: {median_util:.1f}%')
    ax.legend()
    
    plt.tight_layout()
    return fig

def print_summary_statistics(stats_df):
    """統計サマリーを出力"""
    print("\n" + "=" * 60)
    print("WARP TIMELINE ANALYSIS SUMMARY (Working)")
    print("=" * 60)

    # 全warpを含める
    all_warps = stats_df.copy()
    if 'utilization_percent' not in all_warps.columns:
        all_warps['utilization_percent'] = 0.0
    all_warps['utilization_percent'] = all_warps['utilization_percent'].fillna(0.0)
    
    active_warps = all_warps[all_warps['total_samples'] > 0]
    inactive_warps = all_warps[all_warps['total_samples'] == 0]
    
    print(f"Total Warps: {len(all_warps)}")
    print(f"Active Warps (total_samples > 0): {len(active_warps)}")
    print(f"Inactive Warps (total_samples = 0): {len(inactive_warps)}")
    
    if len(all_warps) > 0:
        print(f"Average Utilization (all warps): {all_warps['utilization_percent'].mean():.2f}%")
        print(f"Utilization Std Dev (all warps): {all_warps['utilization_percent'].std():.2f}%")
        print(f"Min Utilization: {all_warps['utilization_percent'].min():.2f}%")
        print(f"Max Utilization: {all_warps['utilization_percent'].max():.2f}%")
    
    if len(active_warps) > 0:
        print(f"\nActive Warps Only:")
        print(f"Average Utilization: {active_warps['utilization_percent'].mean():.2f}%")
        print(f"Utilization Std Dev: {active_warps['utilization_percent'].std():.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Warp Timeline Visualization (Thread Runtime)')
    parser.add_argument('--app_name', type=str, default='fib', help='Prefix (app name) for CSV and image outputs')
    parser.add_argument(
        '--timeline-mode',
        choices=['heatmap', 'legacy'],
        default='heatmap',
        help='heatmap: all-warps raster heatmap (default); legacy: per-warp rectangle timeline',
    )
    parser.add_argument(
        '--time-bins',
        type=int,
        default=DEFAULT_TIME_BINS,
        help='Number of horizontal time bins for heatmap mode (default: 1500)',
    )
    parser.add_argument(
        '--no-sort-by-busy',
        action='store_true',
        help='Keep warp ID order in heatmap (default: sort by total busy time)',
    )
    parser.add_argument(
        '--max-warps',
        type=int,
        default=15,
        help='Max warps to draw in legacy timeline mode',
    )
    parser.add_argument(
        '--with-utilization',
        action='store_true',
        help='Also print summary stats and save utilization histogram (slower)',
    )
    args = parser.parse_args()

    print("Warp Timeline Visualization Tool (Thread Runtime)")
    print("=" * 40)

    try:
        img_dir = os.path.join(args.app_name, "img")
        os.makedirs(img_dir, exist_ok=True)
        timeline_df, stats_df, strong_state = load_and_process_data(
            args.app_name,
            compute_utilization=args.with_utilization,
        )
        if args.with_utilization:
            print_summary_statistics(stats_df)
        print("\nGenerating timeline visualization...")

        # Timeline heatmap (default) or legacy rectangles
        if args.timeline_mode == 'heatmap':
            timeline_fig = create_timeline_heatmap_plot(
                timeline_df,
                stats_df,
                strong_state,
                app_name=args.app_name,
                n_bins=args.time_bins,
                sort_by_busy=not args.no_sort_by_busy,
            )
        else:
            timeline_fig = create_timeline_plot(
                timeline_df,
                stats_df,
                strong_state,
                app_name=args.app_name,
                max_warps=args.max_warps,
            )
        if timeline_fig:
            out_path = os.path.join(img_dir, f"{args.app_name}_timeline.{OUTPUT_FORMAT}")
            timeline_fig.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {out_path}")

        if args.with_utilization:
            util_fig = create_utilization_histogram(stats_df, app_name=args.app_name)
            if util_fig:
                out_path = os.path.join(img_dir, f"{args.app_name}_utilization.{OUTPUT_FORMAT}")
                util_fig.savefig(out_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {out_path}")

        print("\nVisualization complete!")

    except FileNotFoundError as e:
        print(f"Error: Could not find required CSV files: {e}")
        print(f"Make sure warp_timeline_working.csv and warp_statistics_working.csv exist in {args.app_name}/profile/ directory")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

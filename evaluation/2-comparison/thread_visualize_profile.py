import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
plt.style.use("~/plot_style/profile.mplstyle")

DATA_MAX_LIMIT = 30000

# App name to title string mapping
APP_TITLES = {
    'fib': 'Fibonacci (n=35)',
    'nq': 'N-Queens (n=16)',
    'cilksort': 'CilkSort (Array Size=10^8)',
    'mergesort': 'MergeSort (Array Size=2×10^5)',
    'tree_load_compute': 'Tree Load Compute',
}

OUTPUT_FORMAT = "pdf"  # "png" or "pdf"

def compute_utilization_from_timeline(timeline_df, strong_state):
    if 'warp_id' not in timeline_df.columns or 'relative_time_ms' not in timeline_df.columns or 'state_description' not in timeline_df.columns:
        return pd.DataFrame(columns=['warp_id', 'utilization_percent'])
    util_rows = []
    for warp_id, grp in timeline_df.groupby('warp_id'):
        g = grp.sort_values('relative_time_ms').reset_index(drop=True)
        if g.empty:
            util_rows.append({'warp_id': warp_id, 'utilization_percent': 0.0})
            continue
        first_time = float(g['relative_time_ms'].iloc[0])
        last_time = float(g['relative_time_ms'].iloc[-1])
        total_span = max(0.0, last_time - first_time)
        if total_span <= 0.0:
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
            working_time += max(0.0, last_time - working_start_time)
        
        util = max(0.0, min(100.0, (working_time / total_span) * 100.0)) if total_span > 0.0 else 0.0
        util_rows.append({'warp_id': warp_id, 'utilization_percent': util})
    return pd.DataFrame(util_rows)

def load_and_process_data(app_name):
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

    # Ensure utilization_percent exists (compute if missing)
    if 'utilization_percent' not in stats_df.columns:
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
    fig, ax = plt.subplots(figsize=(20, fig_height))

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

    ax.grid(True, alpha=0.3)
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
    args = parser.parse_args()

    print("Warp Timeline Visualization Tool (Thread Runtime)")
    print("=" * 40)

    try:
        img_dir = os.path.join(args.app_name, "img")
        os.makedirs(img_dir, exist_ok=True)
        timeline_df, stats_df, strong_state = load_and_process_data(args.app_name)
        print_summary_statistics(stats_df)
        print("\nGenerating visualizations...")

        # 1. タイムライン図（warpと時間軸のworking/not working）
        timeline_fig = create_timeline_plot(timeline_df, stats_df, strong_state, app_name=args.app_name, max_warps=15)
        if timeline_fig:
            out_path = os.path.join(img_dir, f"{args.app_name}_timeline.{OUTPUT_FORMAT}")
            timeline_fig.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {out_path}")

        # 2. Warpごとのworking時間割合のヒストグラム
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

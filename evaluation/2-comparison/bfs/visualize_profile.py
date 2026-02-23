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
    'bfs': 'BFS (Breadth-First Search)',
}

OUTPUT_FORMAT = "png"  # "png" or "pdf"

def compute_utilization_from_timeline(timeline_df, strong_state):
    """Compute utilization (percentage of time in strong_state) per block_id.
    
    Utilization is calculated as working_time / program_total_time,
    where program_total_time is the time span from the first to the last
    event in the entire timeline (not per-worker lifetime).
    
    Args:
        timeline_df: DataFrame containing timeline data for ALL workers.
                     Must include all workers to correctly calculate program_total_time.
        strong_state: The state that counts as "working" (e.g., "Working").
    
    Returns:
        DataFrame with columns ['block_id', 'utilization_percent'].
    """
    if 'block_id' not in timeline_df.columns or 'relative_time_ms' not in timeline_df.columns or 'state_description' not in timeline_df.columns:
        return pd.DataFrame(columns=['block_id', 'utilization_percent'])
    
    # Calculate program total time (first to last event across all workers)
    program_first_time = float(timeline_df['relative_time_ms'].min())
    program_last_time = float(timeline_df['relative_time_ms'].max())
    program_total_time = max(0.0, program_last_time - program_first_time)
    
    if program_total_time <= 0.0:
        # Fallback: return 0.0 for all if no valid time span
        return pd.DataFrame([
            {'block_id': block_id, 'utilization_percent': 0.0}
            for block_id in timeline_df['block_id'].unique()
        ])
    
    util_rows = []
    for block_id, grp in timeline_df.groupby('block_id'):
        g = grp.sort_values('relative_time_ms').reset_index(drop=True)
        if g.empty:
            util_rows.append({'block_id': block_id, 'utilization_percent': 0.0})
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
        util_rows.append({'block_id': block_id, 'utilization_percent': util})
    return pd.DataFrame(util_rows)

def load_and_process_data(app_name):
    """CSVデータ（working）を読み込んで処理（bfsディレクトリ内から呼び出す前提）"""
    print(f"Loading working data for app='{app_name}'...")

    # bfsディレクトリ内から呼び出す前提で、profileディレクトリを直接参照
    profile_dir = "profile"
    # Block runtime uses block_timeline/block_statistics
    primary_tl = os.path.join(profile_dir, f"{app_name}_block_timeline_working.csv")
    primary_st = os.path.join(profile_dir, f"{app_name}_block_statistics_working.csv")
    fallback_tl = os.path.join(profile_dir, "block_timeline_working.csv")
    fallback_st = os.path.join(profile_dir, "block_statistics_working.csv")
    strong_state = "Working"

    tl_path = primary_tl if os.path.exists(primary_tl) else fallback_tl
    st_path = primary_st if os.path.exists(primary_st) else fallback_st

    print(f"  Timeline CSV: {tl_path}")
    print(f"  Statistics CSV: {st_path}")

    timeline_df = pd.read_csv(tl_path)
    stats_df = pd.read_csv(st_path)

    # Ensure utilization_percent exists (compute if missing)
    if 'utilization_percent' not in stats_df.columns:
        util_df = compute_utilization_from_timeline(timeline_df, strong_state=strong_state)
        if not util_df.empty:
            stats_df = stats_df.merge(util_df, on='block_id', how='left')
            stats_df['utilization_percent'] = stats_df['utilization_percent'].fillna(0.0)
            print("Computed utilization_percent from timeline")
        else:
            print("Warning: utilization_percent not found and cannot be computed")

    return timeline_df, stats_df, strong_state

def create_timeline_plot(timeline_df, stats_df, strong_state, app_name=None, max_blocks=None):
    print("Creating timeline visualization...")

    # Ensure utilization_percent exists
    if 'utilization_percent' not in stats_df.columns:
        stats_df = stats_df.copy()
        stats_df['utilization_percent'] = 0.0
    
    # Get active blocks sorted by utilization (descending)
    active_stats = stats_df[stats_df['total_samples'] > 0].copy()
    active_stats_sorted = active_stats.sort_values('utilization_percent', ascending=False)
    
    # Get top 2 blocks by utilization (must be included)
    top_2_blocks = active_stats_sorted['block_id'].head(2).tolist()
    print(f"Top 2 blocks by utilization: {top_2_blocks}")
    if len(top_2_blocks) >= 2:
        top1_util = active_stats_sorted['utilization_percent'].iloc[0]
        top2_util = active_stats_sorted['utilization_percent'].iloc[1]
        print(f"  Block {top_2_blocks[0]}: {top1_util:.2f}%")
        print(f"  Block {top_2_blocks[1]}: {top2_util:.2f}%")
    
    # Get all active blocks (original order by block_id)
    all_active_blocks = stats_df[stats_df['total_samples'] > 0]['block_id'].tolist()
    
    if max_blocks is not None:
        # Select blocks ensuring top 2 are included
        selected_blocks = []
        remaining_slots = max_blocks
        
        # First, add top 2 blocks
        for block in top_2_blocks:
            if block not in selected_blocks:
                selected_blocks.append(block)
                remaining_slots -= 1
        
        # Then fill remaining slots with other blocks (in original order)
        for block in all_active_blocks:
            if remaining_slots <= 0:
                break
            if block not in selected_blocks:
                selected_blocks.append(block)
                remaining_slots -= 1
        
        # Sort by block_id for consistent visualization
        active_blocks = sorted(selected_blocks)
    else:
        active_blocks = all_active_blocks

    filtered_df = timeline_df[timeline_df['block_id'].isin(active_blocks)].copy()

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

    fig_height = max(8, len(active_blocks) * 0.3)
    fig, ax = plt.subplots(figsize=(20, fig_height))

    # Blue (Working), Orange (NotWorking)
    colors = {'Working': '#1f77b4', 'NotWorking': '#ff7f0e'}
    weak_color = colors.get('NotWorking', '#ff7f0e')

    total_duration = global_max_time - global_min_time
    for i, block_id in enumerate(active_blocks):
        block_data = filtered_df[filtered_df['block_id'] == block_id].sort_values('normalized_time')

        if len(block_data) == 0:
            rect = patches.Rectangle(
                (0, i - 0.4), total_duration, 0.8,
                linewidth=0, facecolor=weak_color, alpha=0.5
            )
            ax.add_patch(rect)
            continue

        first_time = block_data['normalized_time'].iloc[0]
        if first_time > 0:
            rect = patches.Rectangle(
                (0, i - 0.4), first_time, 0.8,
                linewidth=0, facecolor=weak_color, alpha=0.5
            )
            ax.add_patch(rect)

        prev_state = None
        start_time = None
        for _, row in block_data.iterrows():
            current_state = row['state_description']
            current_time = row['normalized_time']

            if prev_state is not None and prev_state != current_state:
                duration = current_time - start_time
                if prev_state == 'Working' and max_tasks is not None and cmap is not None and norm is not None:
                    seg_mask = (block_data['normalized_time'] >= start_time) & (block_data['normalized_time'] <= current_time)
                    seg_vals = pd.to_numeric(block_data.loc[seg_mask, 'tasks_in_batch'], errors='coerce') if 'tasks_in_batch' in block_data.columns else None
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

        if prev_state is not None and len(block_data) > 0:
            last_time = block_data['normalized_time'].iloc[-1]
            duration = last_time - start_time
            if prev_state == 'Working' and max_tasks is not None and cmap is not None and norm is not None:
                seg_mask = (block_data['normalized_time'] >= start_time) & (block_data['normalized_time'] <= last_time)
                seg_vals = pd.to_numeric(block_data.loc[seg_mask, 'tasks_in_batch'], errors='coerce') if 'tasks_in_batch' in block_data.columns else None
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

        last_recorded_time = block_data['normalized_time'].iloc[-1]
        
        # データの件数がMAXに達したかどうかをチェック
        max_data_reached = False
        if len(block_data) >= DATA_MAX_LIMIT:
            max_data_reached = True
        
        # データがMAXに達していない場合のみ、残り時間を塗る
        if last_recorded_time < total_duration and not max_data_reached:
            rect = patches.Rectangle(
                (last_recorded_time, i - 0.4), total_duration - last_recorded_time, 0.8,
                linewidth=0, facecolor=weak_color, alpha=0.5
            )
            ax.add_patch(rect)

    ax.set_xlim(0, total_duration)
    ax.set_ylim(-0.5, len(active_blocks) - 0.5)
    ax.set_yticks(range(len(active_blocks)))
    # Mark top 2 blocks with ★ in the label
    ytick_labels = []
    for bid in active_blocks:
        if len(top_2_blocks) > 0 and bid == top_2_blocks[0]:
            ytick_labels.append(f'★ Block {bid} (Top 1)')
        elif len(top_2_blocks) > 1 and bid == top_2_blocks[1]:
            ytick_labels.append(f'★ Block {bid} (Top 2)')
        else:
            ytick_labels.append(f'Block {bid}')
    ax.set_yticklabels(ytick_labels)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Blocks')
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
    """Blockごとのworking時間割合のヒストグラムを作成"""
    print("Creating utilization histogram...")

    # 全blockを含める（total_samples = 0のblockも含む、utilization = 0%）
    all_blocks = stats_df.copy()
    # utilization_percentが存在しない場合は0.0を設定
    if 'utilization_percent' not in all_blocks.columns:
        all_blocks['utilization_percent'] = 0.0
    all_blocks['utilization_percent'] = all_blocks['utilization_percent'].fillna(0.0)
    
    if len(all_blocks) == 0:
        print("No blocks found")
        return None
    
    active_blocks = all_blocks

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(active_blocks['utilization_percent'], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    ax.set_xlabel('Task Execution Time Ratio (%)')
    ax.set_ylabel('Number of Blocks')
    title_suffix = APP_TITLES.get(app_name, app_name) if app_name else ''
    title = f'Distribution of Task Execution Time Ratio per Block:\n{title_suffix}'
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # 統計情報を表示
    mean_util = active_blocks['utilization_percent'].mean()
    median_util = active_blocks['utilization_percent'].median()
    ax.axvline(mean_util, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_util:.1f}%')
    ax.axvline(median_util, color='green', linestyle='--', linewidth=2, label=f'Median: {median_util:.1f}%')
    ax.legend()
    
    plt.tight_layout()
    return fig

def print_summary_statistics(stats_df):
    """統計サマリーを出力"""
    print("\n" + "=" * 60)
    print("BLOCK TIMELINE ANALYSIS SUMMARY (Working)")
    print("=" * 60)

    # 全blockを含める
    all_blocks = stats_df.copy()
    if 'utilization_percent' not in all_blocks.columns:
        all_blocks['utilization_percent'] = 0.0
    all_blocks['utilization_percent'] = all_blocks['utilization_percent'].fillna(0.0)
    
    active_blocks = all_blocks[all_blocks['total_samples'] > 0]
    inactive_blocks = all_blocks[all_blocks['total_samples'] == 0]
    
    print(f"Total Blocks: {len(all_blocks)}")
    print(f"Active Blocks (total_samples > 0): {len(active_blocks)}")
    print(f"Inactive Blocks (total_samples = 0): {len(inactive_blocks)}")
    
    if len(all_blocks) > 0:
        print(f"Average Utilization (all blocks): {all_blocks['utilization_percent'].mean():.2f}%")
        print(f"Utilization Std Dev (all blocks): {all_blocks['utilization_percent'].std():.2f}%")
        print(f"Min Utilization: {all_blocks['utilization_percent'].min():.2f}%")
        print(f"Max Utilization: {all_blocks['utilization_percent'].max():.2f}%")
    
    if len(active_blocks) > 0:
        print(f"\nActive Blocks Only:")
        print(f"Average Utilization: {active_blocks['utilization_percent'].mean():.2f}%")
        print(f"Utilization Std Dev: {active_blocks['utilization_percent'].std():.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Block Timeline Visualization (BFS)')
    parser.add_argument('--app_name', type=str, default='bfs', help='Prefix (app name) for CSV and image outputs')
    args = parser.parse_args()

    print("Block Timeline Visualization Tool (BFS)")
    print("=" * 40)

    try:
        # bfsディレクトリ内から呼び出す前提で、imgディレクトリを直接参照
        img_dir = "img"
        os.makedirs(img_dir, exist_ok=True)
        timeline_df, stats_df, strong_state = load_and_process_data(args.app_name)
        print_summary_statistics(stats_df)
        print("\nGenerating visualizations...")

        # 1. タイムライン図（blockと時間軸のworking/not working）
        timeline_fig = create_timeline_plot(timeline_df, stats_df, strong_state, app_name=args.app_name, max_blocks=15)
        if timeline_fig:
            out_path = os.path.join(img_dir, f"{args.app_name}_timeline.{OUTPUT_FORMAT}")
            timeline_fig.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {out_path}")

        # 2. Blockごとのworking時間割合のヒストグラム
        util_fig = create_utilization_histogram(stats_df, app_name=args.app_name)
        if util_fig:
            out_path = os.path.join(img_dir, f"{args.app_name}_utilization.{OUTPUT_FORMAT}")
            util_fig.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {out_path}")

        print("\nVisualization complete!")

    except FileNotFoundError as e:
        print(f"Error: Could not find required CSV files: {e}")
        print(f"Make sure block_timeline_working.csv and block_statistics_working.csv exist in profile/ directory")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

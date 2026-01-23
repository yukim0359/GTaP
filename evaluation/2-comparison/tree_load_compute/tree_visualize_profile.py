import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

# Use the same style as other profile visualizers, if available
try:
    plt.style.use(os.path.expanduser("~/plot_style/profile.mplstyle"))
except OSError:
    # Fallback to default style if custom style is missing
    pass

DATA_MAX_LIMIT = 30000
# 上限を thread_visualize_profile.py / block_visualize_profile.py に合わせておく
MAX_WARPS_TO_PLOT = 15
MAX_BLOCKS_TO_PLOT = 15
OUTPUT_FORMAT = "pdf"  # "png" or "pdf"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(SCRIPT_DIR, "profile")
IMG_DIR = os.path.join(SCRIPT_DIR, "img")

def _extract_working_durations(timeline_df, strong_state, id_col):
    """タイムラインデータから各working期間の継続時間(ms)を抽出"""
    if (
        id_col not in timeline_df.columns
        or "relative_time_ms" not in timeline_df.columns
        or "state_description" not in timeline_df.columns
    ):
        return []

    durations = []
    for _, grp in timeline_df.groupby(id_col):
        g = grp.sort_values("relative_time_ms").reset_index(drop=True)
        if g.empty:
            continue

        working_start_time = None
        for _, row in g.iterrows():
            state = row["state_description"]
            t = float(row["relative_time_ms"])

            if state == strong_state:
                if working_start_time is None:
                    working_start_time = t
            else:
                if working_start_time is not None:
                    duration = max(0.0, t - working_start_time)
                    if duration > 0.0:
                        durations.append(duration)
                    working_start_time = None

        # 最後の状態がWorkingの場合、最後のタイムスタンプまで
        if working_start_time is not None:
            last_time = float(g["relative_time_ms"].iloc[-1])
            duration = max(0.0, last_time - working_start_time)
            if duration > 0.0:
                durations.append(duration)

    return durations


def _compute_utilization_from_timeline(timeline_df, strong_state, id_col):
    """Compute utilization (percentage of time in strong_state) per id_col."""
    if (
        id_col not in timeline_df.columns
        or "relative_time_ms" not in timeline_df.columns
        or "state_description" not in timeline_df.columns
    ):
        return pd.DataFrame(columns=[id_col, "utilization_percent"])

    util_rows = []
    for ident, grp in timeline_df.groupby(id_col):
        g = grp.sort_values("relative_time_ms").reset_index(drop=True)
        if g.empty:
            util_rows.append({id_col: ident, "utilization_percent": 0.0})
            continue

        first_time = float(g["relative_time_ms"].iloc[0])
        last_time = float(g["relative_time_ms"].iloc[-1])
        total_span = max(0.0, last_time - first_time)
        if total_span <= 0.0:
            util_rows.append({id_col: ident, "utilization_percent": 0.0})
            continue

        working_time = 0.0
        working_start_time = None

        for _, row in g.iterrows():
            state = row["state_description"]
            t = float(row["relative_time_ms"])

            if state == strong_state:
                if working_start_time is None:
                    working_start_time = t
            else:
                if working_start_time is not None:
                    working_time += max(0.0, t - working_start_time)
                    working_start_time = None

        if working_start_time is not None:
            working_time += max(0.0, last_time - working_start_time)

        util = (
            max(0.0, min(100.0, (working_time / total_span) * 100.0))
            if total_span > 0.0
            else 0.0
        )
        util_rows.append({id_col: ident, "utilization_percent": util})

    return pd.DataFrame(util_rows)


def _load_thread_data():
    """Load thread-runtime (warp) profile for tree_load_compute."""
    print("Loading thread (warp) profile data for tree_load_compute...")
    tl_path = os.path.join(PROFILE_DIR, "tree_thread_warp_timeline_working.csv")
    st_path = os.path.join(PROFILE_DIR, "tree_thread_warp_statistics_working.csv")

    timeline_df = pd.read_csv(tl_path)
    stats_df = pd.read_csv(st_path)
    strong_state = "Working"

    if "utilization_percent" not in stats_df.columns:
        util_df = _compute_utilization_from_timeline(
            timeline_df, strong_state=strong_state, id_col="warp_id"
        )
        if not util_df.empty:
            stats_df = stats_df.merge(util_df, on="warp_id", how="left")
            stats_df["utilization_percent"] = stats_df["utilization_percent"].fillna(
                0.0
            )
            print("Computed utilization_percent for warp profile from timeline")

    return timeline_df, stats_df, strong_state


def _load_block_data():
    """Load block-runtime (block) profile for tree_load_compute."""
    print("Loading block (block) profile data for tree_load_compute...")
    tl_path = os.path.join(PROFILE_DIR, "tree_block_block_timeline_working.csv")
    st_path = os.path.join(PROFILE_DIR, "tree_block_block_statistics_working.csv")

    timeline_df = pd.read_csv(tl_path)
    stats_df = pd.read_csv(st_path)
    strong_state = "Working"

    if "utilization_percent" not in stats_df.columns:
        util_df = _compute_utilization_from_timeline(
            timeline_df, strong_state=strong_state, id_col="block_id"
        )
        if not util_df.empty:
            stats_df = stats_df.merge(util_df, on="block_id", how="left")
            stats_df["utilization_percent"] = stats_df["utilization_percent"].fillna(
                0.0
            )
            print("Computed utilization_percent for block profile from timeline")

    return timeline_df, stats_df, strong_state


def _create_timeline_plot(
    timeline_df,
    stats_df,
    strong_state,
    id_col,
    label_prefix,
    max_entries=None,
    color_by_tasks_in_batch=True,
    custom_title=None,
):
    print(f"Creating {label_prefix} timeline visualization...")

    active_ids = stats_df[stats_df["total_samples"] > 0][id_col].tolist()
    if max_entries is not None:
        active_ids = active_ids[:max_entries]
    if not active_ids:
        print("No active entries to visualize")
        return None

    filtered_df = timeline_df[timeline_df[id_col].isin(active_ids)].copy()
    if len(filtered_df) == 0:
        print("No data to visualize after filtering")
        return None

    global_min_time = timeline_df["relative_time_ms"].min()
    global_max_time = timeline_df["relative_time_ms"].max()
    filtered_df["normalized_time"] = filtered_df["relative_time_ms"] - global_min_time

    max_tasks = None
    cmap = None
    norm = None
    if color_by_tasks_in_batch and "tasks_in_batch" in timeline_df.columns:
        try:
            max_tasks_val = pd.to_numeric(
                timeline_df["tasks_in_batch"], errors="coerce"
            ).max()
            if pd.notna(max_tasks_val) and float(max_tasks_val) > 0.0:
                max_tasks = float(max_tasks_val)
                cmap = plt.cm.Blues
                norm = mpl.colors.Normalize(vmin=0.0, vmax=max_tasks)
        except Exception:
            max_tasks = None

    fig_height = max(8, len(active_ids) * 0.3)
    fig, ax = plt.subplots(figsize=(20, fig_height))

    colors = {"Working": "#1f77b4", "NotWorking": "#ff7f0e"}
    weak_color = colors.get("NotWorking", "#ff7f0e")

    total_duration = global_max_time - global_min_time
    for i, ident in enumerate(active_ids):
        entry_data = filtered_df[filtered_df[id_col] == ident].sort_values(
            "normalized_time"
        )

        if len(entry_data) == 0:
            rect = patches.Rectangle(
                (0, i - 0.4),
                total_duration,
                0.8,
                linewidth=0,
                facecolor=weak_color,
                alpha=0.5,
            )
            ax.add_patch(rect)
            continue

        first_time = entry_data["normalized_time"].iloc[0]
        if first_time > 0:
            rect = patches.Rectangle(
                (0, i - 0.4),
                first_time,
                0.8,
                linewidth=0,
                facecolor=weak_color,
                alpha=0.5,
            )
            ax.add_patch(rect)

        prev_state = None
        start_time = None
        for _, row in entry_data.iterrows():
            current_state = row["state_description"]
            current_time = row["normalized_time"]

            if prev_state is not None and prev_state != current_state:
                duration = current_time - start_time
                if (
                    color_by_tasks_in_batch
                    and prev_state == strong_state
                    and max_tasks is not None
                    and cmap is not None
                    and norm is not None
                ):
                    seg_mask = (entry_data["normalized_time"] >= start_time) & (
                        entry_data["normalized_time"] <= current_time
                    )
                    seg_vals = (
                        pd.to_numeric(
                            entry_data.loc[seg_mask, "tasks_in_batch"],
                            errors="coerce",
                        )
                        if "tasks_in_batch" in entry_data.columns
                        else None
                    )
                    seg_max = (
                        float(seg_vals.max())
                        if seg_vals is not None
                        and not seg_vals.empty
                        and pd.notna(seg_vals.max())
                        else 0.0
                    )
                    color = cmap(norm(seg_max))
                    alpha = 0.9
                else:
                    color = colors.get(prev_state, "#888888")
                    alpha = 0.8 if prev_state == strong_state else 0.5
                rect = patches.Rectangle(
                    (start_time, i - 0.4),
                    duration,
                    0.8,
                    linewidth=0,
                    facecolor=color,
                    alpha=alpha,
                )
                ax.add_patch(rect)

            if prev_state != current_state:
                start_time = current_time
                prev_state = current_state

        if prev_state is not None and len(entry_data) > 0:
            last_time = entry_data["normalized_time"].iloc[-1]
            duration = last_time - start_time
            if (
                color_by_tasks_in_batch
                and prev_state == strong_state
                and max_tasks is not None
                and cmap is not None
                and norm is not None
            ):
                seg_mask = (entry_data["normalized_time"] >= start_time) & (
                    entry_data["normalized_time"] <= last_time
                )
                seg_vals = (
                    pd.to_numeric(
                        entry_data.loc[seg_mask, "tasks_in_batch"], errors="coerce"
                    )
                    if "tasks_in_batch" in entry_data.columns
                    else None
                )
                seg_max = (
                    float(seg_vals.max())
                    if seg_vals is not None
                    and not seg_vals.empty
                    and pd.notna(seg_vals.max())
                    else 0.0
                )
                color = cmap(norm(seg_max))
                alpha = 0.9
            else:
                color = colors.get(prev_state, "#888888")
                alpha = 0.8 if prev_state == strong_state else 0.5
            rect = patches.Rectangle(
                (start_time, i - 0.4),
                duration,
                0.8,
                linewidth=0,
                facecolor=color,
                alpha=alpha,
            )
            ax.add_patch(rect)

        last_recorded_time = entry_data["normalized_time"].iloc[-1]
        max_data_reached = len(entry_data) >= DATA_MAX_LIMIT
        if last_recorded_time < total_duration and not max_data_reached:
            rect = patches.Rectangle(
                (last_recorded_time, i - 0.4),
                total_duration - last_recorded_time,
                0.8,
                linewidth=0,
                facecolor=weak_color,
                alpha=0.5,
            )
            ax.add_patch(rect)

    ax.set_xlim(0, total_duration)
    ax.set_ylim(-0.5, len(active_ids) - 0.5)
    ax.set_yticks(range(len(active_ids)))
    ax.set_yticklabels([f"{label_prefix} {ident}" for ident in active_ids])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(label_prefix + "s")
    if custom_title:
        ax.set_title(f"Worker Timeline Visualization: {custom_title}")
    else:
        ax.set_title("Worker Timeline Visualization: Synthetic Tree")

    ax.grid(True, alpha=0.3)
    legend_elements = [
        patches.Patch(color=colors["Working"], alpha=0.8, label="Executing taskfn"),
        patches.Patch(color=colors["NotWorking"], alpha=0.5, label="Not executing taskfn"),
    ]

    if color_by_tasks_in_batch and "tasks_in_batch" in filtered_df.columns:
        working_df = filtered_df[filtered_df["state_description"] == strong_state]
        if len(working_df) > 0:
            tasks_vals = pd.to_numeric(
                working_df["tasks_in_batch"], errors="coerce"
            ).dropna()
            if len(tasks_vals) > 0:
                avg_tasks = tasks_vals.mean()
                legend_elements.append(
                    patches.Patch(
                        color="none",
                        label=f"Avg tasks per batch: {avg_tasks:.2f}",
                    )
                )

    ax.legend(handles=legend_elements, loc="upper right")

    if color_by_tasks_in_batch and max_tasks and cmap is not None and norm is not None:
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("tasks in batch")

    plt.tight_layout()
    return fig


def _create_utilization_histogram(stats_df, label_prefix):
    print(f"Creating {label_prefix} utilization histogram...")

    all_entries = stats_df.copy()
    if "utilization_percent" not in all_entries.columns:
        all_entries["utilization_percent"] = 0.0
    all_entries["utilization_percent"] = all_entries["utilization_percent"].fillna(0.0)

    if len(all_entries) == 0:
        print("No entries found")
        return None

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(
        all_entries["utilization_percent"],
        bins=20,
        alpha=0.7,
        color="lightblue",
        edgecolor="black",
    )
    ax.set_xlabel("Task Execution Time Ratio (%)")
    ax.set_ylabel(f"Number of {label_prefix}s")
    ax.set_title(
        f"Distribution of Task Execution Time Ratio per {label_prefix}:\n"
        "Synthetic Tree"
    )
    ax.grid(True, alpha=0.3)

    mean_util = all_entries["utilization_percent"].mean()
    median_util = all_entries["utilization_percent"].median()
    ax.axvline(
        mean_util,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_util:.1f}%",
    )
    ax.axvline(
        median_util,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_util:.1f}%",
    )
    ax.legend()

    plt.tight_layout()
    return fig


def _create_working_duration_histogram(working_durations, label_prefix, title):
    """各working期間の継続時間(ms)のヒストグラムを作成"""
    print(f"Creating {label_prefix} working duration histogram...")

    if not working_durations:
        print("No working durations found")
        return None

    durations_ms = pd.Series(working_durations)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(durations_ms, bins=50, alpha=0.7, color="lightgreen", edgecolor="black")
    ax.set_xlabel("Task Execution Time (ms)")
    ax.set_ylabel("Number of Execution Periods")
    ax.set_title(
        f"Distribution of Task Execution Time per Loop:\n"
        f"{title}"
    )
    ax.grid(True, alpha=0.3)

    mean_dur = durations_ms.mean()
    median_dur = durations_ms.median()
    ax.axvline(
        mean_dur,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_dur:.3f} ms",
    )
    ax.axvline(
        median_dur,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_dur:.3f} ms",
    )
    ax.legend()

    plt.tight_layout()
    return fig


def visualize_thread(custom_title=None):
    timeline_df, stats_df, strong_state = _load_thread_data()
    os.makedirs(IMG_DIR, exist_ok=True)

    default_title = "Synthetic Tree (Thread-level workers)"
    timeline_title = custom_title if custom_title else default_title

    working_durations = _extract_working_durations(
        timeline_df, strong_state=strong_state, id_col="warp_id"
    )

    tl_fig = _create_timeline_plot(
        timeline_df,
        stats_df,
        strong_state,
        id_col="warp_id",
        label_prefix="Thread",
        max_entries=MAX_WARPS_TO_PLOT,
        color_by_tasks_in_batch=True,
        custom_title=timeline_title,
    )
    if tl_fig:
        out_path = os.path.join(IMG_DIR, f"tree_thread_timeline.{OUTPUT_FORMAT}")
        tl_fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")

    util_fig = _create_utilization_histogram(stats_df, label_prefix="Warp")
    if util_fig:
        out_path = os.path.join(IMG_DIR, f"tree_thread_utilization.{OUTPUT_FORMAT}")
        util_fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")

    dur_fig = _create_working_duration_histogram(
        working_durations, label_prefix="Thread", title=default_title
    )
    if dur_fig:
        out_path = os.path.join(IMG_DIR, f"tree_thread_working_duration.{OUTPUT_FORMAT}")
        dur_fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")


def visualize_block(custom_title=None):
    timeline_df, stats_df, strong_state = _load_block_data()
    os.makedirs(IMG_DIR, exist_ok=True)

    default_title = "Synthetic Tree (Block-level workers)"
    timeline_title = custom_title if custom_title else default_title

    working_durations = _extract_working_durations(
        timeline_df, strong_state=strong_state, id_col="block_id"
    )

    tl_fig = _create_timeline_plot(
        timeline_df,
        stats_df,
        strong_state,
        id_col="block_id",
        label_prefix="Block",
        max_entries=MAX_BLOCKS_TO_PLOT,
        color_by_tasks_in_batch=False,
        custom_title=timeline_title,
    )
    if tl_fig:
        out_path = os.path.join(IMG_DIR, f"tree_block_timeline.{OUTPUT_FORMAT}")
        tl_fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")

    util_fig = _create_utilization_histogram(stats_df, label_prefix="Block")
    if util_fig:
        out_path = os.path.join(IMG_DIR, f"tree_block_utilization.{OUTPUT_FORMAT}")
        util_fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")

    dur_fig = _create_working_duration_histogram(working_durations, label_prefix="Block", title=default_title)
    if dur_fig:
        out_path = os.path.join(IMG_DIR, f"tree_block_working_duration.{OUTPUT_FORMAT}")
        dur_fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Timeline Visualization for tree_load_compute (thread & block)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["thread", "block", "both"],
        default="both",
        help="Which profile to visualize",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title to show on figures (e.g., run config or dataset)",
    )
    args = parser.parse_args()

    print("tree_load_compute Profile Visualization")
    print("=" * 40)

    try:
        if args.mode in ("thread", "both"):
            visualize_thread(custom_title=args.title)
        if args.mode in ("block", "both"):
            visualize_block(custom_title=args.title)
        print("\nVisualization complete!")
    except FileNotFoundError as e:
        print(f"Error: Could not find required CSV files: {e}")
        print(f"Make sure CSVs exist under {PROFILE_DIR}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()



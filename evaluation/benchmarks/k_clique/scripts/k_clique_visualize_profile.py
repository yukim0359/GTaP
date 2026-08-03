import argparse
import os

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd

try:
    plt.style.use(os.path.expanduser("~/plot_style/profile.mplstyle"))
except OSError:
    pass

VARIANTS = {
    "orientation": {
        "app_name": "k_clique_orientation",
        "title": "K-Clique Orientation",
    },
    "pivot": {
        "app_name": "k_clique_pivot",
        "title": "K-Clique Pivot",
    },
}

APP_NAME = VARIANTS["orientation"]["app_name"]
APP_TITLE = VARIANTS["orientation"]["title"]
PROFILE_TAG = ""
DATA_MAX_LIMIT = 30000
OUTPUT_FORMAT = "pdf"
MAX_WARPS_TO_PLOT = 15
MAX_BLOCKS_TO_PLOT = 15
MANDATORY_TOP_BUSY_ENTRIES = 3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
K_CLIQUE_DIR = os.path.dirname(SCRIPT_DIR)
PROFILE_DIR = os.path.join(K_CLIQUE_DIR, "profile")
IMG_DIR = os.path.join(K_CLIQUE_DIR, "img")

MODES = {
    "working": {
        "suffix": "working",
        "strong_state": "Working",
        "weak_state": "NotWorking",
        "legend_on": "Executing taskfn",
        "legend_off": "Not executing taskfn",
        "hist_xlabel": "Task Execution Time Ratio (%)",
        "hist_title": "Distribution of Task Execution Time Ratio",
        "dur_xlabel": "Task Execution Time (ms)",
        "dur_title": "Distribution of Task Execution Time per Loop",
    },
    "having_task": {
        "suffix": "having_task",
        "strong_state": "Having",
        "weak_state": "NotHaving",
        "legend_on": "Having task",
        "legend_off": "Not having task",
        "hist_xlabel": "Task Having Time Ratio (%)",
        "hist_title": "Distribution of Task Having Time Ratio",
        "dur_xlabel": "Task Having Time (ms)",
        "dur_title": "Distribution of Task Having Time per Loop",
    },
}

WORKERS = {
    "thread": {
        "id_col": "warp_id",
        "label_prefix": "Warp",
        "max_entries": MAX_WARPS_TO_PLOT,
        "title_suffix": "Thread-level workers",
        "out_prefix": "thread",
        "csv_primary_stem": "warp",
        "csv_fallback_prefix": "",
    },
    "block": {
        "id_col": "block_id",
        "label_prefix": "Block",
        "max_entries": MAX_BLOCKS_TO_PLOT,
        "title_suffix": "Block-level workers",
        "out_prefix": "block",
        "csv_primary_stem": "block_block",
        "csv_fallback_prefix": "block_",
    },
}


def compute_utilization_from_timeline(timeline_df, strong_state, id_col):
    required = {id_col, "relative_time_ms", "state_description"}
    if not required.issubset(timeline_df.columns):
        return pd.DataFrame(columns=[id_col, "utilization_percent"])

    program_first_time = float(timeline_df["relative_time_ms"].min())
    program_last_time = float(timeline_df["relative_time_ms"].max())
    program_total_time = max(0.0, program_last_time - program_first_time)

    if program_total_time <= 0.0:
        return pd.DataFrame(
            [
                {id_col: ident, "utilization_percent": 0.0}
                for ident in timeline_df[id_col].unique()
            ]
        )

    util_rows = []
    for ident, grp in timeline_df.groupby(id_col):
        g = grp.sort_values("relative_time_ms").reset_index(drop=True)
        if g.empty:
            util_rows.append({id_col: ident, "utilization_percent": 0.0})
            continue

        active_time = 0.0
        active_start = None
        for _, row in g.iterrows():
            state = row["state_description"]
            t = float(row["relative_time_ms"])
            if state == strong_state:
                if active_start is None:
                    active_start = t
            elif active_start is not None:
                active_time += max(0.0, t - active_start)
                active_start = None

        if active_start is not None:
            last_time = float(g["relative_time_ms"].iloc[-1])
            active_time += max(0.0, last_time - active_start)

        util = max(0.0, min(100.0, (active_time / program_total_time) * 100.0))
        util_rows.append({id_col: ident, "utilization_percent": util})

    return pd.DataFrame(util_rows)


def compute_active_time_from_timeline(timeline_df, strong_state, id_col):
    required = {id_col, "relative_time_ms", "state_description"}
    if not required.issubset(timeline_df.columns):
        return pd.DataFrame(columns=[id_col, "active_time_ms"])

    rows = []
    for ident, grp in timeline_df.groupby(id_col):
        g = grp.sort_values("relative_time_ms").reset_index(drop=True)
        active_time = 0.0
        active_start = None

        for _, row in g.iterrows():
            state = row["state_description"]
            t = float(row["relative_time_ms"])
            if state == strong_state:
                if active_start is None:
                    active_start = t
            elif active_start is not None:
                active_time += max(0.0, t - active_start)
                active_start = None

        if active_start is not None and not g.empty:
            active_time += max(0.0, float(g["relative_time_ms"].iloc[-1]) - active_start)

        rows.append({id_col: ident, "active_time_ms": active_time})

    return pd.DataFrame(rows)


def extract_active_durations(timeline_df, strong_state, id_col):
    required = {id_col, "relative_time_ms", "state_description"}
    if not required.issubset(timeline_df.columns):
        return []

    durations = []
    for _, grp in timeline_df.groupby(id_col):
        g = grp.sort_values("relative_time_ms").reset_index(drop=True)
        if g.empty:
            continue

        active_start = None
        for _, row in g.iterrows():
            state = row["state_description"]
            t = float(row["relative_time_ms"])
            if state == strong_state:
                if active_start is None:
                    active_start = t
            elif active_start is not None:
                duration = max(0.0, t - active_start)
                if duration > 0.0:
                    durations.append(duration)
                active_start = None

        if active_start is not None:
            last_time = float(g["relative_time_ms"].iloc[-1])
            duration = max(0.0, last_time - active_start)
            if duration > 0.0:
                durations.append(duration)

    return durations


def profile_csv_paths(worker, mode):
    suffix = MODES[mode]["suffix"]
    worker_cfg = WORKERS[worker]
    primary_stem = worker_cfg["csv_primary_stem"]
    fallback_prefix = worker_cfg["csv_fallback_prefix"]

    primary_timeline = os.path.join(
        PROFILE_DIR, f"{APP_NAME}_{primary_stem}_timeline_{suffix}{PROFILE_TAG}.csv"
    )
    primary_stats = os.path.join(
        PROFILE_DIR, f"{APP_NAME}_{primary_stem}_statistics_{suffix}{PROFILE_TAG}.csv"
    )
    fallback_timeline = os.path.join(
        PROFILE_DIR, f"{fallback_prefix}timeline_{suffix}.csv"
    )
    fallback_stats = os.path.join(
        PROFILE_DIR, f"{fallback_prefix}statistics_{suffix}.csv"
    )

    timeline_path = primary_timeline if os.path.exists(primary_timeline) else fallback_timeline
    stats_path = primary_stats if os.path.exists(primary_stats) else fallback_stats
    if PROFILE_TAG and not os.path.exists(primary_timeline):
        timeline_path = primary_timeline
    if PROFILE_TAG and not os.path.exists(primary_stats):
        stats_path = primary_stats
    return timeline_path, stats_path


def load_data(worker, mode):
    mode_cfg = MODES[mode]
    worker_cfg = WORKERS[worker]
    strong_state = mode_cfg["strong_state"]
    id_col = worker_cfg["id_col"]
    timeline_path, stats_path = profile_csv_paths(worker, mode)

    print(f"Loading: {timeline_path}")
    print(f"Loading: {stats_path}")

    timeline_df = pd.read_csv(timeline_path)
    stats_df = pd.read_csv(stats_path)

    if "utilization_percent" not in stats_df.columns:
        util_df = compute_utilization_from_timeline(timeline_df, strong_state, id_col)
        if not util_df.empty:
            stats_df = stats_df.merge(util_df, on=id_col, how="left")
            stats_df["utilization_percent"] = stats_df["utilization_percent"].fillna(0.0)

    active_time_df = compute_active_time_from_timeline(timeline_df, strong_state, id_col)
    if not active_time_df.empty:
        stats_df = stats_df.drop(columns=["active_time_ms"], errors="ignore")
        stats_df = stats_df.merge(active_time_df, on=id_col, how="left")
        stats_df["active_time_ms"] = stats_df["active_time_ms"].fillna(0.0)

    return timeline_df, stats_df


def select_active_entries(stats_df, id_col, label_prefix, max_entries):
    active_stats = stats_df[stats_df["total_samples"] > 0].copy()
    if active_stats.empty:
        return []

    if "active_time_ms" in active_stats.columns:
        top_entries = (
            active_stats.sort_values("active_time_ms", ascending=False)[id_col]
            .head(MANDATORY_TOP_BUSY_ENTRIES)
            .tolist()
        )
    else:
        top_entries = active_stats[id_col].head(MANDATORY_TOP_BUSY_ENTRIES).tolist()

    print(f"Top {len(top_entries)} busy {label_prefix.lower()}s by active time: {top_entries}")
    if "active_time_ms" in active_stats.columns:
        for ident in top_entries:
            active_time = active_stats.loc[
                active_stats[id_col] == ident, "active_time_ms"
            ].iloc[0]
            print(f"  {label_prefix} {ident}: {active_time:.6f} ms")

    active_entries = active_stats[id_col].tolist()
    if max_entries is None:
        return active_entries

    max_entries = max(max_entries, len(top_entries))
    selected = []
    for ident in top_entries:
        if ident not in selected:
            selected.append(ident)

    for ident in active_entries:
        if len(selected) >= max_entries:
            break
        if ident not in selected:
            selected.append(ident)

    return sorted(selected)


def add_state_rect(
    ax,
    entry_data,
    state,
    start_time,
    end_time,
    row_idx,
    strong_state,
    colors,
    cmap,
    norm,
    max_tasks,
):
    duration = end_time - start_time
    if duration <= 0:
        return

    if (
        state == strong_state
        and max_tasks is not None
        and cmap is not None
        and norm is not None
        and "tasks_in_batch" in entry_data.columns
    ):
        seg_mask = (entry_data["normalized_time"] >= start_time) & (
            entry_data["normalized_time"] <= end_time
        )
        seg_vals = pd.to_numeric(entry_data.loc[seg_mask, "tasks_in_batch"], errors="coerce")
        seg_max = float(seg_vals.max()) if not seg_vals.empty and pd.notna(seg_vals.max()) else 0.0
        color = cmap(norm(seg_max))
        alpha = 0.9
    else:
        color = colors.get(state, "#888888")
        alpha = 0.8 if state == strong_state else 0.5

    ax.add_patch(
        patches.Rectangle(
            (start_time, row_idx - 0.4),
            duration,
            0.8,
            linewidth=0,
            facecolor=color,
            alpha=alpha,
        )
    )


def create_timeline_plot(timeline_df, stats_df, worker, mode, max_entries):
    mode_cfg = MODES[mode]
    worker_cfg = WORKERS[worker]
    strong_state = mode_cfg["strong_state"]
    weak_state = mode_cfg["weak_state"]
    id_col = worker_cfg["id_col"]
    label_prefix = worker_cfg["label_prefix"]

    active_entries = select_active_entries(stats_df, id_col, label_prefix, max_entries)
    if not active_entries:
        print(f"No active {label_prefix.lower()}s found for worker={worker}, mode={mode}")
        return None

    filtered_df = timeline_df[timeline_df[id_col].isin(active_entries)].copy()
    if filtered_df.empty:
        print(f"No timeline rows after filtering for worker={worker}, mode={mode}")
        return None

    global_min_time = timeline_df["relative_time_ms"].min()
    global_max_time = timeline_df["relative_time_ms"].max()
    total_duration = global_max_time - global_min_time
    filtered_df["normalized_time"] = filtered_df["relative_time_ms"] - global_min_time

    max_tasks = None
    cmap = None
    norm = None
    if "tasks_in_batch" in timeline_df.columns:
        max_tasks_val = pd.to_numeric(timeline_df["tasks_in_batch"], errors="coerce").max()
        if pd.notna(max_tasks_val) and float(max_tasks_val) > 0.0:
            max_tasks = float(max_tasks_val)
            cmap = plt.cm.Blues
            norm = mpl.colors.Normalize(vmin=0.0, vmax=max_tasks)

    fig_height = max(8, len(active_entries) * 0.3)
    base_width, _ = plt.rcParams.get("figure.figsize", [6.4, 4.8])
    fig, ax = plt.subplots(figsize=(base_width * 1.6, fig_height))

    colors = {strong_state: "#1f77b4", weak_state: "#ff7f0e"}
    weak_color = colors[weak_state]

    for row_idx, ident in enumerate(active_entries):
        entry_data = filtered_df[filtered_df[id_col] == ident].sort_values(
            "normalized_time"
        )

        if entry_data.empty:
            ax.add_patch(
                patches.Rectangle(
                    (0, row_idx - 0.4),
                    total_duration,
                    0.8,
                    linewidth=0,
                    facecolor=weak_color,
                    alpha=0.5,
                )
            )
            continue

        first_time = entry_data["normalized_time"].iloc[0]
        if first_time > 0:
            ax.add_patch(
                patches.Rectangle(
                    (0, row_idx - 0.4),
                    first_time,
                    0.8,
                    linewidth=0,
                    facecolor=weak_color,
                    alpha=0.5,
                )
            )

        prev_state = None
        start_time = None
        for _, row in entry_data.iterrows():
            current_state = row["state_description"]
            current_time = row["normalized_time"]

            if prev_state is not None and prev_state != current_state:
                add_state_rect(
                    ax,
                    entry_data,
                    prev_state,
                    start_time,
                    current_time,
                    row_idx,
                    strong_state,
                    colors,
                    cmap,
                    norm,
                    max_tasks,
                )

            if prev_state != current_state:
                start_time = current_time
                prev_state = current_state

        if prev_state is not None:
            last_time = entry_data["normalized_time"].iloc[-1]
            add_state_rect(
                ax,
                entry_data,
                prev_state,
                start_time,
                last_time,
                row_idx,
                strong_state,
                colors,
                cmap,
                norm,
                max_tasks,
            )

        last_recorded_time = entry_data["normalized_time"].iloc[-1]
        if last_recorded_time < total_duration and len(entry_data) < DATA_MAX_LIMIT:
            ax.add_patch(
                patches.Rectangle(
                    (last_recorded_time, row_idx - 0.4),
                    total_duration - last_recorded_time,
                    0.8,
                    linewidth=0,
                    facecolor=weak_color,
                    alpha=0.5,
                )
            )

    ax.set_xlim(0, total_duration)
    ax.set_ylim(-0.5, len(active_entries) - 0.5)
    ax.set_yticks(range(len(active_entries)))
    ax.set_yticklabels([f"{label_prefix} {ident}" for ident in active_entries])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(f"{label_prefix}s")
    ax.set_title(
        f"Worker Timeline: {APP_TITLE} ({worker_cfg['title_suffix']}, {mode})"
    )
    ax.grid(True, alpha=0)

    legend_elements = [
        patches.Patch(color=colors[strong_state], alpha=0.8, label=mode_cfg["legend_on"]),
        patches.Patch(color=colors[weak_state], alpha=0.5, label=mode_cfg["legend_off"]),
    ]
    if "tasks_in_batch" in filtered_df.columns:
        active_df = filtered_df[filtered_df["state_description"] == strong_state]
        tasks_vals = pd.to_numeric(active_df["tasks_in_batch"], errors="coerce").dropna()
        if not tasks_vals.empty:
            legend_elements.append(
                patches.Patch(
                    color="none",
                    label=f"Avg tasks per batch: {tasks_vals.mean():.2f}",
                )
            )
    ax.legend(handles=legend_elements, loc="upper right")

    if max_tasks and cmap is not None and norm is not None:
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("tasks in batch")

    plt.tight_layout()
    return fig


def create_utilization_histogram(stats_df, worker, mode):
    mode_cfg = MODES[mode]
    worker_cfg = WORKERS[worker]
    label_prefix = worker_cfg["label_prefix"]

    all_entries = stats_df.copy()
    if "utilization_percent" not in all_entries.columns:
        all_entries["utilization_percent"] = 0.0
    all_entries["utilization_percent"] = all_entries["utilization_percent"].fillna(0.0)

    if all_entries.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(
        all_entries["utilization_percent"],
        bins=20,
        alpha=0.7,
        color="lightblue",
        edgecolor="black",
    )
    ax.set_xlabel(mode_cfg["hist_xlabel"])
    ax.set_ylabel(f"Number of {label_prefix}s")
    ax.set_title(
        f"{mode_cfg['hist_title']} per {label_prefix}:\n"
        f"{APP_TITLE} ({worker_cfg['title_suffix']}, {mode})"
    )
    ax.grid(True, alpha=0.3)

    mean_util = all_entries["utilization_percent"].mean()
    median_util = all_entries["utilization_percent"].median()
    ax.axvline(mean_util, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_util:.1f}%")
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


def create_active_duration_histogram(durations, worker, mode):
    mode_cfg = MODES[mode]
    worker_cfg = WORKERS[worker]
    label_prefix = worker_cfg["label_prefix"]

    if not durations:
        print(f"No active durations to plot for worker={worker}, mode={mode}")
        return None

    durations_ms = pd.Series(durations)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(durations_ms, bins=50, alpha=0.7, color="lightgreen", edgecolor="black")
    ax.set_xlabel(mode_cfg["dur_xlabel"])
    ax.set_ylabel("Number of Periods")
    ax.set_title(
        f"{mode_cfg['dur_title']}:\n"
        f"{APP_TITLE} ({worker_cfg['title_suffix']}, {mode})"
    )
    ax.grid(True, alpha=0.3)

    mean_dur = durations_ms.mean()
    median_dur = durations_ms.median()
    ax.axvline(
        mean_dur, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_dur:.3f} ms"
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


def create_tasks_in_batch_histogram(timeline_df, worker, mode):
    mode_cfg = MODES[mode]
    worker_cfg = WORKERS[worker]
    strong_state = mode_cfg["strong_state"]

    if "tasks_in_batch" not in timeline_df.columns:
        print(f"No tasks_in_batch column to plot for worker={worker}, mode={mode}")
        return None

    active_df = timeline_df[timeline_df["state_description"] == strong_state]
    task_lengths = pd.to_numeric(active_df["tasks_in_batch"], errors="coerce").dropna()
    task_lengths = task_lengths[task_lengths > 0]
    if task_lengths.empty:
        print(f"No task length samples to plot for worker={worker}, mode={mode}")
        return None

    max_len = int(task_lengths.max())
    bins = min(50, max_len) if max_len > 1 else 1

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(task_lengths, bins=bins, alpha=0.7, color="#9ecae1", edgecolor="black")
    ax.set_xlabel("Tasks in Batch")
    ax.set_ylabel("Number of Samples")
    ax.set_title(
        "Distribution of Task Batch Lengths:\n"
        f"{APP_TITLE} ({worker_cfg['title_suffix']}, {mode})"
    )
    ax.grid(True, alpha=0.3)

    mean_len = task_lengths.mean()
    median_len = task_lengths.median()
    ax.axvline(
        mean_len,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_len:.2f}",
    )
    ax.axvline(
        median_len,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_len:.2f}",
    )
    ax.legend()

    plt.tight_layout()
    return fig


def print_summary(stats_df, worker, mode):
    worker_cfg = WORKERS[worker]
    label_prefix = worker_cfg["label_prefix"]

    all_entries = stats_df.copy()
    if "utilization_percent" not in all_entries.columns:
        all_entries["utilization_percent"] = 0.0
    all_entries["utilization_percent"] = all_entries["utilization_percent"].fillna(0.0)

    active_entries = all_entries[all_entries["total_samples"] > 0]
    inactive_entries = all_entries[all_entries["total_samples"] == 0]

    print("\n" + "=" * 60)
    print(f"{APP_NAME.upper()} PROFILE SUMMARY: worker={worker}, mode={mode}")
    print("=" * 60)
    print(f"Total {label_prefix}s: {len(all_entries)}")
    print(f"Active {label_prefix}s (total_samples > 0): {len(active_entries)}")
    print(f"Inactive {label_prefix}s (total_samples = 0): {len(inactive_entries)}")
    if len(all_entries) > 0:
        print(f"Average Utilization: {all_entries['utilization_percent'].mean():.2f}%")
        print(f"Std Dev Utilization: {all_entries['utilization_percent'].std():.2f}%")
        print(f"Min Utilization: {all_entries['utilization_percent'].min():.2f}%")
        print(f"Max Utilization: {all_entries['utilization_percent'].max():.2f}%")


def run_visualization(worker, mode, max_entries):
    timeline_df, stats_df = load_data(worker, mode)
    print_summary(stats_df, worker, mode)

    os.makedirs(IMG_DIR, exist_ok=True)
    suffix = MODES[mode]["suffix"]
    worker_cfg = WORKERS[worker]
    out_prefix = worker_cfg["out_prefix"]
    strong_state = MODES[mode]["strong_state"]
    id_col = worker_cfg["id_col"]

    tag_suffix = PROFILE_TAG if PROFILE_TAG else ""

    timeline_fig = create_timeline_plot(timeline_df, stats_df, worker, mode, max_entries)
    if timeline_fig:
        out_path = os.path.join(
            IMG_DIR, f"{APP_NAME}{tag_suffix}_{out_prefix}_{suffix}_timeline.{OUTPUT_FORMAT}"
        )
        timeline_fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")

    util_fig = create_utilization_histogram(stats_df, worker, mode)
    if util_fig:
        out_path = os.path.join(
            IMG_DIR, f"{APP_NAME}{tag_suffix}_{out_prefix}_{suffix}_utilization.{OUTPUT_FORMAT}"
        )
        util_fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")

    durations = extract_active_durations(timeline_df, strong_state, id_col)
    dur_fig = create_active_duration_histogram(durations, worker, mode)
    if dur_fig:
        out_path = os.path.join(
            IMG_DIR, f"{APP_NAME}{tag_suffix}_{out_prefix}_{suffix}_duration.{OUTPUT_FORMAT}"
        )
        dur_fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")

    task_len_fig = create_tasks_in_batch_histogram(timeline_df, worker, mode)
    if task_len_fig:
        out_path = os.path.join(
            IMG_DIR, f"{APP_NAME}{tag_suffix}_{out_prefix}_{suffix}_task_length.{OUTPUT_FORMAT}"
        )
        task_len_fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")


def main():
    global APP_NAME, APP_TITLE, PROFILE_DIR, IMG_DIR, OUTPUT_FORMAT, PROFILE_TAG

    parser = argparse.ArgumentParser(description="generic k-clique GTaP profile visualization")
    parser.add_argument(
        "--variant",
        choices=["orientation", "pivot", "both"],
        default="both",
        help="Which k-clique GTaP variant profile to visualize",
    )
    parser.add_argument(
        "--app-name",
        default=None,
        help="Profile CSV prefix written by gtap_export_profile",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Title used in figures",
    )
    parser.add_argument(
        "--profile-dir",
        default=PROFILE_DIR,
        help="Directory containing profile CSV files",
    )
    parser.add_argument(
        "--img-dir",
        default=IMG_DIR,
        help="Directory for generated figures",
    )
    parser.add_argument(
        "--format",
        default=OUTPUT_FORMAT,
        help="Output figure format",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix appended to profile CSV filenames, e.g. _DBLP_k7_edge_o1b",
    )
    parser.add_argument(
        "--worker",
        choices=["thread", "block", "both"],
        default="both",
        help="Which worker runtime profile to visualize",
    )
    parser.add_argument(
        "--mode",
        choices=["working", "having_task", "both"],
        default="both",
        help="Which profile metric to visualize",
    )
    parser.add_argument(
        "--max-warps",
        type=int,
        default=MAX_WARPS_TO_PLOT,
        help="Maximum number of active warps to draw in thread timeline",
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=MAX_BLOCKS_TO_PLOT,
        help="Maximum number of active blocks to draw in block timeline",
    )
    args = parser.parse_args()

    if args.variant == "both" and args.app_name is not None:
        parser.error("--app-name can only be used with a single --variant")

    PROFILE_DIR = args.profile_dir
    IMG_DIR = args.img_dir
    OUTPUT_FORMAT = args.format
    PROFILE_TAG = args.tag

    variants = ["orientation", "pivot"] if args.variant == "both" else [args.variant]
    workers = ["thread", "block"] if args.worker == "both" else [args.worker]
    modes = ["working", "having_task"] if args.mode == "both" else [args.mode]

    for variant in variants:
        variant_cfg = VARIANTS[variant]
        APP_NAME = args.app_name if args.app_name is not None else variant_cfg["app_name"]
        APP_TITLE = args.title if args.title is not None else variant_cfg["title"]

        print(f"{APP_TITLE} Profile Visualization")
        print("=" * 40)
        for worker in workers:
            max_entries = args.max_warps if worker == "thread" else args.max_blocks
            for mode in modes:
                try:
                    run_visualization(worker, mode, max_entries)
                except FileNotFoundError as e:
                    print(f"Skipping variant={variant}, worker={worker}, mode={mode}: missing CSV ({e})")
                except Exception as e:
                    print(f"Error for variant={variant}, worker={worker}, mode={mode}: {e}")

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

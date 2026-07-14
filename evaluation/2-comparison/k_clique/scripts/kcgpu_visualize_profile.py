#!/usr/bin/env python3
"""Visualize KCGPU profile CSVs."""

import argparse
import csv
import glob
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd

try:
    plt.style.use(os.path.expanduser("~/plot_style/profile.mplstyle"))
except OSError:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
K_CLIQUE_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_PROFILE_DIR = os.path.join(K_CLIQUE_DIR, "profile", "kcgpu")
DEFAULT_IMG_DIR = os.path.join(K_CLIQUE_DIR, "img", "kcgpu")
MAX_WORKERS_TO_PLOT = 0
MANDATORY_TOP_BUSY_WORKERS = 3
TIMELINE_CHUNK_ROWS = 500_000
SM_TIMELINE_BINS = 1200


def _worker_col(df):
    if "slot_id" in df.columns:
        return "slot_id"
    if "worker_id" in df.columns:
        return "worker_id"
    return "warp_id"


def _timeline_paths(profile_dir, app_name, tag):
    timeline_path = os.path.join(profile_dir, f"{app_name}_warp_timeline_working{tag}.csv")
    stats_path = os.path.join(profile_dir, f"{app_name}_warp_statistics_working{tag}.csv")
    return timeline_path, stats_path


def _infer_slots_per_sm(stats_df, id_col, sm_count=132):
    active = stats_df[stats_df["total_samples"] > 0]
    if active.empty:
        return 1
    max_slot = int(active[id_col].max())
    return max(1, (max_slot + sm_count) // sm_count)


def _compute_active_time(timeline_df, id_col):
    rows = []
    durations = []
    for ident, grp in timeline_df.groupby(id_col):
        g = grp.sort_values("relative_time_ms")
        active_start = None
        active_time = 0.0
        for _, row in g.iterrows():
            state = row["state_description"]
            t = float(row["relative_time_ms"])
            if state == "Working":
                if active_start is None:
                    active_start = t
            elif active_start is not None:
                duration = max(0.0, t - active_start)
                active_time += duration
                if duration > 0.0:
                    durations.append(duration)
                active_start = None
        if active_start is not None and not g.empty:
            duration = max(0.0, float(g["relative_time_ms"].iloc[-1]) - active_start)
            active_time += duration
            if duration > 0.0:
                durations.append(duration)
        rows.append({id_col: ident, "active_time_ms": active_time})
    return pd.DataFrame(rows), durations


def _read_selected_timeline(timeline_path, id_col, active_workers):
    chunks = []
    worker_set = set(active_workers)
    for chunk in pd.read_csv(timeline_path, chunksize=TIMELINE_CHUNK_ROWS):
        selected = chunk[chunk[id_col].isin(worker_set)]
        if not selected.empty:
            chunks.append(selected.copy())
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _load_timeline(profile_dir, app_name, tag, max_workers):
    timeline_path, stats_path = _timeline_paths(profile_dir, app_name, tag)
    print(f"Loading KCGPU worker-slot timeline: {timeline_path}")
    print(f"Loading KCGPU worker-slot statistics: {stats_path}")

    stats_df = pd.read_csv(stats_path)
    header_df = pd.read_csv(timeline_path, nrows=0)
    id_col = _worker_col(header_df)
    stats_id_col = _worker_col(stats_df)
    if stats_id_col != id_col:
        stats_df = stats_df.rename(columns={stats_id_col: id_col})
    if "utilization_percent" not in stats_df.columns:
        stats_df["utilization_percent"] = 0.0

    active_workers = _select_workers(stats_df, id_col, max_workers)
    if not active_workers:
        return pd.DataFrame(), stats_df, [], id_col, []

    timeline_df = _read_selected_timeline(timeline_path, id_col, active_workers)
    if timeline_df.empty:
        return timeline_df, stats_df, [], id_col, active_workers

    active_df, durations = _compute_active_time(timeline_df, id_col)
    stats_df = stats_df.drop(columns=["active_time_ms"], errors="ignore")
    stats_df = stats_df.merge(active_df, on=id_col, how="left")
    stats_df["active_time_ms"] = stats_df["active_time_ms"].fillna(0.0)
    selected_mask = stats_df[id_col].isin(active_workers)
    active_stats = stats_df[stats_df["total_samples"] > 0]
    if not active_stats.empty and {"first_activity_ms", "last_activity_ms"}.issubset(stats_df.columns):
        first_ms = float(active_stats["first_activity_ms"].min())
        last_ms = float(active_stats["last_activity_ms"].max())
    else:
        first_ms = float(timeline_df["relative_time_ms"].min())
        last_ms = float(timeline_df["relative_time_ms"].max())
    selected_duration = max(0.0, last_ms - first_ms)
    if selected_duration > 0.0:
        stats_df.loc[selected_mask, "utilization_percent"] = (
            stats_df.loc[selected_mask, "active_time_ms"] / selected_duration
        ) * 100.0
    stats_df["utilization_percent"] = stats_df["utilization_percent"].clip(0.0, 100.0)

    return timeline_df, stats_df, durations, id_col, active_workers


def _select_workers(stats_df, id_col, max_workers):
    active = stats_df[stats_df["total_samples"] > 0].copy()
    if active.empty:
        return []

    rank_col = "duration_ms" if "duration_ms" in active.columns else "recorded_samples"
    top_busy = (
        active.sort_values(rank_col, ascending=False)[id_col]
        .head(MANDATORY_TOP_BUSY_WORKERS)
        .tolist()
    )
    print(f"Top {len(top_busy)} KCGPU worker slots by {rank_col}: {top_busy}")
    for ident in top_busy:
        rank_value = active.loc[active[id_col] == ident, rank_col].iloc[0]
        print(f"  Slot {ident}: {rank_value}")

    if max_workers is None or max_workers <= 0:
        print(f"Warning: plotting all {len(active)} active slots may use a lot of memory")
        return sorted(active[id_col].tolist())

    selected = []
    for ident in top_busy + active[id_col].tolist():
        if ident not in selected:
            selected.append(ident)
        if len(selected) >= max(max_workers, len(top_busy)):
            break
    return sorted(selected)


def plot_worker_timeline(profile_dir, img_dir, app_name, tag, output_format, max_workers):
    timeline_df, stats_df, durations, id_col, active_workers = _load_timeline(
        profile_dir, app_name, tag, max_workers
    )
    if not active_workers or timeline_df.empty:
        print("No active KCGPU worker slots found")
        return

    active_stats = stats_df[stats_df["total_samples"] > 0]
    if not active_stats.empty and {"first_activity_ms", "last_activity_ms"}.issubset(stats_df.columns):
        global_min = float(active_stats["first_activity_ms"].min())
        global_max = float(active_stats["last_activity_ms"].max())
    else:
        global_min = float(timeline_df["relative_time_ms"].min())
        global_max = float(timeline_df["relative_time_ms"].max())
    total_duration = max(0.0, global_max - global_min)
    filtered = timeline_df[timeline_df[id_col].isin(active_workers)].copy()
    filtered["normalized_time"] = filtered["relative_time_ms"] - global_min

    os.makedirs(img_dir, exist_ok=True)
    selected_stats = stats_df[stats_df[id_col].isin(active_workers)].copy()

    fig_height = max(8, len(active_workers) * 0.3)
    base_width, _ = plt.rcParams.get("figure.figsize", [6.4, 4.8])
    fig, ax = plt.subplots(figsize=(base_width * 1.6, fig_height))
    colors = {"Working": "#1f77b4", "NotWorking": "#ff7f0e"}
    weak_color = colors["NotWorking"]

    for row_idx, ident in enumerate(active_workers):
        worker_df = filtered[filtered[id_col] == ident].sort_values("normalized_time")
        if worker_df.empty:
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

        first_time = worker_df["normalized_time"].iloc[0]
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
        for _, row in worker_df.iterrows():
            state = row["state_description"]
            t = float(row["normalized_time"])
            if prev_state is not None and prev_state != state:
                duration = max(0.0, t - start_time)
                if duration > 0.0:
                    ax.add_patch(
                        patches.Rectangle(
                            (start_time, row_idx - 0.4),
                            duration,
                            0.8,
                            linewidth=0,
                            facecolor=colors.get(prev_state, "#888888"),
                            alpha=0.8 if prev_state == "Working" else 0.5,
                        )
                    )
            if prev_state != state:
                prev_state = state
                start_time = t

        if prev_state is not None and start_time is not None:
            end_time = float(worker_df["normalized_time"].iloc[-1])
            duration = max(0.0, end_time - start_time)
            if duration > 0.0:
                ax.add_patch(
                    patches.Rectangle(
                        (start_time, row_idx - 0.4),
                        duration,
                        0.8,
                        linewidth=0,
                        facecolor=colors.get(prev_state, "#888888"),
                        alpha=0.8 if prev_state == "Working" else 0.5,
                    )
                )

        last_recorded_time = worker_df["normalized_time"].iloc[-1]
        if last_recorded_time < total_duration:
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
    ax.set_ylim(-0.5, len(active_workers) - 0.5)
    ax.set_yticks(range(len(active_workers)))
    ax.set_yticklabels([f"Slot {ident}" for ident in active_workers])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Slots")
    ax.set_title(
        f"Worker Timeline: KCGPU (Resident-slot workers, working)"
    )
    ax.grid(True, alpha=0)
    ax.legend(
        handles=[
            patches.Patch(color=colors["Working"], alpha=0.8, label="Processing queue item"),
            patches.Patch(color=colors["NotWorking"], alpha=0.5, label="Not processing"),
        ],
        loc="upper right",
    )
    fig.tight_layout()
    out_path = os.path.join(img_dir, f"{app_name}{tag}_slot_working_timeline.{output_format}")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(
        selected_stats["utilization_percent"],
        bins=20,
        alpha=0.7,
        color="lightblue",
        edgecolor="black",
    )
    mean_util = selected_stats["utilization_percent"].mean()
    median_util = selected_stats["utilization_percent"].median()
    ax.axvline(mean_util, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_util:.1f}%")
    ax.axvline(
        median_util,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_util:.1f}%",
    )
    ax.set_xlabel("Working time ratio (%)")
    ax.set_ylabel("Number of Slots")
    ax.set_title(
        "Distribution of Working Time Ratio per Slot:\n"
        "KCGPU (Resident-slot workers, working)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_path = os.path.join(img_dir, f"{app_name}{tag}_slot_working_utilization.{output_format}")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    if durations:
        fig, ax = plt.subplots(figsize=(12, 8))
        durations_ms = pd.Series(durations)
        ax.hist(durations_ms, bins=50, alpha=0.7, color="lightgreen", edgecolor="black")
        mean_dur = durations_ms.mean()
        median_dur = durations_ms.median()
        ax.axvline(mean_dur, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_dur:.3f} ms")
        ax.axvline(
            median_dur,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_dur:.3f} ms",
        )
        ax.set_xlabel("Working interval duration (ms)")
        ax.set_ylabel("Number of Periods")
        ax.set_title(
            "Distribution of Working Time per Loop:\n"
            "KCGPU (Resident-slot workers, working)"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        out_path = os.path.join(img_dir, f"{app_name}{tag}_slot_working_duration.{output_format}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    total_workers = len(stats_df)
    active_count = int((stats_df["total_samples"] > 0).sum())
    print("\n" + "=" * 60)
    print("KCGPU WORKER-SLOT PROFILE SUMMARY")
    print("=" * 60)
    print(f"Total slots: {total_workers}")
    print(f"Active slots (total_samples > 0): {active_count}")
    print(f"Inactive slots (total_samples = 0): {total_workers - active_count}")
    print(f"Plotted slots: {len(active_workers)}")
    print(f"Average plotted-slot utilization: {selected_stats['utilization_percent'].mean():.2f}%")
    print(f"Max plotted-slot utilization: {selected_stats['utilization_percent'].max():.2f}%")


def plot_sm_balance(profile_dir, img_dir, app_name, tag, output_format):
    pattern = os.path.join(profile_dir, f"{app_name}_sm_balance{tag}.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f"No SM balance CSV matching {pattern}")
        return

    sm_path = matches[-1]
    print(f"Loading SM balance: {sm_path}")
    sm_df = pd.read_csv(sm_path)
    active = sm_df[sm_df["node_visits"] > 0].copy()
    if active.empty:
        print("No active SMs in SM balance CSV")
        return

    stats_pattern = os.path.join(profile_dir, f"{app_name}_warp_statistics_working{tag}.csv")
    stats_matches = sorted(glob.glob(stats_pattern))
    if stats_matches:
        stats_df = pd.read_csv(stats_matches[-1])
        active_slots = stats_df[stats_df["total_samples"] > 0].copy()
        if not active_slots.empty and "duration_ms" in active_slots.columns:
            slot_col = _worker_col(active_slots)
            sm_count = max(len(sm_df), 1)
            max_slot = int(active_slots[slot_col].max())
            slots_per_sm = max(1, (max_slot + sm_count) // sm_count)
            active_slots["sm_id"] = active_slots[slot_col] // slots_per_sm
            sm_duration = active_slots.groupby("sm_id")["duration_ms"].sum()
            active["slot_duration_ms"] = active["sm_id"].map(sm_duration).fillna(0.0)
            avg_duration = active["slot_duration_ms"].mean()
            if avg_duration > 0.0:
                active["slot_duration_ratio_to_avg"] = active["slot_duration_ms"] / avg_duration

    os.makedirs(img_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].bar(active["sm_id"], active["node_ratio_to_avg"], color="#2ca02c", alpha=0.85)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Node visits / avg")
    axes[0].set_title("SM Node-Visit Balance:\nKCGPU")
    axes[0].grid(True, alpha=0.3)

    if "slot_duration_ratio_to_avg" in active.columns:
        time_ratio_col = "slot_duration_ratio_to_avg"
        time_ylabel = "Slot duration / avg"
        time_title = "SM Slot-Duration Balance:\nKCGPU"
    else:
        time_ratio_col = "time_ratio_to_avg" if "time_ratio_to_avg" in active.columns else "cycle_ratio_to_avg"
        time_ylabel = "Search time / avg"
        time_title = "SM Search-Time Balance:\nKCGPU"
    axes[1].bar(active["sm_id"], active[time_ratio_col], color="#d62728", alpha=0.85)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("SM id")
    axes[1].set_ylabel(time_ylabel)
    axes[1].set_title(time_title)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(img_dir, f"{app_name}{tag}_sm_balance.{output_format}")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    spread_nodes = active["node_ratio_to_avg"].max() - active["node_ratio_to_avg"].min()
    spread_time = active[time_ratio_col].max() - active[time_ratio_col].min()
    print(f"SM spread (max-min ratio): nodes={spread_nodes:.3f}, time={spread_time:.3f}")


def plot_slot_balance(profile_dir, img_dir, app_name, tag, output_format):
    _, stats_path = _timeline_paths(profile_dir, app_name, tag)
    if not os.path.exists(stats_path):
        print(f"No KCGPU slot statistics CSV: {stats_path}")
        return

    print(f"Loading KCGPU slot balance: {stats_path}")
    stats_df = pd.read_csv(stats_path)
    id_col = _worker_col(stats_df)
    active = stats_df[stats_df["total_samples"] > 0].copy()
    if active.empty or "duration_ms" not in active.columns:
        print("No active KCGPU slots in statistics CSV")
        return

    active = active.sort_values(id_col)
    avg_duration = active["duration_ms"].mean()
    if avg_duration <= 0.0:
        print("No positive KCGPU slot duration in statistics CSV")
        return

    active["duration_ratio_to_avg"] = active["duration_ms"] / avg_duration

    os.makedirs(img_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(active[id_col], active["duration_ratio_to_avg"], color="#1f77b4", alpha=0.85, width=1.0)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Worker slot id")
    ax.set_ylabel("Slot duration / avg")
    ax.set_title("Worker-Slot Duration Balance:\nKCGPU")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(img_dir, f"{app_name}{tag}_slot_balance.{output_format}")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    top = active.nlargest(5, "duration_ms")
    spread = active["duration_ratio_to_avg"].max() - active["duration_ratio_to_avg"].min()
    print(f"Slot spread (max-min ratio): duration={spread:.3f}")
    print("Top KCGPU worker slots by duration_ms:")
    for _, row in top.iterrows():
        print(f"  Slot {int(row[id_col])}: {row['duration_ms']:.6f} ms")


def _add_interval_to_sm_timeline(timeline, sm_id, start_ms, end_ms, first_ms, bin_width_ms):
    if end_ms <= start_ms or bin_width_ms <= 0.0:
        return

    bins = len(timeline[sm_id])
    start_idx = int((start_ms - first_ms) / bin_width_ms)
    end_idx = int((end_ms - first_ms) / bin_width_ms)
    if start_idx < 0:
        start_idx = 0
    if end_idx >= bins:
        end_idx = bins - 1
    if start_idx > end_idx:
        return

    for b in range(start_idx, end_idx + 1):
        bin_start = first_ms + b * bin_width_ms
        bin_end = bin_start + bin_width_ms
        overlap = min(end_ms, bin_end) - max(start_ms, bin_start)
        if overlap > 0.0:
            timeline[sm_id][b] += overlap / bin_width_ms


def plot_sm_working_timeline(profile_dir, img_dir, app_name, tag, output_format, bins):
    timeline_path, stats_path = _timeline_paths(profile_dir, app_name, tag)
    if not os.path.exists(stats_path):
        print(f"No KCGPU slot statistics CSV for SM timeline: {stats_path}")
        return
    if not os.path.exists(timeline_path):
        print(f"No KCGPU slot timeline CSV for SM timeline: {timeline_path}")
        return

    print(f"Loading KCGPU SM working timeline stats: {stats_path}")
    stats_df = pd.read_csv(stats_path)
    id_col = _worker_col(stats_df)
    active = stats_df[stats_df["total_samples"] > 0].copy()
    required = {"first_activity_ms", "last_activity_ms"}
    if active.empty or not required.issubset(active.columns):
        print("No active slot first/last activity data for SM timeline")
        return

    sm_count = 132
    slots_per_sm = _infer_slots_per_sm(active, id_col, sm_count)
    active["sm_id"] = active[id_col] // slots_per_sm
    first_ms = float(active["first_activity_ms"].min())
    last_ms = float(active["last_activity_ms"].max())
    duration_ms = max(0.0, last_ms - first_ms)
    if duration_ms <= 0.0:
        print("No positive KCGPU SM timeline duration")
        return

    bins = max(1, int(bins))
    bin_width = duration_ms / bins
    timeline = [[0.0 for _ in range(bins)] for _ in range(sm_count)]

    active_start_by_slot = {}
    print(f"Loading KCGPU SM working timeline events: {timeline_path}")
    with open(timeline_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("Empty KCGPU slot timeline CSV")
            return
        if "slot_id" in reader.fieldnames:
            event_id_col = "slot_id"
        elif "worker_id" in reader.fieldnames:
            event_id_col = "worker_id"
        else:
            event_id_col = "warp_id"

        for row in reader:
            slot_id = int(row[event_id_col])
            sm_id = slot_id // slots_per_sm
            if sm_id < 0 or sm_id >= sm_count:
                continue
            t = float(row["relative_time_ms"])
            state = int(row["state"])
            if state == 1:
                if slot_id not in active_start_by_slot:
                    active_start_by_slot[slot_id] = t
            elif slot_id in active_start_by_slot:
                start = active_start_by_slot.pop(slot_id)
                _add_interval_to_sm_timeline(timeline, sm_id, start, t, first_ms, bin_width)

    for slot_id, start in active_start_by_slot.items():
        sm_id = slot_id // slots_per_sm
        if 0 <= sm_id < sm_count:
            _add_interval_to_sm_timeline(timeline, sm_id, start, last_ms, first_ms, bin_width)

    os.makedirs(img_dir, exist_ok=True)
    fig_height = max(8, sm_count * 0.055)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    im = ax.imshow(
        timeline,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=[0.0, duration_ms / 1000.0, -0.5, sm_count - 0.5],
        cmap="viridis",
        vmin=0,
        vmax=slots_per_sm,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SM id")
    ax.set_title("SM Working Timeline (Binned Active Slots):\nKCGPU")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Active worker slots")
    fig.tight_layout()
    out_path = os.path.join(img_dir, f"{app_name}{tag}_sm_working_timeline.{output_format}")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize KCGPU working/not-working timelines")
    parser.add_argument("--profile-dir", default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--img-dir", default=DEFAULT_IMG_DIR)
    parser.add_argument("--variant", choices=["orientation", "pivot"], default="orientation")
    parser.add_argument("--app-name", default=None)
    parser.add_argument("--tag", default="", help="Profile tag suffix, e.g. _DBLP_k7_o1b")
    parser.add_argument("--format", default="pdf")
    parser.add_argument("--skip-timeline", action="store_true")
    parser.add_argument(
        "--sm-timeline-only",
        action="store_true",
        help="Generate only the SM working timeline figure (skip balance and slot plots).",
    )
    parser.add_argument("--sm-timeline-bins", type=int, default=SM_TIMELINE_BINS)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS_TO_PLOT,
        help="Maximum active worker slots to draw in the timeline; 0 means all active slots.",
    )
    args = parser.parse_args()

    os.makedirs(args.img_dir, exist_ok=True)
    app_name = args.app_name or f"kcgpu_{args.variant}"

    plot_sm_working_timeline(
        args.profile_dir,
        args.img_dir,
        app_name,
        args.tag,
        args.format,
        args.sm_timeline_bins,
    )

    if args.sm_timeline_only:
        return

    # SM balance is cheap; draw it first so PBS jobs still get this figure if timeline OOMs.
    plot_sm_balance(args.profile_dir, args.img_dir, app_name, args.tag, args.format)
    plot_slot_balance(args.profile_dir, args.img_dir, app_name, args.tag, args.format)

    if not args.skip_timeline:
        plot_worker_timeline(
            args.profile_dir,
            args.img_dir,
            app_name,
            args.tag,
            args.format,
            args.max_workers,
        )


if __name__ == "__main__":
    main()

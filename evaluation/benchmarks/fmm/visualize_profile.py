#!/usr/bin/env python3
"""Visualize GTaP warp profiles from fmm_profile (gtap_fmm.cu)."""

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

DATA_MAX_LIMIT = 30000
OUTPUT_FORMAT = "pdf"  # "png" or "pdf"
MAX_WARPS_TO_PLOT = 15

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(SCRIPT_DIR, "profile")
IMG_DIR = os.path.join(SCRIPT_DIR, "img")

# Must match the result directory passed to gtap_export_profile(...).
PROFILE_PREFIX = "fmm3d_dtt"

PHASES = {
    "dtt": {
        "prefix": PROFILE_PREFIX,
        "title": "FMM3D GTaP DTT (fmm3d_dtt_fill_kernel)",
    },
}

ALL_PHASES = list(PHASES.keys())

MODES = {
    "working": {
        "suffix": "working",
        "strong_state": "Working",
        "weak_state": "NotWorking",
        "legend_on": "Executing taskfn",
        "legend_off": "Not executing taskfn",
        "hist_xlabel": "Task Execution Time Ratio (%)",
        "hist_title": "Distribution of Task Execution Time Ratio per Warp",
    },
    "having_task": {
        "suffix": "having_task",
        "strong_state": "Having",
        "weak_state": "NotHaving",
        "legend_on": "Having task",
        "legend_off": "Not having task",
        "hist_xlabel": "Task Having Time Ratio (%)",
        "hist_title": "Distribution of Task Having Time Ratio per Warp",
    },
}


def compute_utilization_from_timeline(timeline_df, strong_state):
    if (
        "warp_id" not in timeline_df.columns
        or "relative_time_ms" not in timeline_df.columns
        or "state_description" not in timeline_df.columns
    ):
        return pd.DataFrame(columns=["warp_id", "utilization_percent"])

    program_first_time = float(timeline_df["relative_time_ms"].min())
    program_last_time = float(timeline_df["relative_time_ms"].max())
    program_total_time = max(0.0, program_last_time - program_first_time)

    if program_total_time <= 0.0:
        return pd.DataFrame(
            [
                {"warp_id": warp_id, "utilization_percent": 0.0}
                for warp_id in timeline_df["warp_id"].unique()
            ]
        )

    util_rows = []
    for warp_id, grp in timeline_df.groupby("warp_id"):
        g = grp.sort_values("relative_time_ms").reset_index(drop=True)
        if g.empty:
            util_rows.append({"warp_id": warp_id, "utilization_percent": 0.0})
            continue

        active_time = 0.0
        active_start = None
        for _, row in g.iterrows():
            state = row["state_description"]
            t = float(row["relative_time_ms"])
            if state == strong_state:
                if active_start is None:
                    active_start = t
            else:
                if active_start is not None:
                    active_time += max(0.0, t - active_start)
                    active_start = None

        if active_start is not None:
            last_time = float(g["relative_time_ms"].iloc[-1])
            active_time += max(0.0, last_time - active_start)

        util = max(0.0, min(100.0, (active_time / program_total_time) * 100.0))
        util_rows.append({"warp_id": warp_id, "utilization_percent": util})

    return pd.DataFrame(util_rows)


def resolve_profile_prefix(phase, mode):
    phase_cfg = PHASES[phase]
    mode_cfg = MODES[mode]
    suffix = mode_cfg["suffix"]
    prefix = phase_cfg["prefix"]

    timeline_path = os.path.join(PROFILE_DIR, f"{prefix}_warp_timeline_{suffix}.csv")
    stats_path = os.path.join(PROFILE_DIR, f"{prefix}_warp_statistics_{suffix}.csv")
    if os.path.exists(timeline_path) and os.path.exists(stats_path):
        return prefix, timeline_path, stats_path

    raise FileNotFoundError(
        "No profile CSV pair found for "
        f"phase={phase}, mode={mode}.\n"
        f"Expected:\n"
        f"  {timeline_path}\n"
        f"  {stats_path}\n"
        "Build and run: make fmm_profile && ./bin/fmm_profile <N> <theta>"
    )


def load_data(phase, mode):
    mode_cfg = MODES[mode]
    strong_state = mode_cfg["strong_state"]
    prefix, timeline_path, stats_path = resolve_profile_prefix(phase, mode)
    print(f"Profile label: {prefix}")
    print(f"Loading: {timeline_path}")
    print(f"Loading: {stats_path}")

    timeline_df = pd.read_csv(timeline_path)
    stats_df = pd.read_csv(stats_path)

    if "utilization_percent" not in stats_df.columns:
        util_df = compute_utilization_from_timeline(timeline_df, strong_state=strong_state)
        if not util_df.empty:
            stats_df = stats_df.merge(util_df, on="warp_id", how="left")
            stats_df["utilization_percent"] = stats_df["utilization_percent"].fillna(0.0)

    return timeline_df, stats_df, prefix


def create_timeline_plot(timeline_df, stats_df, phase, mode, max_warps=None):
    mode_cfg = MODES[mode]
    strong_state = mode_cfg["strong_state"]
    weak_state = mode_cfg["weak_state"]

    active_warps = stats_df[stats_df["total_samples"] > 0]["warp_id"].tolist()
    if max_warps is not None:
        active_warps = active_warps[:max_warps]

    if not active_warps:
        print(f"No active warps found for phase={phase}, mode={mode}")
        return None

    filtered_df = timeline_df[timeline_df["warp_id"].isin(active_warps)].copy()
    if filtered_df.empty:
        print(f"No timeline rows after filtering for phase={phase}, mode={mode}")
        return None

    global_min_time = timeline_df["relative_time_ms"].min()
    global_max_time = timeline_df["relative_time_ms"].max()
    filtered_df["normalized_time"] = filtered_df["relative_time_ms"] - global_min_time

    max_tasks = None
    cmap = None
    norm = None
    if "tasks_in_batch" in timeline_df.columns:
        max_tasks_val = pd.to_numeric(
            timeline_df["tasks_in_batch"], errors="coerce"
        ).max()
        if pd.notna(max_tasks_val) and float(max_tasks_val) > 0.0:
            max_tasks = float(max_tasks_val)
            cmap = plt.cm.Blues
            norm = mpl.colors.Normalize(vmin=0.0, vmax=max_tasks)

    fig_height = max(8, len(active_warps) * 0.3)
    _w, _h = plt.rcParams.get("figure.figsize", [6.4, 4.8])
    fig_width = _w * 1.6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    colors = {strong_state: "#1f77b4", weak_state: "#ff7f0e"}
    weak_color = colors.get(weak_state, "#ff7f0e")
    total_duration = global_max_time - global_min_time

    for i, warp_id in enumerate(active_warps):
        warp_data = filtered_df[filtered_df["warp_id"] == warp_id].sort_values(
            "normalized_time"
        )

        if warp_data.empty:
            rect = patches.Rectangle(
                (0, i - 0.4), total_duration, 0.8, linewidth=0, facecolor=weak_color, alpha=0.5
            )
            ax.add_patch(rect)
            continue

        first_time = warp_data["normalized_time"].iloc[0]
        if first_time > 0:
            rect = patches.Rectangle(
                (0, i - 0.4), first_time, 0.8, linewidth=0, facecolor=weak_color, alpha=0.5
            )
            ax.add_patch(rect)

        prev_state = None
        start_time = None
        for _, row in warp_data.iterrows():
            current_state = row["state_description"]
            current_time = row["normalized_time"]

            if prev_state is not None and prev_state != current_state:
                duration = current_time - start_time
                if (
                    prev_state == strong_state
                    and max_tasks is not None
                    and cmap is not None
                    and norm is not None
                    and "tasks_in_batch" in warp_data.columns
                ):
                    seg_mask = (warp_data["normalized_time"] >= start_time) & (
                        warp_data["normalized_time"] <= current_time
                    )
                    seg_vals = pd.to_numeric(
                        warp_data.loc[seg_mask, "tasks_in_batch"], errors="coerce"
                    )
                    seg_max = (
                        float(seg_vals.max())
                        if not seg_vals.empty and pd.notna(seg_vals.max())
                        else 0.0
                    )
                    color = cmap(norm(seg_max))
                    alpha = 0.9
                else:
                    color = colors.get(prev_state, "#888888")
                    alpha = 0.8 if prev_state == strong_state else 0.5

                rect = patches.Rectangle(
                    (start_time, i - 0.4), duration, 0.8, linewidth=0, facecolor=color, alpha=alpha
                )
                ax.add_patch(rect)

            if prev_state != current_state:
                start_time = current_time
                prev_state = current_state

        if prev_state is not None and len(warp_data) > 0:
            last_time = warp_data["normalized_time"].iloc[-1]
            duration = last_time - start_time
            if (
                prev_state == strong_state
                and max_tasks is not None
                and cmap is not None
                and norm is not None
                and "tasks_in_batch" in warp_data.columns
            ):
                seg_mask = (warp_data["normalized_time"] >= start_time) & (
                    warp_data["normalized_time"] <= last_time
                )
                seg_vals = pd.to_numeric(
                    warp_data.loc[seg_mask, "tasks_in_batch"], errors="coerce"
                )
                seg_max = (
                    float(seg_vals.max())
                    if not seg_vals.empty and pd.notna(seg_vals.max())
                    else 0.0
                )
                color = cmap(norm(seg_max))
                alpha = 0.9
            else:
                color = colors.get(prev_state, "#888888")
                alpha = 0.8 if prev_state == strong_state else 0.5

            rect = patches.Rectangle(
                (start_time, i - 0.4), duration, 0.8, linewidth=0, facecolor=color, alpha=alpha
            )
            ax.add_patch(rect)

        last_recorded_time = warp_data["normalized_time"].iloc[-1]
        max_data_reached = len(warp_data) >= DATA_MAX_LIMIT
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
    ax.set_ylim(-0.5, len(active_warps) - 0.5)
    ax.set_yticks(range(len(active_warps)))
    ax.set_yticklabels([f"Warp {wid}" for wid in active_warps])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Warps")
    ax.set_title(f"Worker Timeline: {PHASES[phase]['title']} ({mode})")

    legend_elements = [
        patches.Patch(color=colors[strong_state], alpha=0.8, label=mode_cfg["legend_on"]),
        patches.Patch(color=colors[weak_state], alpha=0.5, label=mode_cfg["legend_off"]),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    if max_tasks and cmap is not None and norm is not None:
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("tasks in batch")

    plt.tight_layout()
    return fig


def create_utilization_histogram(stats_df, phase, mode):
    mode_cfg = MODES[mode]
    all_warps = stats_df.copy()
    if "utilization_percent" not in all_warps.columns:
        all_warps["utilization_percent"] = 0.0
    all_warps["utilization_percent"] = all_warps["utilization_percent"].fillna(0.0)

    if all_warps.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(
        all_warps["utilization_percent"],
        bins=20,
        alpha=0.7,
        color="lightblue",
        edgecolor="black",
    )
    ax.set_xlabel(mode_cfg["hist_xlabel"])
    ax.set_ylabel("Number of Warps")
    ax.set_title(f"{mode_cfg['hist_title']}:\n{PHASES[phase]['title']} ({mode})")
    ax.grid(True, alpha=0.3)

    mean_util = all_warps["utilization_percent"].mean()
    median_util = all_warps["utilization_percent"].median()
    ax.axvline(mean_util, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_util:.1f}%")
    ax.axvline(
        median_util, color="green", linestyle="--", linewidth=2, label=f"Median: {median_util:.1f}%"
    )
    ax.legend()

    plt.tight_layout()
    return fig


def print_summary(stats_df, phase, mode):
    all_warps = stats_df.copy()
    if "utilization_percent" not in all_warps.columns:
        all_warps["utilization_percent"] = 0.0
    all_warps["utilization_percent"] = all_warps["utilization_percent"].fillna(0.0)

    active_warps = all_warps[all_warps["total_samples"] > 0]
    inactive_warps = all_warps[all_warps["total_samples"] == 0]

    print("\n" + "=" * 60)
    print(f"FMM3D profile summary: {PHASES[phase]['title']} / {mode}")
    print("=" * 60)
    print(f"Total Warps: {len(all_warps)}")
    print(f"Active Warps (total_samples > 0): {len(active_warps)}")
    print(f"Inactive Warps (total_samples = 0): {len(inactive_warps)}")
    if len(all_warps) > 0:
        print(f"Average Utilization: {all_warps['utilization_percent'].mean():.2f}%")
        print(f"Std Dev Utilization: {all_warps['utilization_percent'].std():.2f}%")
        print(f"Min Utilization: {all_warps['utilization_percent'].min():.2f}%")
        print(f"Max Utilization: {all_warps['utilization_percent'].max():.2f}%")


def run_visualization(phase, mode, max_warps):
    timeline_df, stats_df, prefix = load_data(phase, mode)
    print_summary(stats_df, phase, mode)

    os.makedirs(IMG_DIR, exist_ok=True)
    mode_suffix = MODES[mode]["suffix"]

    timeline_fig = create_timeline_plot(
        timeline_df, stats_df, phase=phase, mode=mode, max_warps=max_warps
    )
    if timeline_fig:
        out_path = os.path.join(IMG_DIR, f"{prefix}_{mode_suffix}_timeline.{OUTPUT_FORMAT}")
        timeline_fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")

    util_fig = create_utilization_histogram(stats_df, phase=phase, mode=mode)
    if util_fig:
        out_path = os.path.join(IMG_DIR, f"{prefix}_{mode_suffix}_utilization.{OUTPUT_FORMAT}")
        util_fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GTaP warp profiles from gtap_fmm.cu (fmm_profile build)."
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=ALL_PHASES,
        default="dtt",
        help=f"Profile phase (prefix {PROFILE_PREFIX})",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["working", "having_task", "both"],
        default="both",
        help="Which state metric to visualize",
    )
    parser.add_argument(
        "--max-warps",
        type=int,
        default=MAX_WARPS_TO_PLOT,
        help="Maximum number of active warps to draw in timeline",
    )
    args = parser.parse_args()

    modes = ["working", "having_task"] if args.mode == "both" else [args.mode]

    print("FMM3D profile visualization (gtap_fmm.cu / fmm_profile)")
    print("=" * 40)
    for mode in modes:
        run_visualization(args.phase, mode, max_warps=args.max_warps)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

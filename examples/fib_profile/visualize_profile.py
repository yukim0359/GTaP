#!/usr/bin/env python3
"""
Visualize GTaP warp-level profile data for the Fibonacci example.

After running `./bin/fib`, the runtime writes CSV files under `./profile/`:
  - fib_warp_timeline_working.csv   -- state-change events per warp
  - fib_warp_statistics_working.csv -- per-warp summary statistics

This script reads those CSVs and produces two figures under `./img/`:
  - fib_timeline.png     -- warp activity over time (Working / Not executing)
  - fib_utilization.png  -- histogram of per-warp task execution time ratio

Usage::

    python3 visualize_profile.py [--profile-dir ./profile] [--app fib]
                                 [--max-warps 15] [--output-dir ./img]
                                 [--format png|pdf]

Requirements: pandas, matplotlib
"""
import os
import argparse
from typing import Optional

import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -------------------------------------------------------------------------
# Style
# -------------------------------------------------------------------------

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 20,
    "font.family": "sans-serif",
    "axes.labelsize": 25,
    "axes.titlesize": 25,
    "axes.labelweight": "semibold",
    "axes.titleweight": "semibold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "lines.linewidth": 3.0,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

APP_TITLE = "Fibonacci"          # title suffix shown in figure headings
OUTPUT_FORMAT = "png"            # default output format: "png" or "pdf"
DATA_MAX_LIMIT = 30000           # max profile samples per warp (matches runtime default)

# Timeline colors
COL_WORKING = "#1f77b4"          # blue  -- warp is executing a task function
COL_IDLE = "#ff7f0e"             # orange -- warp is not executing a task function


# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------

def _compute_utilization(timeline_df: pd.DataFrame, strong_state: str = "Working") -> pd.DataFrame:
    """
    Compute per-warp utilization as fraction of total program time in *strong_state*.
    
    Returns a DataFrame with columns ``['warp_id', 'utilization_percent']``.
    """
    required = {"warp_id", "relative_time_ms", "state_description"}
    if not required.issubset(timeline_df.columns):
        return pd.DataFrame(columns=["warp_id", "utilization_percent"])

    t_min = float(timeline_df["relative_time_ms"].min())
    t_max = float(timeline_df["relative_time_ms"].max())
    program_total = max(0.0, t_max - t_min)
    if program_total <= 0.0:
        return pd.DataFrame([
            {"warp_id": wid, "utilization_percent": 0.0}
            for wid in timeline_df["warp_id"].unique()
        ])

    rows = []
    for warp_id, grp in timeline_df.groupby("warp_id"):
        g = grp.sort_values("relative_time_ms").reset_index(drop=True)
        working_time = 0.0
        start = None
        for _, row in g.iterrows():
            t = float(row["relative_time_ms"])
            if row["state_description"] == strong_state:
                if start is None:
                    start = t
            else:
                if start is not None:
                    working_time += max(0.0, t - start)
                    start = None
        if start is not None:
            working_time += max(0.0, float(g["relative_time_ms"].iloc[-1]) - start)
        util = min(100.0, max(0.0, working_time / program_total * 100.0))
        rows.append({"warp_id": warp_id, "utilization_percent": util})
    return pd.DataFrame(rows)


def load_data(profile_dir: str, app_name: str):
    """
    Load timeline and statistics CSVs from *profile_dir*.

    Expected files:
      - ``{app_name}_warp_timeline_working.csv``
      - ``{app_name}_warp_statistics_working.csv``

    Returns ``(timeline_df, stats_df, strong_state)``.
    """
    tl_path = os.path.join(profile_dir, f"{app_name}_warp_timeline_working.csv")
    st_path = os.path.join(profile_dir, f"{app_name}_warp_statistics_working.csv")
    if not os.path.exists(tl_path):
        raise FileNotFoundError(f"Timeline CSV not found: {tl_path}")
    if not os.path.exists(st_path):
        raise FileNotFoundError(f"Statistics CSV not found: {st_path}")

    timeline_df = pd.read_csv(tl_path)
    stats_df = pd.read_csv(st_path)
    strong_state = "Working"

    if "utilization_percent" not in stats_df.columns:
        util_df = _compute_utilization(timeline_df, strong_state)
        if not util_df.empty:
            stats_df = stats_df.merge(util_df, on="warp_id", how="left")
            stats_df["utilization_percent"] = stats_df["utilization_percent"].fillna(0.0)

    return timeline_df, stats_df, strong_state


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def create_timeline_plot(timeline_df: pd.DataFrame, stats_df: pd.DataFrame,
                         strong_state: str, app_name: str,
                         max_warps: Optional[int] = None) -> Optional[plt.Figure]:
    """
    Plot per-warp activity over time.

    Each warp row is painted blue (working) or orange (idle). When
    ``tasks_in_batch`` is present in the CSV, the blue shade encodes the
    number of tasks processed per batch (darker = more tasks). A colorbar
    is added in that case.

    Returns the Figure, or None if there is no data.
    """
    active = stats_df[stats_df["total_samples"] > 0]["warp_id"].tolist()
    if max_warps is not None:
        active = active[:max_warps]
    filtered = timeline_df[timeline_df["warp_id"].isin(active)].copy()
    if filtered.empty:
        return None

    t_min = timeline_df["relative_time_ms"].min()
    t_max = timeline_df["relative_time_ms"].max()
    filtered["norm_time"] = filtered["relative_time_ms"] - t_min
    total_dur = t_max - t_min

    # Optionally encode tasks_in_batch as blue intensity
    max_tasks = cmap = norm = None
    if "tasks_in_batch" in timeline_df.columns:
        try:
            mv = pd.to_numeric(timeline_df["tasks_in_batch"], errors="coerce").max()
            if pd.notna(mv) and float(mv) > 0.0:
                max_tasks = float(mv)
                cmap = plt.cm.Blues
                norm = mpl.colors.Normalize(vmin=0.0, vmax=max_tasks)
        except Exception:
            pass

    colors = {"Working": COL_WORKING, "NotWorking": COL_IDLE}
    _w, _ = plt.rcParams.get("figure.figsize", [6.4, 4.8])
    fig, ax = plt.subplots(figsize=(_w * 1.6, max(8, len(active) * 0.3)))

    def _seg_color(state, warp_data, t_start, t_end):
        if state == "Working" and max_tasks is not None:
            mask = (warp_data["norm_time"] >= t_start) & (warp_data["norm_time"] <= t_end)
            vals = pd.to_numeric(warp_data.loc[mask, "tasks_in_batch"], errors="coerce")
            mv = float(vals.max()) if not vals.empty and pd.notna(vals.max()) else 0.0
            return cmap(norm(mv)), 0.9
        return colors.get(state, "#888888"), (0.8 if state == strong_state else 0.5)

    for i, warp_id in enumerate(active):
        wd = filtered[filtered["warp_id"] == warp_id].sort_values("norm_time")

        if wd.empty:
            ax.add_patch(patches.Rectangle((0, i - 0.4), total_dur, 0.8,
                                           linewidth=0, facecolor=COL_IDLE, alpha=0.5))
            continue

        first_t = wd["norm_time"].iloc[0]
        if first_t > 0:
            ax.add_patch(patches.Rectangle((0, i - 0.4), first_t, 0.8,
                                           linewidth=0, facecolor=COL_IDLE, alpha=0.5))

        prev_state = start_t = None
        for _, row in wd.iterrows():
            cur_state = row["state_description"]
            cur_t = row["norm_time"]
            if prev_state is not None and prev_state != cur_state:
                col, alp = _seg_color(prev_state, wd, start_t, cur_t)
                ax.add_patch(patches.Rectangle((start_t, i - 0.4), cur_t - start_t, 0.8,
                                               linewidth=0, facecolor=col, alpha=alp))
            if prev_state != cur_state:
                start_t, prev_state = cur_t, cur_state

        if prev_state is not None:
            last_t = wd["norm_time"].iloc[-1]
            col, alp = _seg_color(prev_state, wd, start_t, last_t)
            ax.add_patch(patches.Rectangle((start_t, i - 0.4), last_t - start_t, 0.8,
                                           linewidth=0, facecolor=col, alpha=alp))

        last_t = wd["norm_time"].iloc[-1]
        if last_t < total_dur and len(wd) < DATA_MAX_LIMIT:
            ax.add_patch(patches.Rectangle((last_t, i - 0.4), total_dur - last_t, 0.8,
                                           linewidth=0, facecolor=COL_IDLE, alpha=0.5))

    ax.set_xlim(0, total_dur)
    ax.set_ylim(-0.5, len(active) - 0.5)
    ax.set_yticks(range(len(active)))
    ax.set_yticklabels([f"Warp {w}" for w in active])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Warps")
    ax.set_title(f"Worker Timeline Visualization: {APP_TITLE}")
    ax.grid(True, alpha=0)

    legend = [
        patches.Patch(color=COL_WORKING, alpha=0.8, label="Executing taskfn"),
        patches.Patch(color=COL_IDLE, alpha=0.5, label="Not executing taskfn"),
    ]
    if "tasks_in_batch" in filtered.columns:
        w_df = filtered[filtered["state_description"] == strong_state]
        vals = pd.to_numeric(w_df["tasks_in_batch"], errors="coerce").dropna()
        if len(vals) > 0:
            legend.append(patches.Patch(color="none", label=f"Avg tasks per batch: {vals.mean():.2f}"))
    ax.legend(handles=legend, loc="upper right")

    if max_tasks is not None:
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("tasks in batch")

    plt.tight_layout()
    return fig


def create_utilization_histogram(stats_df: pd.DataFrame,
                                 app_name: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot a histogram of per-warp task execution time ratio.

    Returns the Figure, or None if there is no data.
    """
    df = stats_df.copy()
    df["utilization_percent"] = df.get("utilization_percent", 0.0).fillna(0.0)
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(df["utilization_percent"], bins=20, alpha=0.7, color="lightblue", edgecolor="black")
    ax.set_xlabel("Task Execution Time Ratio (%)")
    ax.set_ylabel("Number of Warps")
    ax.set_title(f"Distribution of Task Execution Time Ratio per Warp:\n{APP_TITLE}")
    ax.grid(True, alpha=0.3)

    mean_v = df["utilization_percent"].mean()
    med_v = df["utilization_percent"].median()
    ax.axvline(mean_v, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_v:.1f}%")
    ax.axvline(med_v, color="green", linestyle="--", linewidth=2, label=f"Median: {med_v:.1f}%")
    ax.legend()
    plt.tight_layout()
    return fig


# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------

def print_summary(stats_df: pd.DataFrame) -> None:
    """Print a per-warp utilization summary to stdout."""
    df = stats_df.copy()
    df["utilization_percent"] = df.get("utilization_percent", 0.0).fillna(0.0)
    active = df[df["total_samples"] > 0]
    print("\n" + "=" * 60)
    print("WARP TIMELINE ANALYSIS SUMMARY (Working)")
    print("=" * 60)
    print(f"Total Warps:    {len(df)}")
    print(f"Active Warps:   {len(active)}")
    print(f"Inactive Warps: {len(df) - len(active)}")
    if len(df) > 0:
        print(f"Avg Utilization (all):    {df['utilization_percent'].mean():.2f}%")
        print(f"Std Dev (all):            {df['utilization_percent'].std():.2f}%")
    if len(active) > 0:
        print(f"Avg Utilization (active): {active['utilization_percent'].mean():.2f}%")


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize GTaP warp profile CSVs for the Fibonacci example."
    )
    parser.add_argument("--profile-dir", default="./profile",
                        help="Directory containing profile CSV files (default: ./profile)")
    parser.add_argument("--app", default="fib",
                        help="App name used as CSV filename prefix (default: fib)")
    parser.add_argument("--max-warps", type=int, default=15,
                        help="Max number of warps shown in the timeline (default: 15)")
    parser.add_argument("--output-dir", default="./img",
                        help="Directory to save output figures (default: ./img)")
    parser.add_argument("--format", default=OUTPUT_FORMAT, choices=["png", "pdf"],
                        help=f"Output figure format (default: {OUTPUT_FORMAT})")
    args = parser.parse_args()

    print("Warp Timeline Visualization Tool (GTaP Thread Runtime)")
    print("=" * 40)
    print(f"Profile dir : {args.profile_dir}")
    print(f"App name    : {args.app}")

    timeline_df, stats_df, strong_state = load_data(args.profile_dir, args.app)
    print_summary(stats_df)
    print("\nGenerating visualizations...")

    os.makedirs(args.output_dir, exist_ok=True)
    fmt = args.format

    fig = create_timeline_plot(timeline_df, stats_df, strong_state,
                               app_name=args.app, max_warps=args.max_warps)
    if fig:
        path = os.path.join(args.output_dir, f"{args.app}_timeline.{fmt}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    fig = create_utilization_histogram(stats_df, app_name=args.app)
    if fig:
        path = os.path.join(args.output_dir, f"{args.app}_utilization.{fmt}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

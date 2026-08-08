#!/usr/bin/env python3
"""Visualize thread- or block-mode profiles from the Fibonacci example."""

import argparse
import json
import math
import os
from typing import Optional

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


OUTPUT_FORMAT = "png"
IDLE_COLOR = "#ff7f0e"
DEFAULT_TIME_BINS = 1500
MIN_PIXELS_PER_WORKER = 2.0


def make_timeline_colormap(max_tasks: float):
    color_map = plt.cm.Blues.copy()
    idle_rgb = mpl.colors.to_rgb(IDLE_COLOR)
    idle_on_white = tuple(0.5 * channel + 0.5 for channel in idle_rgb)
    color_map.set_under(idle_on_white)
    normalization = mpl.colors.Normalize(
        vmin=1e-6, vmax=max(1.0, max_tasks))
    return color_map, normalization

plt.rcParams.update({
    "figure.figsize": (12, 8),
    "font.size": 16,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_profile(profile_dir: str):
    profile_path = os.path.join(profile_dir, "profile.json")
    intervals_path = os.path.join(
        profile_dir, "task_execution_intervals.csv")
    summary_path = os.path.join(
        profile_dir, "task_execution_aggregates.csv")

    for path in (profile_path, intervals_path, summary_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Profile file not found: {path}")

    with open(profile_path, encoding="utf-8") as file:
        profile = json.load(file)
    intervals = pd.read_csv(intervals_path)
    summary = pd.read_csv(summary_path)

    mode = profile.get("mode")
    if mode not in ("thread", "block"):
        raise ValueError(f"Unsupported profile mode: {mode!r}")
    id_column = "warp_id" if mode == "thread" else "block_id"
    required_intervals = {id_column, "start_ns", "end_ns"}
    required_summary = {
        id_column, "intervals_recorded", "intervals_dropped"
    }
    if not required_intervals.issubset(intervals.columns):
        missing = sorted(required_intervals - set(intervals.columns))
        raise ValueError(f"Missing interval columns: {', '.join(missing)}")
    if not required_summary.issubset(summary.columns):
        missing = sorted(required_summary - set(summary.columns))
        raise ValueError(f"Missing summary columns: {', '.join(missing)}")

    intervals = intervals.dropna(subset=["start_ns", "end_ns"]).copy()
    intervals["start_ns"] = pd.to_numeric(intervals["start_ns"])
    intervals["end_ns"] = pd.to_numeric(intervals["end_ns"])
    if "batch_task_count" in intervals.columns:
        intervals["batch_task_count"] = pd.to_numeric(
            intervals["batch_task_count"], errors="coerce").fillna(0)
    intervals = intervals[intervals["end_ns"] >= intervals["start_ns"]]
    return profile, intervals, summary, mode, id_column


def compute_utilization(intervals: pd.DataFrame, summary: pd.DataFrame,
                        id_column: str) -> pd.DataFrame:
    result = summary[[id_column, "intervals_recorded",
                      "intervals_dropped"]].copy()
    if intervals.empty:
        result["utilization_percent"] = 0.0
        return result

    first_ns = intervals["start_ns"].min()
    last_ns = intervals["end_ns"].max()
    elapsed_ns = max(0, last_ns - first_ns)
    executing = intervals
    if "batch_task_count" in executing.columns:
        executing = executing[executing["batch_task_count"] > 0]
    durations = (executing["end_ns"] - executing["start_ns"]).groupby(
        executing[id_column]).sum()
    result["executing_ns"] = result[id_column].map(durations).fillna(0)
    result["utilization_percent"] = (
        result["executing_ns"] / elapsed_ns * 100.0
        if elapsed_ns else 0.0
    )
    return result


def create_timeline(intervals: pd.DataFrame, summary: pd.DataFrame,
                    mode: str, id_column: str,
                    max_workers: Optional[int], time_bins: int,
                    save_dpi: Optional[int]) -> Optional[plt.Figure]:
    if intervals.empty:
        return None

    origin_ns = intervals["start_ns"].min()
    end_ns = intervals["end_ns"].max()
    duration_ms = (end_ns - origin_ns) / 1_000_000.0

    executing = intervals
    if mode == "thread" and "batch_task_count" in intervals.columns:
        executing = intervals[intervals["batch_task_count"] > 0]
    busy_ns = (executing["end_ns"] - executing["start_ns"]).groupby(
        executing[id_column]).sum().to_dict()
    workers = [int(value) for value in summary[id_column].tolist()]
    workers.sort(key=lambda value: (-busy_ns.get(value, 0), value))
    if max_workers is not None and max_workers > 0:
        workers = workers[:max_workers]

    worker_to_row = {worker_id: row
                     for row, worker_id in enumerate(workers)}
    matrix = np.zeros((len(workers), time_bins), dtype=np.float32)
    span_ns = max(1, end_ns - origin_ns)
    has_batch_counts = mode == "thread" and "batch_task_count" in intervals.columns
    rows = intervals[id_column].map(worker_to_row).to_numpy(
        dtype=np.float64)
    starts_ns = intervals["start_ns"].to_numpy(dtype=np.int64)
    ends_ns = intervals["end_ns"].to_numpy(dtype=np.int64)
    values = (intervals["batch_task_count"].to_numpy(dtype=np.float32)
              if has_batch_counts
              else np.ones(len(intervals), dtype=np.float32))

    valid = np.isfinite(rows) & (values > 0)
    rows = rows[valid].astype(np.intp)
    starts_ns = starts_ns[valid]
    ends_ns = ends_ns[valid]
    values = values[valid]
    first_bins = ((starts_ns - origin_ns) * time_bins // span_ns).clip(
        0, time_bins - 1).astype(np.intp)
    last_bins = (((ends_ns - origin_ns) * time_bins + span_ns - 1)
                 // span_ns).clip(0, time_bins).astype(np.intp)
    last_bins = np.maximum(last_bins, first_bins + 1)

    lengths = last_bins - first_bins
    expanded_rows = np.repeat(rows, lengths)
    expanded_values = np.repeat(values, lengths)
    interval_offsets = np.repeat(
        np.cumsum(lengths, dtype=np.int64) - lengths, lengths)
    expanded_bins = (
        np.repeat(first_bins, lengths)
        + np.arange(lengths.sum(), dtype=np.int64)
        - interval_offsets)
    np.maximum.at(
        matrix, (expanded_rows, expanded_bins), expanded_values)

    max_tasks = (max(1.0, float(intervals["batch_task_count"].max()))
                 if has_batch_counts else 1.0)
    color_map, normalization = make_timeline_colormap(max_tasks)
    figure, axis = plt.subplots(figsize=(14, 8))
    image = axis.imshow(
        matrix, aspect="auto", origin="upper", interpolation="nearest",
        cmap=color_map, norm=normalization,
        extent=[0, duration_ms, len(workers), 0], rasterized=True)

    unit = "Warp" if mode == "thread" else "Block"
    axis.set_xlim(0, duration_ms)
    axis.set_ylim(len(workers), 0)
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel(f"{unit}s (by busy time)")
    axis.set_title(f"Fibonacci Task Execution Timeline ({mode} mode)")
    axis.grid(False)
    legend_handles = [
        patches.Patch(color=plt.cm.Blues(0.8),
                      label="Executing taskfn"),
        patches.Patch(color=IDLE_COLOR, alpha=0.5,
                      label="Not executing taskfn"),
    ]
    if has_batch_counts:
        color_bar = figure.colorbar(
            image, ax=axis, fraction=0.03, pad=0.02, extend="min")
        color_bar.set_label("Tasks in batch")
        average = float(intervals["batch_task_count"].mean())
        legend_handles.append(patches.Patch(
            color="none", label=f"Avg tasks per batch: {average:.2f}"))
    axis.legend(handles=legend_handles, loc="upper right")
    if save_dpi is None:
        axes_height_inches = figure.get_size_inches()[1] * 0.75
        save_dpi = max(300, math.ceil(
            len(workers) * MIN_PIXELS_PER_WORKER / axes_height_inches))
    figure._gtap_save_dpi = save_dpi
    print(f"Timeline workers: {len(workers)}, bins: {time_bins}, "
          f"save dpi: {save_dpi}")
    figure.tight_layout()
    return figure


def create_utilization_histogram(utilization: pd.DataFrame,
                                 mode: str) -> Optional[plt.Figure]:
    if utilization.empty:
        return None
    unit = "Warp" if mode == "thread" else "Block"
    figure, axis = plt.subplots()
    axis.hist(utilization["utilization_percent"], bins=20,
              color="lightblue", edgecolor="black")
    axis.set_xlabel("Task Execution Time Ratio (%)")
    axis.set_ylabel(f"Number of {unit}s")
    axis.set_title(f"Task Execution Utilization ({mode} mode)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def print_summary(utilization: pd.DataFrame, mode: str) -> None:
    unit = "warps" if mode == "thread" else "blocks"
    active = utilization[utilization["intervals_recorded"] > 0]
    print(f"Mode: {mode}")
    print(f"Total {unit}: {len(utilization)}")
    print(f"Active {unit}: {len(active)}")
    print(f"Intervals recorded: {utilization['intervals_recorded'].sum()}")
    print(f"Intervals dropped: {utilization['intervals_dropped'].sum()}")
    if not active.empty:
        print(f"Average active utilization: "
              f"{active['utilization_percent'].mean():.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a GTaP Fibonacci profile.")
    parser.add_argument("--mode", choices=["thread", "block"],
                        default="thread",
                        help="Profile to visualize (default: thread)")
    parser.add_argument("--profile-dir",
                        help="Result directory (default: ./profile/fib_<mode>)")
    parser.add_argument("--max-workers", type=int, default=0,
                        help="Show only the busiest N workers (0 = all)")
    parser.add_argument("--time-bins", type=int, default=DEFAULT_TIME_BINS,
                        help=f"Horizontal heatmap bins (default: {DEFAULT_TIME_BINS})")
    parser.add_argument("--dpi", type=int,
                        help="Output DPI (default: auto, at least 2 pixels per worker)")
    parser.add_argument("--output-dir", default="./img",
                        help="Figure output directory (default: ./img)")
    parser.add_argument("--format", choices=["png", "pdf"],
                        default=OUTPUT_FORMAT)
    args = parser.parse_args()

    profile_dir = args.profile_dir or os.path.join(
        ".", "profile", f"fib_{args.mode}")
    _, intervals, summary, mode, id_column = load_profile(profile_dir)
    if mode != args.mode:
        raise ValueError(
            f"Requested {args.mode} mode, but profile.json says {mode}")

    utilization = compute_utilization(intervals, summary, id_column)
    print(f"Profile directory: {profile_dir}")
    print_summary(utilization, mode)
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = f"fib_{mode}"

    timeline = create_timeline(
        intervals, summary, mode, id_column, args.max_workers,
        args.time_bins, args.dpi)
    if timeline is not None:
        path = os.path.join(
            args.output_dir, f"{prefix}_timeline.{args.format}")
        timeline.savefig(path, dpi=timeline._gtap_save_dpi)
        plt.close(timeline)
        print(f"Saved: {path}")

    histogram = create_utilization_histogram(utilization, mode)
    if histogram is not None:
        path = os.path.join(
            args.output_dir, f"{prefix}_utilization.{args.format}")
        histogram.savefig(path)
        plt.close(histogram)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()

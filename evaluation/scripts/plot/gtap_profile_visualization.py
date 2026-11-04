"""Shared visualization helpers for the current GTaP profile export format."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd


DEFAULT_TIME_BINS = 1500
MAX_TASKS_PER_BATCH = 32
IDLE_COLOR = "#ffbb7d"


def load_profile(profile_dir: Path, expected_mode: str | None = None):
    profile_dir = Path(profile_dir)
    with (profile_dir / "profile.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    intervals = pd.read_csv(profile_dir / "task_execution_intervals.csv")
    aggregates = pd.read_csv(profile_dir / "task_execution_aggregates.csv")
    mode = metadata.get("mode")
    if mode not in ("thread", "block"):
        raise ValueError(f"Invalid profile mode: {mode!r}")
    if expected_mode is not None and mode != expected_mode:
        raise ValueError(f"Expected {expected_mode} profile, found {mode}")
    id_column = "warp_id" if mode == "thread" else "block_id"
    required = {id_column, "start_ns", "end_ns"}
    missing = required - set(intervals.columns)
    if missing:
        raise ValueError(f"Interval CSV missing columns: {sorted(missing)}")
    intervals = intervals.dropna(subset=list(required)).copy()
    intervals["start_ms"] = intervals["start_ns"] / 1_000_000.0
    intervals["end_ms"] = intervals["end_ns"] / 1_000_000.0
    if "batch_task_count" not in intervals:
        intervals["batch_task_count"] = 1
    if id_column != "warp_id":
        intervals = intervals.rename(columns={id_column: "warp_id"})
        aggregates = aggregates.rename(columns={id_column: "warp_id"})
    return metadata, intervals, aggregates


def profile_time_bounds(intervals: pd.DataFrame) -> tuple[float, float]:
    return float(intervals["start_ms"].min()), float(intervals["end_ms"].max())


def compute_busy_time_per_warp(intervals, _t_min=None, _t_max=None,
                               strong_state=None):
    del _t_min, _t_max, strong_state
    durations = intervals["end_ms"] - intervals["start_ms"]
    return durations.groupby(intervals["warp_id"]).sum().to_dict()


def ordered_warp_ids(aggregates, busy_times, sort_by_busy=True):
    ids = [int(value) for value in aggregates["warp_id"].tolist()]
    if sort_by_busy:
        ids.sort(key=lambda value: (-busy_times.get(value, 0.0), value))
    else:
        ids.sort()
    return ids


def build_warp_heatmap_matrix(intervals, warp_ids, *, n_bins, t_min, t_max,
                              strong_state=None):
    del strong_state
    matrix = np.zeros((len(warp_ids), n_bins), dtype=np.float32)
    if not warp_ids or t_max <= t_min:
        return matrix
    rows = {worker: row for row, worker in enumerate(warp_ids)}
    bin_width = (t_max - t_min) / n_bins
    for item in intervals.itertuples(index=False):
        row = rows.get(int(item.warp_id))
        if row is None:
            continue
        first = max(0, min(n_bins - 1, int((item.start_ms - t_min) / bin_width)))
        last = max(first + 1, min(n_bins, int(np.ceil((item.end_ms - t_min) / bin_width))))
        matrix[row, first:last] = np.maximum(
            matrix[row, first:last], float(item.batch_task_count))
    return matrix


def tasks_in_batch_scalar_mappable(max_tasks=MAX_TASKS_PER_BATCH):
    cmap = mpl.colormaps["Blues"].copy()
    cmap.set_under(IDLE_COLOR)
    result = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=1e-6, vmax=float(max_tasks)), cmap=cmap)
    result.set_array([])
    return result


TIMELINE_HEATMAP_CMAP = mpl.colormaps["Blues"].copy()
TIMELINE_HEATMAP_CMAP.set_under(IDLE_COLOR)
TIMELINE_HEATMAP_NORM = mpl.colors.Normalize(
    vmin=1e-6, vmax=float(MAX_TASKS_PER_BATCH))
COLORBAR_LABEL_TASKS = "Tasks in batch"


def configure_tasks_in_batch_colorbar(colorbar, max_tasks=MAX_TASKS_PER_BATCH):
    colorbar.set_ticks([0, 8, 16, 24, int(max_tasks)])
    colorbar.set_label(COLORBAR_LABEL_TASKS)

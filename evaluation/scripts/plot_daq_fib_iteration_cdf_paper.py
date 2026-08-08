#!/usr/bin/env python3
"""Paper figure: CDF of task-execution interval durations (Fibonacci).

Uses current-format profiles from evaluation/daq/fib (n=40, cutoff=10 by default) and
compares 1-queue (without DAQ) vs 3-queue DAQ (cutoff / non-cutoff / continuation).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parents[1]
COMPARE_DIR = EVAL_DIR / "benchmarks"
IMG_DIR = EVAL_DIR / "img"
DAQ_FIB_DIR = EVAL_DIR / "daq" / "fib"
WORKSPACE_ROOT = EVAL_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from plot_style.gtap_colors import COL_GTAP_THREAD, COL_OMP  # noqa: E402

FIG_SCALE = 0.9
FIG_WIDTH_IN = (239.75 / 72.0) * FIG_SCALE  # ACM single column × 0.9
FIG_HEIGHT_IN = FIG_WIDTH_IN * 0.72 * 0.8
OUTPUT_FORMAT = "pdf"
P99_QUANTILE = 0.99

PAPER_RC = {
    "font.size": 8,
    "font.weight": "normal",
    "axes.labelsize": 8,
    "axes.labelweight": "normal",
    "axes.titlesize": 8,
    "axes.titleweight": "normal",
    "figure.labelsize": 8,
    "figure.labelweight": "normal",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.25,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

COL_WITHOUT_DAQ = COL_GTAP_THREAD
COL_WITH_DAQ = COL_OMP


@dataclass(frozen=True)
class SeriesSpec:
    key: str
    label: str
    profile_dir: Path
    color: str


def extract_working_durations(intervals: pd.DataFrame) -> np.ndarray:
    """Return task-execution interval durations in milliseconds."""
    required = {"start_ns", "end_ns"}
    missing = required - set(intervals.columns)
    if missing:
        raise ValueError(f"timeline CSV missing columns: {sorted(missing)}")

    durations = (intervals["end_ns"] - intervals["start_ns"]) / 1_000_000.0
    return durations[durations > 0.0].to_numpy(dtype=np.float64)


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([]), np.array([])
    x = np.sort(values)
    y = np.arange(1, x.size + 1, dtype=np.float64) / float(x.size)
    return x, y


def load_iteration_durations(profile_dir: Path) -> np.ndarray:
    df = pd.read_csv(profile_dir / "task_execution_intervals.csv")
    return extract_working_durations(df)


def default_series() -> tuple[SeriesSpec, SeriesSpec]:
    profile_dir = DAQ_FIB_DIR / "profile"
    return (
        SeriesSpec(
            key="without_daq",
            label="w/o DAQ (1 queue)",
            profile_dir=profile_dir / "fib_queue_1",
            color=COL_WITHOUT_DAQ,
        ),
        SeriesSpec(
            key="with_daq",
            label="w/ DAQ (3 queues)",
            profile_dir=profile_dir / "fib_queue_3",
            color=COL_WITH_DAQ,
        ),
    )


def plot_cdf(
    series_data: list[tuple[SeriesSpec, np.ndarray]],
    *,
    title: str | None,
    x_max: float | None,
    show_grid: bool,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    xmax = 0.0
    for spec, durations in series_data:
        if durations.size == 0:
            continue
        x, y = empirical_cdf(durations)
        p99 = float(np.quantile(durations, P99_QUANTILE))
        xmax = max(xmax, float(x[-1]), p99)
        ax.plot(x, y, color=spec.color, label=spec.label)
        y_at_p99 = float(np.mean(durations <= p99))
        ax.plot(
            [p99, p99],
            [0.0, y_at_p99],
            color=spec.color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.85,
        )

    if x_max is not None:
        xmax = x_max
    elif xmax <= 0.0:
        xmax = 1.0
    ax.set_xlim(left=0.0, right=xmax * 1.05)
    ax.set_ylim(0.0, 1.025)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlabel("Warp batch time (ms)")
    ax.set_ylabel("CDF")
    if title:
        ax.set_title(title)
    if show_grid:
        ax.grid(True, axis="both")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(pad=0.4)
    return fig


def parse_args() -> argparse.Namespace:
    without, with_daq = default_series()
    parser = argparse.ArgumentParser(
        description="Plot Fibonacci DAQ execution-interval CDF (paper figure).",
    )
    parser.add_argument(
        "--without-daq-profile",
        dest="without_daq_profile",
        type=Path,
        default=without.profile_dir,
        help="1-queue GTaP profile directory",
    )
    parser.add_argument(
        "--with-daq-profile",
        dest="with_daq_profile",
        type=Path,
        default=with_daq.profile_dir,
        help="3-queue GTaP profile directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=IMG_DIR / f"daq_fib_iteration_cdf.{OUTPUT_FORMAT}",
        help="Output figure path",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional figure title (default: none)",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Optional xmax override (ms)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Save DPI",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Disable grid",
    )
    parser.add_argument(
        "--print-stats",
        action="store_true",
        help="Print summary statistics to stdout",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plt.rcParams.update(PAPER_RC)

    without, with_daq = default_series()
    series = (
        SeriesSpec(without.key, without.label, args.without_daq_profile, without.color),
        SeriesSpec(with_daq.key, with_daq.label, args.with_daq_profile, with_daq.color),
    )

    series_data: list[tuple[SeriesSpec, np.ndarray]] = []
    for spec in series:
        intervals_path = spec.profile_dir / "task_execution_intervals.csv"
        if not intervals_path.is_file():
            print(f"ERROR: missing intervals CSV: {intervals_path}", file=sys.stderr)
            return 1
        durations = load_iteration_durations(spec.profile_dir)
        series_data.append((spec, durations))
        if args.print_stats:
            p99 = float(np.quantile(durations, P99_QUANTILE)) if durations.size else float("nan")
            print(
                f"{spec.label}: n={durations.size} "
                f"median={np.median(durations):.6f} ms "
                f"p99={p99:.6f} ms "
                f"max={durations.max() if durations.size else float('nan'):.6f} ms"
            )

    fig = plot_cdf(
        series_data,
        title=args.title,
        x_max=args.x_max,
        show_grid=not args.no_grid,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        format=OUTPUT_FORMAT,
        dpi=args.dpi,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

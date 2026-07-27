#!/usr/bin/env python3
"""Paper figure: hetero_tree warp/block timeline heatmaps (4 configs).

Same layout as plot_k_clique_pivot_timeline_paper.py (3.33 in wide, paper RC,
shared colorbar style). Expects CSVs under hetero_tree/profile/ from a PROFILE
build, e.g.:

  hetero_tree/profile/hetero_tree_thread_warp_timeline_working.csv
  hetero_tree/profile/hetero_tree_thread_daq_warp_timeline_working.csv
  hetero_tree/profile/hetero_tree_block_block_timeline_working.csv
  hetero_tree/profile/hetero_tree_block_cutoff_block_timeline_working.csv

Default: one PDF per config under hetero_tree/img/.
Use --stack for a 4-row comparison figure.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import transforms
from matplotlib.gridspec import GridSpec

EVAL_DIR = Path(__file__).resolve().parents[1]
COMPARE_DIR = EVAL_DIR / "benchmarks"
IMG_DIR = EVAL_DIR / "img"
HETERO_DIR = COMPARE_DIR / "hetero_tree"
sys.path.insert(0, str(COMPARE_DIR))

from thread_visualize_profile import (  # noqa: E402
    DEFAULT_TIME_BINS,
    TIMELINE_HEATMAP_CMAP,
    TIMELINE_HEATMAP_NORM,
    build_warp_heatmap_matrix,
    compute_busy_time_per_warp,
    ordered_warp_ids,
    tasks_in_batch_scalar_mappable,
    COLORBAR_LABEL_TASKS,
    configure_tasks_in_batch_colorbar,
)

FIG_WIDTH_IN = 239.75 / 72.0  # ≈ 3.33 in
DEFAULT_SAVE_DPI = 600
AXES_HEIGHT_FRAC = 0.975 - 0.075
OUTPUT_FORMAT = "pdf"
COLORBAR_HEIGHT_SHRINK = 0.88
COLORBAR_LABEL_OFFSET_PT = (-3.0, -2.0)
DEFAULT_MIN_PX_PER_WARP = 2.0
STRONG_STATE = "Working"


@dataclass(frozen=True)
class PanelSpec:
    app_name: str
    title: str
    worker: str  # "warp" | "block"
    max_workers: int = 0  # 0 = all


DEFAULT_PANELS: tuple[PanelSpec, ...] = (
    PanelSpec("hetero_tree_thread", "Thread (wo DAQ)", "warp"),
    PanelSpec("hetero_tree_thread_daq", "Thread (DAQ)", "warp"),
    PanelSpec("hetero_tree_block", "Block", "block"),
    PanelSpec("hetero_tree_block_cutoff", "Block (cutoff)", "block"),
)

PAPER_RC = {
    "font.size": 7,
    "font.weight": "normal",
    "axes.labelsize": 7,
    "axes.labelweight": "normal",
    "axes.titlesize": 7,
    "axes.titleweight": "normal",
    "figure.labelsize": 7,
    "figure.labelweight": "normal",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def dpi_for_min_px_per_row(
    fig_height_in: float,
    n_rows: int,
    *,
    min_px_per_row: float = DEFAULT_MIN_PX_PER_WARP,
    floor_dpi: int = DEFAULT_SAVE_DPI,
) -> int:
    axes_h_in = float(fig_height_in) * AXES_HEIGHT_FRAC
    if axes_h_in <= 0.0 or n_rows <= 0:
        return int(floor_dpi)
    return max(int(floor_dpi), int(math.ceil(n_rows * min_px_per_row / axes_h_in)))


def resolve_profile_csv_paths(
    *,
    profile_dir: Path,
    app_name: str,
    worker: str,
) -> tuple[Path, Path]:
    if worker == "block":
        tl = profile_dir / f"{app_name}_block_timeline_working.csv"
        st = profile_dir / f"{app_name}_block_statistics_working.csv"
    else:
        tl = profile_dir / f"{app_name}_warp_timeline_working.csv"
        st = profile_dir / f"{app_name}_warp_statistics_working.csv"
    return tl, st


def load_profile(
    *,
    profile_dir: Path,
    app_name: str,
    worker: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    timeline_path, stats_path = resolve_profile_csv_paths(
        profile_dir=profile_dir,
        app_name=app_name,
        worker=worker,
    )
    if not timeline_path.exists():
        raise FileNotFoundError(f"Missing timeline CSV: {timeline_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing statistics CSV: {stats_path}")

    print(f"Loading: {timeline_path}")
    print(f"Loading: {stats_path}")
    timeline_df = pd.read_csv(timeline_path)
    stats_df = pd.read_csv(stats_path)

    # Reuse warp heatmap helpers: alias block_id -> warp_id.
    if worker == "block":
        if "block_id" not in timeline_df.columns:
            raise ValueError(f"'block_id' missing in {timeline_path}")
        if "block_id" not in stats_df.columns:
            raise ValueError(f"'block_id' missing in {stats_path}")
        timeline_df = timeline_df.rename(columns={"block_id": "warp_id"})
        stats_df = stats_df.rename(columns={"block_id": "warp_id"})
    else:
        if "warp_id" not in timeline_df.columns:
            raise ValueError(f"'warp_id' missing in {timeline_path}")
        if "warp_id" not in stats_df.columns:
            raise ValueError(f"'warp_id' missing in {stats_path}")

    return timeline_df, stats_df


def _prepare_heatmap(
    *,
    profile_dir: Path,
    spec: PanelSpec,
    n_bins: int,
    sort_by_busy: bool,
):
    timeline_df, stats_df = load_profile(
        profile_dir=profile_dir,
        app_name=spec.app_name,
        worker=spec.worker,
    )
    t_min = float(timeline_df["relative_time_ms"].min())
    t_max = float(timeline_df["relative_time_ms"].max())
    total_duration = max(0.0, t_max - t_min)
    if total_duration <= 0.0:
        raise ValueError(f"No timeline span for {spec.app_name}")

    busy_times = compute_busy_time_per_warp(
        timeline_df, t_min, t_max, strong_state=STRONG_STATE,
    )
    worker_ids = ordered_warp_ids(stats_df, busy_times, sort_by_busy=sort_by_busy)
    total_workers = len(worker_ids)
    if spec.max_workers > 0:
        worker_ids = worker_ids[: spec.max_workers]

    matrix = build_warp_heatmap_matrix(
        timeline_df,
        worker_ids,
        n_bins=n_bins,
        t_min=t_min,
        t_max=t_max,
        strong_state=STRONG_STATE,
    )
    return matrix, total_duration, len(worker_ids), total_workers


def _ylabel_for(worker: str) -> str:
    unit = "Blocks" if worker == "block" else "Warps"
    return f"{unit} (sorted by total busy time)"


def plot_one_timeline(
    *,
    spec: PanelSpec,
    output_path: Path,
    profile_dir: Path,
    n_bins: int,
    sort_by_busy: bool,
    panel_height_ratio: float,
    save_dpi: int | None,
    xlim_s: float | None,
) -> None:
    plt.rcParams.update(PAPER_RC)

    fig_height = FIG_WIDTH_IN * panel_height_ratio
    print(f"Building heatmap for {spec.app_name}...")
    matrix, duration_ms, n_shown, _n_total = _prepare_heatmap(
        profile_dir=profile_dir,
        spec=spec,
        n_bins=n_bins,
        sort_by_busy=sort_by_busy,
    )
    if save_dpi is None:
        save_dpi = dpi_for_min_px_per_row(fig_height, n_shown)
        print(
            f"Auto dpi={save_dpi} for ≥{DEFAULT_MIN_PX_PER_WARP:g} px/row "
            f"({n_shown} rows)"
        )

    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_height))
    gs = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.0, 0.04],
        left=0.155,
        right=0.995,
        bottom=0.075,
        top=0.975,
        wspace=0.08,
    )
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    cbar_pos = cax.get_position()
    cbar_h = cbar_pos.height * COLORBAR_HEIGHT_SHRINK
    cax.set_position([
        cbar_pos.x0,
        cbar_pos.y0 + (cbar_pos.height - cbar_h) / 2,
        cbar_pos.width,
        cbar_h,
    ])

    duration_s = duration_ms / 1000.0
    axis_xlim_s = duration_s if xlim_s is None else max(duration_s, xlim_s)

    ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap=TIMELINE_HEATMAP_CMAP,
        norm=TIMELINE_HEATMAP_NORM,
        extent=[0.0, duration_s, n_shown, 0.0],
        rasterized=True,
    )
    ax.set_xlim(0.0, axis_xlim_s)
    ax.set_ylim(n_shown, 0.0)
    ax.set_xlabel("Time (s)")
    ax.set_title(spec.title, loc="left", pad=1)
    ax.grid(False)

    label_fp = ax.xaxis.label.get_fontproperties()
    fig.supylabel(_ylabel_for(spec.worker), fontproperties=label_fp)

    cbar = fig.colorbar(tasks_in_batch_scalar_mappable(), cax=cax, extend="min")
    cbar.set_label(COLORBAR_LABEL_TASKS, fontproperties=label_fp)
    configure_tasks_in_batch_colorbar(cbar)
    dx_pt, dy_pt = COLORBAR_LABEL_OFFSET_PT
    cbar.ax.yaxis.label.set_transform(
        cbar.ax.yaxis.label.get_transform()
        + transforms.ScaledTranslation(dx_pt / 72, dy_pt / 72, fig.dpi_scale_trans)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_stack_timelines(
    *,
    panels: tuple[PanelSpec, ...],
    output_path: Path,
    profile_dir: Path,
    n_bins: int,
    sort_by_busy: bool,
    panel_height_ratio: float,
    save_dpi: int,
    shared_xlim: bool,
) -> None:
    """4-row stack (sort_timelines style) with optional shared x max."""
    plt.rcParams.update(PAPER_RC)

    n_panels = len(panels)
    fig_height = FIG_WIDTH_IN * panel_height_ratio * n_panels
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_height))
    gs = GridSpec(
        n_panels,
        2,
        figure=fig,
        width_ratios=[1.0, 0.04],
        height_ratios=[1.0] * n_panels,
        left=0.155,
        right=0.995,
        bottom=0.075,
        top=0.975,
        wspace=0.08,
        hspace=0.40,
    )
    axes = [fig.add_subplot(gs[i, 0]) for i in range(n_panels)]
    cax = fig.add_subplot(gs[:, 1])
    cbar_pos = cax.get_position()
    cbar_h = cbar_pos.height * COLORBAR_HEIGHT_SHRINK
    cax.set_position([
        cbar_pos.x0,
        cbar_pos.y0 + (cbar_pos.height - cbar_h) / 2,
        cbar_pos.width,
        cbar_h,
    ])

    prepared = []
    max_duration_s = 0.0
    for spec in panels:
        print(f"Building heatmap for {spec.app_name}...")
        matrix, duration_ms, n_shown, _ = _prepare_heatmap(
            profile_dir=profile_dir,
            spec=spec,
            n_bins=n_bins,
            sort_by_busy=sort_by_busy,
        )
        duration_s = duration_ms / 1000.0
        max_duration_s = max(max_duration_s, duration_s)
        prepared.append((spec, matrix, duration_s, n_shown))

    for ax, (spec, matrix, duration_s, n_shown) in zip(axes, prepared):
        xlim = max_duration_s if shared_xlim else duration_s
        ax.imshow(
            matrix,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
            cmap=TIMELINE_HEATMAP_CMAP,
            norm=TIMELINE_HEATMAP_NORM,
            extent=[0.0, duration_s, n_shown, 0.0],
            rasterized=True,
        )
        ax.set_xlim(0.0, xlim)
        ax.set_ylim(n_shown, 0.0)
        ax.set_title(spec.title, loc="left", pad=1)
        ax.grid(False)

    axes[-1].set_xlabel("Time (s)")
    label_fp = axes[-1].xaxis.label.get_fontproperties()
    # Mixed warp/block: generic label
    fig.supylabel("Workers (sorted by total busy time)", fontproperties=label_fp)

    cbar = fig.colorbar(tasks_in_batch_scalar_mappable(), cax=cax, extend="min")
    cbar.set_label(COLORBAR_LABEL_TASKS, fontproperties=label_fp)
    configure_tasks_in_batch_colorbar(cbar)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {output_path}")


def _select_panels(only: str | None) -> tuple[PanelSpec, ...]:
    if not only:
        return DEFAULT_PANELS
    key = only.strip().lower().replace("-", "_")
    aliases = {
        "thread": "hetero_tree_thread",
        "thread_wo": "hetero_tree_thread",
        "thread_daq": "hetero_tree_thread_daq",
        "daq": "hetero_tree_thread_daq",
        "block": "hetero_tree_block",
        "block_cutoff": "hetero_tree_block_cutoff",
        "cutoff": "hetero_tree_block_cutoff",
    }
    app = aliases.get(key, key)
    for spec in DEFAULT_PANELS:
        if spec.app_name == app:
            return (spec,)
    raise ValueError(f"Unknown --only {only!r}; choose from {list(aliases)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="hetero_tree warp/block timeline heatmaps (paper style).",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=HETERO_DIR / "profile",
        help="Directory with hetero_tree_*_{warp,block}_timeline_working.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HETERO_DIR / "img",
        help="Directory for per-config PDFs (default: hetero_tree/img)",
    )
    parser.add_argument(
        "--stack",
        action="store_true",
        help="Emit one 4-row stacked PDF instead of four separate files",
    )
    parser.add_argument(
        "--stack-output",
        type=Path,
        default=None,
        help="Stacked PDF path (default: hetero_tree/img/hetero_tree_timelines_stack.pdf)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Plot a single config: thread|thread_daq|block|block_cutoff",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Show only the busiest N warps/blocks (0 = all)",
    )
    parser.add_argument(
        "--time-bins",
        type=int,
        default=DEFAULT_TIME_BINS,
        help=f"Horizontal time bins (default: {DEFAULT_TIME_BINS})",
    )
    parser.add_argument(
        "--no-sort-by-busy",
        action="store_true",
        help="Keep worker ID order instead of sorting by total busy time",
    )
    parser.add_argument(
        "--panel-height-ratio",
        type=float,
        default=0.38,
        help="Panel height as a multiple of figure width (default: 0.38)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help=f"savefig dpi (default: auto ≥{DEFAULT_MIN_PX_PER_WARP:g} px/row)",
    )
    parser.add_argument(
        "--shared-xlim",
        action="store_true",
        help="With --stack, pad all panels to the longest duration",
    )
    args = parser.parse_args()

    panels = _select_panels(args.only)
    if args.max_workers > 0:
        panels = tuple(
            PanelSpec(p.app_name, p.title, p.worker, args.max_workers) for p in panels
        )

    profile_dir = args.profile_dir.resolve()
    sort_by_busy = not args.no_sort_by_busy

    if args.stack:
        out = (
            args.stack_output.resolve()
            if args.stack_output is not None
            else (HETERO_DIR / "img" / f"hetero_tree_timelines_stack.{OUTPUT_FORMAT}").resolve()
        )
        plot_stack_timelines(
            panels=panels,
            output_path=out,
            profile_dir=profile_dir,
            n_bins=args.time_bins,
            sort_by_busy=sort_by_busy,
            panel_height_ratio=args.panel_height_ratio,
            save_dpi=args.dpi or DEFAULT_SAVE_DPI,
            shared_xlim=args.shared_xlim,
        )
        return

    out_dir = args.output_dir.resolve()
    for spec in panels:
        out = out_dir / f"{spec.app_name}_timeline.{OUTPUT_FORMAT}"
        plot_one_timeline(
            spec=spec,
            output_path=out,
            profile_dir=profile_dir,
            n_bins=args.time_bins,
            sort_by_busy=sort_by_busy,
            panel_height_ratio=args.panel_height_ratio,
            save_dpi=args.dpi,
            xlim_s=None,
        )


if __name__ == "__main__":
    main()

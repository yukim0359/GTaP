#!/usr/bin/env python3
"""Paper figure: MergeSort + CilkSort warp timeline heatmaps (stacked, shared axes/colorbar).

Output PDF is cropped tight to content (bbox_inches='tight').
Re-profile MergeSort at N=10^7 before publishing; crop busy warps for the MergeSort panel.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# benchmarks/ holds benchmark CSVs and thread_visualize_profile.py.
EVAL_DIR = Path(__file__).resolve().parents[1]
COMPARE_DIR = EVAL_DIR / "benchmarks"
IMG_DIR = EVAL_DIR / "img"
sys.path.insert(0, str(COMPARE_DIR))

from thread_visualize_profile import (  # noqa: E402
    DEFAULT_TIME_BINS,
    TIMELINE_HEATMAP_CMAP,
    TIMELINE_HEATMAP_NORM,
    build_warp_heatmap_matrix,
    compute_busy_time_per_warp,
    load_and_process_data,
    ordered_warp_ids,
    tasks_in_batch_scalar_mappable,
    COLORBAR_LABEL_TASKS,
    configure_tasks_in_batch_colorbar,
)

# ACM column width: figsize is in inches; LaTeX should include at 1:1 (width=FIG_WIDTH_IN).
FIG_WIDTH_IN = 239.75 / 72.0  # ≈ 3.33 in
DEFAULT_SAVE_DPI = 600
OUTPUT_FORMAT = "pdf"
COLORBAR_HEIGHT_SHRINK = 0.72  # fraction of heatmap column height

# 8pt paper typography.
PAPER_RC = {
    "font.size": 8,
    "font.weight": "semibold",
    "axes.labelsize": 8,
    "axes.labelweight": "semibold",
    "axes.titlesize": 8,
    "axes.titleweight": "semibold",
    "figure.labelsize": 8,
    "figure.labelweight": "semibold",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


@dataclass(frozen=True)
class PanelSpec:
    app_name: str
    title: str
    max_warps: int  # 0 = show all warps (sorted by busy time)


DEFAULT_PANELS = (
    PanelSpec(
        "mergesort",
        "MergeSort (Array Size=200,000)",
        16,
    ),
    PanelSpec(
        "cilksort",
        "CilkSort (Array Size=100,000,000)",
        0,
    ),
)


def _prepare_heatmap(
    app_name: str,
    *,
    n_bins: int,
    sort_by_busy: bool,
    max_warps: int,
):
    timeline_df, stats_df, strong_state = load_and_process_data(app_name)
    t_min = float(timeline_df["relative_time_ms"].min())
    t_max = float(timeline_df["relative_time_ms"].max())
    total_duration = max(0.0, t_max - t_min)
    if total_duration <= 0.0:
        raise ValueError(f"No timeline span for app={app_name!r}")

    busy_times = compute_busy_time_per_warp(
        timeline_df, t_min, t_max, strong_state=strong_state,
    )
    warp_ids = ordered_warp_ids(stats_df, busy_times, sort_by_busy=sort_by_busy)
    total_warps = len(warp_ids)
    if max_warps > 0:
        warp_ids = warp_ids[:max_warps]

    matrix = build_warp_heatmap_matrix(
        timeline_df,
        warp_ids,
        n_bins=n_bins,
        t_min=t_min,
        t_max=t_max,
        strong_state=strong_state,
    )
    return matrix, total_duration, len(warp_ids), total_warps


def plot_sort_timelines_paper(
    *,
    output_path: Path,
    panels: tuple[PanelSpec, ...] = DEFAULT_PANELS,
    n_bins: int = DEFAULT_TIME_BINS,
    sort_by_busy: bool = True,
    panel_height_ratio: float = 0.38,
    save_dpi: int = DEFAULT_SAVE_DPI,
):
    """Build a 2-row figure inside a fixed figsize; per-panel time axis."""
    plt.rcParams.update(PAPER_RC)

    n_panels = len(panels)
    fig_height = FIG_WIDTH_IN * panel_height_ratio * n_panels
    print(
        f"Figure size (fixed): {FIG_WIDTH_IN:.3f} x {fig_height:.3f} in @ {save_dpi} dpi"
    )

    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_height))
    # Heatmaps in column 0; colorbar in column 1 — all inside figsize (no tight crop).
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

    images = []
    for ax, spec in zip(axes, panels):
        print(f"Building heatmap for {spec.app_name}...")
        matrix, duration, n_shown, _n_total = _prepare_heatmap(
            spec.app_name,
            n_bins=n_bins,
            sort_by_busy=sort_by_busy,
            max_warps=spec.max_warps,
        )
        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
            cmap=TIMELINE_HEATMAP_CMAP,
            norm=TIMELINE_HEATMAP_NORM,
            extent=[0.0, duration, n_shown, 0.0],
            rasterized=True,
        )
        images.append(im)
        ax.set_xlim(0.0, duration)
        ax.set_ylim(n_shown, 0.0)
        ax.set_title(spec.title, loc="left", pad=1)
        ax.grid(False)

    axes[-1].set_xlabel("Time (ms)")
    label_fp = axes[-1].xaxis.label.get_fontproperties()
    fig.supylabel("Warps (sorted by total busy time)", fontproperties=label_fp)

    cbar = fig.colorbar(tasks_in_batch_scalar_mappable(), cax=cax, extend="min")
    cbar.set_label(COLORBAR_LABEL_TASKS, fontproperties=label_fp)
    configure_tasks_in_batch_colorbar(cbar)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {output_path} ({FIG_WIDTH_IN:.3f} x {fig_height:.3f} in)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stack MergeSort and CilkSort warp timeline heatmaps for the paper.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=IMG_DIR / f"sort_timelines_paper.{OUTPUT_FORMAT}",
        help="Output PDF path (default: evaluation/img/sort_timelines_paper.pdf)",
    )
    parser.add_argument(
        "--time-bins",
        type=int,
        default=DEFAULT_TIME_BINS,
        help=f"Horizontal time bins per panel (default: {DEFAULT_TIME_BINS})",
    )
    parser.add_argument(
        "--no-sort-by-busy",
        action="store_true",
        help="Keep warp ID order instead of sorting by total busy time",
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
        default=DEFAULT_SAVE_DPI,
        help=f"savefig dpi for rasterized heatmap (default: {DEFAULT_SAVE_DPI})",
    )
    parser.add_argument(
        "--mergesort-max-warps",
        type=int,
        default=16,
        help="Show only the busiest N warps in the MergeSort panel (0 = all)",
    )
    parser.add_argument(
        "--cilksort-max-warps",
        type=int,
        default=0,
        help="Show only the busiest N warps in the CilkSort panel (0 = all)",
    )
    parser.add_argument(
        "--mergesort-title",
        type=str,
        default="MergeSort (Array Size=200,000)",
        help="Panel title for MergeSort",
    )
    parser.add_argument(
        "--cilksort-title",
        type=str,
        default="CilkSort (Array Size=100,000,000)",
        help="Panel title for CilkSort",
    )
    args = parser.parse_args()

    panels = (
        PanelSpec("mergesort", args.mergesort_title, args.mergesort_max_warps),
        PanelSpec("cilksort", args.cilksort_title, args.cilksort_max_warps),
    )

    os.chdir(COMPARE_DIR)
    plot_sort_timelines_paper(
        output_path=args.output.resolve(),
        panels=panels,
        n_bins=args.time_bins,
        sort_by_busy=not args.no_sort_by_busy,
        panel_height_ratio=args.panel_height_ratio,
        save_dpi=args.dpi,
    )


if __name__ == "__main__":
    main()

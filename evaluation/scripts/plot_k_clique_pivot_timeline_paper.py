#!/usr/bin/env python3
"""Paper figure: k-clique pivoting warp timeline heatmap (GTaP thread workers).

Same layout as plot_sort_timelines_paper.py (3.33 in wide, 8pt, shared colorbar).
Expects profile CSVs under k_clique/profile/ from gtap_pivot with GTAP_PROFILE=1, e.g.:

  k_clique/profile/k_clique_pivot_warp_timeline_working.csv
  k_clique/profile/k_clique_pivot_warp_statistics_working.csv

Optional --profile-tag (e.g. _DBLP_k7) is appended before .csv.

X-axis modes (--xscale):
  native       GTaP's own duration (default output: gtap_k_clique_pivot_timeline.pdf)
  align-kcgpu  pad xlim to KCGPU span; post-finish is white
               (default output: gtap_k_clique_pivot_timeline_align_kcgpu.pdf)
Optional overrides: --xlim-s SECONDS, --align-kcgpu-stats PATH.
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
K_CLIQUE_DIR = COMPARE_DIR / "k_clique"
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
DEFAULT_SAVE_DPI = 600  # floor when auto dpi is unused / too small
# GridSpec vertical span in plot_k_clique_pivot_timeline_paper (bottom=0.075, top=0.975).
AXES_HEIGHT_FRAC = 0.975 - 0.075
OUTPUT_FORMAT = "pdf"
COLORBAR_HEIGHT_SHRINK = 0.88
COLORBAR_LABEL_OFFSET_PT = (-3.0, -2.0)  # (left, down) in points
DEFAULT_APP_NAME = "k_clique_pivot"
STRONG_STATE = "Working"


# Default: 2 px/warp so a single busy tail row survives PDF rasterization.
DEFAULT_MIN_PX_PER_WARP = 2.0
# Match plot_kcgpu_sm_timeline_paper.DEFAULT_PROFILE_TAG for shared x-axis runs.
DEFAULT_ALIGN_KCGPU_STATS = (
    K_CLIQUE_DIR
    / "profile"
    / "kcgpu"
    / "kcgpu_pivot_warp_statistics_working_as-Skitter_k9_edge_degen_p1b.csv"
)
POST_FINISH_FACECOLOR = "white"
XSCALE_NATIVE = "native"
XSCALE_ALIGN_KCGPU = "align-kcgpu"
XSCALE_CHOICES = (XSCALE_NATIVE, XSCALE_ALIGN_KCGPU)
DEFAULT_OUTPUT_BY_XSCALE = {
    XSCALE_NATIVE: IMG_DIR / f"gtap_k_clique_pivot_timeline.{OUTPUT_FORMAT}",
    XSCALE_ALIGN_KCGPU: (
        IMG_DIR / f"gtap_k_clique_pivot_timeline_align_kcgpu.{OUTPUT_FORMAT}"
    ),
}


def dpi_for_min_px_per_row(
    fig_height_in: float,
    n_rows: int,
    *,
    min_px_per_row: float = DEFAULT_MIN_PX_PER_WARP,
    floor_dpi: int = DEFAULT_SAVE_DPI,
) -> int:
    """Pick savefig dpi so each heatmap row gets ≥ min_px_per_row (figsize unchanged)."""
    axes_h_in = float(fig_height_in) * AXES_HEIGHT_FRAC
    if axes_h_in <= 0.0 or n_rows <= 0:
        return int(floor_dpi)
    return max(int(floor_dpi), int(math.ceil(n_rows * min_px_per_row / axes_h_in)))


def duration_s_from_kcgpu_stats(stats_path: Path) -> float:
    """KCGPU timeline span (s); same first/last rule as plot_kcgpu_sm_timeline_paper."""
    stats_df = pd.read_csv(stats_path)
    if "total_samples" in stats_df.columns:
        active = stats_df[stats_df["total_samples"] > 0]
    else:
        active = stats_df
    required = {"first_activity_ms", "last_activity_ms"}
    if active.empty or not required.issubset(active.columns):
        raise ValueError(
            f"KCGPU stats need first/last_activity_ms: {stats_path}"
        )
    first_ms = float(active["first_activity_ms"].min())
    last_ms = float(active["last_activity_ms"].max())
    duration_ms = max(0.0, last_ms - first_ms)
    if duration_ms <= 0.0:
        raise ValueError(f"Non-positive KCGPU duration in {stats_path}")
    return duration_ms / 1000.0


def resolve_xlim_s(
    *,
    xscale: str,
    xlim_s: float | None,
    align_kcgpu_stats: Path | None,
) -> float | None:
    """Resolve shared xlim; None means use GTaP's native duration only."""
    if xlim_s is not None:
        if xlim_s <= 0.0:
            raise ValueError(f"--xlim-s must be positive, got {xlim_s}")
        return float(xlim_s)

    if xscale == XSCALE_NATIVE and align_kcgpu_stats is None:
        return None

    stats_path = align_kcgpu_stats
    if stats_path is None and xscale == XSCALE_ALIGN_KCGPU:
        stats_path = DEFAULT_ALIGN_KCGPU_STATS
    if stats_path is None:
        return None

    path = stats_path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"KCGPU stats for x-axis align: {path}")
    dur = duration_s_from_kcgpu_stats(path)
    print(f"Align xlim from KCGPU stats {path.name}: {dur:.6f} s")
    return dur


def default_output_path(xscale: str) -> Path:
    try:
        return DEFAULT_OUTPUT_BY_XSCALE[xscale]
    except KeyError as exc:
        raise ValueError(f"Unknown --xscale {xscale!r}") from exc


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


@dataclass(frozen=True)
class PanelSpec:
    max_warps: int  # 0 = all warps (sorted by busy time)


DEFAULT_PANEL = PanelSpec(max_warps=0)


def _normalize_profile_tag(profile_tag: str) -> str:
    if not profile_tag:
        return ""
    return profile_tag if profile_tag.startswith("_") else f"_{profile_tag}"


def resolve_profile_csv_paths(
    *,
    profile_dir: Path,
    app_name: str,
    profile_tag: str,
) -> tuple[Path, Path]:
    tag = _normalize_profile_tag(profile_tag)
    primary_tl = profile_dir / f"{app_name}_warp_timeline_working{tag}.csv"
    primary_st = profile_dir / f"{app_name}_warp_statistics_working{tag}.csv"
    fallback_tl = profile_dir / f"warp_timeline_working{tag}.csv"
    fallback_st = profile_dir / f"warp_statistics_working{tag}.csv"

    timeline_path = primary_tl if primary_tl.exists() else fallback_tl
    stats_path = primary_st if primary_st.exists() else fallback_st
    return timeline_path, stats_path


def load_k_clique_pivot_data(
    *,
    profile_dir: Path,
    app_name: str,
    profile_tag: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    timeline_path, stats_path = resolve_profile_csv_paths(
        profile_dir=profile_dir,
        app_name=app_name,
        profile_tag=profile_tag,
    )
    if not timeline_path.exists():
        raise FileNotFoundError(f"Missing timeline CSV: {timeline_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing statistics CSV: {stats_path}")

    print(f"Loading: {timeline_path}")
    print(f"Loading: {stats_path}")
    timeline_df = pd.read_csv(timeline_path)
    stats_df = pd.read_csv(stats_path)

    if "warp_id" not in timeline_df.columns:
        raise ValueError(f"'warp_id' column missing in {timeline_path}")
    if "warp_id" not in stats_df.columns:
        raise ValueError(f"'warp_id' column missing in {stats_path}")

    return timeline_df, stats_df, STRONG_STATE


def _prepare_heatmap(
    *,
    profile_dir: Path,
    app_name: str,
    profile_tag: str,
    n_bins: int,
    sort_by_busy: bool,
    max_warps: int,
):
    timeline_df, stats_df, strong_state = load_k_clique_pivot_data(
        profile_dir=profile_dir,
        app_name=app_name,
        profile_tag=profile_tag,
    )
    t_min = float(timeline_df["relative_time_ms"].min())
    t_max = float(timeline_df["relative_time_ms"].max())
    total_duration = max(0.0, t_max - t_min)
    if total_duration <= 0.0:
        raise ValueError("No timeline span in k-clique pivot profile")

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


def plot_k_clique_pivot_timeline_paper(
    *,
    output_path: Path,
    panel: PanelSpec = DEFAULT_PANEL,
    profile_dir: Path,
    app_name: str,
    profile_tag: str,
    n_bins: int = DEFAULT_TIME_BINS,
    sort_by_busy: bool = True,
    panel_height_ratio: float = 0.38,
    save_dpi: int | None = None,
    xscale: str = XSCALE_NATIVE,
    xlim_s: float | None = None,
    align_kcgpu_stats: Path | None = None,
) -> None:
    plt.rcParams.update(PAPER_RC)

    fig_height = FIG_WIDTH_IN * panel_height_ratio
    shared_xlim_s = resolve_xlim_s(
        xscale=xscale,
        xlim_s=xlim_s,
        align_kcgpu_stats=align_kcgpu_stats,
    )

    print("Building k-clique pivot heatmap...")
    matrix, duration, n_shown, _n_total = _prepare_heatmap(
        profile_dir=profile_dir,
        app_name=app_name,
        profile_tag=profile_tag,
        n_bins=n_bins,
        sort_by_busy=sort_by_busy,
        max_warps=panel.max_warps,
    )
    if save_dpi is None:
        save_dpi = dpi_for_min_px_per_row(fig_height, n_shown)
        print(
            f"Auto dpi={save_dpi} for ≥{DEFAULT_MIN_PX_PER_WARP:g} px/warp "
            f"({n_shown} rows, axes≈{fig_height * AXES_HEIGHT_FRAC:.3f} in)"
        )
    print(
        f"Figure size (fixed): {FIG_WIDTH_IN:.3f} x {fig_height:.3f} in @ {save_dpi} dpi "
        f"(≈{save_dpi * fig_height * AXES_HEIGHT_FRAC / max(n_shown, 1):.2f} px/warp)"
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

    duration_s = duration / 1000.0
    axis_xlim_s = duration_s if shared_xlim_s is None else max(duration_s, shared_xlim_s)
    # Heatmap only covers GTaP's own run; padding to a longer shared xlim is white
    # (not orange idle), so post-finish time is visually "no kernel" rather than idle warps.
    ax.set_facecolor(POST_FINISH_FACECOLOR)
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
    ax.grid(False)
    if shared_xlim_s is not None and axis_xlim_s > duration_s + 1e-9:
        print(
            f"Shared xlim={axis_xlim_s:.6f} s; GTaP data ends at {duration_s:.6f} s; "
            f"padding is {POST_FINISH_FACECOLOR}"
        )

    label_fp = ax.xaxis.label.get_fontproperties()
    fig.supylabel("Warps (sorted by total busy time)", fontproperties=label_fp)

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
    print(f"Saved: {output_path} ({FIG_WIDTH_IN:.3f} x {fig_height:.3f} in)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="k-clique pivot warp timeline heatmap for the paper.",
    )
    parser.add_argument(
        "--xscale",
        choices=XSCALE_CHOICES,
        default=XSCALE_NATIVE,
        help=(
            "X-axis mode: 'native' = GTaP duration only; "
            "'align-kcgpu' = pad to KCGPU span with white after finish "
            f"(default: {XSCALE_NATIVE})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PDF path (default depends on --xscale: "
            f"{DEFAULT_OUTPUT_BY_XSCALE[XSCALE_NATIVE].name} or "
            f"{DEFAULT_OUTPUT_BY_XSCALE[XSCALE_ALIGN_KCGPU].name})"
        ),
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=K_CLIQUE_DIR / "profile",
        help="Directory with k_clique_pivot_warp_timeline_working*.csv",
    )
    parser.add_argument(
        "--app-name",
        default=DEFAULT_APP_NAME,
        help=f"Profile CSV prefix (default: {DEFAULT_APP_NAME})",
    )
    parser.add_argument(
        "--profile-tag",
        default="",
        help="Optional suffix before .csv, e.g. _DBLP_k7",
    )
    parser.add_argument(
        "--max-warps",
        type=int,
        default=0,
        help="Show only the busiest N warps (0 = all)",
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
        default=None,
        help=(
            "savefig dpi for rasterized heatmap "
            f"(default: auto so each warp row is ≥{DEFAULT_MIN_PX_PER_WARP:g} px; "
            "figsize unchanged)"
        ),
    )
    parser.add_argument(
        "--xlim-s",
        type=float,
        default=None,
        help="Override x-axis max in seconds (pads with white after GTaP finishes)",
    )
    parser.add_argument(
        "--align-kcgpu-stats",
        type=Path,
        default=None,
        help=(
            "KCGPU warp statistics CSV for align-kcgpu xlim "
            f"(default when --xscale align-kcgpu: {DEFAULT_ALIGN_KCGPU_STATS.name})"
        ),
    )
    args = parser.parse_args()

    output_path = (
        args.output.resolve()
        if args.output is not None
        else default_output_path(args.xscale).resolve()
    )

    plot_k_clique_pivot_timeline_paper(
        output_path=output_path,
        panel=PanelSpec(max_warps=args.max_warps),
        profile_dir=args.profile_dir.resolve(),
        app_name=args.app_name,
        profile_tag=args.profile_tag,
        n_bins=args.time_bins,
        sort_by_busy=not args.no_sort_by_busy,
        panel_height_ratio=args.panel_height_ratio,
        save_dpi=args.dpi,
        xscale=args.xscale,
        xlim_s=args.xlim_s,
        align_kcgpu_stats=args.align_kcgpu_stats,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paper figure: FMM 3-way end-to-end phase stack (GTaP / worklist / Host OMP).

Reads summary CSV from fmm/csv/ and renders a horizontal stacked-bar breakdown by
default at ACM single-column width (3.33 in) with 8pt typography. Phases are
collapsed to seven paper segments (Tree/setup, DTT init, DTT, DTT H2D, M2L,
P2P/L2P, Other). Use --vertical for the legacy column-oriented layout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

EVAL_DIR = Path(__file__).resolve().parents[1]
COMPARE_DIR = EVAL_DIR / "benchmarks"
IMG_DIR = EVAL_DIR / "img"
FMM_DIR = COMPARE_DIR / "fmm"
sys.path.insert(0, str(FMM_DIR))

from plot_phase_stack import (  # noqa: E402
    PAPER_DTT_PHASES,
    PAPER_EDGE_COLORS,
    PAPER_PHASES,
    PAPER_COLORS,
    RunBreakdown,
    collapse_run_to_paper,
    format_theta_label,
    gtap_baseline_index,
    paper_display_total,
    paper_dtt_band_bounds,
    paper_dtt_section_ms,
    paper_stack_phase_order,
    read_summary_csv,
)

FIG_WIDTH_IN = 239.75 / 72.0  # ≈ 3.33 in (ACM single column)
_VERTICAL_PLOT_HEIGHT_RATIO = 0.60 * 1.1 * 0.9
_HORIZONTAL_PLOT_HEIGHT_PER_RUN = 0.11 * 1.2
_HORIZONTAL_PLOT_BASE_HEIGHT_RATIO = 0.14
_LEGEND_HEIGHT_IN = 0.52
_LEGEND_GAP_FRAC = 0.05
_HORIZONTAL_LEGEND_GAP_FRAC = 0.11
BAR_WIDTH = 0.40 * 1.1
BAR_HEIGHT = BAR_WIDTH
BAR_HEIGHT_H = 0.30 * 0.8 * 0.8
BAR_ROW_STEP = 0.40 * 1.2
LEGEND_FONT_SIZE = 8
OUTPUT_FORMAT = "pdf"

Layout = Literal["horizontal", "vertical"]

PAPER_XLABELS = {
    "GTaP DTT": "GTaP",
    "GPU worklist DTT": "GPU worklist",
    "Host OpenMP DTT": "CPU OpenMP",
}

PAPER_YLABELS = {
    "GTaP DTT": "GTaP",
    "GPU worklist DTT": "GPU\nworklist",
    "Host OpenMP DTT": "CPU\nOpenMP",
}

_HORIZONTAL_RUN_ORDER = (
    "GTaP DTT",
    "GPU worklist DTT",
    "Host OpenMP DTT",
)

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
    "lines.linewidth": 1.0,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _default_csv(n: int, theta: float) -> Path:
    theta_lbl = format_theta_label(theta)
    return FMM_DIR / "csv" / f"fmm_3way_stack_N{n}_theta{theta_lbl}_summary.csv"


def _default_output(n: int, theta: float, *, layout: Layout) -> Path:
    theta_lbl = format_theta_label(theta)
    stem = f"fmm_3way_stack_N{n}_theta{theta_lbl}"
    if layout == "vertical":
        stem += "_v"
    return IMG_DIR / f"{stem}.{OUTPUT_FORMAT}"


def _figure_height(n_runs: int, *, layout: Layout) -> float:
    if layout == "vertical":
        return FIG_WIDTH_IN * _VERTICAL_PLOT_HEIGHT_RATIO + _LEGEND_HEIGHT_IN
    plot_h = FIG_WIDTH_IN * (
        _HORIZONTAL_PLOT_BASE_HEIGHT_RATIO + _HORIZONTAL_PLOT_HEIGHT_PER_RUN * n_runs
    )
    return plot_h + _LEGEND_HEIGHT_IN


def _axis_label(run_label: str) -> str:
    return PAPER_XLABELS.get(run_label, run_label)


def _axis_label_horizontal(run_label: str) -> str:
    return PAPER_YLABELS.get(run_label, run_label)


def _sort_runs_horizontal(runs: list[RunBreakdown]) -> list[RunBreakdown]:
    order = {label: idx for idx, label in enumerate(_HORIZONTAL_RUN_ORDER)}
    return sorted(runs, key=lambda run: order.get(run.label, len(order)))


def _paper_phase_filter(include_init: bool) -> list[str]:
    if include_init:
        return list(PAPER_PHASES)
    return [phase for phase in PAPER_PHASES if phase != "DTT init"]


def _bar_edge_kwargs(phase: str) -> dict:
    if phase in PAPER_DTT_PHASES:
        return {
            "edgecolor": PAPER_EDGE_COLORS[phase],
            "linewidth": 0.55,
            "zorder": 5,
        }
    return {
        "edgecolor": "none",
        "linewidth": 0.0,
        "zorder": 3,
    }


def _add_phase_legend(
    fig: plt.Figure,
    fig_height_in: float,
    used_phases: list[str],
    phase_filter: list[str],
    *,
    subplot_bottom: float,
    legend_gap_frac: float = _LEGEND_GAP_FRAC,
) -> None:
    legend_phases = paper_stack_phase_order(
        {phase: (1.0 if phase in set(used_phases) else 0.0) for phase in PAPER_PHASES},
        phase_filter,
    )
    legend_handles = [Patch(facecolor=PAPER_COLORS[p], label=p) for p in legend_phases]
    axes_bottom = _LEGEND_HEIGHT_IN / fig_height_in
    fig.legend(
        handles=legend_handles,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        bbox_transform=fig.transFigure,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=0.7,
        handletextpad=0.25,
        columnspacing=0.55,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(
        bottom=subplot_bottom + axes_bottom + legend_gap_frac,
    )


def _plot_fmm_3way_stack_vertical(
    runs: list[RunBreakdown],
    *,
    output_path: Path,
    include_init: bool = True,
    show_diff_lines: bool = True,
) -> None:
    phase_filter = _paper_phase_filter(include_init)
    x = list(range(len(runs)))
    bottoms = [0.0] * len(runs)
    used_phases: list[str] = []

    n_runs = len(runs)
    bar_half = BAR_WIDTH * 0.5
    fig_height_in = _figure_height(n_runs, layout="vertical")
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, fig_height_in))

    for i, run in enumerate(runs):
        run_order = paper_stack_phase_order(run.phases, phase_filter)
        for phase in run_order:
            value = run.phases.get(phase, 0.0)
            if value == 0.0:
                continue
            ax.bar(
                x[i],
                value,
                bottom=bottoms[i],
                width=BAR_WIDTH,
                color=PAPER_COLORS[phase],
                **_bar_edge_kwargs(phase),
            )
            bottoms[i] += value
            if phase not in used_phases:
                used_phases.append(phase)

    dtt_bounds = [paper_dtt_band_bounds(run, phase_filter) for run in runs]
    if show_diff_lines and len(runs) >= 2:
        for edge_idx in (0, 1):
            for i in range(len(runs) - 1):
                y0 = dtt_bounds[i][edge_idx]
                y1 = dtt_bounds[i + 1][edge_idx]
                ax.plot(
                    [x[i] + bar_half, x[i + 1] - bar_half],
                    [y0, y1],
                    color="#444444",
                    linestyle=(0, (2, 2)),
                    linewidth=0.7,
                    alpha=0.65,
                    zorder=4,
                )

    gtap_idx = gtap_baseline_index(runs)
    gtap_dtt_ms = paper_dtt_section_ms(runs[gtap_idx], phase_filter)
    for i, run in enumerate(runs):
        if i == gtap_idx or gtap_dtt_ms <= 0.0:
            continue
        ratio = paper_dtt_section_ms(run, phase_filter) / gtap_dtt_ms
        bot, top = dtt_bounds[i]
        if top <= bot:
            continue
        ax.text(
            x[i] + bar_half * 1.2,
            (bot + top) * 0.5,
            f"×{ratio:.2f}",
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#c00000",
            zorder=10,
        )

    ymax = max(max(bottoms), max(paper_display_total(run, include_init) for run in runs))
    ms_label_gap = ymax * 0.022
    for i, run in enumerate(runs):
        ax.text(
            i,
            bottoms[i] + ms_label_gap,
            f"{paper_display_total(run, include_init):.1f} ms",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="semibold",
        )

    labels = [_axis_label(run.label) for run in runs]
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.50, (n_runs - 1) + bar_half * 1.2 + 0.40)
    ax.set_ylim(0.0, ymax * 1.08)
    ax.set_ylabel("Time (ms)")
    ax.grid(axis="y", linestyle="--", alpha=plt.rcParams["grid.alpha"])

    fig.subplots_adjust(left=0.16, right=0.98, top=0.98, bottom=0.0)
    _add_phase_legend(fig, fig_height_in, used_phases, phase_filter, subplot_bottom=0.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path} ({FIG_WIDTH_IN:.3f} x {fig_height_in:.3f} in, vertical)")


def _plot_fmm_3way_stack_horizontal(
    runs: list[RunBreakdown],
    *,
    output_path: Path,
    include_init: bool = True,
    show_diff_lines: bool = True,
) -> None:
    runs = _sort_runs_horizontal(runs)
    phase_filter = _paper_phase_filter(include_init)
    y = [i * BAR_ROW_STEP for i in range(len(runs))]
    lefts = [0.0] * len(runs)
    used_phases: list[str] = []

    n_runs = len(runs)
    bar_half = BAR_HEIGHT_H * 0.5
    fig_height_in = _figure_height(n_runs, layout="horizontal")
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, fig_height_in))

    for i, run in enumerate(runs):
        run_order = paper_stack_phase_order(run.phases, phase_filter)
        for phase in run_order:
            value = run.phases.get(phase, 0.0)
            if value == 0.0:
                continue
            ax.barh(
                y[i],
                value,
                left=lefts[i],
                height=BAR_HEIGHT_H,
                color=PAPER_COLORS[phase],
                **_bar_edge_kwargs(phase),
            )
            lefts[i] += value
            if phase not in used_phases:
                used_phases.append(phase)

    dtt_bounds = [paper_dtt_band_bounds(run, phase_filter) for run in runs]
    if show_diff_lines and len(runs) >= 2:
        for edge_idx in (0, 1):
            for i in range(len(runs) - 1):
                x0 = dtt_bounds[i][edge_idx]
                x1 = dtt_bounds[i + 1][edge_idx]
                ax.plot(
                    [x0, x1],
                    [y[i] + bar_half, y[i + 1] - bar_half],
                    color="#444444",
                    linestyle=(0, (2, 2)),
                    linewidth=0.7,
                    alpha=0.65,
                    zorder=4,
                )

    gtap_idx = gtap_baseline_index(runs)
    gtap_dtt_ms = paper_dtt_section_ms(runs[gtap_idx], phase_filter)
    for i, run in enumerate(runs):
        if i == gtap_idx or gtap_dtt_ms <= 0.0:
            continue
        ratio = paper_dtt_section_ms(run, phase_filter) / gtap_dtt_ms
        left, right = dtt_bounds[i]
        if right <= left:
            continue
        ax.text(
            (left + right) * 0.5,
            y[i] - bar_half * 1.25,
            f"×{ratio:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#c00000",
            zorder=10,
        )

    xmax = max(max(lefts), max(paper_display_total(run, include_init) for run in runs))
    for i, run in enumerate(runs):
        ax.text(
            lefts[i],
            y[i] - bar_half,
            f"{paper_display_total(run, include_init):.1f} ms",
            ha="right",
            va="bottom",
            fontsize=8,
            fontweight="semibold",
        )

    labels = [_axis_label_horizontal(run.label) for run in runs]
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", pad=1)
    ax.invert_yaxis()
    y_span = (n_runs - 1) * BAR_ROW_STEP
    ax.set_xlim(0.0, xmax * 1.04)
    ax.set_ylim(y_span + bar_half + 0.12, -bar_half * 1.25 - 0.15)
    ax.set_xlabel("Time (ms)")
    ax.grid(axis="x", linestyle="--", alpha=plt.rcParams["grid.alpha"])

    fig.subplots_adjust(left=0.19, right=0.98, top=0.98, bottom=0.0)
    _add_phase_legend(
        fig,
        fig_height_in,
        used_phases,
        phase_filter,
        subplot_bottom=0.0,
        legend_gap_frac=_HORIZONTAL_LEGEND_GAP_FRAC,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path} ({FIG_WIDTH_IN:.3f} x {fig_height_in:.3f} in, horizontal)")


def plot_fmm_3way_stack_paper(
    runs: list[RunBreakdown],
    *,
    output_path: Path,
    include_init: bool = True,
    show_diff_lines: bool = True,
    layout: Layout = "horizontal",
) -> None:
    plt.rcParams.update(PAPER_RC)
    paper_runs = [collapse_run_to_paper(run) for run in runs]
    if layout == "vertical":
        _plot_fmm_3way_stack_vertical(
            paper_runs,
            output_path=output_path,
            include_init=include_init,
            show_diff_lines=show_diff_lines,
        )
    else:
        _plot_fmm_3way_stack_horizontal(
            paper_runs,
            output_path=output_path,
            include_init=include_init,
            show_diff_lines=show_diff_lines,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FMM 3-way phase stack for paper (3.33 in, 8pt).",
    )
    parser.add_argument("--n", type=int, default=50_000_000)
    parser.add_argument("--theta", type=float, default=0.2)
    parser.add_argument(
        "--csv-in",
        type=Path,
        default=None,
        help="Summary CSV (default: fmm/csv/fmm_3way_stack_N<N>_theta<THETA>_summary.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            f"Output path (default: img/fmm_3way_stack_N<N>_theta<THETA>.{OUTPUT_FORMAT}; "
            f"vertical adds _v suffix)."
        ),
    )
    parser.add_argument(
        "--layout",
        choices=("horizontal", "vertical"),
        default="horizontal",
        help="Stack orientation (default: horizontal).",
    )
    parser.add_argument(
        "--vertical",
        action="store_true",
        help="Shortcut for --layout vertical (legacy column stack).",
    )
    parser.add_argument(
        "--no-init",
        action="store_true",
        help="Exclude DTT init from the stacked bar.",
    )
    parser.add_argument(
        "--no-diff-lines",
        action="store_true",
        help="Do not draw dotted DTT-band connectors.",
    )
    args = parser.parse_args()

    layout: Layout = "vertical" if args.vertical else args.layout

    csv_path = (args.csv_in or _default_csv(args.n, args.theta)).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    data = read_summary_csv(str(csv_path))
    if not data.runs:
        raise ValueError(f"No runs in {csv_path}")

    output_path = (
        args.output or _default_output(args.n, args.theta, layout=layout)
    ).resolve()

    plot_fmm_3way_stack_paper(
        data.runs,
        output_path=output_path,
        include_init=not args.no_init,
        show_diff_lines=not args.no_diff_lines,
        layout=layout,
    )


if __name__ == "__main__":
    main()

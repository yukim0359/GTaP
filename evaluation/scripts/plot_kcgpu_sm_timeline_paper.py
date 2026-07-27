#!/usr/bin/env python3
"""Paper figure: KCGPU SM working timeline heatmap (binned active slots).

Monochrome greens for active slots (1..slots_per_sm); idle (0) uses a distinct
lavender under-color (not the Blues/orange scheme of the GTaP warp timeline).
SMs are sorted by total busy time, like GTaP warp timelines.

Expects profile CSVs under k_clique/profile/kcgpu/, e.g.:

  kcgpu/kcgpu_pivot_warp_timeline_working_as-Skitter_k9_edge_degen_p1b.csv
  kcgpu/kcgpu_pivot_warp_statistics_working_as-Skitter_k9_edge_degen_p1b.csv

Optional --profile-tag (e.g. _as-Skitter_k9_edge_degen_p1b) is appended before .csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

EVAL_DIR = Path(__file__).resolve().parents[1]
COMPARE_DIR = EVAL_DIR / "2-comparison"
IMG_DIR = EVAL_DIR / "img"
K_CLIQUE_DIR = COMPARE_DIR / "k_clique"
K_CLIQUE_SCRIPTS = K_CLIQUE_DIR / "scripts"
sys.path.insert(0, str(K_CLIQUE_SCRIPTS))

from kcgpu_visualize_profile import (  # noqa: E402
    SM_TIMELINE_BINS,
    _add_interval_to_sm_timeline,
    _infer_slots_per_sm,
    _timeline_paths,
    _worker_col,
)

FIG_WIDTH_IN = 239.75 / 72.0  # ≈ 3.33 in
DEFAULT_SAVE_DPI = 600
OUTPUT_FORMAT = "pdf"
COLORBAR_HEIGHT_SHRINK = 0.88
PANEL_HEIGHT_SCALE = 0.82  # shrink auto height vs. kcgpu_visualize_profile aspect
DEFAULT_SM_COUNT = 132
# Distinct from GTaP timeline (Blues + orange idle).
SM_ACTIVE_CMAP = plt.cm.Greens
SM_IDLE_COLOR = "#d4c4e8"  # soft lavender
COLORBAR_LABEL = "Active worker slots (0 = idle)"
DEFAULT_PROFILE_TAG = "_as-Skitter_k9_edge_degen_p1b"


def make_sm_slots_colormap(slots_per_sm: int):
    """Greens for any positive occupancy; idle (exactly 0) via cmap.set_under."""
    cmap = SM_ACTIVE_CMAP.copy()
    cmap.set_under(SM_IDLE_COLOR)
    # Tiny positive vmin: values == 0 use under; any occupancy uses Greens.
    norm = mpl.colors.Normalize(vmin=1e-6, vmax=float(max(slots_per_sm, 1)))
    return cmap, norm


def sm_slots_scalar_mappable(slots_per_sm: int):
    cmap, norm = make_sm_slots_colormap(slots_per_sm)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm


def configure_sm_slots_colorbar(cbar, slots_per_sm: int) -> None:
    step = 4 if slots_per_sm >= 8 else 1
    ticks = list(range(0, int(slots_per_sm) + 1, step))
    if ticks[-1] != int(slots_per_sm):
        ticks.append(int(slots_per_sm))
    cbar.set_ticks(ticks)

GRAPH_ALIASES = {
    "dblp": "DBLP",
    "com-dblp": "DBLP",
    "skitter": "as-Skitter",
    "as-skitter": "as-Skitter",
    "orkut": "Orkut",
    "com-orkut": "com-Orkut",
    "livejournal": "com-LiveJournal",
    "com-livejournal": "com-LiveJournal",
    "facebook": "ego-Facebook",
    "ego-facebook": "ego-Facebook",
}

VARIANT_DEFAULTS = {
    "pivot": {"process": "edge", "orient": "degen", "q": "p1b"},
    "orientation": {"process": "edge", "orient": "degree", "q": "o1b"},
}

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
    max_sms: int  # 0 = all SMs (sorted by busy time)


DEFAULT_PANEL = PanelSpec(max_sms=0)


def _normalize_profile_tag(profile_tag: str) -> str:
    if not profile_tag:
        return ""
    return profile_tag if profile_tag.startswith("_") else f"_{profile_tag}"


def _canonical_graph_name(graph: str) -> str:
    key = graph.strip().lower()
    return GRAPH_ALIASES.get(key, graph)


def _build_profile_tag(*, graph: str, k: int, variant: str) -> str:
    defaults = VARIANT_DEFAULTS[variant]
    graph_name = _canonical_graph_name(graph)
    return f"_{graph_name}_k{k}_{defaults['process']}_{defaults['orient']}_{defaults['q']}"


def _infer_variant_from_tag(profile_tag: str) -> str | None:
    tag = _normalize_profile_tag(profile_tag).lower()
    if tag.endswith("_o1b") or tag.endswith("_o2b") or "_degree_" in tag:
        return "orientation"
    if tag.endswith("_p1b") or tag.endswith("_p2b") or "_degen_" in tag:
        return "pivot"
    return None


def _fix_profile_tag_for_variant(profile_tag: str, variant: str) -> str:
    """Repair common tag suffix mismatches (e.g. degen + o1b with pivot variant)."""
    tag = _normalize_profile_tag(profile_tag)
    defaults = VARIANT_DEFAULTS[variant]
    parts = tag.split("_")
    if len(parts) < 2:
        return tag

    graph = parts[1]
    graph = _canonical_graph_name(graph)
    parts[1] = graph

    k_idx = next((i for i, part in enumerate(parts) if part.startswith("k") and part[1:].isdigit()), None)
    if k_idx is None:
        return "_" + "_".join(parts[1:])

    k = parts[k_idx]
    return _build_profile_tag(graph=graph, k=int(k[1:]), variant=variant)


def _profile_csv_paths(profile_dir: Path, app_name: str, profile_tag: str) -> tuple[Path, Path]:
    tag = _normalize_profile_tag(profile_tag)
    tl_str, st_str = _timeline_paths(str(profile_dir), app_name, tag)
    return Path(tl_str), Path(st_str)


def _list_matching_stats(profile_dir: Path, app_name: str, needle: str) -> list[Path]:
    pattern = f"{app_name}_warp_statistics_working*{needle}*.csv"
    return sorted(profile_dir.glob(pattern))


def resolve_kcgpu_profile(
    *,
    profile_dir: Path,
    app_name: str,
    profile_tag: str,
    variant: str,
) -> tuple[Path, Path, str, str]:
    candidates: list[tuple[str, str]] = []
    tag = _normalize_profile_tag(profile_tag)
    candidates.append((app_name, tag))

    fixed_tag = _fix_profile_tag_for_variant(tag, variant)
    if fixed_tag != tag:
        candidates.append((app_name, fixed_tag))

    inferred = _infer_variant_from_tag(tag)
    if inferred and inferred != variant:
        alt_app = f"kcgpu_{inferred}"
        candidates.append((alt_app, _fix_profile_tag_for_variant(tag, inferred)))

    seen: set[tuple[str, str]] = set()
    for candidate_app, candidate_tag in candidates:
        key = (candidate_app, candidate_tag)
        if key in seen:
            continue
        seen.add(key)
        timeline_path, stats_path = _profile_csv_paths(profile_dir, candidate_app, candidate_tag)
        if timeline_path.exists() and stats_path.exists():
            if candidate_tag != tag or candidate_app != app_name:
                print(
                    f"Resolved profile: app={candidate_app}, tag={candidate_tag} "
                    f"(requested app={app_name}, tag={tag})"
                )
            return timeline_path, stats_path, candidate_tag, candidate_app

    needle = tag.strip("_").split("_")[0] if tag else ""
    if needle.lower() in GRAPH_ALIASES:
        needle = _canonical_graph_name(needle)
    matches = _list_matching_stats(profile_dir, app_name, needle) if needle else []
    hint = ""
    if matches:
        hint = "\nNearby matches:\n  " + "\n  ".join(p.name for p in matches[:8])

    timeline_path, stats_path = _profile_csv_paths(profile_dir, app_name, tag)
    raise FileNotFoundError(
        f"Missing KCGPU profile CSVs for app={app_name}, tag={tag}.\n"
        f"  expected stats: {stats_path}\n"
        f"  expected timeline: {timeline_path}\n"
        f"Hint: graph names are often prefixed (e.g. Skitter -> as-Skitter); "
        f"pivot uses ..._degen_p1b, orientation uses ..._degree_o1b.{hint}"
    )


def compute_busy_time_per_sm(timeline: list[list[float]], bin_width_ms: float) -> dict[int, float]:
    """Total slot-ms of activity per SM (integral of binned active-slot fraction)."""
    busy: dict[int, float] = {}
    for sm_id, row in enumerate(timeline):
        busy[sm_id] = sum(value * bin_width_ms for value in row)
    return busy


def ordered_sm_ids(
    sm_count: int,
    busy_times: dict[int, float],
    *,
    sort_by_busy: bool,
) -> list[int]:
    sm_ids = list(range(sm_count))
    if not sort_by_busy:
        return sm_ids
    return sorted(sm_ids, key=lambda sm_id: (-busy_times.get(sm_id, 0.0), sm_id))


def build_sm_timeline_matrix(
    *,
    profile_dir: Path,
    app_name: str,
    profile_tag: str,
    variant: str,
    n_bins: int,
    sm_count: int = DEFAULT_SM_COUNT,
) -> tuple[list[list[float]], float, int, list[int], dict[int, float]]:
    timeline_path, stats_path, _resolved_tag, _resolved_app = resolve_kcgpu_profile(
        profile_dir=profile_dir,
        app_name=app_name,
        profile_tag=profile_tag,
        variant=variant,
    )
    print(f"Loading: {stats_path}")
    print(f"Loading: {timeline_path}")
    stats_df = pd.read_csv(stats_path)
    id_col = _worker_col(stats_df)
    active = stats_df[stats_df["total_samples"] > 0].copy()
    required = {"first_activity_ms", "last_activity_ms"}
    if active.empty or not required.issubset(active.columns):
        raise ValueError("No active slot first/last activity data for SM timeline")

    slots_per_sm = _infer_slots_per_sm(active, id_col, sm_count)
    first_ms = float(active["first_activity_ms"].min())
    last_ms = float(active["last_activity_ms"].max())
    duration_ms = max(0.0, last_ms - first_ms)
    if duration_ms <= 0.0:
        raise ValueError("No positive KCGPU SM timeline duration")

    bins = max(1, int(n_bins))
    bin_width = duration_ms / bins
    timeline = [[0.0 for _ in range(bins)] for _ in range(sm_count)]

    active_start_by_slot: dict[int, float] = {}
    with open(timeline_path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty timeline CSV: {timeline_path}")
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
                _add_interval_to_sm_timeline(
                    timeline, sm_id, start, t, first_ms, bin_width,
                )

    for slot_id, start in active_start_by_slot.items():
        sm_id = slot_id // slots_per_sm
        if 0 <= sm_id < sm_count:
            _add_interval_to_sm_timeline(
                timeline, sm_id, start, last_ms, first_ms, bin_width,
            )

    busy_times = compute_busy_time_per_sm(timeline, bin_width)
    return timeline, duration_ms, slots_per_sm, sm_count, busy_times


def _prepare_heatmap(
    *,
    profile_dir: Path,
    app_name: str,
    profile_tag: str,
    variant: str,
    n_bins: int,
    sort_by_busy: bool,
    max_sms: int,
    sm_count: int,
) -> tuple[list[list[float]], float, int, int, int]:
    timeline, duration_ms, slots_per_sm, sm_count, busy_times = build_sm_timeline_matrix(
        profile_dir=profile_dir,
        app_name=app_name,
        profile_tag=profile_tag,
        variant=variant,
        n_bins=n_bins,
        sm_count=sm_count,
    )
    sm_ids = ordered_sm_ids(sm_count, busy_times, sort_by_busy=sort_by_busy)
    total_sms = len(sm_ids)
    if max_sms > 0:
        sm_ids = sm_ids[:max_sms]

    matrix = [timeline[sm_id] for sm_id in sm_ids]
    return matrix, duration_ms, slots_per_sm, len(sm_ids), total_sms


def plot_kcgpu_sm_timeline_paper(
    *,
    output_path: Path,
    panel: PanelSpec = DEFAULT_PANEL,
    profile_dir: Path,
    app_name: str,
    profile_tag: str,
    variant: str,
    n_bins: int = SM_TIMELINE_BINS,
    sort_by_busy: bool = True,
    panel_height_ratio: float | None = None,
    sm_count: int = DEFAULT_SM_COUNT,
    save_dpi: int = DEFAULT_SAVE_DPI,
) -> None:
    plt.rcParams.update(PAPER_RC)

    matrix, duration_ms, slots_per_sm, n_shown, _n_total = _prepare_heatmap(
        profile_dir=profile_dir,
        app_name=app_name,
        profile_tag=profile_tag,
        variant=variant,
        n_bins=n_bins,
        sort_by_busy=sort_by_busy,
        max_sms=panel.max_sms,
        sm_count=sm_count,
    )
    if panel_height_ratio is None:
        panel_height_ratio = (
            max(0.38, sm_count * 0.055 / 14.0) * PANEL_HEIGHT_SCALE
        )

    fig_height = FIG_WIDTH_IN * panel_height_ratio
    print(
        f"Figure size (fixed): {FIG_WIDTH_IN:.3f} x {fig_height:.3f} in @ {save_dpi} dpi"
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

    print("Building KCGPU SM working heatmap...")
    cmap, norm = make_sm_slots_colormap(slots_per_sm)
    ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=[0.0, duration_ms / 1000.0, n_shown, 0.0],
        rasterized=True,
    )
    ax.set_xlim(0.0, duration_ms / 1000.0)
    ax.set_ylim(n_shown, 0.0)
    ax.set_xlabel("Time (s)")
    ax.grid(False)

    label_fp = ax.xaxis.label.get_fontproperties()
    if sort_by_busy:
        fig.supylabel("SMs (sorted by total busy time)", fontproperties=label_fp)
    else:
        fig.supylabel("SM id", fontproperties=label_fp)

    cbar = fig.colorbar(
        sm_slots_scalar_mappable(slots_per_sm),
        cax=cax,
        extend="min",
    )
    cbar.set_label(COLORBAR_LABEL, fontproperties=label_fp)
    configure_sm_slots_colorbar(cbar, slots_per_sm)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {output_path} ({FIG_WIDTH_IN:.3f} x {fig_height:.3f} in)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KCGPU SM working timeline heatmap for the paper.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=IMG_DIR / f"kcgpu_pivot_sm_timeline.{OUTPUT_FORMAT}",
        help="Output PDF path",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=K_CLIQUE_DIR / "profile" / "kcgpu",
        help="Directory with kcgpu_*_warp_timeline_working*.csv",
    )
    parser.add_argument(
        "--variant",
        choices=["orientation", "pivot"],
        default="pivot",
        help="KCGPU variant (selects default app name and profile tag pattern)",
    )
    parser.add_argument(
        "--app-name",
        default=None,
        help="Profile CSV prefix (default: kcgpu_<variant>)",
    )
    parser.add_argument(
        "--graph",
        default=None,
        help="Graph name (e.g. Skitter, DBLP, Orkut); builds --profile-tag with --k",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Clique size k; used with --graph to build --profile-tag",
    )
    parser.add_argument(
        "--profile-tag", "--profile_tag",
        dest="profile_tag",
        default=None,
        help=f"Suffix before .csv (default: {DEFAULT_PROFILE_TAG}, or from --graph/--k)",
    )
    parser.add_argument(
        "--max-sms",
        type=int,
        default=0,
        help="Show only the busiest N SMs (0 = all)",
    )
    parser.add_argument(
        "--time-bins",
        type=int,
        default=SM_TIMELINE_BINS,
        help=f"Horizontal time bins (default: {SM_TIMELINE_BINS})",
    )
    parser.add_argument(
        "--no-sort-by-busy",
        action="store_true",
        help="Keep SM id order instead of sorting by total busy time",
    )
    parser.add_argument(
        "--panel-height-ratio",
        type=float,
        default=None,
        help="Panel height as a multiple of figure width (default: auto from SM count)",
    )
    parser.add_argument(
        "--sm-count",
        type=int,
        default=DEFAULT_SM_COUNT,
        help=f"Number of SMs on the device (default: {DEFAULT_SM_COUNT})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_SAVE_DPI,
        help=f"savefig dpi for rasterized heatmap (default: {DEFAULT_SAVE_DPI})",
    )
    args = parser.parse_args()

    if (args.graph is None) ^ (args.k is None):
        parser.error("--graph and --k must be given together")

    app_name = args.app_name or f"kcgpu_{args.variant}"
    if args.graph is not None and args.k is not None:
        profile_tag = _build_profile_tag(graph=args.graph, k=args.k, variant=args.variant)
    else:
        profile_tag = args.profile_tag or DEFAULT_PROFILE_TAG

    plot_kcgpu_sm_timeline_paper(
        output_path=args.output.resolve(),
        panel=PanelSpec(max_sms=args.max_sms),
        profile_dir=args.profile_dir.resolve(),
        app_name=app_name,
        profile_tag=profile_tag,
        variant=args.variant,
        n_bins=args.time_bins,
        sort_by_busy=not args.no_sort_by_busy,
        panel_height_ratio=args.panel_height_ratio,
        sm_count=args.sm_count,
        save_dpi=args.dpi,
    )


if __name__ == "__main__":
    main()

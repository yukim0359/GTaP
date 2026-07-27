#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

THESIS_STYLE = ["~/plot_style/thesis_plt.mplstyle"]

# Stack bar chart (fmm_3way_stack_*.png): avoid thesis_plt global sizes (title 20pt, bbox tight, …).
STACK_PLOT_RC = {
    "font.size": 10,
    "font.weight": "normal",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "lines.markeredgewidth": 1.0,
    "savefig.bbox": "standard",
    "savefig.dpi": 220,
}

# Bottom-to-top stack order (matches gtap_fmm.cu pipeline timeline with FMM3D_GPU_UPWARD=1).
PHASES = [
    "Tree",
    "Metadata",
    "Transfer/bind",
    "Upward",
    "DTT init",
    "DTT",
    "DTT H2D",
    "Body pack",
    "Tree pack",
    "M2L",
    "L2L",
    "Eval",
    "Phi D2H",
    "Other",
]

DTT_PHASES = frozenset({"DTT init", "DTT", "DTT H2D"})

# Paper figure: collapse fine-grained PHASES into seven stack segments.
PAPER_PHASES = [
    "Tree/setup",
    "DTT init",
    "DTT",
    "DTT H2D",
    "M2L",
    "L2P/P2P",
    "Other",
]

PAPER_GROUP_SOURCES = {
    "Tree/setup": ("Tree", "Metadata", "Transfer/bind", "Upward"),
    "DTT init": ("DTT init",),
    "DTT": ("DTT",),
    "DTT H2D": ("DTT H2D",),
    "M2L": ("M2L",),
    "L2P/P2P": ("Eval",),
    "Other": ("L2L", "Phi D2H", "Body pack", "Tree pack", "Other"),
}

PAPER_COMMON_PHASES = frozenset({"Tree/setup", "M2L", "L2P/P2P", "Other"})
PAPER_DTT_PHASES = frozenset({"DTT init", "DTT", "DTT H2D"})

SHORT_LABELS = {
    "GTaP DTT": "GTaP",
    "GPU worklist DTT": "Worklist",
    "Host OpenMP DTT": "Host OMP",
    "Host DTT": "Host OMP",
}


def stack_phase_order(phases: Dict[str, float], phase_filter: List[str]) -> List[str]:
    """Canonical pipeline order for stacked bars and legend."""
    allowed = set(phase_filter)
    return [
        phase
        for phase in PHASES
        if phase in allowed and phases.get(phase, 0.0) != 0.0
    ]


def _blend_toward_white(hex_color: str, white_frac: float) -> str:
    r, g, b = to_rgb(hex_color)
    mix = max(0.0, min(1.0, white_frac))
    return "#{:02x}{:02x}{:02x}".format(
        int((r + (1.0 - r) * mix) * 255),
        int((g + (1.0 - g) * mix) * 255),
        int((b + (1.0 - b) * mix) * 255),
    )


PAPER_COLORS = {
    "Tree/setup": _blend_toward_white("#4e79a7", 0.72),
    "DTT init": "#D97706",
    "DTT": "#1D4ED8",
    "DTT H2D": "#EA580C",
    "M2L": _blend_toward_white("#e15759", 0.72),
    "L2P/P2P": _blend_toward_white("#edc948", 0.70),
    "Other": _blend_toward_white("#b0b0b0", 0.62),
}

PAPER_EDGE_COLORS = {
    "Tree/setup": "#d8dee3",
    "DTT init": "#b45309",
    "DTT": "#1e3a8a",
    "DTT H2D": "#9a3412",
    "M2L": "#ead0d0",
    "L2P/P2P": "#efe5b7",
    "Other": "#dddddd",
}


# How much to blend common-phase hues toward white (lower = more saturated).
_COMMON_MUTE_WHITE = 0.55

# Shared base hues; common phases are muted so DTT segments read as the focus.
_COMMON_BASE = {
    "Tree": "#4e79a7",
    "Upward": "#59a14f",
    "Metadata": "#af7aa1",
    "Transfer/bind": "#bab0ab",
    "Body pack": "#ff9da7",
    "Tree pack": "#b07aa1",
    "M2L": "#e15759",
    "L2L": "#76b7b2",
    "Eval": "#edc948",
    "Phi D2H": "#86bcb6",
    "Other": "#b0b0b0",
}

_DTT_HERO = {
    "DTT init": "#D97706",  # amber: one-shot setup
    "DTT": "#1D4ED8",       # strong blue: core list construction
    "DTT H2D": "#EA580C",   # orange: host handoff
}

COLORS = {
    **_DTT_HERO,
    **{
        phase: _blend_toward_white(color, _COMMON_MUTE_WHITE)
        for phase, color in _COMMON_BASE.items()
    },
}

# Binaries built by gtap/evaluation/benchmarks/fmm/Makefile
DEFAULT_BIN_GTAP = "./bin/gtap_dtt_fmm"
DEFAULT_BIN_WORKLIST = "./bin/worklist_dtt_fmm"
DEFAULT_BIN_OMP = "./bin/omp_dtt_fmm"


@dataclass
class RunBreakdown:
    label: str
    phases: Dict[str, float]
    phase_order: List[str]
    total: float
    dtt_mode: str
    num_runs: int = 1


@dataclass
class PlotData:
    runs: List[RunBreakdown]
    raw_by_label: Dict[str, List[RunBreakdown]] = field(default_factory=dict)
    n: int = 0
    theta: float = 0.0
    source_csv: str = ""


def csv_summary_path(csv_out: str) -> str:
    if csv_out.endswith("_summary.csv"):
        return csv_out
    if csv_out.endswith(".csv"):
        return csv_out[:-4] + "_summary.csv"
    return csv_out + "_summary.csv"


def csv_runs_path(csv_out: str) -> str:
    summary = csv_summary_path(csv_out)
    if summary.endswith("_summary.csv"):
        return summary[: -len("_summary.csv")] + "_runs.csv"
    return summary[:-4] + "_runs.csv"


def write_csv(data: PlotData, csv_out: str) -> None:
    summary_path = csv_summary_path(csv_out)
    runs_path = csv_runs_path(csv_out)
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(runs_path) or ".", exist_ok=True)

    summary_cols = [
        "N",
        "theta",
        "label",
        "dtt_mode",
        "num_runs",
        "execution_time_ms",
        *PHASES,
        "phase_order",
    ]
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_cols)
        writer.writeheader()
        for run in data.runs:
            row = {
                "N": data.n,
                "theta": data.theta,
                "label": run.label,
                "dtt_mode": run.dtt_mode,
                "num_runs": run.num_runs,
                "execution_time_ms": f"{run.total:.6f}",
                "phase_order": "|".join(run.phase_order),
            }
            for phase in PHASES:
                row[phase] = f"{run.phases.get(phase, 0.0):.6f}"
            writer.writerow(row)

    runs_cols = [
        "N",
        "theta",
        "label",
        "run_idx",
        "dtt_mode",
        "execution_time_ms",
        *PHASES,
        "phase_order",
    ]
    with open(runs_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=runs_cols)
        writer.writeheader()
        for label, raw_runs in data.raw_by_label.items():
            for idx, run in enumerate(raw_runs, start=1):
                row = {
                    "N": data.n,
                    "theta": data.theta,
                    "label": label,
                    "run_idx": idx,
                    "dtt_mode": run.dtt_mode,
                    "execution_time_ms": f"{run.total:.6f}",
                    "phase_order": "|".join(run.phase_order),
                }
                for phase in PHASES:
                    row[phase] = f"{run.phases.get(phase, 0.0):.6f}"
                writer.writerow(row)

    print(f"Saved: {summary_path}")
    if data.raw_by_label:
        print(f"Saved: {runs_path}")


def phase_value_from_row(row: Dict[str, str], phase: str) -> float:
    raw = row.get(phase)
    if raw is None and phase == "DTT init":
        raw = row.get("Runtime init")
    return float(raw or 0.0)


def read_summary_csv(path: str) -> PlotData:
    runs: List[RunBreakdown] = []
    n = 0
    theta = 0.0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("label"):
                continue
            n = int(float(row["N"]))
            theta = float(row["theta"])
            phases = {phase: phase_value_from_row(row, phase) for phase in PHASES}
            phase_order = stack_phase_order(phases, PHASES)
            runs.append(
                RunBreakdown(
                    label=row["label"],
                    phases=phases,
                    phase_order=phase_order,
                    total=float(row["execution_time_ms"]),
                    dtt_mode=row.get("dtt_mode") or "unknown",
                    num_runs=int(float(row.get("num_runs") or 1)),
                )
            )
    return PlotData(runs=runs, n=n, theta=theta, source_csv=path)


def run_cmd(cmd: List[str], env: Dict[str, str]) -> str:
    p = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed: "
            + " ".join(cmd)
            + f"\nexit code: {p.returncode}\n"
            + p.stdout
            + p.stderr
        )
    return p.stdout + p.stderr


def find_ms(text: str, pattern: str, default: float = 0.0) -> float:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default


def format_theta_label(theta: float) -> str:
    """Label/filename theta with the decimal point removed (0.3 -> 03, 0.35 -> 035)."""
    text = format(theta, "f").rstrip("0")
    if text.endswith("."):
        text = text[:-1]
    return text.replace(".", "")


def timeline_phase(line: str) -> str:
    if "DTT init" in line or "fixed-cap buffers + runtime setup" in line:
        return "DTT init"
    if "GTaP init/warm runtime allocation" in line:
        return "DTT init"
    if "host DTT list buffer alloc" in line or "worklist pair buffer alloc" in line:
        return "DTT init"
    if "build adaptive tree" in line:
        return "Tree"
    if "upward pass (P2M/M2M)" in line:
        return "Upward"
    if "build traversal metadata lists" in line:
        return "Metadata"
    if (
        "DTT lists" in line
        or "DTT traversal + fused" in line
        or "DTT traversal (fused" in line
        or "DTT traversal core" in line
        or "DTT worklist setup" in line
        or "DTT list construction" in line
        or "DTT reset/counts" in line
        or "DTT list stats" in line
        or "DTT post/check" in line
        or "DTT setup" in line
    ):
        return "DTT"
    if "DTT list counts" in line or "DTT list payloads" in line:
        return "DTT H2D"
    if "pack body arrays" in line or "pack body/tree arrays" in line:
        return "Body pack"
    if "pack tree nodes" in line:
        return "Tree pack"
    if (
        "upload static arrays and bind symbols" in line
        or "copy tree for L2L" in line
        or "copy tree after L2L" in line
    ):
        return "Transfer/bind"
    if "M2L kernel" in line or "M2L over target cells" in line:
        return "M2L"
    if "L2L downward pass" in line:
        return "L2L"
    if "L2P + P2P eval" in line:
        return "Eval"
    if "copy phi/acc result arrays" in line or "copy phi result" in line:
        return "Phi D2H"
    return ""


def parse_timeline_phases(text: str) -> Dict[str, float]:
    """Sum pipeline timeline lines into phase buckets."""
    phases = {k: 0.0 for k in PHASES}
    in_timeline = False
    for line in text.splitlines():
        if line.startswith("=== Pipeline timeline"):
            in_timeline = True
            continue
        if in_timeline and line.startswith("==="):
            break
        if not in_timeline:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("setup="):
            continue
        match = re.search(r":\s*([0-9.]+)\s*$", stripped)
        if not match:
            continue
        phase = timeline_phase(line)
        if phase:
            phases[phase] += float(match.group(1))
    return phases


def align_phases_to_total(phases: Dict[str, float], total: float) -> Dict[str, float]:
    """Make stacked phases sum to execution time (add Other or scale down)."""
    aligned = dict(phases)
    aligned["Other"] = 0.0
    stacked = sum(aligned.get(phase, 0.0) for phase in PHASES if phase != "Other")
    gap = total - stacked
    if abs(gap) < 0.05:
        return aligned
    if gap > 0.0:
        aligned["Other"] = gap
        return aligned
    if stacked <= 0.0:
        return aligned
    scale = total / stacked
    for phase in PHASES:
        if phase == "Other":
            continue
        value = aligned.get(phase, 0.0)
        if value != 0.0:
            aligned[phase] = value * scale
    aligned["Other"] = 0.0
    return aligned


def parse_phase_order(text: str, phases: Dict[str, float]) -> List[str]:
    return stack_phase_order(phases, PHASES)


def parse_output(text: str, label: str) -> RunBreakdown:
    phases = parse_timeline_phases(text)

    total = find_ms(text, r"Execution time:\s*([0-9.]+)\s*ms")
    if total == 0.0:
        total = sum(v for k, v in phases.items() if k != "Other")
    phases = align_phases_to_total(phases, total)

    mode_match = re.search(r"DTT mode:\s*([^\n]+)", text)
    if mode_match:
        dtt_mode = mode_match.group(1).strip()
    elif "GPU worklist fixed-cap" in text:
        dtt_mode = "GPU worklist fixed-cap lists"
    elif "GTaP task traversal" in text or "GTaP fused" in text:
        dtt_mode = "GTaP DTT"
    elif "CPU/Cilk host" in text:
        dtt_mode = "CPU/Cilk host DTT"
    elif "CPU/OpenMP host" in text:
        dtt_mode = "CPU/OpenMP host DTT"
    elif "fixed-cap single-pass" in text:
        dtt_mode = "GPU/GTaP fixed-cap single-pass"
    elif "exafmm-beta fused" in text:
        dtt_mode = "exafmm-beta fused DTT"
    elif "CSR two-pass" in text:
        dtt_mode = "GPU/GTaP CSR two-pass"
    else:
        dtt_mode = "unknown"

    return RunBreakdown(
        label=label,
        phases=phases,
        phase_order=parse_phase_order(text, phases),
        total=total,
        dtt_mode=dtt_mode,
    )


def mean_value(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def median_value(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def collapse_to_paper_phases(phases: Dict[str, float]) -> Dict[str, float]:
    collapsed = {phase: 0.0 for phase in PAPER_PHASES}
    for paper_phase, sources in PAPER_GROUP_SOURCES.items():
        collapsed[paper_phase] = sum(phases.get(source, 0.0) for source in sources)
    return collapsed


def paper_phase_order(phases: Dict[str, float]) -> List[str]:
    return [phase for phase in PAPER_PHASES if phases.get(phase, 0.0) != 0.0]


def paper_stack_phase_order(
    phases: Dict[str, float],
    phase_filter: Optional[List[str]] = None,
) -> List[str]:
    allowed = set(phase_filter) if phase_filter is not None else set(PAPER_PHASES)
    return [
        phase
        for phase in PAPER_PHASES
        if phase in allowed and phases.get(phase, 0.0) != 0.0
    ]


def collapse_run_to_paper(run: RunBreakdown) -> RunBreakdown:
    phases = collapse_to_paper_phases(run.phases)
    return RunBreakdown(
        label=run.label,
        phases=phases,
        phase_order=paper_phase_order(phases),
        total=run.total,
        dtt_mode=run.dtt_mode,
        num_runs=run.num_runs,
    )


def paper_display_total(run: RunBreakdown, include_init: bool) -> float:
    if include_init:
        return run.total
    return run.total - run.phases.get("DTT init", 0.0)


def paper_dtt_band_bounds(run: RunBreakdown, phase_filter: List[str]) -> tuple[float, float]:
    """Bottom/top (ms) of the stacked DTT-related band for one paper-phase bar."""
    bottom_y = 0.0
    dtt_bottom = None
    dtt_top = 0.0
    for phase in paper_stack_phase_order(run.phases, phase_filter):
        value = run.phases.get(phase, 0.0)
        if phase in PAPER_DTT_PHASES:
            if dtt_bottom is None:
                dtt_bottom = bottom_y
            bottom_y += value
            dtt_top = bottom_y
        else:
            bottom_y += value
    if dtt_bottom is None:
        return 0.0, 0.0
    return dtt_bottom, dtt_top


def paper_dtt_section_ms(run: RunBreakdown, phase_filter: List[str]) -> float:
    allowed = set(phase_filter)
    return sum(run.phases.get(phase, 0.0) for phase in PAPER_DTT_PHASES if phase in allowed)


def short_label(label: str) -> str:
    return SHORT_LABELS.get(label, label)


def format_paper_n(n: int) -> str:
    if n >= 1_000_000 and n % 1_000_000 == 0:
        exp = 0
        value = n
        while value % 10 == 0 and value >= 10:
            value //= 10
            exp += 1
        return rf"$N={value}\times10^{{{exp}}}$"
    return rf"$N={n:,}$"


def format_paper_theta(theta: float) -> str:
    return rf"$\theta={format(theta, 'g')}$"


def format_time_s(ms: float) -> str:
    return f"{ms / 1000.0:.3f} s"


def format_axis_s(ms: float, _pos=None) -> str:
    seconds = ms / 1000.0
    if seconds >= 10.0:
        return f"{seconds:.0f}"
    if seconds >= 1.0:
        return f"{seconds:.1f}"
    return f"{seconds:.2f}"


def paper_run_total(run: RunBreakdown) -> float:
    return max(run.total, sum(run.phases.get(phase, 0.0) for phase in PAPER_PHASES))


def read_runs_csv_grouped(path: str) -> Dict[str, List[RunBreakdown]]:
    grouped: Dict[str, List[RunBreakdown]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("label"):
                continue
            label = row["label"]
            phases = {phase: phase_value_from_row(row, phase) for phase in PHASES}
            grouped.setdefault(label, []).append(
                RunBreakdown(
                    label=label,
                    phases=phases,
                    phase_order=stack_phase_order(phases, PHASES),
                    total=float(row["execution_time_ms"]),
                    dtt_mode=row.get("dtt_mode") or "unknown",
                    num_runs=1,
                )
            )
    return grouped


def apply_controlled_paper_breakdown(data: PlotData, runs_csv: str) -> List[RunBreakdown]:
    """Use pooled medians for common phases; per-method medians for DTT phases."""
    grouped = read_runs_csv_grouped(runs_csv)
    if not grouped:
        raise ValueError(f"no runs found in {runs_csv!r}")

    pooled_common: Dict[str, List[float]] = {phase: [] for phase in PAPER_COMMON_PHASES}
    per_label_dtt: Dict[str, Dict[str, List[float]]] = {}

    for label, runs in grouped.items():
        per_label_dtt.setdefault(label, {phase: [] for phase in PAPER_DTT_PHASES})
        for run in runs:
            paper = collapse_to_paper_phases(run.phases)
            for phase in PAPER_COMMON_PHASES:
                pooled_common[phase].append(paper[phase])
            for phase in PAPER_DTT_PHASES:
                per_label_dtt[label][phase].append(paper[phase])

    common_medians = {phase: median_value(samples) for phase, samples in pooled_common.items()}
    controlled: List[RunBreakdown] = []
    for run in data.runs:
        label = run.label
        phases = dict(common_medians)
        for phase in PAPER_DTT_PHASES:
            phases[phase] = median_value(per_label_dtt.get(label, {}).get(phase, []))
        total = sum(phases.values())
        controlled.append(
            RunBreakdown(
                label=label,
                phases=phases,
                phase_order=paper_phase_order(phases),
                total=total,
                dtt_mode=run.dtt_mode,
                num_runs=run.num_runs,
            )
        )
    return controlled


def runs_for_paper(data: PlotData, paper: bool, controlled: bool, runs_csv: str) -> List[RunBreakdown]:
    if not paper:
        return data.runs
    if controlled:
        if not runs_csv:
            raise SystemExit("--controlled requires --runs-csv (per-run rows for median pooling)")
        return apply_controlled_paper_breakdown(data, runs_csv)
    collapsed: List[RunBreakdown] = []
    for run in data.runs:
        paper_phases = collapse_to_paper_phases(run.phases)
        collapsed.append(
            RunBreakdown(
                label=run.label,
                phases=paper_phases,
                phase_order=paper_phase_order(paper_phases),
                total=sum(paper_phases.values()),
                dtt_mode=run.dtt_mode,
                num_runs=run.num_runs,
            )
        )
    return collapsed


def aggregate_breakdowns(label: str, runs: List[RunBreakdown]) -> RunBreakdown:
    if not runs:
        raise ValueError(f"no runs to aggregate for {label!r}")
    if len(runs) == 1:
        run = runs[0]
        run.num_runs = 1
        return run

    totals = [r.total for r in runs]
    total = mean_value(totals)

    phases: Dict[str, float] = {}
    for phase in PHASES:
        samples = [r.phases.get(phase, 0.0) for r in runs]
        phases[phase] = mean_value(samples)

    phase_order = stack_phase_order(phases, PHASES)

    return RunBreakdown(
        label=label,
        phases=phases,
        phase_order=phase_order,
        total=total,
        dtt_mode=runs[0].dtt_mode,
        num_runs=len(runs),
    )


def display_total(run: RunBreakdown, include_init: bool) -> float:
    if include_init:
        return run.total
    return run.total - run.phases.get("DTT init", 0.0)


def dtt_band_bounds(run: RunBreakdown, phase_filter: List[str]) -> tuple[float, float]:
    """Bottom/top y (ms) of the stacked DTT-related band for one bar."""
    bottom_y = 0.0
    dtt_bottom = None
    dtt_top = 0.0
    for phase in stack_phase_order(run.phases, phase_filter):
        value = run.phases.get(phase, 0.0)
        if phase in DTT_PHASES:
            if dtt_bottom is None:
                dtt_bottom = bottom_y
            bottom_y += value
            dtt_top = bottom_y
        else:
            bottom_y += value
    if dtt_bottom is None:
        return 0.0, 0.0
    return dtt_bottom, dtt_top


def dtt_section_ms(run: RunBreakdown, phase_filter: List[str]) -> float:
    allowed = set(phase_filter)
    return sum(run.phases.get(phase, 0.0) for phase in DTT_PHASES if phase in allowed)


def gtap_baseline_index(runs: List[RunBreakdown]) -> int:
    for i, run in enumerate(runs):
        if "GTaP" in run.label or "GTAP" in run.label.upper():
            return i
    return 0


def plot(
    runs: List[RunBreakdown],
    out: str,
    title: str,
    include_init: bool,
    show_diff_lines: bool,
) -> None:
    with plt.style.context(STACK_PLOT_RC):
        _plot_stack_bars(runs, out, title, include_init, show_diff_lines)


def _plot_stack_bars(
    runs: List[RunBreakdown],
    out: str,
    title: str,
    include_init: bool,
    show_diff_lines: bool,
) -> None:
    phase_filter = PHASES if include_init else [p for p in PHASES if p != "DTT init"]
    x = list(range(len(runs)))
    bottoms = [0.0] * len(runs)
    used_phases = []

    n_runs = len(runs)
    fig_w = min(10.0, max(6.0, 1.85 * n_runs + 2.0))
    fig_h = 5.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for i, r in enumerate(runs):
        run_order = stack_phase_order(r.phases, phase_filter)
        for phase in run_order:
            value = r.phases.get(phase, 0.0)
            if value == 0.0:
                continue
            is_dtt = phase in DTT_PHASES
            ax.bar(
                x[i],
                value,
                bottom=bottoms[i],
                width=0.62,
                color=COLORS[phase],
                edgecolor=("#1e3a8a" if phase == "DTT" else "#9a3412") if is_dtt else "white",
                linewidth=0.55 if is_dtt else 0.25,
                zorder=5 if is_dtt else 3,
            )
            bottoms[i] += value
            if phase not in used_phases:
                used_phases.append(phase)

    bar_half = 0.31
    dtt_bounds = [dtt_band_bounds(r, phase_filter) for r in runs]
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
                    linewidth=0.8,
                    alpha=0.65,
                    zorder=4,
                )

    gtap_idx = gtap_baseline_index(runs)
    gtap_dtt_ms = dtt_section_ms(runs[gtap_idx], phase_filter)
    for i, run in enumerate(runs):
        if i == gtap_idx or gtap_dtt_ms <= 0.0:
            continue
        ratio = dtt_section_ms(run, phase_filter) / gtap_dtt_ms
        bot, top = dtt_bounds[i]
        if top <= bot:
            continue
        ax.text(
            x[i] + bar_half + 0.04,
            (bot + top) * 0.5,
            f"×{ratio:.2f}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#c00000",
            zorder=10,
        )

    ymax = max(max(bottoms), max(display_total(r, include_init) for r in runs))
    ms_label_gap = ymax * 0.025
    for i, r in enumerate(runs):
        shown_total = display_total(r, include_init)
        ax.text(
            i,
            bottoms[i] + ms_label_gap,
            f"{shown_total:.1f} ms",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    labels = [r.label for r in runs]
    ax.set_xticks(x, labels)
    # Pad inside the axes frame so totals / ratio labels clear the top/right spines.
    ax.set_xlim(-0.55, (n_runs - 1) + bar_half + 0.48)
    ax.set_ylim(0.0, ymax * 1.10)
    ax.set_ylabel("Time (ms)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    used = set(used_phases)
    legend_phases = stack_phase_order(
        {phase: (1.0 if phase in used else 0.0) for phase in PHASES},
        phase_filter,
    )
    legend_handles = [Patch(facecolor=COLORS[p], label=p) for p in legend_phases]
    legend = ax.legend(
        handles=legend_handles,
        ncol=min(4, max(1, len(legend_phases))),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(
        out,
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.06,
        bbox_extra_artists=[legend],
    )
    print(f"Saved: {out}")


def plot_paper_horizontal(
    ax,
    runs: List[RunBreakdown],
    *,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    xlim_ms: float = 0.0,
) -> List[str]:
    y = list(range(len(runs)))
    left = [0.0] * len(runs)
    used_phases: List[str] = []
    xmax = xlim_ms or (max((paper_run_total(run) for run in runs), default=0.0))
    axis_max = xmax * 1.12 if xmax > 0.0 else 1.0

    for phase in PAPER_PHASES:
        widths = [run.phases.get(phase, 0.0) for run in runs]
        if not any(widths):
            continue
        ax.barh(
            y,
            widths,
            left=left,
            height=0.52,
            color=PAPER_COLORS[phase],
            edgecolor=PAPER_EDGE_COLORS[phase],
            linewidth=0.45,
            zorder=4,
            label=phase,
        )
        left = [left[i] + widths[i] for i in range(len(runs))]
        used_phases.append(phase)

    for i, run in enumerate(runs):
        ax.text(
            left[i] + axis_max * 0.012,
            y[i],
            format_time_s(run.total),
            va="center",
            ha="left",
            fontsize=8.0,
            color="#333333",
            clip_on=False,
        )

    ax.set_yticks(y)
    if show_ylabel:
        ax.set_yticklabels([short_label(run.label) for run in runs])
    else:
        ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(0.0, axis_max)
    ax.set_xlabel("Runtime (s)" if show_xlabel else "")
    ax.xaxis.set_major_formatter(FuncFormatter(format_axis_s))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
    ax.grid(axis="x", linestyle="-", alpha=0.18, linewidth=0.7)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", length=3, width=0.7, labelbottom=show_xlabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    return used_phases


def plot_paper_panels(
    panels: List[tuple[PlotData, str]],
    out: str,
    *,
    controlled: bool,
    runs_csv_by_summary: Dict[str, str],
    caption_out: str = "",
) -> None:
    with plt.style.context(THESIS_STYLE):
        _plot_paper_panels_impl(
            panels, out, controlled=controlled,
            runs_csv_by_summary=runs_csv_by_summary, caption_out=caption_out,
        )


def _plot_paper_panels_impl(
    panels: List[tuple[PlotData, str]],
    out: str,
    *,
    controlled: bool,
    runs_csv_by_summary: Dict[str, str],
    caption_out: str = "",
) -> None:
    n_panels = len(panels)
    panel_runs: List[tuple[PlotData, str, List[RunBreakdown]]] = []
    for data, panel_tag in panels:
        runs_csv = runs_csv_by_summary.get(data.source_csv or "", "")
        runs = runs_for_paper(data, paper=True, controlled=controlled, runs_csv=runs_csv)
        panel_runs.append((data, panel_tag, runs))
    shared_xmax = max(
        (paper_run_total(run) for _, _, runs in panel_runs for run in runs),
        default=0.0,
    )

    fig_h = max(2.3, 0.64 * len(panel_runs[0][2]) + 0.9)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(7.2, fig_h * n_panels),
        sharex=True,
        squeeze=False,
    )
    used_phases: List[str] = []

    for idx, (data, panel_tag, runs) in enumerate(panel_runs):
        ax = axes[idx, 0]
        panel_used = plot_paper_horizontal(
            ax,
            runs,
            show_ylabel=True,
            show_xlabel=(idx == n_panels - 1),
            xlim_ms=shared_xmax,
        )
        for phase in panel_used:
            if phase not in used_phases:
                used_phases.append(phase)
        subtitle = format_paper_theta(data.theta)
        panel_title = f"{panel_tag}   {subtitle}" if panel_tag else subtitle
        ax.set_title(panel_title, loc="left", pad=5, fontsize=9)

    legend_handles = [Patch(facecolor=PAPER_COLORS[p], label=p) for p in used_phases]
    fig.legend(
        handles=legend_handles,
        ncol=min(len(used_phases), 6),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        frameon=False,
        handlelength=1.25,
        columnspacing=1.1,
        handletextpad=0.35,
    )
    n_for_title = panels[0][0].n
    fig.suptitle(
        f"End-to-end FMM3D runtime breakdown ({format_paper_n(n_for_title)})",
        y=0.99,
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.16, right=0.92, top=0.90, bottom=0.14, hspace=0.34)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Saved: {out}")

    if caption_out:
        breakdown_kind = "controlled" if controlled else "measured"
        lines = [
            "Suggested caption:",
            "End-to-end FMM3D runtime breakdown for "
            f"{format_paper_n(panels[0][0].n)}. "
            f"Bars use a {breakdown_kind} phase breakdown: "
            + (
                "common phases (tree/setup, M2L, eval, other) share pooled "
                "medians across methods, while DTT-related phases use "
                "method-specific medians."
                if controlled
                else "each bar reports independently measured end-to-end runs; "
                "common phases therefore include run-to-run variation."
            ),
        ]
        caption_text = "\n".join(lines)
        with open(caption_out, "w", encoding="utf-8") as f:
            f.write(caption_text + "\n")
        print(f"Saved: {caption_out}")
        print(caption_text)


def plot_paper_single(
    runs: List[RunBreakdown],
    out: str,
    *,
    title: str,
    controlled: bool,
) -> None:
    with plt.style.context(THESIS_STYLE):
        _plot_paper_single_impl(runs, out, title=title, controlled=controlled)


def _plot_paper_single_impl(
    runs: List[RunBreakdown],
    out: str,
    *,
    title: str,
    controlled: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, max(2.6, 0.78 * len(runs) + 1.05)))
    used = plot_paper_horizontal(ax, runs, show_ylabel=True, show_xlabel=True)
    ax.set_title(title, loc="left", pad=8, fontsize=10.5)
    legend_handles = [Patch(facecolor=PAPER_COLORS[p], label=p) for p in used]
    fig.legend(
        handles=legend_handles,
        ncol=min(len(used), 6),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
        handlelength=1.25,
        columnspacing=1.1,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.16, right=0.92, top=0.83, bottom=0.22)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Saved: {out}")


def pick_arg(primary: str, fallback: str, default: str) -> str:
    if primary:
        return primary
    if fallback:
        return fallback
    return default


def split_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def run_bin_many(
    bin_path: str, label: str, n: int, theta: float, num_runs: int, env: Dict[str, str]
) -> tuple[RunBreakdown, List[RunBreakdown]]:
    parsed: List[RunBreakdown] = []
    for run_idx in range(num_runs):
        text = run_cmd([bin_path, str(n), str(theta)], env)
        parsed.append(parse_output(text, label))
        if num_runs > 1:
            print(f"  {label}: run {run_idx + 1}/{num_runs} total={parsed[-1].total:.1f} ms")
    return aggregate_breakdowns(label, parsed), parsed


def read_or_run(args, env: Dict[str, str]) -> PlotData:
    bins = list(args.bin or [])
    labels = list(args.label or [])
    logs = list(args.log or [])
    bins.extend(split_csv(args.bins))
    labels.extend(split_csv(args.labels))
    logs.extend(split_csv(args.logs))

    if bins or logs:
        runs: List[RunBreakdown] = []
        raw_by_label: Dict[str, List[RunBreakdown]] = {}
        if args.no_run:
            if not logs:
                raise SystemExit("--no-run requires --log/--logs when using multi-run mode")
            for i, log in enumerate(logs):
                label = labels[i] if i < len(labels) else os.path.basename(log)
                with open(log, "r", encoding="utf-8") as f:
                    runs.append(parse_output(f.read(), label))
            return PlotData(runs=runs, n=args.n, theta=args.theta)

        if not bins:
            raise SystemExit("multi-run mode requires --bin/--bins unless --no-run is used")
        if args.runs > 1 and args.no_run:
            raise SystemExit("--runs > 1 requires executing binaries (omit --no-run)")
        for i, bin_path in enumerate(bins):
            label = labels[i] if i < len(labels) else os.path.basename(bin_path)
            if args.runs > 1:
                print(f"Running {label} ({bin_path}) x{args.runs}...")
            agg, raw = run_bin_many(bin_path, label, args.n, args.theta, args.runs, env)
            runs.append(agg)
            raw_by_label[label] = raw
        return PlotData(runs=runs, raw_by_label=raw_by_label, n=args.n, theta=args.theta)

    runs = []
    raw_by_label: Dict[str, List[RunBreakdown]] = {}
    left_bin = pick_arg(args.left_bin, args.gtap_bin, DEFAULT_BIN_GTAP)
    right_bin = pick_arg(args.right_bin, args.omp_bin, DEFAULT_BIN_OMP)
    left_label = pick_arg(args.left_label, args.gtap_label, "GTaP DTT")
    right_label = pick_arg(args.right_label, args.omp_label, "Host OMP DTT")
    left_log = pick_arg(args.left_log, args.omp_log, "")
    right_log = pick_arg(args.right_log, args.gtap_log, "")

    if args.no_run:
        if not left_log or not right_log:
            raise SystemExit("--no-run requires --left-log and --right-log")
        with open(left_log, "r", encoding="utf-8") as f:
            runs.append(parse_output(f.read(), left_label))
        with open(right_log, "r", encoding="utf-8") as f:
            runs.append(parse_output(f.read(), right_label))
        return PlotData(runs=runs, n=args.n, theta=args.theta)

    if args.runs > 1:
        print(f"Running {left_label} ({left_bin}) x{args.runs}...")
        left_agg, left_raw = run_bin_many(left_bin, left_label, args.n, args.theta, args.runs, env)
        runs.append(left_agg)
        raw_by_label[left_label] = left_raw
        print(f"Running {right_label} ({right_bin}) x{args.runs}...")
        right_agg, right_raw = run_bin_many(right_bin, right_label, args.n, args.theta, args.runs, env)
        runs.append(right_agg)
        raw_by_label[right_label] = right_raw
    else:
        left_text = run_cmd([left_bin, str(args.n), str(args.theta)], env)
        right_text = run_cmd([right_bin, str(args.n), str(args.theta)], env)
        left_run = parse_output(left_text, left_label)
        right_run = parse_output(right_text, right_label)
        runs.append(left_run)
        runs.append(right_run)
        raw_by_label[left_label] = [left_run]
        raw_by_label[right_label] = [right_run]
    return PlotData(runs=runs, raw_by_label=raw_by_label, n=args.n, theta=args.theta)


def panel_tag_for_theta(theta: float, heavy_theta: float = 0.2) -> str:
    if abs(theta - heavy_theta) < 1e-9:
        return "(a) DTT-heavy regime"
    return "(b) DTT-light regime"


def resolve_runs_csv(summary_csv: str, runs_csv: str) -> str:
    if runs_csv:
        return runs_csv
    return csv_runs_path(summary_csv)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot FMM phase stacked bars from pipeline output.")
    ap.add_argument("--n", type=int, default=20_000_000)
    ap.add_argument("--theta", type=float, default=0.3)
    ap.add_argument("--left-bin", default="")
    ap.add_argument("--right-bin", default="")
    ap.add_argument("--left-label", default="")
    ap.add_argument("--right-label", default="")
    ap.add_argument("--bin", action="append", default=[], help="Binary to run. Repeat for 3+ bars.")
    ap.add_argument("--label", action="append", default=[], help="Label for a --bin/--log entry. Repeat in the same order.")
    ap.add_argument("--log", action="append", default=[], help="Log file to parse. Repeat for 3+ bars with --no-run.")
    ap.add_argument("--bins", default="", help="Comma-separated binaries to run.")
    ap.add_argument("--labels", default="", help="Comma-separated labels.")
    ap.add_argument("--logs", default="", help="Comma-separated log files to parse with --no-run.")
    ap.add_argument("--out", default="img/fmm_gtap_dtt_vs_omp_dtt_stack.png")
    ap.add_argument("--csv-out", default="", help="Write summary/runs CSV for later replot.")
    ap.add_argument("--csv-in", default="", help="Replot from a summary CSV written by --csv-out.")
    ap.add_argument("--title", default="")
    ap.add_argument("--include-init", action="store_true",
                    help="Include DTT init (fixed-cap buffers + runtime setup) in the stacked bar.")
    ap.add_argument("--no-diff-lines", action="store_true", help="Do not draw dotted cumulative phase-boundary connectors.")
    ap.add_argument("--runs", type=int, default=1,
                    help="Repeat each binary this many times; bar phases and the "
                         "execution-time label use per-run averages.")
    ap.add_argument("--left-log", default="")
    ap.add_argument("--right-log", default="")
    # Backward-compatible aliases for older OMP-vs-GTaP usage.
    ap.add_argument("--omp-bin", default="", help=f"Alias for --right-bin (default: {DEFAULT_BIN_OMP})")
    ap.add_argument("--gtap-bin", default="", help=f"Alias for --left-bin (default: {DEFAULT_BIN_GTAP})")
    ap.add_argument("--omp-label", default="")
    ap.add_argument("--gtap-label", default="")
    ap.add_argument("--omp-log", default="")
    ap.add_argument("--gtap-log", default="")
    ap.add_argument("--no-run", action="store_true", help="Parse logs instead of running binaries.")
    ap.add_argument("--paper", action="store_true",
                    help="Paper figure: collapse phases to ~6 groups and use horizontal stacked bars.")
    ap.add_argument("--controlled", action="store_true",
                    help="Paper figure: common phases use pooled medians; DTT phases per method.")
    ap.add_argument("--runs-csv", default="",
                    help="Per-run CSV for --controlled (default: sibling *_runs.csv of --csv-in).")
    ap.add_argument("--panel-csv", action="append", default=[],
                    help="Summary CSV for a multi-panel paper figure. Repeat for each panel.")
    ap.add_argument("--panel-tag", action="append", default=[],
                    help="Optional panel tag, e.g. '(a) DTT-heavy regime'.")
    ap.add_argument("--caption-out", default="", help="Write suggested figure caption text.")
    args = ap.parse_args()

    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")

    env = dict(os.environ)
    cuda_root = env.get("CUDA_PATH") or env.get("CUDA_HOME") or ""
    prepend = []
    if cuda_root:
        prepend.append(os.path.join(cuda_root, "lib64"))
    # Optional site-specific lib (e.g. gcc-toolset); leave unset on other machines.
    toolset_lib = env.get("GCC_TOOLSET_LIB64", "")
    if toolset_lib:
        prepend.append(toolset_lib)
    if prepend:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(prepend + ([existing] if existing else []))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.panel_csv:
        if not args.no_run:
            raise SystemExit("--panel-csv requires --no-run")
        if not args.paper:
            raise SystemExit("--panel-csv requires --paper")
        panels: List[tuple[PlotData, str]] = []
        runs_csv_by_summary: Dict[str, str] = {}
        for idx, summary_path in enumerate(args.panel_csv):
            panel_data = read_summary_csv(summary_path)
            if args.panel_tag and idx < len(args.panel_tag):
                tag = args.panel_tag[idx]
            else:
                tag = panel_tag_for_theta(panel_data.theta)
            panels.append((panel_data, tag))
            runs_csv_by_summary[summary_path] = resolve_runs_csv(summary_path, args.runs_csv)
        plot_paper_panels(
            panels,
            args.out,
            controlled=args.controlled,
            runs_csv_by_summary=runs_csv_by_summary,
            caption_out=args.caption_out,
        )
        return

    if args.csv_in:
        if not args.no_run:
            raise SystemExit("--csv-in requires --no-run")
        data = read_summary_csv(args.csv_in)
    else:
        data = read_or_run(args, env)

    runs = data.runs
    theta_label = format_theta_label(data.theta if data.theta else args.theta)
    n_for_title = data.n if data.n else args.n
    title = args.title or f"FMM phase breakdown (N={n_for_title}, theta={theta_label})"
    if args.runs > 1 and not args.csv_in:
        title += f", average of {args.runs} runs"
    if args.csv_out and not args.csv_in:
        write_csv(data, args.csv_out)

    if args.paper:
        runs_csv = resolve_runs_csv(data.source_csv or args.csv_in, args.runs_csv)
        paper_runs = runs_for_paper(data, paper=True, controlled=args.controlled, runs_csv=runs_csv)
        if not args.title:
            title = f"End-to-end FMM3D runtime breakdown ({format_paper_n(n_for_title)}, {format_paper_theta(data.theta or args.theta)})"
        plot_paper_single(
            paper_runs,
            args.out,
            title=title,
            controlled=args.controlled,
        )
        if args.caption_out:
            breakdown_kind = "controlled" if args.controlled else "measured"
            caption = (
                f"Suggested caption:\n"
                f"End-to-end FMM3D runtime breakdown for {format_paper_n(n_for_title)} "
                f"at {format_paper_theta(data.theta or args.theta)}. "
                f"Bars use a {breakdown_kind} phase breakdown."
            )
            with open(args.caption_out, "w", encoding="utf-8") as f:
                f.write(caption + "\n")
            print(f"Saved: {args.caption_out}")
        return

    plot(runs, args.out, title, args.include_init, not args.no_diff_lines)


if __name__ == "__main__":
    main()

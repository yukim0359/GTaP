#!/usr/bin/env python3
"""Generate LaTeX benchmark table from k579 e2e best summary CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
K_CLIQUE_DIR = SCRIPT_DIR.parent
K579_DIR = K_CLIQUE_DIR / "k579"
DEFAULT_INPUT = K_CLIQUE_DIR / "k579" / "results" / "gtap_kcgpu_k579_e2e_init_included_best_summary.csv"
DEFAULT_OUTPUT = K579_DIR / "latex" / "k579_benchmark_tables.tex"

GRAPH_ORDER = ["DBLP", "as-Skitter", "Orkut"]
GRAPH_LABELS = {
    "DBLP": "com-dblp",
    "as-Skitter": "as-skitter",
    "Orkut": "com-orkut",
}
K_VALUES = [5, 7, 9]
VARIANTS = ["orientation", "pivot"]


def _float(value: str | float | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    return float(value)


def _load_summary(path: Path) -> dict[tuple[str, int, str], dict]:
    table: dict[tuple[str, int, str], dict] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["graph"], int(row["k"]), row["gtap_variant"])
            table[key] = row
    return table


def _fmt_seconds(ms: float | None) -> str:
    if ms is None:
        return ""
    seconds = ms / 1000.0
    if seconds >= 100.0:
        return f"{seconds:.3f}"
    if seconds >= 1.0:
        return f"{seconds:.3f}"
    return f"{seconds:.3f}"


def _fmt_cells(gtap_ms: float | None, kcgpu_ms: float | None) -> tuple[str, str]:
    gtap_s = _fmt_seconds(gtap_ms)
    kcgpu_s = _fmt_seconds(kcgpu_ms)
    if not gtap_s or not kcgpu_s:
        return gtap_s, kcgpu_s
    if gtap_ms is not None and kcgpu_ms is not None:
        if gtap_ms < kcgpu_ms:
            return f"\\textbf{{{gtap_s}}}", kcgpu_s
        if kcgpu_ms < gtap_ms:
            return gtap_s, f"\\textbf{{{kcgpu_s}}}"
    return gtap_s, kcgpu_s


def render_table(table: dict[tuple[str, int, str], dict]) -> str:
    lines: list[str] = []
    lines.append("% k-clique k=5,7,9 benchmark table (end-to-end time in seconds)")
    lines.append("% GTaP e2e: orient + workspace + count phase (includes gtap_initialize)")
    lines.append("% KCGPU e2e: preprocess + ctor + count (best q per graph×k; k579_min variants)")
    lines.append("% Source: orientation_e2e + pivot_e2e logs (job 2361719/2361720, 20 repeats avg)")
    lines.append("% Table style from Almasri et al., arXiv:2104.13209 (fig/4-evaluation/tab-results.tex)")
    lines.append("% Requires: \\usepackage{multirow}")
    lines.append("\\begin{table}[tb]")
    lines.append("  \\centering")
    lines.append("  \\caption{")
    lines.append(
        "    End-to-end execution time of GTaP and \\textsc{KCGPU} $k$-clique counting "
        "implementations (seconds; counts verified against \\textsc{KCGPU})."
    )
    lines.append("  }")
    lines.append("  \\label{tab:k579-e2e-time}")
    lines.append("  \\small")
    lines.append("  \\begin{tabular}{l|c|rr|rr|}")
    lines.append("    \\cline{3-6}")
    lines.append(
        "    \\multicolumn{1}{l}{} & \\multicolumn{1}{c|}{} & "
        "\\multicolumn{4}{c|}{Execution Time (s)} \\\\ \\hline"
    )
    lines.append(
        "    \\multicolumn{1}{|c|}{\\multirow{2}{*}{Graph}} & "
        "\\multicolumn{1}{c|}{\\multirow{2}{*}{$k$}} &"
    )
    lines.append(
        "    \\multicolumn{2}{c|}{Graph Orientation} &"
        "    \\multicolumn{2}{c|}{Pivoting} \\\\ \\cline{3-6}"
    )
    lines.append(
        "    \\multicolumn{1}{|l|}{} & \\multicolumn{1}{c|}{} &"
        "    \\multicolumn{1}{c|}{GTaP} & \\multicolumn{1}{c|}{\\textsc{KCGPU}} &"
        "    \\multicolumn{1}{c|}{GTaP} & \\multicolumn{1}{c|}{\\textsc{KCGPU}} \\\\ \\hline"
    )

    for graph in GRAPH_ORDER:
        label = GRAPH_LABELS[graph]
        for idx, k in enumerate(K_VALUES):
            orient = table.get((graph, k, "orientation"))
            pivot = table.get((graph, k, "pivot"))
            o_gtap, o_kcgpu = _fmt_cells(
                _float(orient.get("gtap_count_e2e_ms")) if orient else None,
                _float(orient.get("kcgpu_best_ms")) if orient else None,
            )
            p_gtap, p_kcgpu = _fmt_cells(
                _float(pivot.get("gtap_count_e2e_ms")) if pivot else None,
                _float(pivot.get("kcgpu_best_ms")) if pivot else None,
            )
            if idx == 0:
                graph_cell = f"\\multicolumn{{1}}{{|l|}}{{\\multirow{{3}}{{*}}{{{label}}}}}"
            else:
                graph_cell = "\\multicolumn{1}{|l|}{}"
            suffix = " \\\\ \\hline" if idx == len(K_VALUES) - 1 else " \\\\ \\cline{2-6}"
            lines.append(
                f"    {graph_cell} & \\multicolumn{{1}}{{c|}}{{{k}}} &"
                f"    \\multicolumn{{1}}{{r|}}{{{o_gtap}}} & \\multicolumn{{1}}{{r|}}{{{o_kcgpu}}} &"
                f"    \\multicolumn{{1}}{{r|}}{{{p_gtap}}} & \\multicolumn{{1}}{{r|}}{{{p_kcgpu}}}{suffix}"
            )

    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: file not found: {args.input}", file=sys.stderr)
        return 1

    tex = render_table(_load_summary(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tex)
    print(tex)
    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

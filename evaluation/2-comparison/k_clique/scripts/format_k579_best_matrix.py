#!/usr/bin/env python3
"""Pivot k579 best summary: rows=graph×k, columns=orientation vs pivot."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
K_CLIQUE_DIR = SCRIPT_DIR.parent
K579_RESULTS = K_CLIQUE_DIR / "k579" / "results"
DEFAULT_INPUT = K579_RESULTS / "gtap_kcgpu_k579_e2e_best_summary.csv"
DEFAULT_OUTPUT = K579_RESULTS / "gtap_kcgpu_k579_e2e_best_matrix.csv"

GRAPH_ORDER = ["DBLP", "as-Skitter", "Orkut"]
K_VALUES = [5, 7, 9]
VARIANTS = ["orientation", "pivot"]


def _load_summary(path: Path) -> dict[tuple[str, int, str], dict]:
    table: dict[tuple[str, int, str], dict] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["graph"], int(row["k"]), row["gtap_variant"])
            table[key] = row
    return table


def _fmt_num(value: str | float | None, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}f}"


def _cell(row: dict | None) -> dict[str, str]:
    if row is None:
        return {"gtap_ms": "", "kcgpu_ms": ""}
    return {
        "gtap_ms": _fmt_num(row.get("gtap_count_e2e_ms")),
        "kcgpu_ms": _fmt_num(row.get("kcgpu_best_ms")),
    }


def build_matrix_rows(table: dict[tuple[str, int, str], dict]) -> list[dict]:
    rows: list[dict] = []
    for graph in GRAPH_ORDER:
        for k in K_VALUES:
            out: dict[str, str] = {"graph": graph, "k": str(k)}
            for variant in VARIANTS:
                cell = _cell(table.get((graph, k, variant)))
                out[f"{variant}_gtap_ms"] = cell["gtap_ms"]
                out[f"{variant}_kcgpu_ms"] = cell["kcgpu_ms"]
            rows.append(out)
    return rows


def fieldnames() -> list[str]:
    cols = ["graph", "k"]
    for variant in VARIANTS:
        cols.append(f"{variant}_gtap_ms")
        cols.append(f"{variant}_kcgpu_ms")
    return cols


def write_csv(path: Path, rows: list[dict]) -> None:
    names = fieldnames()
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in names})


def print_table(rows: list[dict]) -> None:
    print(
        f"{'graph':<12} {'k':>2}  "
        f"{'orient GTAP':>12} {'orient KCGPU':>12}  "
        f"{'pivot GTAP':>12} {'pivot KCGPU':>12}"
    )
    print("-" * 68)
    last_graph = None
    for row in rows:
        graph = row["graph"]
        if last_graph is not None and graph != last_graph:
            print()
        last_graph = graph
        print(
            f"{graph:<12} {int(row['k']):>2}  "
            f"{row['orientation_gtap_ms']:>12} {row['orientation_kcgpu_ms']:>12}  "
            f"{row['pivot_gtap_ms']:>12} {row['pivot_kcgpu_ms']:>12}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pivot k579 best summary into graph×k rows and orientation/pivot columns."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input best summary CSV (default: {DEFAULT_INPUT.name})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output matrix CSV (default: {DEFAULT_OUTPUT.name})",
    )
    parser.add_argument("--no-csv", action="store_true", help="Print table only.")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: file not found: {args.input}", file=sys.stderr)
        return 1

    rows = build_matrix_rows(_load_summary(args.input))
    print_table(rows)
    if not args.no_csv:
        write_csv(args.output, rows)
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

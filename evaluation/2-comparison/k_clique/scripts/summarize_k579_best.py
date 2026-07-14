#!/usr/bin/env python3
"""Summarize GTAP vs best-KCGPU e2e times from k579 result CSVs."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
K_CLIQUE_DIR = SCRIPT_DIR.parent
K579_RESULTS = K_CLIQUE_DIR / "k579" / "results"
DEFAULT_ORIENTATION_CSV = K579_RESULTS / "gtap_kcgpu_orientation_k579_e2e_results_avg.csv"
DEFAULT_PIVOT_CSV = K579_RESULTS / "gtap_kcgpu_pivot_k579_e2e_results_avg.csv"
DEFAULT_OUTPUT_CSV = K579_RESULTS / "gtap_kcgpu_k579_e2e_best_summary.csv"

GRAPH_ORDER = {"DBLP": 0, "as-Skitter": 1, "Orkut": 2}


def _float(value: str | float | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    return float(value)


def _gtap_e2e_ms(row: dict) -> float | None:
    return _float(row.get("gtap_count_e2e_ms"))


def _load_best_rows(path: Path, require_match: bool) -> dict[tuple[str, int], dict]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if require_match and row.get("match") != "YES":
                continue
            graph = row["graph"]
            k = int(row["k"])
            groups[(graph, k)].append(row)

    best_by_key: dict[tuple[str, int], dict] = {}
    for key, rows in groups.items():
        if not rows:
            continue
        ref = rows[0]
        best = min(rows, key=lambda r: _float(r["kcgpu_ms"]) or float("inf"))
        gtap_count_e2e_ms = _gtap_e2e_ms(ref)
        gtap_count_phase_ms = _float(ref.get("gtap_count_phase_ms"))
        gtap_kernel_ms = _float(ref.get("gtap_kernel_ms"))
        kcgpu_best_ms = _float(best["kcgpu_ms"])
        best_by_key[key] = {
            "gtap_variant": ref["gtap_variant"],
            "graph": key[0],
            "k": key[1],
            "gtap_heavy": ref["gtap_heavy"],
            "gtap_second_heavy": ref["gtap_second_heavy"],
            "gtap_third_heavy": ref["gtap_third_heavy"],
            "gtap_count": ref["gtap_count"],
            "gtap_count_e2e_ms": gtap_count_e2e_ms,
            "gtap_count_phase_ms": gtap_count_phase_ms,
            "gtap_kernel_ms": gtap_kernel_ms,
            "kcgpu_best_ms": kcgpu_best_ms,
            "kcgpu_best_q": best["kcgpu_q"],
            "kcgpu_orient": best["kcgpu_orient"],
            "match_all": all(r.get("match") == "YES" for r in rows),
            "variant_count": len(rows),
        }
        if gtap_count_e2e_ms and kcgpu_best_ms and kcgpu_best_ms > 0:
            best_by_key[key]["gtap_e2e_over_kcgpu_best"] = gtap_count_e2e_ms / kcgpu_best_ms
        else:
            best_by_key[key]["gtap_e2e_over_kcgpu_best"] = None
        if gtap_kernel_ms and kcgpu_best_ms and kcgpu_best_ms > 0:
            best_by_key[key]["gtap_kernel_over_kcgpu_best"] = gtap_kernel_ms / kcgpu_best_ms
        else:
            best_by_key[key]["gtap_kernel_over_kcgpu_best"] = None
        if gtap_count_phase_ms and kcgpu_best_ms and kcgpu_best_ms > 0:
            best_by_key[key]["gtap_count_phase_over_kcgpu_best"] = (
                gtap_count_phase_ms / kcgpu_best_ms
            )
        else:
            best_by_key[key]["gtap_count_phase_over_kcgpu_best"] = None
    return best_by_key


def _sort_key(item: dict) -> tuple:
    return (GRAPH_ORDER.get(item["graph"], 99), item["k"], item["gtap_variant"])


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "gtap_variant",
        "graph",
        "k",
        "gtap_heavy",
        "gtap_second_heavy",
        "gtap_third_heavy",
        "gtap_count",
        "gtap_count_e2e_ms",
        "gtap_count_phase_ms",
        "gtap_kernel_ms",
        "kcgpu_best_ms",
        "kcgpu_best_q",
        "kcgpu_orient",
        "gtap_e2e_over_kcgpu_best",
        "gtap_kernel_over_kcgpu_best",
        "gtap_count_phase_over_kcgpu_best",
        "variant_count",
        "match_all",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {name: row.get(name, "") for name in fieldnames}
            for name in ("gtap_count_e2e_ms", "gtap_count_phase_ms", "gtap_kernel_ms", "kcgpu_best_ms"):
                value = out.get(name)
                if value not in ("", None):
                    out[name] = f"{float(value):.3f}"
            for name in (
                "gtap_e2e_over_kcgpu_best",
                "gtap_kernel_over_kcgpu_best",
                "gtap_count_phase_over_kcgpu_best",
            ):
                value = out.get(name)
                if value not in ("", None):
                    out[name] = f"{float(value):.6f}"
            writer.writerow(out)


def _print_table(rows: list[dict]) -> None:
    header = (
        f"{'variant':<12} {'graph':<12} {'k':>2} "
        f"{'gtap_e2e':>12} {'kcgpu_e2e':>12} "
        f"{'best_q':>6} {'e2e/best':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        e2e_ratio = row.get("gtap_e2e_over_kcgpu_best")
        e2e_ratio_s = f"{e2e_ratio:.4f}" if e2e_ratio is not None else "n/a"
        gtap_e2e = row.get("gtap_count_e2e_ms")
        kcgpu_best = row.get("kcgpu_best_ms")
        print(
            f"{row['gtap_variant']:<12} {row['graph']:<12} {row['k']:>2} "
            f"{float(gtap_e2e):>12.3f} {float(kcgpu_best):>12.3f} "
            f"{row['kcgpu_best_q']:>6} {e2e_ratio_s:>10}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract best KCGPU e2e and GTAP e2e times per graph/k from k579 CSVs."
    )
    parser.add_argument(
        "--orientation-csv",
        type=Path,
        default=DEFAULT_ORIENTATION_CSV,
        help=f"Orientation averaged CSV (default: {DEFAULT_ORIENTATION_CSV.name})",
    )
    parser.add_argument(
        "--pivot-csv",
        type=Path,
        default=DEFAULT_PIVOT_CSV,
        help=f"Pivot averaged CSV (default: {DEFAULT_PIVOT_CSV.name})",
    )
    parser.add_argument(
        "--only",
        choices=["orientation", "pivot"],
        help="Load only one variant CSV (uses --orientation-csv or --pivot-csv).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output summary CSV (default: {DEFAULT_OUTPUT_CSV.name})",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Print table only; do not write output CSV.",
    )
    parser.add_argument(
        "--include-mismatch",
        action="store_true",
        help="Include rows even when match != YES.",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    if args.only == "orientation":
        paths = [args.orientation_csv]
    elif args.only == "pivot":
        paths = [args.pivot_csv]
    else:
        paths = [args.orientation_csv, args.pivot_csv]

    for path in paths:
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            return 1
        rows.extend(_load_best_rows(path, require_match=not args.include_mismatch).values())

    rows.sort(key=_sort_key)
    if not rows:
        print("No rows found.", file=sys.stderr)
        return 1

    _print_table(rows)
    if not args.no_csv:
        _write_csv(args.output, rows)
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

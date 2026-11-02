#!/usr/bin/env python3
"""Median GTAP/KCGPU k579 benchmark rows over repeat runs."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

NUMERIC_FIELDS = [
    "gtap_heavy",
    "gtap_second_heavy",
    "gtap_third_heavy",
    "gtap_orientation_max_tasks_per_warp",
    "gtap_pivot_max_tasks_per_warp",
    "gtap_count_e2e_ms",
    "gtap_initialize_ms",
    "gtap_count_phase_ms",
    "gtap_kernel_ms",
    "gtap_preprocess_transfer_ms",
    "kcgpu_ms",
    "gtap_count_e2e_ms_over_kcgpu_ms",
    "gtap_count_phase_ms_over_kcgpu_ms",
    "gtap_kernel_ms_over_kcgpu_ms",
]

GROUP_FIELDS = [
    "graph",
    "k",
    "gtap_variant",
    "gtap_heavy",
    "gtap_second_heavy",
    "gtap_third_heavy",
    "gtap_orientation_max_tasks_per_warp",
    "gtap_pivot_max_tasks_per_warp",
    "gtap_count",
    "gtap_status",
    "kcgpu_orient",
    "kcgpu_process",
    "kcgpu_element",
    "kcgpu_q",
    "kcgpu_alloc",
    "kcgpu_small_graph",
    "kcgpu_sort",
    "kcgpu_count",
    "graph_path",
]


def _float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return float(value)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def _stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def _exit_ok(value: str | None) -> bool:
    if value is None:
        return True
    value = value.strip()
    return value == "" or value == "0"


def is_success(row: dict) -> bool:
    return _exit_ok(row.get("gtap_exit")) and _exit_ok(row.get("kcgpu_exit"))


def _unique_exit(items: list[dict], field: str) -> str:
    values = {item.get(field, "") for item in items if item.get(field, "") != ""}
    return values.pop() if len(values) == 1 else ""


def _group_key(row: dict) -> tuple[str, ...]:
    return tuple(row.get(name, "") for name in GROUP_FIELDS)


def average_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)

    averaged: list[dict] = []
    for key, items in sorted(groups.items()):
        valid = [item for item in items if is_success(item)]
        source = valid
        out = {name: value for name, value in zip(GROUP_FIELDS, key)}
        out["repeat"] = "avg"
        out["repeats"] = str(len(valid))
        out["repeats_total"] = str(len(items))

        for field in NUMERIC_FIELDS:
            values = [_float(item.get(field)) for item in source]
            nums = [v for v in values if v is not None]
            median = _median(nums)
            out[field] = f"{median:.3f}" if median is not None else ""
            stdev = _stdev(nums)
            out[f"{field}_stdev"] = f"{stdev:.3f}" if stdev is not None else ""

        matches = [item.get("match", "") for item in source if item.get("match")]
        if matches and all(m == "YES" for m in matches):
            out["match"] = "YES"
        elif matches and all(m == "NO" for m in matches):
            out["match"] = "NO"
        elif matches:
            out["match"] = "PARTIAL"
        else:
            out["match"] = ""

        exit_items = valid if valid else items
        out["gtap_exit"] = _unique_exit(exit_items, "gtap_exit")
        out["kcgpu_exit"] = _unique_exit(exit_items, "kcgpu_exit")
        averaged.append(out)
    return averaged


def fieldnames(rows: list[dict]) -> list[str]:
    base = [
        "graph",
        "k",
        "repeat",
        "repeats",
        "repeats_total",
        "gtap_variant",
        "gtap_heavy",
        "gtap_second_heavy",
        "gtap_third_heavy",
        "gtap_orientation_max_tasks_per_warp",
        "gtap_pivot_max_tasks_per_warp",
        "gtap_count",
        "gtap_count_e2e_ms",
        "gtap_count_e2e_ms_stdev",
        "gtap_count_phase_ms",
        "gtap_count_phase_ms_stdev",
        "gtap_kernel_ms",
        "gtap_kernel_ms_stdev",
        "gtap_preprocess_transfer_ms",
        "gtap_preprocess_transfer_ms_stdev",
        "gtap_status",
        "kcgpu_orient",
        "kcgpu_process",
        "kcgpu_element",
        "kcgpu_q",
        "kcgpu_alloc",
        "kcgpu_small_graph",
        "kcgpu_sort",
        "kcgpu_count",
        "kcgpu_ms",
        "kcgpu_ms_stdev",
        "match",
        "gtap_count_e2e_ms_over_kcgpu_ms",
        "gtap_count_phase_ms_over_kcgpu_ms",
        "gtap_kernel_ms_over_kcgpu_ms",
        "gtap_exit",
        "kcgpu_exit",
        "graph_path",
    ]
    seen = set(base)
    for row in rows:
        for name in row:
            if name not in seen:
                base.append(name)
                seen.add(name)
    return base


def _fmt_ms(value: str | None) -> str:
    if value is None or not str(value).strip():
        return f"{'N/A':>12}"
    return f"{float(value):>12.3f}"


def print_table(rows: list[dict]) -> None:
    print(
        f"{'graph':<12} {'k':>2} {'q':>5} {'n':>2} "
        f"{'gtap_e2e':>12} {'kcgpu_e2e':>12} {'e2e_ratio':>10}"
    )
    print("-" * 58)
    for row in rows:
        ratio = _float(row.get("gtap_count_e2e_ms_over_kcgpu_ms"))
        ratio_s = f"{ratio:.4f}" if ratio is not None else "n/a"
        print(
            f"{row['graph']:<12} {int(row['k']):>2} {row['kcgpu_q']:>5} "
            f"{int(row['repeats']):>2} {_fmt_ms(row.get('gtap_count_e2e_ms'))} "
            f"{_fmt_ms(row.get('kcgpu_ms'))} {ratio_s:>10}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Raw results CSV with repeat column",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output median CSV",
    )
    parser.add_argument("--no-csv", action="store_true", help="Print table only.")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: file not found: {args.input}", file=sys.stderr)
        return 1

    with args.input.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("No rows found.", file=sys.stderr)
        return 1

    averaged = average_rows(rows)
    print_table(averaged)
    if not args.no_csv:
        names = fieldnames(averaged)
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
            writer.writeheader()
            for row in averaged:
                writer.writerow(row)
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

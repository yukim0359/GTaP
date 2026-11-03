#!/usr/bin/env python3
"""Build k579/results/e2e_table.csv from orientation/pivot per-run CSVs.

Per graph×k×variant: median over repeats, then keep the fastest KCGPU q.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
K_CLIQUE_DIR = SCRIPT_DIR.parent.parent
K579_RESULTS = K_CLIQUE_DIR / "k579" / "results"

DEFAULT_ORIENTATION_RUNS = K579_RESULTS / "orientation_runs.csv"
DEFAULT_PIVOT_RUNS = K579_RESULTS / "pivot_runs.csv"
DEFAULT_OUTPUT = K579_RESULTS / "e2e_table.csv"

GRAPH_ORDER = {"DBLP": 0, "as-Skitter": 1, "Orkut": 2}

NUMERIC_FIELDS = [
    "gtap_count_e2e_ms",
    "gtap_count_phase_ms",
    "gtap_kernel_ms",
    "kcgpu_ms",
]

GROUP_FIELDS = [
    "graph",
    "k",
    "gtap_variant",
    "gtap_heavy",
    "gtap_second_heavy",
    "gtap_third_heavy",
    "gtap_count",
    "kcgpu_orient",
    "kcgpu_q",
]


def _float(value: str | float | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    return float(value)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def _exit_ok(value: str | None) -> bool:
    if value is None:
        return True
    value = value.strip()
    return value == "" or value == "0"


def _is_success(row: dict) -> bool:
    return _exit_ok(row.get("gtap_exit")) and _exit_ok(row.get("kcgpu_exit"))


def _group_key(row: dict) -> tuple[str, ...]:
    return tuple(row.get(name, "") for name in GROUP_FIELDS)


def median_by_config(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)

    out_rows: list[dict] = []
    for key, items in groups.items():
        valid = [item for item in items if _is_success(item)]
        source = valid if valid else items
        out = {name: value for name, value in zip(GROUP_FIELDS, key)}
        for field in NUMERIC_FIELDS:
            nums = [v for v in (_float(item.get(field)) for item in source) if v is not None]
            med = _median(nums)
            out[field] = med
        matches = [item.get("match", "") for item in source if item.get("match")]
        out["match"] = (
            "YES"
            if matches and all(m == "YES" for m in matches)
            else ("NO" if matches and all(m == "NO" for m in matches) else "PARTIAL")
        )
        out_rows.append(out)
    return out_rows


def best_per_graph_k(median_rows: list[dict], *, require_match: bool) -> list[dict]:
    groups: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in median_rows:
        if require_match and row.get("match") != "YES":
            continue
        key = (row["graph"], int(row["k"]), row["gtap_variant"])
        groups[key].append(row)

    best_rows: list[dict] = []
    for (graph, k, variant), rows in groups.items():
        ref = rows[0]
        best = min(rows, key=lambda r: _float(r["kcgpu_ms"]) or float("inf"))
        gtap_e2e = _float(ref.get("gtap_count_e2e_ms"))
        gtap_phase = _float(ref.get("gtap_count_phase_ms"))
        gtap_kernel = _float(ref.get("gtap_kernel_ms"))
        kcgpu_best = _float(best.get("kcgpu_ms"))
        row = {
            "gtap_variant": variant,
            "graph": graph,
            "k": k,
            "gtap_heavy": ref["gtap_heavy"],
            "gtap_second_heavy": ref["gtap_second_heavy"],
            "gtap_third_heavy": ref["gtap_third_heavy"],
            "gtap_count": ref["gtap_count"],
            "gtap_count_e2e_ms": gtap_e2e,
            "gtap_count_phase_ms": gtap_phase,
            "gtap_kernel_ms": gtap_kernel,
            "kcgpu_best_ms": kcgpu_best,
            "kcgpu_best_q": best["kcgpu_q"],
            "kcgpu_orient": best["kcgpu_orient"],
            "variant_count": len(rows),
            "match_all": all(r.get("match") == "YES" for r in rows),
            "gtap_e2e_over_kcgpu_best": (
                gtap_e2e / kcgpu_best if gtap_e2e and kcgpu_best and kcgpu_best > 0 else None
            ),
            "gtap_kernel_over_kcgpu_best": (
                gtap_kernel / kcgpu_best
                if gtap_kernel and kcgpu_best and kcgpu_best > 0
                else None
            ),
            "gtap_count_phase_over_kcgpu_best": (
                gtap_phase / kcgpu_best if gtap_phase and kcgpu_best and kcgpu_best > 0 else None
            ),
        }
        best_rows.append(row)
    best_rows.sort(key=lambda r: (GRAPH_ORDER.get(r["graph"], 99), r["k"], r["gtap_variant"]))
    return best_rows


def _read_runs(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_table(path: Path, rows: list[dict]) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {name: row.get(name, "") for name in fieldnames}
            for name in (
                "gtap_count_e2e_ms",
                "gtap_count_phase_ms",
                "gtap_kernel_ms",
                "kcgpu_best_ms",
            ):
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
        ratio = row.get("gtap_e2e_over_kcgpu_best")
        ratio_s = f"{ratio:.4f}" if ratio is not None else "n/a"
        print(
            f"{row['gtap_variant']:<12} {row['graph']:<12} {row['k']:>2} "
            f"{float(row['gtap_count_e2e_ms']):>12.3f} {float(row['kcgpu_best_ms']):>12.3f} "
            f"{row['kcgpu_best_q']:>6} {ratio_s:>10}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orientation-runs",
        type=Path,
        default=DEFAULT_ORIENTATION_RUNS,
    )
    parser.add_argument(
        "--pivot-runs",
        type=Path,
        default=DEFAULT_PIVOT_RUNS,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--include-mismatch",
        action="store_true",
        help="Include rows even when match != YES.",
    )
    args = parser.parse_args()

    paths = [args.orientation_runs, args.pivot_runs]
    for path in paths:
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            return 1

    median_rows: list[dict] = []
    for path in paths:
        median_rows.extend(median_by_config(_read_runs(path)))

    rows = best_per_graph_k(median_rows, require_match=not args.include_mismatch)
    if not rows:
        print("No rows found.", file=sys.stderr)
        return 1

    _print_table(rows)
    _write_table(args.output, rows)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Re-parse k579 e2e benchmark logs with gtap_initialize included in GTaP e2e."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
K_CLIQUE_DIR = SCRIPT_DIR.parent
K579_DIR = K_CLIQUE_DIR / "k579"
RESULTS_DIR = K579_DIR / "results"

GRAPH_ORDER = {"DBLP": 0, "as-Skitter": 1, "Orkut": 2}
GRAPH_LABELS = {
    "DBLP": "com-dblp",
    "as-Skitter": "as-skitter",
    "Orkut": "com-orkut",
}
K_VALUES = [5, 7, 9]

RE_GTAP = {
    "count": re.compile(r"GTaP count: (\d+)"),
    "e2e": re.compile(r"GTaP count e2e time: ([0-9.]+) ms"),
    "phase": re.compile(r"GTaP count phase time: ([0-9.]+) ms"),
    "kernel": re.compile(r"GTaP kernel time: ([0-9.]+) ms"),
    "orient": re.compile(r"GTaP orient time: ([0-9.]+) ms"),
    "workspace": re.compile(r"GTaP workspace alloc time: ([0-9.]+) ms"),
    "initialize": re.compile(r"GTaP gtap_initialize time: ([0-9.]+) ms"),
    "status": re.compile(r"Validation: (\S+)"),
}
RE_KCGPU_E2E = re.compile(r"count end-to-end time ([0-9.]+) s")
RE_KCGPU_COUNT = re.compile(r"Counter = ([0-9,]+)")
RE_LOG_KEY = re.compile(r"^(?P<graph>(?:as-Skitter|DBLP|Orkut))_k(?P<k>\d+)_r(?P<repeat>\d+)_")

ORIENT_KCGPU_Q_BY_K = {
    5: ["o4b"],
    7: ["o1b", "o2b"],
    9: ["o1b"],
}
PIVOT_KCGPU_Q = ["p1b"]


def _float(value: str | float | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    return float(value)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def parse_log_key(name: str) -> tuple[str, int, int] | None:
    match = RE_LOG_KEY.match(name)
    if not match:
        return None
    return match.group("graph"), int(match.group("k")), int(match.group("repeat"))


def parse_gtap_log(path: Path) -> dict | None:
    text = path.read_text()
    fields: dict[str, str | None] = {}
    for key, rx in RE_GTAP.items():
        hit = rx.search(text)
        fields[key] = hit.group(1) if hit else None
    if not fields["e2e"]:
        return None

    old_e2e = float(fields["e2e"])
    init_ms = float(fields["initialize"]) if fields["initialize"] else 0.0
    new_e2e = old_e2e + init_ms
    orient = _float(fields["orient"])
    workspace = _float(fields["workspace"])
    phase = _float(fields["phase"])
    if orient is not None and workspace is not None and phase is not None:
        comp_e2e = orient + workspace + phase
        if abs(comp_e2e - new_e2e) > 0.05:
            print(
                f"WARN {path.name}: old+init={new_e2e:.3f} vs components={comp_e2e:.3f}",
                file=sys.stderr,
            )

    return {
        "gtap_count": fields["count"],
        "gtap_count_e2e_ms": new_e2e,
        "gtap_initialize_ms": init_ms,
        "gtap_count_phase_ms": _float(fields["phase"]),
        "gtap_kernel_ms": _float(fields["kernel"]),
        "gtap_status": fields["status"] or "",
    }


def parse_kcgpu_log(path: Path) -> dict | None:
    text = path.read_text()
    e2e = RE_KCGPU_E2E.search(text)
    cnt = RE_KCGPU_COUNT.search(text)
    if not e2e:
        return None
    return {
        "kcgpu_ms": float(e2e.group(1)) * 1000.0,
        "kcgpu_count": cnt.group(1).replace(",", "") if cnt else None,
    }


def kcgpu_q_from_name(name: str) -> str | None:
    match = re.search(r"_kcgpu_edge_([a-z0-9]+)\.log$", name)
    return match.group(1) if match else None


def load_variant_rows(log_dir: Path, variant: str, gtap_suffix: str) -> list[dict]:
    rows: list[dict] = []
    for gtap_log in sorted(log_dir.glob(f"*_gtap_{gtap_suffix}.log")):
        key = parse_log_key(gtap_log.name)
        if not key:
            continue
        graph, k, repeat = key
        gtap = parse_gtap_log(gtap_log)
        if not gtap:
            continue

        allowed_q = ORIENT_KCGPU_Q_BY_K[k] if variant == "orientation" else PIVOT_KCGPU_Q
        for q in allowed_q:
            kcgpu_log = log_dir / f"{graph}_k{k}_r{repeat}_kcgpu_edge_{q}.log"
            if not kcgpu_log.exists():
                continue
            kcgpu = parse_kcgpu_log(kcgpu_log)
            if not kcgpu:
                continue
            match = (
                "YES"
                if gtap["gtap_count"] and kcgpu["kcgpu_count"] == gtap["gtap_count"]
                else "NO"
            )
            rows.append(
                {
                    "graph": graph,
                    "k": k,
                    "repeat": repeat,
                    "gtap_variant": variant,
                    "kcgpu_q": q,
                    "match": match,
                    **gtap,
                    **kcgpu,
                }
            )
    return rows


def average_by_group(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["graph"], row["k"], row["gtap_variant"], row["kcgpu_q"])].append(row)

    averaged: list[dict] = []
    for key, group_rows in groups.items():
        graph, k, gtap_variant, kcgpu_q = key
        e2e_vals = [r["gtap_count_e2e_ms"] for r in group_rows]
        init_vals = [r["gtap_initialize_ms"] for r in group_rows]
        phase_vals = [r["gtap_count_phase_ms"] for r in group_rows if r["gtap_count_phase_ms"] is not None]
        kernel_vals = [r["gtap_kernel_ms"] for r in group_rows if r["gtap_kernel_ms"] is not None]
        kcgpu_vals = [r["kcgpu_ms"] for r in group_rows]
        ratio_vals = [r["gtap_count_e2e_ms"] / r["kcgpu_ms"] for r in group_rows if r["kcgpu_ms"] > 0]
        averaged.append(
            {
                "graph": graph,
                "k": k,
                "gtap_variant": gtap_variant,
                "kcgpu_q": kcgpu_q,
                "repeats": len(group_rows),
                "gtap_count": group_rows[0]["gtap_count"],
                "gtap_count_e2e_ms": _mean(e2e_vals),
                "gtap_count_e2e_ms_stdev": _stdev(e2e_vals),
                "gtap_initialize_ms": _mean(init_vals),
                "gtap_count_phase_ms": _mean(phase_vals) if phase_vals else None,
                "gtap_kernel_ms": _mean(kernel_vals) if kernel_vals else None,
                "kcgpu_ms": _mean(kcgpu_vals),
                "kcgpu_ms_stdev": _stdev(kcgpu_vals),
                "gtap_count_e2e_ms_over_kcgpu_ms": _mean(ratio_vals),
                "match_all": all(r["match"] == "YES" for r in group_rows),
                "variant_count": 1,
            }
        )
    return sorted(
        averaged,
        key=lambda r: (GRAPH_ORDER.get(r["graph"], 99), r["k"], r["gtap_variant"], r["kcgpu_q"]),
    )


def best_summary(avg_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in avg_rows:
        if not row["match_all"]:
            continue
        groups[(row["graph"], row["k"], row["gtap_variant"])].append(row)

    best_rows: list[dict] = []
    for key, rows in groups.items():
        graph, k, gtap_variant = key
        ref = rows[0]
        best = min(rows, key=lambda r: r["kcgpu_ms"])
        gtap_e2e = ref["gtap_count_e2e_ms"]
        kcgpu_best = best["kcgpu_ms"]
        ratio = gtap_e2e / kcgpu_best if kcgpu_best > 0 else None
        best_rows.append(
            {
                "gtap_variant": gtap_variant,
                "graph": graph,
                "k": k,
                "gtap_count": ref["gtap_count"],
                "gtap_count_e2e_ms": gtap_e2e,
                "gtap_initialize_ms": ref["gtap_initialize_ms"],
                "gtap_count_phase_ms": ref["gtap_count_phase_ms"],
                "gtap_kernel_ms": ref["gtap_kernel_ms"],
                "kcgpu_best_ms": kcgpu_best,
                "kcgpu_best_q": best["kcgpu_q"],
                "gtap_e2e_over_kcgpu_best": ratio,
                "variant_count": len(rows),
                "match_all": True,
            }
        )
    return sorted(
        best_rows,
        key=lambda r: (GRAPH_ORDER.get(r["graph"], 99), r["k"], r["gtap_variant"]),
    )


def write_avg_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "graph",
        "k",
        "gtap_variant",
        "kcgpu_q",
        "repeats",
        "gtap_count",
        "gtap_count_e2e_ms",
        "gtap_count_e2e_ms_stdev",
        "gtap_initialize_ms",
        "gtap_count_phase_ms",
        "gtap_kernel_ms",
        "kcgpu_ms",
        "kcgpu_ms_stdev",
        "gtap_count_e2e_ms_over_kcgpu_ms",
        "match_all",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {name: row.get(name, "") for name in fieldnames}
            for name in (
                "gtap_count_e2e_ms",
                "gtap_count_e2e_ms_stdev",
                "gtap_initialize_ms",
                "gtap_count_phase_ms",
                "gtap_kernel_ms",
                "kcgpu_ms",
                "kcgpu_ms_stdev",
                "gtap_count_e2e_ms_over_kcgpu_ms",
            ):
                value = out.get(name)
                if value not in ("", None):
                    out[name] = f"{float(value):.3f}"
            writer.writerow(out)


def write_best_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "gtap_variant",
        "graph",
        "k",
        "gtap_count",
        "gtap_count_e2e_ms",
        "gtap_initialize_ms",
        "gtap_count_phase_ms",
        "gtap_kernel_ms",
        "kcgpu_best_ms",
        "kcgpu_best_q",
        "gtap_e2e_over_kcgpu_best",
        "variant_count",
        "match_all",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {name: row.get(name, "") for name in fieldnames}
            for name in (
                "gtap_count_e2e_ms",
                "gtap_initialize_ms",
                "gtap_count_phase_ms",
                "gtap_kernel_ms",
                "kcgpu_best_ms",
            ):
                value = out.get(name)
                if value not in ("", None):
                    out[name] = f"{float(value):.3f}"
            ratio = out.get("gtap_e2e_over_kcgpu_best")
            if ratio not in ("", None):
                out["gtap_e2e_over_kcgpu_best"] = f"{float(ratio):.6f}"
            writer.writerow(out)


def process_variant(log_dir: Path, variant: str, gtap_suffix: str, out_prefix: str) -> list[dict]:
    rows = load_variant_rows(log_dir, variant, gtap_suffix)
    if not rows:
        print(f"ERROR: no rows parsed from {log_dir}", file=sys.stderr)
        return []

    avg_rows = average_by_group(rows)
    best_rows = best_summary(avg_rows)
    write_avg_csv(RESULTS_DIR / f"{out_prefix}_avg.csv", avg_rows)
    write_best_csv(RESULTS_DIR / f"{out_prefix}_best_summary.csv", best_rows)
    print(f"Parsed {len(rows)} rows from {log_dir}")
    print(f"Wrote {RESULTS_DIR / f'{out_prefix}_avg.csv'}")
    print(f"Wrote {RESULTS_DIR / f'{out_prefix}_best_summary.csv'}")
    return best_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=["orientation", "pivot", "both"],
        default="both",
    )
    args = parser.parse_args()

    best_rows: list[dict] = []
    if args.only in ("orientation", "both"):
        best_rows.extend(
            process_variant(
                K579_DIR / "logs" / "orientation_e2e",
                "orientation",
                "orientation",
                "gtap_kcgpu_orientation_k579_e2e_init_included",
            )
        )
    if args.only in ("pivot", "both"):
        best_rows.extend(
            process_variant(
                K579_DIR / "logs" / "pivot_e2e",
                "pivot",
                "pivot",
                "gtap_kcgpu_pivot_k579_e2e_init_included",
            )
        )

    if not best_rows:
        return 1

    combined = sorted(
        best_rows,
        key=lambda r: (GRAPH_ORDER.get(r["graph"], 99), r["k"], r["gtap_variant"]),
    )
    combined_path = RESULTS_DIR / "gtap_kcgpu_k579_e2e_init_included_best_summary.csv"
    write_best_csv(combined_path, combined)
    print(f"Wrote {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

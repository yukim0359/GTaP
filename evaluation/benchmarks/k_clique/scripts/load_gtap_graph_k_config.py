#!/usr/bin/env python3
"""Load per graph×k GTAP tuning from gtap_graph_k_config.csv."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
K_CLIQUE_DIR = SCRIPT_DIR.parent
DEFAULT_CONFIG = K_CLIQUE_DIR / "data" / "gtap_graph_k_config.csv"

# Fallback when a CSV cell is empty. heavy/second/third match Makefile
# smoke defaults. max_tasks_per_warp matches the binary runtime defaults.
BUILD_DEFAULTS = {
    "heavy": 32,
    "second_heavy": 32,
    "third_heavy": 32,
    "orientation_max_tasks_per_warp": 150_000,
    "pivot_max_tasks_per_warp": 200_000,
}

FIELD_TO_ENV = {
    "heavy": "GTAP_HEAVY_CANDIDATES",
    "second_heavy": "GTAP_SECOND_HEAVY_CANDIDATES",
    "third_heavy": "GTAP_THIRD_HEAVY_CANDIDATES",
    "orientation_max_tasks_per_warp": "GTAP_ORIENTATION_MAX_TASKS_PER_WARP",
    "pivot_max_tasks_per_warp": "GTAP_PIVOT_MAX_TASKS_PER_WARP",
}


def _normalize_graph(name: str) -> str:
    aliases = {
        "com-DBLP": "DBLP",
        "Skitter": "as-Skitter",
        "com-Orkut": "Orkut",
    }
    return aliases.get(name, name)


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return int(value)


def load_config_table(path: Path) -> dict[tuple[str, int], dict[str, int]]:
    table: dict[tuple[str, int], dict[str, int]] = {}
    with path.open(newline="") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            row = next(csv.reader([line]))
            if not row or row[0].startswith("#"):
                continue
            if row[0] == "graph":
                continue
            if len(row) < 2:
                continue
            graph = _normalize_graph(row[0].strip())
            k = int(row[1].strip())
            fields = [
                "heavy",
                "second_heavy",
                "third_heavy",
                "orientation_max_tasks_per_warp",
                "pivot_max_tasks_per_warp",
            ]
            entry: dict[str, int] = {}
            for idx, field in enumerate(fields, start=2):
                if idx < len(row):
                    parsed = _parse_int(row[idx])
                    if parsed is not None:
                        entry[field] = parsed
            table[(graph, k)] = entry
    return table


def resolve_entry(
    table: dict[tuple[str, int], dict[str, int]],
    graph: str,
    k: int,
) -> dict[str, int]:
    graph = _normalize_graph(graph)
    merged = dict(BUILD_DEFAULTS)
    exact = table.get((graph, k), {})
    merged.update(exact)
    return merged


def to_env(entry: dict[str, int]) -> dict[str, str]:
    return {FIELD_TO_ENV[key]: str(value) for key, value in entry.items()}


def print_shell_export(env: dict[str, str]) -> None:
    for key in sorted(env):
        print(f'export {key}="{env[key]}"')


def print_make_args(env: dict[str, str], variant: str) -> None:
    if variant == "orientation":
        keys = [
            "GTAP_HEAVY_CANDIDATES",
            "GTAP_SECOND_HEAVY_CANDIDATES",
            "GTAP_THIRD_HEAVY_CANDIDATES",
        ]
    elif variant == "pivot":
        keys = []
    else:
        raise ValueError(f"unknown variant: {variant}")
    for key in keys:
        print(f"{key}={env[key]}")


def print_run_env(env: dict[str, str], variant: str) -> None:
    if variant == "orientation":
        value = env["GTAP_ORIENTATION_MAX_TASKS_PER_WARP"]
    elif variant == "pivot":
        value = env["GTAP_PIVOT_MAX_TASKS_PER_WARP"]
    else:
        raise ValueError(f"unknown variant: {variant}")
    print(f'export GTAP_MAX_TASKS_PER_WARP="{value}"')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", help="Graph name (e.g. DBLP, as-Skitter, Orkut)")
    parser.add_argument("k", type=int, help="k-clique size")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Config CSV (default: {DEFAULT_CONFIG.name})",
    )
    parser.add_argument(
        "--shell-export",
        action="store_true",
        help="Print bash export statements for all fields",
    )
    parser.add_argument(
        "--make-args",
        choices=["orientation", "pivot"],
        help="Print compile-time make VAR=value lines for the given GTAP variant",
    )
    parser.add_argument(
        "--run-env",
        choices=["orientation", "pivot"],
        help="Print export GTAP_MAX_TASKS_PER_WARP=... for the given variant",
    )
    parser.add_argument(
        "--field",
        choices=sorted(FIELD_TO_ENV),
        help="Print a single resolved field value",
    )
    parser.add_argument(
        "--values",
        action="store_true",
        help="Print heavy second_heavy third_heavy orientation_max_tasks_per_warp pivot_max_tasks_per_warp (space-separated)",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 1

    table = load_config_table(args.config)
    entry = resolve_entry(table, args.graph, args.k)
    env = to_env(entry)

    if args.field:
        print(entry[args.field])
    elif args.values:
        print(
            entry["heavy"],
            entry["second_heavy"],
            entry["third_heavy"],
            entry["orientation_max_tasks_per_warp"],
            entry["pivot_max_tasks_per_warp"],
        )
    elif args.make_args:
        print_make_args(env, args.make_args)
    elif args.run_env:
        print_run_env(env, args.run_env)
    elif args.shell_export:
        print_shell_export(env)
    else:
        for key in sorted(entry):
            print(f"{key}={entry[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

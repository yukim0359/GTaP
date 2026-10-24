#!/usr/bin/env python3
"""Generate data/nCr.txt for the pivot nCr lookup table.

Writes GTAP_NCR_ROWS x GTAP_NCR_COLS comma-separated doubles (%.18e),
no trailing comma. Each entry is the correctly rounded float of C(n, r).
The generated file is gitignored; rebuild with `make ncr`.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
K_CLIQUE_DIR = SCRIPT_DIR.parent.parent
DEFAULT_PATH = K_CLIQUE_DIR / "data" / "nCr.txt"
DEFAULT_ROWS = 1001
DEFAULT_COLS = 401


def ncr_double(n: int, r: int) -> float:
    if r < 0 or r > n:
        return 0.0
    return float(math.comb(n, r))


def write_table(path: Path, rows: int, cols: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        for n in range(rows):
            out.write(
                ",".join(f"{ncr_double(n, r):.18e}" for r in range(cols))
            )
            out.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_PATH,
        help=f"output path (default: {DEFAULT_PATH})",
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS)
    args = parser.parse_args()
    if args.rows < 1 or args.cols < 1:
        raise SystemExit("rows and cols must be positive")
    write_table(args.path, args.rows, args.cols)


if __name__ == "__main__":
    main()

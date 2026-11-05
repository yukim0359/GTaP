#!/usr/bin/env python3
"""Convert a SNAP-style undirected edgelist to pattern-symmetric Matrix Market.

Usage:
  edgelist_to_symmetric_mtx.py INPUT.txt OUTPUT.mtx

Skips comment lines (#) and self-loops; keeps the lower triangle (row >=
col), matching SuiteSparse Matrix Collection SNAP/* MTX files; remaps
node IDs to contiguous 1..n in sorted-ID order; writes 1-based indices.
Uses an on-disk sort so large graphs fit in modest memory.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="SNAP edgelist (.txt)")
    parser.add_argument("output", help="Matrix Market (.mtx)")
    args = parser.parse_args()

    nodes: set[int] = set()
    edge_count = 0

    with tempfile.TemporaryDirectory(prefix="snap_mtx_") as tmp:
        raw_path = os.path.join(tmp, "edges.tsv")
        remapped_path = os.path.join(tmp, "edges.remapped.tsv")
        sorted_path = os.path.join(tmp, "edges.sorted.tsv")

        with open(args.input, "r", encoding="utf-8", errors="replace") as inf, open(
            raw_path, "w", encoding="utf-8"
        ) as outf:
            for line in inf:
                if not line or line[0] == "#":
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                u = int(parts[0])
                v = int(parts[1])
                if u == v:
                    continue
                # Store undirected edge with min/max for dedup; remap later.
                if u > v:
                    u, v = v, u
                nodes.add(u)
                nodes.add(v)
                outf.write(f"{u}\t{v}\n")

        id_map = {old: new for new, old in enumerate(sorted(nodes))}
        n = len(id_map)

        with open(raw_path, "r", encoding="utf-8") as inf, open(
            remapped_path, "w", encoding="utf-8"
        ) as outf:
            for line in inf:
                u_s, v_s = line.split()
                a = id_map[int(u_s)]
                b = id_map[int(v_s)]
                # SuiteSparse SNAP/* MTX: lower triangle (row >= col).
                if a < b:
                    a, b = b, a
                outf.write(f"{a}\t{b}\n")

        subprocess.run(
            ["sort", "-u", "-t", "\t", "-k1,1n", "-k2,2n", "-o", sorted_path, remapped_path],
            check=True,
            env={**os.environ, "LC_ALL": "C"},
        )

        with open(sorted_path, "r", encoding="utf-8") as inf, open(
            args.output, "w", encoding="utf-8"
        ) as outf:
            for _ in inf:
                edge_count += 1
            inf.seek(0)
            outf.write("%%MatrixMarket matrix coordinate pattern symmetric\n")
            outf.write(f"{n} {n} {edge_count}\n")
            for line in inf:
                u_s, v_s = line.split()
                outf.write(f"{int(u_s) + 1} {int(v_s) + 1}\n")

    print(f"Wrote {args.output} (n={n}, nnz={edge_count})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Per-edge (per-block) processing-time statistics from a KCGPU profile.

In KCGPU edge mode the kernel is launched with one logical block per edge
(grid = #edges). At any instant each physical worker slot (sm_id*CBPSM+levelPtr)
runs one edge's block; when it finishes it releases the slot and the next edge's
block takes it. The warp timeline CSV records Working/NotWorking state-change
events per slot, so one contiguous "Working" run = one edge's processing time.

This streams the timeline and reports the distribution of per-edge times
(count, mean, percentiles, histogram) plus the longest stragglers.

Workers whose profiler buffer saturated (dropped_samples>0) have truncated/
merged segments; their edges are EXCLUDED from the reliable distribution and
reported separately. Re-run with a larger KCPROFILE_MAX_SAMPLES to get 0 drops.

The output (summary table + log-decade histogram + top-N stragglers) uses a
fixed layout so it is identical for every graph: pass --markdown for a report
you can paste/commit, and --out to save it to a file.

Usage:
  python3 scripts/kcgpu_max_edge_time.py --variant pivot --graph Orkut --k 9
  python3 scripts/kcgpu_max_edge_time.py --variant pivot --graph Orkut --k 9 --markdown
  python3 scripts/kcgpu_max_edge_time.py --variant orientation --graph DBLP --k 9 --markdown --out report.md
  python3 scripts/kcgpu_max_edge_time.py --csv <..._warp_timeline_working_*.csv>
  python3 scripts/kcgpu_max_edge_time.py --variant orientation --graph as-Skitter --k 9 --include-dropped
"""

from __future__ import annotations

import argparse
import csv
import glob
import heapq
import math
import os
import sys
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
K_CLIQUE_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_PROFILE_DIR = os.path.join(K_CLIQUE_DIR, "profile", "kcgpu")

BINS_PER_DECADE = 10


def _id_column(fieldnames):
    for cand in ("slot_id", "worker_id", "warp_id", "block_id"):
        if cand in fieldnames:
            return cand
    raise SystemExit(f"ERROR: no worker id column found in {fieldnames}")


def _load_dropped(timeline_path):
    """Map worker id -> dropped_samples from the sibling statistics CSV."""
    stats_path = timeline_path.replace("_warp_timeline_working",
                                       "_warp_statistics_working")
    if not os.path.exists(stats_path):
        sys.stderr.write(f"WARNING: statistics CSV not found, cannot flag dropped warps: {stats_path}\n")
        return {}, stats_path
    dropped = {}
    with open(stats_path, newline="") as f:
        reader = csv.DictReader(f)
        id_col = _id_column(set(reader.fieldnames))
        for r in reader:
            dropped[r[id_col]] = int(r.get("dropped_samples", 0) or 0)
    return dropped, stats_path


def _resolve_csv(args):
    if args.csv:
        return args.csv
    app = f"kcgpu_{args.variant}"
    pattern = os.path.join(
        args.profile_dir,
        f"{app}_warp_timeline_working_*{args.graph}*k{args.k}_*.csv",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit(
            f"ERROR: no timeline CSV matched: {pattern}\n"
            f"Pass --csv explicitly or check {args.profile_dir}"
        )
    if len(matches) > 1:
        sys.stderr.write("WARNING: multiple matches, using first:\n")
        for m in matches:
            sys.stderr.write(f"  {m}\n")
    return matches[0]


def _bucket(dur_ms):
    return int(math.floor(math.log10(dur_ms) * BINS_PER_DECADE))


def _bucket_rep(key):
    return 10.0 ** ((key + 0.5) / BINS_PER_DECADE)


def _percentile(hist, total, q):
    """Approximate q-quantile (0..1) from a log-spaced histogram."""
    if total == 0:
        return float("nan")
    target = q * total
    cum = 0
    for key in sorted(hist):
        cum += hist[key]
        if cum >= target:
            return _bucket_rep(key)
    return _bucket_rep(max(hist))


def analyze(path, top_n, dropped, include_dropped):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = {name: i for i, name in enumerate(header)}
        id_col = _id_column(set(header))
        i_id = cols[id_col]
        i_time = cols["relative_time_ms"]
        i_state = cols["state"]

        open_start = {}
        last_time = {}
        per_worker_max = {}
        heap = []                       # min-heap of (duration, worker, start)

        st = {                          # reliable (clean) distribution
            "n": 0, "sum": 0.0, "min": math.inf, "max": 0.0,
            "argmax": None, "hist": defaultdict(int),
        }
        dropped_seg_count = 0           # segments on saturated workers (excluded)

        def record(worker, start, end):
            nonlocal dropped_seg_count
            dur = end - start
            if dur <= 0.0:
                return
            if len(heap) < top_n:
                heapq.heappush(heap, (dur, worker, start))
            elif dur > heap[0][0]:
                heapq.heapreplace(heap, (dur, worker, start))
            prev = per_worker_max.get(worker)
            if prev is None or dur > prev[0]:
                per_worker_max[worker] = (dur, start)

            is_clean = dropped.get(worker, 0) == 0
            if not is_clean and not include_dropped:
                dropped_seg_count += 1
                return
            st["n"] += 1
            st["sum"] += dur
            if dur < st["min"]:
                st["min"] = dur
            if dur > st["max"]:
                st["max"] = dur
                st["argmax"] = (worker, start)
            st["hist"][_bucket(dur)] += 1

        for row in reader:
            worker = row[i_id]
            t = float(row[i_time])
            state = row[i_state]
            last_time[worker] = t
            if state == "1":
                if worker not in open_start:
                    open_start[worker] = t
            else:
                start = open_start.pop(worker, None)
                if start is not None:
                    record(worker, start, t)

        for worker, start in open_start.items():
            record(worker, start, last_time.get(worker, start))

    top = sorted(heap, key=lambda x: x[0], reverse=True)
    return {
        "id_col": id_col,
        "stats": st,
        "dropped_seg_count": dropped_seg_count,
        "top": top,
        "per_worker_max": per_worker_max,
    }


# Fixed log-decade window so the histogram has the SAME row layout for every
# graph (empty decades are still printed). Auto-expands only if data exceeds it.
DECADE_MIN_DEFAULT = -3   # [0.001, 0.01) ms
DECADE_MAX_DEFAULT = 4    # ... up to [1e4, 1e5) ms


def _histogram_rows(hist, total, decade_min, decade_max):
    """Contiguous per-decade rows (lo, hi, count, pct, cum_pct).

    Always covers [decade_min, decade_max]; expands to include any populated
    decade outside that window so no edge is dropped from the table.
    """
    by_decade = defaultdict(int)
    for key, c in hist.items():
        by_decade[key // BINS_PER_DECADE] += c
    if by_decade:
        lo = min(decade_min, min(by_decade))
        hi = max(decade_max, max(by_decade))
    else:
        lo, hi = decade_min, decade_max
    rows = []
    cum = 0
    for d in range(lo, hi + 1):
        c = by_decade.get(d, 0)
        cum += c
        rows.append((
            10.0 ** d, 10.0 ** (d + 1), c,
            (100.0 * c / total) if total else 0.0,
            (100.0 * cum / total) if total else 0.0,
        ))
    return rows


def build_summary(args, path, dropped, res):
    """Collect every value the renderers need into one format-agnostic dict."""
    st = res["stats"]
    n = st["n"]
    return {
        "graph": args.graph, "k": args.k, "variant": args.variant,
        "timeline": path, "id_col": res["id_col"],
        "n_workers": len(dropped),
        "n_dropped": sum(1 for v in dropped.values() if v > 0),
        "include_dropped": args.include_dropped,
        "dropped_seg_count": res["dropped_seg_count"],
        "n": n,
        "total_ms": st["sum"],
        "mean": (st["sum"] / n) if n else float("nan"),
        "min": st["min"] if n else float("nan"),
        "p50": _percentile(st["hist"], n, 0.50),
        "p90": _percentile(st["hist"], n, 0.90),
        "p99": _percentile(st["hist"], n, 0.99),
        "p999": _percentile(st["hist"], n, 0.999),
        "max": st["max"],
        "argmax": st["argmax"],
        "hist_rows": _histogram_rows(st["hist"], n, args.decade_min, args.decade_max),
        "top": [(dur, w, start, dropped.get(w, 0) == 0)
                for (dur, w, start) in res["top"]],
    }


def _drop_status(s):
    if s["n_workers"] == 0:
        return "statistics CSV not found; reliability unknown"
    if s["n_dropped"] == 0:
        return f"0 / {s['n_workers']} (all per-edge times reliable)"
    tail = ("INCLUDED (unreliable)" if s["include_dropped"]
            else f"excluded ({s['dropped_seg_count']:,} segments)")
    return (f"{s['n_dropped']} / {s['n_workers']} saturated "
            f"-> their edges {tail}")


def render_text(s, out):
    def p(line=""):
        print(line, file=out)

    p(f"Timeline: {s['timeline']}")
    p(f"Buffer-saturated workers: {_drop_status(s)}")
    p()
    p("=== Per-edge processing time (one edge = one Working segment) ===")
    if s["n"] == 0:
        p("No usable edges found.")
        return
    am = s["argmax"]
    p(f"  edges (segments)  : {s['n']:,}")
    p(f"  total time        : {s['total_ms']:.3f} ms")
    p(f"  mean              : {s['mean']:.4f} ms")
    p(f"  min               : {s['min']:.4f} ms")
    p(f"  median (p50 ~)    : {s['p50']:.4f} ms")
    p(f"  p90 ~             : {s['p90']:.4f} ms")
    p(f"  p99 ~             : {s['p99']:.4f} ms")
    p(f"  p99.9 ~           : {s['p999']:.4f} ms")
    p(f"  max               : {s['max']:.3f} ms"
      + (f"  ({s['id_col']}={am[0]}, start={am[1]:.3f} ms)" if am else ""))
    p("  (p50..p99.9 are log-histogram estimates; min/mean/max/total are exact)")
    p()
    p("Per-edge time histogram (log decades):")
    p(f"  {'range_ms':>22}  {'edges':>14}  {'pct':>6}  cumulative")
    for lo, hi, c, pct, cum in s["hist_rows"]:
        bar = "#" * int(pct / 2)
        p(f"  [{lo:>9.4g}, {hi:>9.4g})  {c:>14,}  {pct:>5.1f}%  {cum:>5.1f}%  {bar}")
    p()
    p(f"Top {len(s['top'])} longest single edges:")
    p(f"  {'rank':>4}  {s['id_col']:>10}  {'duration_ms':>12}  {'start_ms':>12}  reliability")
    for rank, (dur, worker, start, reliable) in enumerate(s["top"], 1):
        rel = "ok" if reliable else "dropped!"
        p(f"  {rank:>4}  {worker:>10}  {dur:>12.3f}  {start:>12.3f}  {rel}")


def render_markdown(s, out):
    def p(line=""):
        print(line, file=out)

    p(f"## {s['graph']} k{s['k']} {s['variant']} \u2014 per-edge processing time")
    p()
    p(f"- timeline: `{os.path.basename(s['timeline'])}`")
    p(f"- buffer-saturated workers: {_drop_status(s)}")
    p()
    if s["n"] == 0:
        p("_No usable edges found._")
        return
    am = s["argmax"]
    maxnote = (f" (`{s['id_col']}={am[0]}`, start={am[1]:.3f} ms)" if am else "")
    p("| metric | value |")
    p("|---|---|")
    p(f"| edges (segments) | {s['n']:,} |")
    p(f"| total edge-time | {s['total_ms']:.3f} ms |")
    p(f"| mean | {s['mean']:.4f} ms |")
    p(f"| min | {s['min']:.4f} ms |")
    p(f"| median (p50~) | {s['p50']:.4f} ms |")
    p(f"| p90~ | {s['p90']:.4f} ms |")
    p(f"| p99~ | {s['p99']:.4f} ms |")
    p(f"| p99.9~ | {s['p999']:.4f} ms |")
    p(f"| max | {s['max']:.3f} ms{maxnote} |")
    p()
    p("_p50..p99.9 are log-histogram estimates; min/mean/max/total are exact._")
    p()
    p("### histogram (log decades)")
    p()
    p("| range (ms) | edges | pct | cumulative |")
    p("|---|---:|---:|---:|")
    for lo, hi, c, pct, cum in s["hist_rows"]:
        p(f"| [{lo:g}, {hi:g}) | {c:,} | {pct:.1f}% | {cum:.1f}% |")
    p()
    p(f"### top {len(s['top'])} longest single edges")
    p()
    p(f"| rank | {s['id_col']} | duration_ms | start_ms | reliability |")
    p("|---:|---:|---:|---:|:--|")
    for rank, (dur, worker, start, reliable) in enumerate(s["top"], 1):
        rel = "ok" if reliable else "dropped!"
        p(f"| {rank} | {worker} | {dur:.3f} | {start:.3f} | {rel} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="path to *_warp_timeline_working_*.csv")
    ap.add_argument("--variant", choices=["pivot", "orientation"], default="pivot")
    ap.add_argument("--graph", default="Orkut")
    ap.add_argument("--k", default="9")
    ap.add_argument("--profile-dir", default=DEFAULT_PROFILE_DIR)
    ap.add_argument("--top", type=int, default=20,
                    help="number of longest edges to list (default 20)")
    ap.add_argument("--include-dropped", action="store_true",
                    help="include buffer-saturated workers in the distribution "
                         "(default: exclude; their per-edge times are unreliable)")
    ap.add_argument("--markdown", action="store_true",
                    help="emit a GitHub-flavored markdown report (stable layout)")
    ap.add_argument("--out", help="write the report to this file instead of stdout")
    ap.add_argument("--decade-min", type=int, default=DECADE_MIN_DEFAULT,
                    help="lowest histogram decade exponent (10^x ms); default -3")
    ap.add_argument("--decade-max", type=int, default=DECADE_MAX_DEFAULT,
                    help="highest histogram decade exponent (10^x ms); default 4")
    args = ap.parse_args()

    path = _resolve_csv(args)
    dropped, _stats_path = _load_dropped(path)
    res = analyze(path, args.top, dropped, args.include_dropped)
    summary = build_summary(args, path, dropped, res)

    render = render_markdown if args.markdown else render_text
    if args.out:
        with open(args.out, "w") as fh:
            render(summary, fh)
        sys.stderr.write(f"wrote {args.out}\n")
    else:
        render(summary, sys.stdout)


if __name__ == "__main__":
    main()

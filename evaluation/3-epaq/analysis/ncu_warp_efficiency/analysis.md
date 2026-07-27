# Warp-efficiency analysis

Source: `warp_efficiency_summary.csv`, generated from Nsight Compute raw CSVs.
The main metric is `smsp__thread_inst_executed_per_inst_executed.ratio`, i.e.
average active lanes per executed instruction.

## Dispatch probe

| variant | active lanes | kernel time ns | interpretation |
|---|---:|---:|---|
| null_uniform | 32.00 | 623,200 | Full warp execution. Dispatch preserves uniform execution for same function and no branch. |
| state_divergent | 8.40 | 2,388,288 | Roughly 4-way split. Runtime is 3.83x slower than null_uniform. |

The direct `__activemask()` output in the raw probe log also shows
`entry_full=128/128`, `body_full=128/128`, and `exit_full=128/128` for
`null-uniform`, with zero clock spread. The state-divergent control splits at
the switch body as intended.

## fib

| cutoff | queue_1 lanes | queue_3 lanes | delta | queue_1 time ns | queue_3 time ns |
|---:|---:|---:|---:|---:|---:|
| 8 | 16.05 | 20.77 | +4.72 | 237,792 | 8,860,608 |
| 10 | 16.03 | 20.91 | +4.88 | 276,768 | 4,948,928 |
| 12 | 16.01 | 19.95 | +3.94 | 424,032 | 3,729,184 |
| 14 | 15.99 | 18.17 | +2.18 | 752,480 | 3,452,768 |

EPAQ raises warp execution efficiency for fib by roughly 2-5 lanes, depending
on cutoff. That supports the intended mechanism: separating task classes makes
the executed instructions less divergent.

The NCU kernel times should not be used as the performance result here: NCU
replay/profiling overhead and the selected profiling region distort absolute
time, especially for these short fib kernels. Use the existing EPAQ timing CSVs
for performance and this NCU run for lane-utilization evidence.

## N-Queens

| cutoff | queue_1 lanes | queue_2 lanes | delta | queue_1 time ns | queue_2 time ns |
|---:|---:|---:|---:|---:|---:|
| 5 | 4.77 | 4.76 | -0.01 | 115,326,528 | 114,628,832 |
| 7 | 4.07 | 4.23 | +0.16 | 117,104,032 | 111,561,728 |
| 9 | 5.09 | 5.43 | +0.34 | 120,289,472 | 112,037,952 |

NQ does not show a meaningful active-lane improvement from EPAQ. This supports
the interpretation that NQ's weak EPAQ effect is not caused by the dispatch
path failing to preserve uniform batches. More likely causes are class-internal
divergence, task granularity, scheduler/queue cost, or a bottleneck other than
SIMT control-flow divergence.

## CilkSort

Latest run: input size `100000000`, cutoffs `64` and `256`.

| cutoff | queue_1 lanes | queue_3 lanes | delta | queue_1 time ns | queue_3 time ns |
|---:|---:|---:|---:|---:|---:|
| 64 | 14.90 | 16.05 | +1.15 | 105,919,584 | 106,731,616 |
| 256 | 14.88 | 16.06 | +1.18 | 105,982,880 | 106,997,536 |

CilkSort shows a small active-lane improvement from EPAQ, about +1.1 lanes in
this run. This is much weaker than fib, but unlike N-Queens it is not zero.

The NCU kernel times are roughly similar between queue_1 and queue_3, with
queue_3 slightly longer in this profiling run. As with fib, these NCU durations
should not be treated as the definitive performance result. They are useful
mainly as context for the warp-efficiency measurement.

Interpretation: CilkSort is an intermediate case. EPAQ improves SIMT lane
utilization a little, but the queue partitioning does not expose as clean a
control-flow win as fib. Remaining costs likely come from merge/sort internals,
memory traffic, task overhead, and class-internal divergence that queue
separation does not eliminate.

## Takeaway

The mechanism-level concern is resolved in favor of GTAP: a uniform batch really
can execute as a full warp through the dispatch path. EPAQ's observed scope is
therefore an algorithm/workload issue, not evidence of hidden divergence in the
indirect-call dispatch itself.

Across the real benchmarks, the active-lane evidence is:

- fib: clear improvement from EPAQ.
- CilkSort: small improvement from EPAQ.
- N-Queens: almost no improvement from EPAQ.

This supports the current interpretation that EPAQ helps when queue separation
actually groups tasks with similar control flow, but it cannot fix divergence
that remains inside each task body or bottlenecks unrelated to SIMT lane
utilization.

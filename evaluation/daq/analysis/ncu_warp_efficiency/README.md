# Nsight Compute warp-efficiency experiment

This experiment profiles average active lanes per executed instruction for:

- the artificial dispatch probe:
  - `null-uniform`
  - `state-divergent`
- fib DAQ off/on:
  - `fib_queue_1`
  - `fib_queue_3`
- N-Queens DAQ off/on:
  - `nq_queue_1`
  - `nq_queue_2`
- CilkSort DAQ off/on:
  - `cilksort_queue_1`
  - `cilksort_queue_3`

Run from a GPU node with Nsight Compute:

```sh
cd evaluation/daq/analysis/ncu_warp_efficiency
./run_ncu_warp_efficiency.sh
```

The summary CSV is:

```text
evaluation/daq/analysis/ncu_warp_efficiency/warp_efficiency_summary.csv
```

Raw Nsight Compute CSV files are kept under:

```text
evaluation/daq/analysis/ncu_warp_efficiency/raw/
```

Default inputs:

- dispatch probe: `compute_iters=256`, `repeats=128`
- fib: `n=40`, cutoffs `8 10 12 14`
- N-Queens: `n=16`, cutoffs `5 7 9`
- CilkSort: `n=1000000`, cutoffs `64 256 1024`

Useful overrides:

```sh
BENCHES=probe ./run_ncu_warp_efficiency.sh
BENCHES=fib FIB_CUTOFFS="10 12" ./run_ncu_warp_efficiency.sh
BENCHES=nq NQ_CUTOFFS="7" ./run_ncu_warp_efficiency.sh
BENCHES=cilksort CILKSORT_SIZE=1000000 CILKSORT_CUTOFFS="64 256" ./run_ncu_warp_efficiency.sh
NCU_EXTRA_ARGS="--kernel-name regex:exec_kernel" ./run_ncu_warp_efficiency.sh
```

Primary metric:

```text
smsp__thread_inst_executed_per_inst_executed.ratio
```

Interpretation:

- `dispatch_probe/null_uniform` should be close to full-lane execution.
- `dispatch_probe/state_divergent` should be much lower and slower, validating
  the control.
- If fib queue_3 improves this metric over queue_1, DAQ is improving SIMT
  lane utilization where it improves time.
- If NQ queue_2 does not improve this metric, NQ's weak DAQ result is likely
  not caused by the dispatch path itself. The remaining candidates are
  class-internal divergence, task granularity, queue/scheduler cost, or a
  non-divergence bottleneck.

#!/bin/bash
# Rebuild and measure the FMM 3-way end-to-end stack (default N=5e7, theta=0.2).
# Optional env: N THETA NUM_RUNS GTAP_MAX_TASKS_PER_WARP GTAP_GRID_SIZE

set -euo pipefail

FMM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1
cd "$FMM_ROOT" || exit 1

N="${N:-50000000}"
THETA="${THETA:-0.2}"
NUM_RUNS="${NUM_RUNS:-20}"
export NUM_RUNS
export GTAP_MAX_TASKS_PER_WARP="${GTAP_MAX_TASKS_PER_WARP:-250000}"
export GTAP_GRID_SIZE="${GTAP_GRID_SIZE:-3168}"

echo "=== FMM e2e 3-way stack: N=${N} theta=${THETA} NUM_RUNS=${NUM_RUNS} GTAP_MAX_TASKS_PER_WARP=${GTAP_MAX_TASKS_PER_WARP} GTAP_GRID_SIZE=${GTAP_GRID_SIZE} ==="
date
hostname
nvidia-smi -L || true

./scripts/stack/plot_fmm_dtt_stack.sh --rebuild "n=${N}" "theta=${THETA}"

echo "=== Done ==="
date
ls -la "csv/fmm_3way_stack_N${N}_theta"*summary.csv || true

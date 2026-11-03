#!/bin/bash
# Pivot vs KCGPU on DBLP / as-Skitter / Orkut (k=5,7,9).
# Reuses an existing KCGPU binary. Writes per-run CSV; paper table is built
# by plot from runs.
#
#   run → k579/results/pivot_runs.csv
#   plot kclique → k579/results/e2e_table.csv (+ LaTeX)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
K_CLIQUE_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
K579_DIR="$K_CLIQUE_DIR/k579"
cd "$K_CLIQUE_DIR" || exit 1

K_VALUES=${K_VALUES:-"5 7 9"}
GRAPHS=${GRAPHS:-"DBLP as-Skitter Orkut"}
REPEATS=${REPEATS:-20}

export KCGPU_PROFILE="${KCGPU_PROFILE:-0}"
export GTAP_ENABLE_PROFILING="${GTAP_ENABLE_PROFILING:-0}"
export KCGPU_FORCE_REBUILD="${KCGPU_FORCE_REBUILD:-0}"
export BUILD_KCGPU="${BUILD_KCGPU:-0}"
export BUILD_GTAP="${BUILD_GTAP:-1}"

export GTAP_VARIANT=pivot
export KCGPU_ORIENT="${KCGPU_ORIENT:-degen}"
export GTAP_ORIENT="${GTAP_ORIENT:-degen}"
export KCGPU_VARIANT_SET="${KCGPU_VARIANT_SET:-pivot_binary}"
export LOG_DIR="${LOG_DIR:-$K579_DIR/logs/pivot_e2e}"
export RESULTS_FILE="${RESULTS_FILE:-$K579_DIR/results/pivot_runs.csv}"

mkdir -p "$LOG_DIR" "$(dirname "$RESULTS_FILE")"

env \
    K_VALUES="$K_VALUES" \
    GRAPHS="$GRAPHS" \
    REPEATS="$REPEATS" \
    GTAP_VARIANT="$GTAP_VARIANT" \
    KCGPU_ORIENT="$KCGPU_ORIENT" \
    GTAP_ORIENT="$GTAP_ORIENT" \
    KCGPU_PROFILE="$KCGPU_PROFILE" \
    GTAP_ENABLE_PROFILING="$GTAP_ENABLE_PROFILING" \
    KCGPU_VARIANT_SET="$KCGPU_VARIANT_SET" \
    KCGPU_FORCE_REBUILD="$KCGPU_FORCE_REBUILD" \
    BUILD_KCGPU="$BUILD_KCGPU" \
    BUILD_GTAP="$BUILD_GTAP" \
    RESULTS_FILE="$RESULTS_FILE" \
    LOG_DIR="$LOG_DIR" \
    "$SCRIPT_DIR/../common/run_benchmark.sh"

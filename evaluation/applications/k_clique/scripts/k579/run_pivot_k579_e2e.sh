#!/bin/bash
# Pivot vs KCGPU on DBLP / as-Skitter / Orkut (k=5,7,9).
# Reuses an existing KCGPU binary; writes k579/results/*e2e*.csv.

set -euo pipefail

K_CLIQUE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1
cd "$K_CLIQUE_DIR" || exit 1

export KCGPU_PROFILE=0
export GTAP_ENABLE_PROFILING=0
export KCGPU_FORCE_REBUILD=0
export BUILD_KCGPU=0
export BUILD_GTAP=1

export RESULTS_FILE="$PWD/k579/results/gtap_kcgpu_pivot_k579_e2e_results.csv"
export AVG_RESULTS_FILE="$PWD/k579/results/gtap_kcgpu_pivot_k579_e2e_results_avg.csv"
export BEST_SUMMARY_FILE="$PWD/k579/results/gtap_kcgpu_pivot_k579_e2e_best_summary.csv"
export LOG_DIR="$PWD/k579/logs/pivot_e2e"

exec bash ./scripts/k579/run_kcgpu_pivot_k579_graph_tuned.sh "$@"

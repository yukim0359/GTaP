#!/bin/bash
# GTAP orientation vs KCGPU (degree-oriented, binary encode) on DBLP / as-Skitter / Orkut
# for k=5,7,9. GTAP build params come from gtap_graph_k_config.csv (per graph×k).
#
# KCGPU variants: edge o1b o2b o4b o8b (oriented_binary)
# Runs REPEATS=20 by default and writes averaged results via summarize_k579_repeats.py.
#
# Usage:
#   ./run_kcgpu_orientation_k579_graph_tuned.sh
#   RUN_GTAP=0 ./run_kcgpu_orientation_k579_graph_tuned.sh   # KCGPU only
#   RUN_KCGPU=0 ./run_kcgpu_orientation_k579_graph_tuned.sh  # GTAP only

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
K_CLIQUE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
K579_DIR="$K_CLIQUE_DIR/k579"

K_VALUES=${K_VALUES:-"5 7 9"}
GRAPHS=${GRAPHS:-"DBLP as-Skitter Orkut"}
REPEATS=${REPEATS:-20}
GTAP_VARIANT=orientation
KCGPU_ORIENT=${KCGPU_ORIENT:-degree}
GTAP_ORIENT=${GTAP_ORIENT:-degree}
KCGPU_PROFILE=${KCGPU_PROFILE:-0}
GTAP_PROFILE=${GTAP_PROFILE:-0}
KCGPU_VARIANT_SET=${KCGPU_VARIANT_SET:-oriented_binary}
RESULTS_FILE=${RESULTS_FILE:-"$K579_DIR/results/gtap_kcgpu_orientation_k579_results.csv"}
AVG_RESULTS_FILE=${AVG_RESULTS_FILE:-"$K579_DIR/results/gtap_kcgpu_orientation_k579_results_avg.csv"}
BEST_SUMMARY_FILE=${BEST_SUMMARY_FILE:-"$K579_DIR/results/gtap_kcgpu_orientation_k579_best_summary.csv"}
LOG_DIR=${LOG_DIR:-"$K579_DIR/logs/orientation"}
SUMMARIZE=${SUMMARIZE:-1}

export GTAP_ORIENT

env \
    K_VALUES="$K_VALUES" \
    GRAPHS="$GRAPHS" \
    REPEATS="$REPEATS" \
    GTAP_VARIANT="$GTAP_VARIANT" \
    KCGPU_ORIENT="$KCGPU_ORIENT" \
    GTAP_ORIENT="$GTAP_ORIENT" \
    KCGPU_PROFILE="$KCGPU_PROFILE" \
    GTAP_PROFILE="$GTAP_PROFILE" \
    KCGPU_VARIANT_SET="$KCGPU_VARIANT_SET" \
    RESULTS_FILE="$RESULTS_FILE" \
    LOG_DIR="$LOG_DIR" \
    "$SCRIPT_DIR/run_benchmark.sh"

if [ "$SUMMARIZE" = "1" ]; then
    python3 "$SCRIPT_DIR/summarize_k579_repeats.py" \
        -i "$RESULTS_FILE" \
        -o "$AVG_RESULTS_FILE"
    python3 "$SCRIPT_DIR/summarize_k579_best.py" \
        --only orientation \
        --orientation-csv "$AVG_RESULTS_FILE" \
        -o "$BEST_SUMMARY_FILE"
fi

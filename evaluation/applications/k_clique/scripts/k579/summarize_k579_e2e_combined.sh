#!/bin/bash
# Build the paper e2e table from per-run CSVs.
#
#   orientation_runs.csv + pivot_runs.csv → e2e_table.csv → LaTeX
#
# Usage:
#   ./scripts/k579/summarize_k579_e2e_combined.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
K579_DIR=$(cd "$SCRIPT_DIR/../../k579" && pwd)
RESULTS_DIR="$K579_DIR/results"

ORIENT_RUNS="${ORIENT_RUNS:-$RESULTS_DIR/orientation_runs.csv}"
PIVOT_RUNS="${PIVOT_RUNS:-$RESULTS_DIR/pivot_runs.csv}"
E2E_TABLE="${E2E_TABLE:-$RESULTS_DIR/e2e_table.csv}"
LATEX_TABLE="${LATEX_TABLE:-$K579_DIR/latex/k579_benchmark_tables.tex}"

mkdir -p "$RESULTS_DIR" "$(dirname "$LATEX_TABLE")"

python3 "$SCRIPT_DIR/build_e2e_table.py" \
    --orientation-runs "$ORIENT_RUNS" \
    --pivot-runs "$PIVOT_RUNS" \
    -o "$E2E_TABLE"

python3 "$SCRIPT_DIR/format_k579_latex_table.py" \
    -i "$E2E_TABLE" \
    -o "$LATEX_TABLE"

echo "Wrote $E2E_TABLE"
echo "Wrote $LATEX_TABLE"

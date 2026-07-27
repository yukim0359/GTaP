#!/bin/bash
# Re-parse e2e logs (GTaP e2e includes gtap_initialize) and build combined summary + LaTeX.
#
# Usage (after both PBS jobs finish):
#   ./scripts/summarize_k579_e2e_combined.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
K579_DIR=$(cd "$SCRIPT_DIR/../k579" && pwd)
RESULTS_DIR="$K579_DIR/results"

COMBINED_SUMMARY="$RESULTS_DIR/gtap_kcgpu_k579_e2e_init_included_best_summary.csv"
MATRIX_CSV="$RESULTS_DIR/gtap_kcgpu_k579_e2e_init_included_best_matrix.csv"
LATEX_TABLE="$K579_DIR/latex/k579_benchmark_tables.tex"

python3 "$SCRIPT_DIR/reparse_k579_e2e_logs.py" --only both

python3 "$SCRIPT_DIR/format_k579_best_matrix.py" \
    -i "$COMBINED_SUMMARY" \
    -o "$MATRIX_CSV"

python3 "$SCRIPT_DIR/format_k579_latex_table.py" \
    -i "$COMBINED_SUMMARY" \
    -o "$LATEX_TABLE"

echo "Wrote $COMBINED_SUMMARY"
echo "Wrote $MATRIX_CSV"
echo "Wrote $LATEX_TABLE"

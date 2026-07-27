#!/bin/bash
# Submit k579 e2e benchmark jobs (orientation + pivot) on the GPU queue.
#
# Usage:
#   ./scripts/submit_k579_e2e.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
K_CLIQUE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

cd "$K_CLIQUE_DIR"
mkdir -p k579/results k579/logs/orientation_e2e k579/logs/pivot_e2e

if ! command -v qsub >/dev/null 2>&1; then
    echo "qsub not found. Run PBS scripts manually on a GPU node:" >&2
    echo "  bash scripts/run_orientation_k579_e2e.pbs" >&2
    echo "  bash scripts/run_pivot_k579_e2e.pbs" >&2
    exit 1
fi

orient_job=$(qsub "$SCRIPT_DIR/run_orientation_k579_e2e.pbs")
pivot_job=$(qsub "$SCRIPT_DIR/run_pivot_k579_e2e.pbs")

echo "Submitted orientation: $orient_job"
echo "Submitted pivot:       $pivot_job (parallel, no depend)"
echo
echo "After both finish:"
echo "  ./scripts/summarize_k579_e2e_combined.sh"

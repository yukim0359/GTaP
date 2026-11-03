#!/bin/bash
# Build, measure, and draw the FMM 3-way end-to-end stack (N=5e7, theta=0.2).
# Writes csv/fmm_3way_stack_N50000000_theta02_{summary,runs}.csv
# and evaluation/img/fmm_3way_stack_N50000000_theta02.pdf.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
FMM_ROOT="$(cd "$script_dir/../.." && pwd)" || exit 1
cd "$FMM_ROOT" || exit 1
# shellcheck source=../omp_host_dtt_env.sh
source "$script_dir/../omp_host_dtt_env.sh"

N=50000000
THETA=0.2
BIN_DIR="bin/N50000000/theta02"
CSV_PREFIX="csv/fmm_3way_stack_N50000000_theta02"
NUM_RUNS="${NUM_RUNS:-20}"

echo "=== FMM e2e 3-way stack: N=${N} theta=${THETA} NUM_RUNS=${NUM_RUNS} ==="
date
hostname
nvidia-smi -L || true

echo "=== Rebuilding theta=${THETA} (N=${N}) ==="
unset GTAP_MAX_TASKS_PER_WARP || true
"$script_dir/../dtt_theta/build_fmm_dtt_theta_bins.sh" "$THETA" "$N"
unset GTAP_MAX_TASKS_PER_WARP || true

for bin in gtap_dtt_fmm worklist_dtt_fmm omp_dtt_fmm; do
  if [[ ! -x "$BIN_DIR/$bin" ]]; then
    echo "Error: missing $BIN_DIR/$bin" >&2
    exit 1
  fi
done

echo "Measuring N=${N} theta=${THETA} (bins from ${BIN_DIR})"
python3 "$script_dir/fmm_phase_stack.py" \
  --n "$N" \
  --theta "$THETA" \
  --runs "$NUM_RUNS" \
  --bin "$BIN_DIR/gtap_dtt_fmm" \
  --label "GTaP DTT" \
  --bin "$BIN_DIR/worklist_dtt_fmm" \
  --label "GPU worklist DTT" \
  --bin "$BIN_DIR/omp_dtt_fmm" \
  --label "Host OpenMP DTT" \
  --csv-out "$CSV_PREFIX" \
  --csv-only \
  --include-init

python3 "$FMM_ROOT/../../scripts/plot_fmm_3way_stack_paper.py" --n "$N" --theta "$THETA"

echo "=== Done ==="
date
ls -la csv/fmm_3way_stack_N50000000_theta02_summary.csv

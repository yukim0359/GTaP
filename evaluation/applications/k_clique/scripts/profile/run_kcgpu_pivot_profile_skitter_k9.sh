#!/bin/bash
# Paper Fig.7 (KCGPU): as-Skitter, k=9, pivoting SM working timeline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K_CLIQUE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_DIR="$(cd "$K_CLIQUE_DIR/../.." && pwd)"
cd "$K_CLIQUE_DIR" || exit 1

export KCGPU_SM_TIMELINE_ONLY=1
export KCGPU_VARIANT=pivot
export KCGPU_ORIENT="${KCGPU_ORIENT:-degen}"
export KCGPU_Q="${KCGPU_Q:-p1b}"
export GRAPH_NAME=as-Skitter
export K=9
export FIG_FORMAT=pdf
# Keep tagged sweep outputs under k_clique/img/kcgpu/; paper name goes to evaluation/img/.
export IMG_ROOT="${IMG_ROOT:-$K_CLIQUE_DIR/img/kcgpu}"
OUTPUT_PDF="$EVAL_DIR/img/kcgpu_pivot_sm_timeline.pdf"

./scripts/profile/run_kcgpu_profile.sh

tag="_${GRAPH_NAME}_k${K}_edge_${KCGPU_ORIENT}_${KCGPU_Q}"
generated="$IMG_ROOT/kcgpu_pivot${tag}_sm_timeline.${FIG_FORMAT}"
if [ ! -f "$generated" ]; then
    echo "ERROR: expected figure missing: $generated" >&2
    exit 1
fi
mkdir -p "$(dirname "$OUTPUT_PDF")"
cp -f "$generated" "$OUTPUT_PDF"
echo "Copied paper figure: $OUTPUT_PDF"

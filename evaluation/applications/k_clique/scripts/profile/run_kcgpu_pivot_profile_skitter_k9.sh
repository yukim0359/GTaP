#!/bin/bash
# Paper Fig.7 (KCGPU): measure as-Skitter, k=9, pivoting profile dump only.
# Draw the paper PDF with: ./eval.sh plot kclique-timeline-kcgpu
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K_CLIQUE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$K_CLIQUE_DIR" || exit 1

export KCGPU_SM_TIMELINE_ONLY=1
export KCGPU_VARIANT=pivot
export KCGPU_ORIENT="${KCGPU_ORIENT:-degen}"
export KCGPU_Q="${KCGPU_Q:-p1b}"
export GRAPH_NAME=as-Skitter
export K=9
export KCGPU_MAKE_FIGURES=0

./scripts/profile/run_kcgpu_profile.sh

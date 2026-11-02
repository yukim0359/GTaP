#!/bin/bash

# Run compute_iters sweeps for K=2 and K=4.
# Usage: qsub compare_hetero_compute_k24.pbs
#        or: bash compare_hetero_compute_k24.sh

COMPARE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
cd "$COMPARE_DIR" || exit 1
K_VALUES=(${K_VALUES:-2 4})
DEPTH=${DEPTH:-25}
NUM_RUNS=${NUM_RUNS:-20}

echo "=== hetero_tree compute sweep for K in: ${K_VALUES[*]} (D=${DEPTH}, runs=${NUM_RUNS}) ==="

for k in "${K_VALUES[@]}"; do
    echo
    echo "########## HETERO_TREE_K=${k} ##########"
    HETERO_TREE_K="$k" DEPTH="$DEPTH" NUM_RUNS="$NUM_RUNS" \
        bash "$COMPARE_DIR/compare_hetero_compute.sh" || exit 1
done

echo "Done."

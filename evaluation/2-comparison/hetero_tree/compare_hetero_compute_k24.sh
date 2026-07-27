#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -W group_list=gc64
#PBS -j oe

# Run compute_iters sweeps for K=2 and K=4, then generate plots.
# Usage: qsub compare_hetero_compute_k24.sh
#        or: bash compare_hetero_compute_k24.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PBS_O_WORKDIR:-$SCRIPT_DIR}"

COMPARE_DIR=$(pwd)
K_VALUES=(${K_VALUES:-2 4})
DEPTH=${DEPTH:-25}
NUM_RUNS=${NUM_RUNS:-20}

echo "=== hetero_tree compute sweep for K in: ${K_VALUES[*]} (D=${DEPTH}, runs=${NUM_RUNS}) ==="

for k in "${K_VALUES[@]}"; do
    echo
    echo "########## HETERO_TREE_K=${k} ##########"
    HETERO_TREE_K="$k" DEPTH="$DEPTH" NUM_RUNS="$NUM_RUNS" \
        bash "$COMPARE_DIR/compare_hetero_compute.sh"
done

echo
echo "=== Plotting ==="
cd "$COMPARE_DIR"
python3 plot_performance_hetero_tree.py

echo "Done."

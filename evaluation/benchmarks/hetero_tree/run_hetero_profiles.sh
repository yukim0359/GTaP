#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -W group_list=gc64
#PBS -j oe

# Collect warp/block working-timeline CSVs for all 4 hetero_tree impls.
# Usage:
#   make -C hetero_tree run-profile
#   DEPTH=25 COMPUTE=4096 bash run_hetero_profiles.sh
# Requires GTAP_ENABLE_PROFILING-enabled binaries (make GTAP_ENABLE_PROFILING=1 -B all).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PBS_O_WORKDIR:-$SCRIPT_DIR}"

COMPARE_DIR=$(pwd)
BIN_DIR="$COMPARE_DIR/bin"
PROFILE_DIR="$COMPARE_DIR/profile"
mkdir -p "$PROFILE_DIR" "$BIN_DIR"

DEPTH=${DEPTH:-25}
COMPUTE=${COMPUTE:-4096}

echo "Building GTAP_ENABLE_PROFILING binaries (GTAP_ENABLE_PROFILING=1) ..."
make -C "$COMPARE_DIR" -B all GTAP_ENABLE_PROFILING=1

run_one() {
    local bin=$1
    local label=$2
    if [ ! -x "$bin" ]; then
        echo "ERROR: missing $bin" >&2
        return 1
    fi
    echo "=== $label: $bin $DEPTH $COMPUTE ==="
    # gtap_export_profile writes ./profile/<app>_*.csv relative to CWD
    "$bin" "$DEPTH" "$COMPUTE"
    echo
}

run_one "$BIN_DIR/gtap_thread_hetero_tree"        "thread wo DAQ"
run_one "$BIN_DIR/gtap_thread_hetero_tree_daq"    "thread DAQ"
run_one "$BIN_DIR/gtap_block_hetero_tree"         "block"
run_one "$BIN_DIR/gtap_block_hetero_tree_cutoff"  "block cutoff"

echo "Profile CSVs under $PROFILE_DIR:"
ls -la "$PROFILE_DIR"/*hetero_tree* 2>/dev/null || ls -la "$PROFILE_DIR"
echo
echo "Plot with:"
echo "  python3 ../../scripts/plot_hetero_tree_timeline_paper.py"
echo "  # or: make -C . plot-profile"

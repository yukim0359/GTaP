#!/bin/bash

# compute_iters sweep for hetero_tree (4 impls): thread wo/DAQ, block, block-cutoff.
# Usage:
#   HETERO_TREE_K=2 qsub -v HETERO_TREE_K compare_hetero_compute.pbs
#   HETERO_TREE_K=4 bash compare_hetero_compute.sh
# Writes: hetero_tree_compute_results_K${HETERO_TREE_K}.csv

BENCHMARK_NAME=hetero_tree

COMPARE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
cd "$COMPARE_DIR" || exit 1
source "$COMPARE_DIR/../../common/benchmark.sh" || exit 1
BIN_DIR="$COMPARE_DIR/bin"
mkdir -p "$BIN_DIR"

HETERO_TREE_K=${HETERO_TREE_K:-2}
DEPTH=${DEPTH:-25}
NUM_RUNS=${NUM_RUNS:-20}

echo "Building hetero_tree binaries (K=${HETERO_TREE_K}) ..."
make -C "$COMPARE_DIR" -B all HETERO_TREE_K="$HETERO_TREE_K" || exit 1

# compute_iters sweep: D fixed.
COMPUTE_VALUES=(64 128 256 512 1024 2048 4096 8192 16384 32768)

RESULTS_FILE="$COMPARE_DIR/${BENCHMARK_NAME}_compute_results_K${HETERO_TREE_K}.csv"
echo "compute_iters,GTAP_thread_med,GTAP_thread_err_low,GTAP_thread_err_high,GTAP_thread_daq_med,GTAP_thread_daq_err_low,GTAP_thread_daq_err_high,GTAP_block_med,GTAP_block_err_low,GTAP_block_err_high,GTAP_block_cutoff_med,GTAP_block_cutoff_err_low,GTAP_block_cutoff_err_high" > "$RESULTS_FILE"

echo "K=${HETERO_TREE_K} DEPTH=${DEPTH} NUM_RUNS=${NUM_RUNS}"
printf "%12s | %25s | %25s | %25s | %25s\n" \
    "compute_iters" "thread wo (ms)" "thread DAQ (ms)" "block (ms)" "block cutoff (ms)"
printf "%12s-+-%25s-+-%25s-+-%25s-+-%25s\n" \
    "------------" "-------------------------" "-------------------------" \
    "-------------------------" "-------------------------"

for compute in "${COMPUTE_VALUES[@]}"; do
    TH_MED=0 TH_ELO=0 TH_EHI=0
    if [ -x "$BIN_DIR/gtap_thread_hetero_tree" ]; then
        read TH_MED _ _ TH_ELO TH_EHI < <(
            benchmark_run_stats "$NUM_RUNS" benchmark_parse_execution_ms \
                "$BIN_DIR/gtap_thread_hetero_tree" "$DEPTH" "$compute"
        )
    fi

    DAQ_MED=0 DAQ_ELO=0 DAQ_EHI=0
    if [ -x "$BIN_DIR/gtap_thread_hetero_tree_daq" ]; then
        read DAQ_MED _ _ DAQ_ELO DAQ_EHI < <(
            benchmark_run_stats "$NUM_RUNS" benchmark_parse_execution_ms \
                "$BIN_DIR/gtap_thread_hetero_tree_daq" "$DEPTH" "$compute"
        )
    fi

    BLK_MED=0 BLK_ELO=0 BLK_EHI=0
    if [ -x "$BIN_DIR/gtap_block_hetero_tree" ]; then
        read BLK_MED _ _ BLK_ELO BLK_EHI < <(
            benchmark_run_stats "$NUM_RUNS" benchmark_parse_execution_ms \
                "$BIN_DIR/gtap_block_hetero_tree" "$DEPTH" "$compute"
        )
    fi

    CUT_MED=0 CUT_ELO=0 CUT_EHI=0
    if [ -x "$BIN_DIR/gtap_block_hetero_tree_cutoff" ]; then
        read CUT_MED _ _ CUT_ELO CUT_EHI < <(
            benchmark_run_stats "$NUM_RUNS" benchmark_parse_execution_ms \
                "$BIN_DIR/gtap_block_hetero_tree_cutoff" "$DEPTH" "$compute"
        )
    fi

    printf "%12d | " "$compute"
    benchmark_fmt_med_iqr "$TH_MED" "$TH_ELO" "$TH_EHI"; printf " | "
    benchmark_fmt_med_iqr "$DAQ_MED" "$DAQ_ELO" "$DAQ_EHI"; printf " | "
    benchmark_fmt_med_iqr "$BLK_MED" "$BLK_ELO" "$BLK_EHI"; printf " | "
    benchmark_fmt_med_iqr "$CUT_MED" "$CUT_ELO" "$CUT_EHI"; printf "\n"

    echo "$compute,$TH_MED,$TH_ELO,$TH_EHI,$DAQ_MED,$DAQ_ELO,$DAQ_EHI,$BLK_MED,$BLK_ELO,$BLK_EHI,$CUT_MED,$CUT_ELO,$CUT_EHI" >> "$RESULTS_FILE"
done

echo "Wrote $RESULTS_FILE"

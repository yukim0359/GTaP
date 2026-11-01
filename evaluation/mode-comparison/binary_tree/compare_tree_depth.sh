#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -W group_list=gc64
#PBS -j oe

# Depth sweep: thread / block / block-cutoff (K=1 binary tree).
BENCHMARK_NAME=binary_tree

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PBS_O_WORKDIR:-$SCRIPT_DIR}"

COMPARE_DIR=$(pwd)
source "$COMPARE_DIR/../../common/benchmark.sh"
BIN_DIR="$COMPARE_DIR/bin"
mkdir -p "$BIN_DIR"

echo "Building GTaP binaries (force rebuild for updated runtime) ..."
make -C "$COMPARE_DIR" -B gtap_thread gtap_block gtap_block_cutoff

DEPTH_VALUES=(12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28)
COMPUTE=${COMPUTE:-2048}
MEMORY=${MEMORY:-0}
NUM_RUNS=${NUM_RUNS:-20}

RESULTS_FILE="$COMPARE_DIR/${BENCHMARK_NAME}_depth_results.csv"
echo "n,GTAP_thread_med,GTAP_thread_err_low,GTAP_thread_err_high,GTAP_block_med,GTAP_block_err_low,GTAP_block_err_high,GTAP_block_cutoff_med,GTAP_block_cutoff_err_low,GTAP_block_cutoff_err_high" > "$RESULTS_FILE"

printf "%6s | %25s | %25s | %25s\n" "n" "thread (ms)" "block (ms)" "block cutoff (ms)"
printf "%6s-+-%25s-+-%25s-+-%25s\n" \
    "------" "-------------------------" "-------------------------" "-------------------------"

for depth in "${DEPTH_VALUES[@]}"; do
    TH_MED=0 TH_ELO=0 TH_EHI=0
    if [ -x "$BIN_DIR/gtap_thread_$BENCHMARK_NAME" ]; then
        read TH_MED _ _ TH_ELO TH_EHI < <(
            benchmark_run_stats "$NUM_RUNS" benchmark_parse_execution_ms \
                "$BIN_DIR/gtap_thread_$BENCHMARK_NAME" "$depth" "$COMPUTE" "$MEMORY"
        )
    fi

    BLK_MED=0 BLK_ELO=0 BLK_EHI=0
    if [ -x "$BIN_DIR/gtap_block_$BENCHMARK_NAME" ]; then
        read BLK_MED _ _ BLK_ELO BLK_EHI < <(
            benchmark_run_stats "$NUM_RUNS" benchmark_parse_execution_ms \
                "$BIN_DIR/gtap_block_$BENCHMARK_NAME" "$depth" "$COMPUTE" "$MEMORY"
        )
    fi

    CUT_MED=0 CUT_ELO=0 CUT_EHI=0
    if [ -x "$BIN_DIR/gtap_block_${BENCHMARK_NAME}_cutoff" ]; then
        read CUT_MED _ _ CUT_ELO CUT_EHI < <(
            benchmark_run_stats "$NUM_RUNS" benchmark_parse_execution_ms \
                "$BIN_DIR/gtap_block_${BENCHMARK_NAME}_cutoff" "$depth" "$COMPUTE" "$MEMORY"
        )
    fi

    printf "%6d | " "$depth"
    benchmark_fmt_med_iqr "$TH_MED" "$TH_ELO" "$TH_EHI"; printf " | "
    benchmark_fmt_med_iqr "$BLK_MED" "$BLK_ELO" "$BLK_EHI"; printf " | "
    benchmark_fmt_med_iqr "$CUT_MED" "$CUT_ELO" "$CUT_EHI"; printf "\n"

    echo "$depth,$TH_MED,$TH_ELO,$TH_EHI,$BLK_MED,$BLK_ELO,$BLK_EHI,$CUT_MED,$CUT_ELO,$CUT_EHI" >> "$RESULTS_FILE"
done

echo "Wrote $RESULTS_FILE"

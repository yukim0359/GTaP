#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -W group_list=gc64
#PBS -j oe

# compute_iters sweep: thread / block / block-cutoff (K=1 binary tree).
BENCHMARK_NAME=binary_tree

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PBS_O_WORKDIR:-$SCRIPT_DIR}"

COMPARE_DIR=$(pwd)
source "$COMPARE_DIR/../../common/benchmark.sh"
BIN_DIR="$COMPARE_DIR/bin"
mkdir -p "$BIN_DIR"

echo "Building GTaP binaries (force rebuild for updated runtime) ..."
make -C "$COMPARE_DIR" -B gtap_thread gtap_block gtap_block_cutoff

COMPUTE_VALUES=(64 128 256 512 1024 2048 4096 8192 16384 32768)
DEPTH=${DEPTH:-25}
MEMORY=${MEMORY:-0}
NUM_RUNS=${NUM_RUNS:-20}

RESULTS_FILE="$COMPARE_DIR/${BENCHMARK_NAME}_compute_results.csv"
echo "compute_iters,GTAP_thread_med,GTAP_thread_err_low,GTAP_thread_err_high,GTAP_block_med,GTAP_block_err_low,GTAP_block_err_high,GTAP_block_cutoff_med,GTAP_block_cutoff_err_low,GTAP_block_cutoff_err_high" > "$RESULTS_FILE"

printf "%12s | %25s | %25s | %25s\n" "compute_iters" "thread (ms)" "block (ms)" "block cutoff (ms)"
printf "%12s-+-%25s-+-%25s-+-%25s\n" \
    "------------" "-------------------------" "-------------------------" "-------------------------"

for compute in "${COMPUTE_VALUES[@]}"; do
    TH_MED=0 TH_ELO=0 TH_EHI=0
    if [ -x "$BIN_DIR/gtap_thread_$BENCHMARK_NAME" ]; then
        read TH_MED _ _ TH_ELO TH_EHI < <(
            benchmark_run_stats "$NUM_RUNS" benchmark_parse_execution_ms \
                "$BIN_DIR/gtap_thread_$BENCHMARK_NAME" "$DEPTH" "$compute" "$MEMORY"
        )
    fi

    BLK_MED=0 BLK_ELO=0 BLK_EHI=0
    if [ -x "$BIN_DIR/gtap_block_$BENCHMARK_NAME" ]; then
        read BLK_MED _ _ BLK_ELO BLK_EHI < <(
            benchmark_run_stats "$NUM_RUNS" benchmark_parse_execution_ms \
                "$BIN_DIR/gtap_block_$BENCHMARK_NAME" "$DEPTH" "$compute" "$MEMORY"
        )
    fi

    CUT_MED=0 CUT_ELO=0 CUT_EHI=0
    if [ -x "$BIN_DIR/gtap_block_${BENCHMARK_NAME}_cutoff" ]; then
        read CUT_MED _ _ CUT_ELO CUT_EHI < <(
            benchmark_run_stats "$NUM_RUNS" benchmark_parse_execution_ms \
                "$BIN_DIR/gtap_block_${BENCHMARK_NAME}_cutoff" "$DEPTH" "$compute" "$MEMORY"
        )
    fi

    printf "%12d | " "$compute"
    benchmark_fmt_med_iqr "$TH_MED" "$TH_ELO" "$TH_EHI"; printf " | "
    benchmark_fmt_med_iqr "$BLK_MED" "$BLK_ELO" "$BLK_EHI"; printf " | "
    benchmark_fmt_med_iqr "$CUT_MED" "$CUT_ELO" "$CUT_EHI"; printf "\n"

    echo "$compute,$TH_MED,$TH_ELO,$TH_EHI,$BLK_MED,$BLK_ELO,$BLK_EHI,$CUT_MED,$CUT_ELO,$CUT_EHI" >> "$RESULTS_FILE"
done

echo "Wrote $RESULTS_FILE"

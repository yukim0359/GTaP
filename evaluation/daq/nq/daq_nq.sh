#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -W group_list=gc64
#PBS -j oe

BENCHMARK_NAME=nq

cd "$PBS_O_WORKDIR"

COMPARE_DIR=$(pwd)
source "$COMPARE_DIR/../../common/benchmark.sh"
PROJECT_ROOT=$(cd "$COMPARE_DIR" && pwd)
BIN_DIR="$PROJECT_ROOT/bin"
mkdir -p "$BIN_DIR"

echo "Building DAQ nq binaries (force rebuild for updated runtime) ..."
make -C "$COMPARE_DIR" -B all

# Test values for cutoff depth
CUTOFF_DEPTHS=(3 4 5 6 7 8 9 10)

# Fixed n value for testing
N_VALUE=16

# Number of runs for averaging
NUM_RUNS=20

# Create results file
RESULTS_FILE="daq_performance_results_$BENCHMARK_NAME.csv"
echo "cutoff,1queue_med,1queue_err_low,1queue_err_high,2queue_med,2queue_err_low,2queue_err_high,Speedup_1queue/2queue" > $RESULTS_FILE

# Pretty header (fixed width columns)
printf "%6s | %25s | %25s | %10s\n" "cutoff" "1queue (ms)" "2queue (ms)" "Speedup"
printf "%6s-+-%25s-+-%25s-+-%10s\n" "------" "-------------------------" "-------------------------" "----------"


for cutoff in "${CUTOFF_DEPTHS[@]}"; do
    # --- 1-queue ---
    QUEUE1_MED=0 QUEUE1_ELO=0 QUEUE1_EHI=0
    if [ -x "$BIN_DIR/${BENCHMARK_NAME}_queue_1" ]; then
        read QUEUE1_MED _ _ QUEUE1_ELO QUEUE1_EHI < <(
            benchmark_run_stats "$NUM_RUNS" benchmark_parse_execution_ms \
                "$BIN_DIR/${BENCHMARK_NAME}_queue_1" "$N_VALUE" "$cutoff"
        )
    fi

    # --- 2-queue ---
    QUEUE2_MED=0 QUEUE2_ELO=0 QUEUE2_EHI=0
    if [ -x "$BIN_DIR/${BENCHMARK_NAME}_queue_2" ]; then
        read QUEUE2_MED _ _ QUEUE2_ELO QUEUE2_EHI < <(
            benchmark_run_stats "$NUM_RUNS" benchmark_parse_execution_ms \
                "$BIN_DIR/${BENCHMARK_NAME}_queue_2" "$N_VALUE" "$cutoff"
        )
    fi

    # Speedup (1-queue / 2-queue) based on medians
    SPEEDUP=0
    if [ "$QUEUE1_MED" != "0" ] && [ "$QUEUE2_MED" != "0" ]; then
        SPEEDUP=$(echo "scale=6; $QUEUE1_MED / $QUEUE2_MED" | bc -l)
    fi

    # Print aligned row
    printf "%6d | " "$cutoff"
    benchmark_fmt_med_iqr "$QUEUE1_MED" "$QUEUE1_ELO" "$QUEUE1_EHI"; printf " | "
    benchmark_fmt_med_iqr "$QUEUE2_MED"  "$QUEUE2_ELO"  "$QUEUE2_EHI";  printf " | "
    if [ "$SPEEDUP" = "0" ]; then
        printf "%10.2f\n" "N/A"
    else
        printf "%10.2f\n" "$SPEEDUP"
    fi

    # CSV
    echo "$cutoff,$QUEUE1_MED,$QUEUE1_ELO,$QUEUE1_EHI,$QUEUE2_MED,$QUEUE2_ELO,$QUEUE2_EHI,$SPEEDUP" >> "$RESULTS_FILE"
done

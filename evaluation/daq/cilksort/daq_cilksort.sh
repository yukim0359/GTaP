#!/bin/bash

BENCHMARK_NAME=cilksort

COMPARE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
cd "$COMPARE_DIR" || exit 1
source "$COMPARE_DIR/../../common/benchmark.sh" || exit 1
PROJECT_ROOT=$(cd "$COMPARE_DIR" && pwd) || exit 1
BIN_DIR="$PROJECT_ROOT/bin"
UTIL_DIR="$PROJECT_ROOT/../../util"
TMP_DIR="$COMPARE_DIR/tmp"

# Test values for cutoff depth
CUTOFF_VALUES=(64 128 256 512 1024 2048 4096)

# array size
SIZE=50000000

# Number of runs for averaging
NUM_RUNS=20

# Create tmp directory
mkdir -p "$TMP_DIR"
make -C "$UTIL_DIR" gen_vector || exit 1

# Create results file
RESULTS_FILE="daq_performance_results_$BENCHMARK_NAME.csv"
echo "cutoff,1queue_med,1queue_err_low,1queue_err_high,3queue_med,3queue_err_low,3queue_err_high,Speedup_1queue/3queue" > $RESULTS_FILE

# Pretty header (fixed width columns)
printf "%6s | %25s | %25s | %10s\n" "cutoff" "1queue (ms)" "3queue (ms)" "Speedup"
printf "%6s-+-%25s-+-%25s-+-%10s\n" "------" "-------------------------" "-------------------------" "----------"

# Generate test data once (unsorted + gen_vector sidecar .sorted.bin)
echo "Generating test data..."
cd "$UTIL_DIR" || exit 1
DATA_FILE="test_data_${SIZE}.bin"
REF_FILE="test_data_${SIZE}.sorted.bin"
./gen_vector "$SIZE" "$TMP_DIR/$DATA_FILE" || exit 1
cd "$COMPARE_DIR" || exit 1
if [ ! -f "$TMP_DIR/$REF_FILE" ]; then
    echo "Error: missing sorted reference $TMP_DIR/$REF_FILE" >&2
    exit 1
fi

for cutoff in "${CUTOFF_VALUES[@]}"; do
    echo "=== Compiling and testing with cutoff=$cutoff ===" >&2

    # Force rebuild: only GTAP_CFLAGS (cutoff) changes between iterations.
    if ! make -C "$COMPARE_DIR" -B queue_1 queue_3 \
        GTAP_CFLAGS="-DTASK_SPAWN_CUTOFF_SORT=$cutoff -DTASK_SPAWN_CUTOFF_MERGE=$cutoff"; then
        echo "Warning: Failed to compile with cutoff=$cutoff" >&2
    fi

    # --- 1-queue ---
    QUEUE1_MED=0 QUEUE1_ELO=0 QUEUE1_EHI=0
    if [ -x "$BIN_DIR/${BENCHMARK_NAME}_queue_1" ]; then
        read QUEUE1_MED _ _ QUEUE1_ELO QUEUE1_EHI < <(
            benchmark_run_stats "$NUM_RUNS" benchmark_parse_execution_ms \
                "$BIN_DIR/${BENCHMARK_NAME}_queue_1" "$TMP_DIR/$DATA_FILE" "$TMP_DIR/$REF_FILE"
        )
    fi

    # --- 3-queue ---
    QUEUE3_MED=0 QUEUE3_ELO=0 QUEUE3_EHI=0
    if [ -x "$BIN_DIR/${BENCHMARK_NAME}_queue_3" ]; then
        read QUEUE3_MED _ _ QUEUE3_ELO QUEUE3_EHI < <(
            benchmark_run_stats "$NUM_RUNS" benchmark_parse_execution_ms \
                "$BIN_DIR/${BENCHMARK_NAME}_queue_3" "$TMP_DIR/$DATA_FILE" "$TMP_DIR/$REF_FILE"
        )
    fi

    # Speedup (1-queue / 3-queue) based on medians
    SPEEDUP=0
    if [ "$QUEUE1_MED" != "0" ] && [ "$QUEUE3_MED" != "0" ]; then
        SPEEDUP=$(echo "scale=6; $QUEUE1_MED / $QUEUE3_MED" | bc -l)
    fi

    # Print aligned row
    printf "%6d | " "$cutoff"
    benchmark_fmt_med_iqr "$QUEUE1_MED" "$QUEUE1_ELO" "$QUEUE1_EHI"; printf " | "
    benchmark_fmt_med_iqr "$QUEUE3_MED"  "$QUEUE3_ELO"  "$QUEUE3_EHI";  printf " | "
    if [ "$SPEEDUP" = "0" ]; then
        printf "%10s\n" "N/A"
    else
        printf "%10.2f\n" "$SPEEDUP"
    fi

    # CSV
    echo "$cutoff,$QUEUE1_MED,$QUEUE1_ELO,$QUEUE1_EHI,$QUEUE3_MED,$QUEUE3_ELO,$QUEUE3_EHI,$SPEEDUP" >> "$RESULTS_FILE"
done

# Cleanup
rm -f "$TMP_DIR/$DATA_FILE" "$TMP_DIR/$REF_FILE"

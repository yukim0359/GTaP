#!/bin/bash

BENCHMARK_NAME=cilksort

COMPARE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
cd "$COMPARE_DIR" || exit 1
source "$COMPARE_DIR/../../common/benchmark.sh" || exit 1
PROJECT_ROOT=$(cd "$COMPARE_DIR" && pwd) || exit 1
BIN_DIR="$PROJECT_ROOT/bin"
MAKEFILE="$COMPARE_DIR/Makefile"
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


# Compile function with cutoff value
compile_with_cutoff() {
    local cutoff=$1
    local queue_type=$2
    local src_file="$COMPARE_DIR/${BENCHMARK_NAME}_queue_${queue_type}.cu"
    local out_file="$BIN_DIR/${BENCHMARK_NAME}_queue_${queue_type}"
    
    if [ ! -f "$src_file" ]; then
        echo "Error: Source file not found: $src_file" >&2
        return 1
    fi
    
    local PROJ_DIR="${PROJ_DIR:-$(cd "$COMPARE_DIR/../../.." && pwd)}"
    local CUDA_PATH="${CUDA_PATH:-${CUDA_HOME:-}}"
    local CUDA_ARCH="${CUDA_ARCH:-sm_90}"
    local CXX_BIN="${CXX_BIN:-$PROJ_DIR/clang-gtap/build/bin/clang++}"
    local GT_INC="${GT_INC:-$PROJ_DIR/runtime}"
    local GTAP_GRID_SIZE="${GTAP_GRID_SIZE:-4000}"
    local GTAP_BLOCK_SIZE="${GTAP_BLOCK_SIZE:-32}"
    local GTAP_MAX_TASKS_PER_WARP="${GTAP_MAX_TASKS_PER_WARP:-90000}"
    local GTAP_CFLAGS="${GTAP_CFLAGS:-}"
    if [ -z "$CUDA_PATH" ]; then
        echo "Error: set CUDA_PATH or CUDA_HOME to the CUDA installation root" >&2
        return 1
    fi
    
    mkdir -p "$BIN_DIR"
    # Do not run a binary left over from a previous cutoff if this build fails.
    rm -f "$out_file"
    
    echo "Compiling ${BENCHMARK_NAME}_queue_${queue_type} with cutoff=$cutoff..." >&2
    
    # Add -Xcuda-ptxas --warn-on-spills only for queue_1 (matching Makefile)
    local extra_flags=""
    if [ "$queue_type" = "1" ]; then
        extra_flags="-Xcuda-ptxas --warn-on-spills"
    fi
    
    "$CXX_BIN" -O3 -x cuda \
      --cuda-path="$CUDA_PATH" \
      --cuda-gpu-arch="$CUDA_ARCH" \
      -Wall -Wextra \
      $extra_flags \
      -I"$GT_INC" \
      -DGTAP_GRID_SIZE="$GTAP_GRID_SIZE" \
      -DGTAP_BLOCK_SIZE="$GTAP_BLOCK_SIZE" \
      -DGTAP_MAX_TASKS_PER_WARP="$GTAP_MAX_TASKS_PER_WARP" \
      -DTASK_SPAWN_CUTOFF_SORT="$cutoff" \
      -DTASK_SPAWN_CUTOFF_MERGE="$cutoff" \
      $GTAP_CFLAGS \
      -L"$CUDA_PATH/lib64" -lcudart \
      "$src_file" -o "$out_file"
}

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
    # Compile with current cutoff value
    echo "=== Compiling and testing with cutoff=$cutoff ===" >&2
    
    # Compile 1-queue version
    if ! compile_with_cutoff "$cutoff" "1"; then
        echo "Warning: Failed to compile queue_1 with cutoff=$cutoff" >&2
    fi
    
    # Compile 3-queue version
    if ! compile_with_cutoff "$cutoff" "3"; then
        echo "Warning: Failed to compile queue_3 with cutoff=$cutoff" >&2
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

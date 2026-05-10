#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -W group_list=gc64
#PBS -j oe

BENCHMARK_NAME=cilksort

cd "$PBS_O_WORKDIR"

COMPARE_DIR=$(pwd)
PROJECT_ROOT=$(cd "$COMPARE_DIR" && pwd)
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

# Create results file
RESULTS_FILE="epaq_performance_results_$BENCHMARK_NAME.csv"
echo "cutoff,1queue_med,1queue_err_low,1queue_err_high,3queue_med,3queue_err_low,3queue_err_high,Speedup_1queue/3queue" > $RESULTS_FILE

# Pretty header (fixed width columns)
printf "%6s | %25s | %25s | %10s\n" "cutoff" "1queue (ms)" "3queue (ms)" "Speedup"
printf "%6s-+-%25s-+-%25s-+-%10s\n" "------" "-------------------------" "-------------------------" "----------"


run_stats() {
    local program=$1
    local data_file=$2
    local grep_pattern=$3
    local times=()

    for i in $(seq 1 $NUM_RUNS); do
        local output
        output=$($program "$data_file" 2>&1)

        local time
        time=$(echo "$output" | grep "$grep_pattern" | sed -n 's/.*: \([0-9.]*\) ms.*/\1/p' | head -n 1)

        if [ -n "$time" ]; then
            times+=("$time")
        fi
    done

    local m=${#times[@]}
    if [ "$m" -lt 5 ]; then
        echo "0 0 0 0 0"
        return
    fi

    # Sort ascending
    IFS=$'\n' sorted=($(printf "%s\n" "${times[@]}" | sort -n))
    unset IFS
    m=${#sorted[@]}

    # Median
    local median
    if (( m % 2 == 1 )); then
        median=${sorted[$((m/2))]}
    else
        median=$(echo "scale=6; (${sorted[$((m/2-1))]} + ${sorted[$((m/2))]}) / 2" | bc -l)
    fi

    # Quantile indices (nearest-rank)
    local q1_idx=$(( (m*25 + 99) / 100 - 1 ))
    local q3_idx=$(( (m*75 + 99) / 100 - 1 ))
    (( q1_idx < 0 )) && q1_idx=0
    (( q3_idx < 0 )) && q3_idx=0
    (( q1_idx >= m )) && q1_idx=$((m-1))
    (( q3_idx >= m )) && q3_idx=$((m-1))

    local q1=${sorted[$q1_idx]}
    local q3=${sorted[$q3_idx]}

    local err_low err_high
    err_low=$(echo "scale=6; $median - $q1" | bc -l)
    err_high=$(echo "scale=6; $q3 - $median" | bc -l)

    echo "$median $q1 $q3 $err_low $err_high"
}

fmt_med_iqr() {
    local med=$1 elo=$2 ehi=$3
    local s
    if [ "$med" = "0" ]; then
        s="N/A"
    else
        s=$(printf "%.3f(+%.3f/%.3f)" "$med" "$ehi" "$elo")
    fi
    printf "%25s" "$s"
}

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
    local GTAP_MAX_TASKS_PER_WARP="${GTAP_MAX_TASKS_PER_WARP:-80000}"
    local GTAP_CFLAGS="${GTAP_CFLAGS:--DGTAP_TERMINATE_ON_FIRST_TASK_FINISH}"
    local GTAP_NUM_QUEUES=$queue_type

    if [ -z "$CUDA_PATH" ]; then
        echo "Error: set CUDA_PATH or CUDA_HOME to the CUDA installation root" >&2
        return 1
    fi
    
    mkdir -p "$BIN_DIR"
    
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
      -DGTAP_GRID_SIZE="$GTAP_GRID_SIZE" -DGTAP_BLOCK_SIZE="$GTAP_BLOCK_SIZE" \
      -DGTAP_MAX_TASKS_PER_WARP="$GTAP_MAX_TASKS_PER_WARP" \
      -DGTAP_NUM_QUEUES="$GTAP_NUM_QUEUES" \
      -DTASK_SPAWN_CUTOFF_SORT="$cutoff" \
      -DTASK_SPAWN_CUTOFF_MERGE="$cutoff" \
      $GTAP_CFLAGS \
      -L"$CUDA_PATH/lib64" -lcudart \
      "$src_file" -o "$out_file"
}

# Generate test data once
echo "Generating test data..."
cd "$UTIL_DIR"
DATA_FILE="test_data_${SIZE}.bin"
if [ ! -f "$TMP_DIR/$DATA_FILE" ]; then
    ./gen_vector "$SIZE" "$TMP_DIR/$DATA_FILE" > /dev/null 2>&1
fi
cd "$COMPARE_DIR"

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
        read QUEUE1_MED _ _ QUEUE1_ELO QUEUE1_EHI < <(run_stats "$BIN_DIR/${BENCHMARK_NAME}_queue_1" "$TMP_DIR/$DATA_FILE" "Execution time")
    fi

    # --- 3-queue ---
    QUEUE3_MED=0 QUEUE3_ELO=0 QUEUE3_EHI=0
    if [ -x "$BIN_DIR/${BENCHMARK_NAME}_queue_3" ]; then
        read QUEUE3_MED _ _ QUEUE3_ELO QUEUE3_EHI < <(run_stats "$BIN_DIR/${BENCHMARK_NAME}_queue_3" "$TMP_DIR/$DATA_FILE" "Execution time")
    fi

    # Speedup (1-queue / 3-queue) based on medians
    SPEEDUP=0
    if [ "$QUEUE1_MED" != "0" ] && [ "$QUEUE3_MED" != "0" ]; then
        SPEEDUP=$(echo "scale=6; $QUEUE1_MED / $QUEUE3_MED" | bc -l)
    fi

    # Print aligned row
    printf "%6d | " "$cutoff"
    fmt_med_iqr "$QUEUE1_MED" "$QUEUE1_ELO" "$QUEUE1_EHI"; printf " | "
    fmt_med_iqr "$QUEUE3_MED"  "$QUEUE3_ELO"  "$QUEUE3_EHI";  printf " | "
    if [ "$SPEEDUP" = "0" ]; then
        printf "%10s\n" "N/A"
    else
        printf "%10.2f\n" "$SPEEDUP"
    fi

    # CSV
    echo "$cutoff,$QUEUE1_MED,$QUEUE1_ELO,$QUEUE1_EHI,$QUEUE3_MED,$QUEUE3_ELO,$QUEUE3_EHI,$SPEEDUP" >> "$RESULTS_FILE"
done

# Cleanup
rm -f "$TMP_DIR/$DATA_FILE"

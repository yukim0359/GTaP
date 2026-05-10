#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -W group_list=gc64
#PBS -j oe

BENCHMARK_NAME=nq

cd "$PBS_O_WORKDIR"

# When submitted from gtap/evaluation/1-worker_scalability/$BENCHMARK_NAME
SCALING_DIR=$(pwd)
PROJECT_ROOT=$(cd "$SCALING_DIR/../../.." && pwd)
RUNTIME_DIR="$PROJECT_ROOT/runtime"
TMP_BIN_DIR="$SCALING_DIR/tmp_bin_${BENCHMARK_NAME}"

# Compiler paths (matching Makefile)
CUDA_PATH="${CUDA_PATH:-${CUDA_HOME:-}}"
CLANG_BIN="${CLANG_BIN:-$PROJECT_ROOT/clang-gtap/build/bin/clang}"

if [ -z "$CUDA_PATH" ]; then
    echo "Error: set CUDA_PATH or CUDA_HOME to the CUDA installation root"
    exit 1
fi

if [ ! -d "$RUNTIME_DIR" ]; then
    echo "Error: Please submit this script from gtap/evaluation/1-worker_scalability/$BENCHMARK_NAME"
    exit 1
fi

if [ ! -f "$CLANG_BIN" ]; then
    echo "Error: clang not found at $CLANG_BIN"
    exit 1
fi

mkdir -p "$TMP_BIN_DIR"

echo "=== ${BENCHMARK_NAME} Worker Scaling Analysis ==="
echo "Testing different GTAP_GRID_SIZE and GTAP_BLOCK_SIZE combinations"
echo "=============================================================="

# Block sizes to test
BLOCK_SIZES=(32 64 128 256)

# Total threads: 2^10 to 2^18 (9 points)
# Total threads = gridDim × blockDim
TOTAL_THREADS_EXPONENTS=(10 11 12 13 14 15 16 17 18)

# Number of runs (matching compare_nq.sh)
NUM_RUNS=20

### nq-specific settings ###
INPUT_N=16
CUTOFF_DEPTH=7

# Results file
RESULTS_FILE="$SCALING_DIR/${BENCHMARK_NAME}_scaling_results.csv"
echo "block_size,total_threads,grid_size,max_tasks_per_warp,ws_med,ws_err_low,ws_err_high,gq_med,gq_err_low,gq_err_high,chaselev_med,chaselev_err_low,chaselev_err_high" > "$RESULTS_FILE"

# Compiler settings (matching Makefile)
CUDA_ARCH="${CUDA_ARCH:-sm_90}"

# Constant: total_threads * MAX_TASKS_PER_WARP = 1024 * 96 * 150000
CONSTANT_TOTAL_THREADS_MAX_TASKS=$((1024 * 96 * 150000))

# Function to compute median and IQR (matching compare_nq.sh)
# Returns: median q1 q3 err_low err_high
compute_stats() {
    local times=("$@")
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

for block_size in "${BLOCK_SIZES[@]}"; do
    echo ""
    echo "=== Block Size: $block_size ==="
    
    for exp in "${TOTAL_THREADS_EXPONENTS[@]}"; do
        total_threads=$((2 ** exp))
        grid_size=$((total_threads / block_size))
        
        # Skip if grid_size is 0 or too small
        if [ "$grid_size" -lt 1 ]; then
            echo "  Skipping total_threads=$total_threads (grid_size=$grid_size < 1)"
            continue
        fi
        
        # Calculate MAX_TASKS_PER_WARP to keep total_threads * MAX_TASKS_PER_WARP constant
        max_tasks_per_warp=$((CONSTANT_TOTAL_THREADS_MAX_TASKS / total_threads))
        
        echo "  Testing: block_size=$block_size, total_threads=$total_threads, grid_size=$grid_size, max_tasks_per_warp=$max_tasks_per_warp"
        
        # Common compile flags (matching Makefile)
        COMMON_FLAGS="-O3 -x cuda --cuda-path=$CUDA_PATH --cuda-gpu-arch=$CUDA_ARCH"
        COMMON_FLAGS="$COMMON_FLAGS -Wall -Wextra -Xcuda-ptxas --warn-on-spills"
        COMMON_FLAGS="$COMMON_FLAGS -I$RUNTIME_DIR"
        COMMON_FLAGS="$COMMON_FLAGS -DGTAP_GRID_SIZE=$grid_size -DGTAP_BLOCK_SIZE=$block_size"
        COMMON_FLAGS="$COMMON_FLAGS -DGTAP_MAX_TASKS_PER_WARP=$max_tasks_per_warp"
        COMMON_FLAGS="$COMMON_FLAGS -DGTAP_NUM_QUEUES=1"
        LINK_FLAGS="-L$CUDA_PATH/lib64 -lcudart"
        
        # Compile WS (default work-stealing) variant
        BINARY_WS="${BENCHMARK_NAME}_ws_${block_size}_${grid_size}"
        BINARY_WS_PATH="$TMP_BIN_DIR/$BINARY_WS"
        
        echo "    Compiling WS variant..."
        $CLANG_BIN $COMMON_FLAGS $LINK_FLAGS \
            -o "$BINARY_WS_PATH" \
            "$SCALING_DIR/gtap_${BENCHMARK_NAME}.cu" 2>&1 | grep -v "warning" || true
        
        if [ ! -f "$BINARY_WS_PATH" ]; then
            echo "    ERROR: WS compilation failed"
            continue
        fi
        
        # Compile GQ variant
        BINARY_GQ="${BENCHMARK_NAME}_gq_${block_size}_${grid_size}"
        BINARY_GQ_PATH="$TMP_BIN_DIR/$BINARY_GQ"
        
        echo "    Compiling GQ variant..."
        $CLANG_BIN $COMMON_FLAGS $LINK_FLAGS \
            -DGQ \
            -o "$BINARY_GQ_PATH" \
            "$SCALING_DIR/gtap_${BENCHMARK_NAME}.cu" 2>&1 | grep -v "warning" || true
        
        if [ ! -f "$BINARY_GQ_PATH" ]; then
            echo "    ERROR: GQ compilation failed"
            rm -f "$BINARY_WS_PATH"
            continue
        fi
        
        # Compile CHASELEV variant
        BINARY_CL="${BENCHMARK_NAME}_cl_${block_size}_${grid_size}"
        BINARY_CL_PATH="$TMP_BIN_DIR/$BINARY_CL"
        
        echo "    Compiling CHASELEV variant..."
        $CLANG_BIN $COMMON_FLAGS $LINK_FLAGS \
            -DCHASELEV \
            -o "$BINARY_CL_PATH" \
            "$SCALING_DIR/gtap_${BENCHMARK_NAME}.cu" 2>&1 | grep -v "warning" || true
        
        if [ ! -f "$BINARY_CL_PATH" ]; then
            echo "    ERROR: CHASELEV compilation failed"
            rm -f "$BINARY_WS_PATH" "$BINARY_GQ_PATH"
            continue
        fi
        
        # Run WS variant
        echo "    Running WS variant $NUM_RUNS times..."
        TIMES_WS=()
        for i in $(seq 1 $NUM_RUNS); do
            output=$("$BINARY_WS_PATH" "$INPUT_N" "$CUTOFF_DEPTH" 2>&1 || true)
            time=$(echo "$output" | grep "Execution time" | sed 's/.*: \([0-9.]*\) ms.*/\1/')
            if [ -n "$time" ]; then
                TIMES_WS+=("$time")
            fi
        done
        
        # Run GQ variant
        echo "    Running GQ variant $NUM_RUNS times..."
        TIMES_GQ=()
        for i in $(seq 1 $NUM_RUNS); do
            output=$("$BINARY_GQ_PATH" "$INPUT_N" "$CUTOFF_DEPTH" 2>&1 || true)
            time=$(echo "$output" | grep "Execution time" | sed 's/.*: \([0-9.]*\) ms.*/\1/')
            if [ -n "$time" ]; then
                TIMES_GQ+=("$time")
            fi
        done
        
        # Run CHASELEV variant
        echo "    Running CHASELEV variant $NUM_RUNS times..."
        TIMES_CL=()
        for i in $(seq 1 $NUM_RUNS); do
            output=$("$BINARY_CL_PATH" "$INPUT_N" "$CUTOFF_DEPTH" 2>&1 || true)
            time=$(echo "$output" | grep "Execution time" | sed 's/.*: \([0-9.]*\) ms.*/\1/')
            if [ -n "$time" ]; then
                TIMES_CL+=("$time")
            fi
        done
        
        if [ ${#TIMES_WS[@]} -lt 5 ] || [ ${#TIMES_GQ[@]} -lt 5 ] || [ ${#TIMES_CL[@]} -lt 5 ]; then
            echo "    ERROR: Not enough timing results (need at least 5)"
            rm -f "$BINARY_WS_PATH" "$BINARY_GQ_PATH" "$BINARY_CL_PATH"
            continue
        fi
        
        # Calculate statistics (median + IQR)
        read WS_MED _ _ WS_ELO WS_EHI < <(compute_stats "${TIMES_WS[@]}")
        read GQ_MED _ _ GQ_ELO GQ_EHI < <(compute_stats "${TIMES_GQ[@]}")
        read CL_MED _ _ CL_ELO CL_EHI < <(compute_stats "${TIMES_CL[@]}")
        
        printf "    WS Result: %.3f (+%.3f/-%.3f) ms\n" "$WS_MED" "$WS_EHI" "$WS_ELO"
        printf "    GQ Result: %.3f (+%.3f/-%.3f) ms\n" "$GQ_MED" "$GQ_EHI" "$GQ_ELO"
        printf "    CHASELEV Result: %.3f (+%.3f/-%.3f) ms\n" "$CL_MED" "$CL_EHI" "$CL_ELO"
        
        # Write to CSV
        echo "$block_size,$total_threads,$grid_size,$max_tasks_per_warp,$WS_MED,$WS_ELO,$WS_EHI,$GQ_MED,$GQ_ELO,$GQ_EHI,$CL_MED,$CL_ELO,$CL_EHI" >> "$RESULTS_FILE"
        
        # Clean up binaries to save space
        rm -f "$BINARY_WS_PATH" "$BINARY_GQ_PATH" "$BINARY_CL_PATH"
    done
done

cd "$SCALING_DIR"

echo ""
echo "=== Results saved to $RESULTS_FILE ==="
echo "CSV contents:"
cat "$RESULTS_FILE"

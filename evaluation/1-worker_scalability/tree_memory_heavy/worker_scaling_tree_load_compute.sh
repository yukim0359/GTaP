#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -W group_list=gc64
#PBS -j oe

BENCHMARK_NAME=tree_memory_heavy

cd "$PBS_O_WORKDIR"

# When submitted from gtap/evaluation/1-worker_scalability/$BENCHMARK_NAME
SCALING_DIR=$(pwd)
PROJECT_ROOT=$(cd "$SCALING_DIR/../../.." && pwd)
RUNTIME_DIR="$PROJECT_ROOT/runtime"
TMP_BIN_DIR="$SCALING_DIR/tmp_bin_${BENCHMARK_NAME}"

# Compiler paths (matching Makefile)
CUDA_PATH="${CUDA_PATH:-/work/opt/local/aarch64/cores/nvidia/25.9/Linux_aarch64/25.9/cuda}"
CXX_BIN="${CXX_BIN:-$PROJECT_ROOT/clang-gtap/build/bin/clang++}"

if [ ! -d "$RUNTIME_DIR" ]; then
    echo "Error: Please submit this script from gtap/evaluation/1-worker_scalability/$BENCHMARK_NAME"
    exit 1
fi

if [ ! -f "$CXX_BIN" ]; then
    echo "Error: clang++ not found at $CXX_BIN"
    exit 1
fi

mkdir -p "$TMP_BIN_DIR"

echo "=== ${BENCHMARK_NAME} Worker Scaling Analysis (Block-level) ==="
echo "Testing different GTAP_GRID_SIZE and GTAP_BLOCK_SIZE combinations"
echo "================================================================="

# Block sizes to test
BLOCK_SIZES=(32 64 128 256)

# Grid sizes to test (worker = block in block-level)
# Maximum grid_size depends on block_size (shared memory constraints):
#   block_size=256 -> max grid_size=1024
#   block_size=128 -> max grid_size=2048
#   block_size=64  -> max grid_size=4096
#   block_size=32  -> max grid_size=8192
GRID_SIZE_VALUES=(16 32 64 128 256 512 1024 2048 4096 8192)

# Function to get max grid size for a given block size
get_max_grid_size() {
    local block_size=$1
    case $block_size in
        32)  echo 8192 ;;
        64)  echo 4096 ;;
        128) echo 2048 ;;
        256) echo 1024 ;;
        *)   echo 1024 ;;
    esac
}

# Number of runs
NUM_RUNS=20

### tree_memory_heavy-specific settings ###
TREE_HEIGHT=20
COMPUTE_ITERS=64
MEM_OPS=1024

# Results file
RESULTS_FILE="$SCALING_DIR/${BENCHMARK_NAME}_scaling_results.csv"
echo "block_size,total_workers,grid_size,max_tasks_per_block,ws_med,ws_err_low,ws_err_high,gq_med,gq_err_low,gq_err_high" > "$RESULTS_FILE"

# Compiler settings (matching Makefile)
CUDA_ARCH="${CUDA_ARCH:-sm_90}"

# Constant: grid_size * MAX_TASKS_PER_BLOCK = 256 * 100000
CONSTANT_GRID_MAX_TASKS=$((256 * 100000))

# Function to compute median and IQR
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
    max_grid=$(get_max_grid_size $block_size)
    
    for grid_size in "${GRID_SIZE_VALUES[@]}"; do
        # Skip if grid_size exceeds maximum for this block_size
        if [ "$grid_size" -gt "$max_grid" ]; then
            echo "  Skipping grid_size=$grid_size (exceeds max=$max_grid for block_size=$block_size)"
            continue
        fi
        
        # Calculate MAX_TASKS_PER_BLOCK to keep grid_size * MAX_TASKS_PER_BLOCK constant
        max_tasks_per_block=$((CONSTANT_GRID_MAX_TASKS / grid_size))
        
        echo "  Testing: block_size=$block_size, grid_size=$grid_size, max_tasks_per_block=$max_tasks_per_block"
        
        # Common compile flags (matching Makefile)
        COMMON_FLAGS="-O3 -x cuda --cuda-path=$CUDA_PATH --cuda-gpu-arch=$CUDA_ARCH"
        COMMON_FLAGS="$COMMON_FLAGS -Wall -Wextra -Xcuda-ptxas --warn-on-spills"
        COMMON_FLAGS="$COMMON_FLAGS -I$RUNTIME_DIR"
        COMMON_FLAGS="$COMMON_FLAGS -DGTAP_GRID_SIZE=$grid_size -DGTAP_BLOCK_SIZE=$block_size"
        COMMON_FLAGS="$COMMON_FLAGS -DGTAP_MAX_TASKS_PER_BLOCK=$max_tasks_per_block -DGTAP_MAX_CHILD_TASKS=2"
        LINK_FLAGS="-L$CUDA_PATH/lib64 -lcudart"
        
        # Compile WS (default work-stealing) variant
        BINARY_WS="${BENCHMARK_NAME}_ws_${block_size}_${grid_size}"
        BINARY_WS_PATH="$TMP_BIN_DIR/$BINARY_WS"
        
        echo "    Compiling WS variant..."
        $CXX_BIN $COMMON_FLAGS $LINK_FLAGS \
            -o "$BINARY_WS_PATH" \
            "$SCALING_DIR/gtap_block_${BENCHMARK_NAME}.cu" 2>&1 | grep -v "warning" || true
        
        if [ ! -f "$BINARY_WS_PATH" ]; then
            echo "    ERROR: WS compilation failed"
            continue
        fi
        
        # Compile GQ variant
        BINARY_GQ="${BENCHMARK_NAME}_gq_${block_size}_${grid_size}"
        BINARY_GQ_PATH="$TMP_BIN_DIR/$BINARY_GQ"
        
        echo "    Compiling GQ variant..."
        $CXX_BIN $COMMON_FLAGS $LINK_FLAGS \
            -DGQ \
            -o "$BINARY_GQ_PATH" \
            "$SCALING_DIR/gtap_block_${BENCHMARK_NAME}.cu" 2>&1 | grep -v "warning" || true
        
        if [ ! -f "$BINARY_GQ_PATH" ]; then
            echo "    ERROR: GQ compilation failed"
            rm -f "$BINARY_WS_PATH"
            continue
        fi
        
        # Run WS variant
        echo "    Running WS variant $NUM_RUNS times..."
        TIMES_WS=()
        for i in $(seq 1 $NUM_RUNS); do
            output=$("$BINARY_WS_PATH" "$TREE_HEIGHT" "$COMPUTE_ITERS" "$MEM_OPS" 2>&1 || true)
            time=$(echo "$output" | grep "Execution time" | sed 's/.*: \([0-9.]*\) ms.*/\1/')
            if [ -n "$time" ]; then
                TIMES_WS+=("$time")
            fi
        done
        
        # Run GQ variant
        echo "    Running GQ variant $NUM_RUNS times..."
        TIMES_GQ=()
        for i in $(seq 1 $NUM_RUNS); do
            output=$("$BINARY_GQ_PATH" "$TREE_HEIGHT" "$COMPUTE_ITERS" "$MEM_OPS" 2>&1 || true)
            time=$(echo "$output" | grep "Execution time" | sed 's/.*: \([0-9.]*\) ms.*/\1/')
            if [ -n "$time" ]; then
                TIMES_GQ+=("$time")
            fi
        done
        
        if [ ${#TIMES_WS[@]} -lt 5 ] || [ ${#TIMES_GQ[@]} -lt 5 ]; then
            echo "    ERROR: Not enough timing results (need at least 5)"
            rm -f "$BINARY_WS_PATH" "$BINARY_GQ_PATH"
            continue
        fi
        
        # Calculate statistics (median + IQR)
        read WS_MED _ _ WS_ELO WS_EHI < <(compute_stats "${TIMES_WS[@]}")
        read GQ_MED _ _ GQ_ELO GQ_EHI < <(compute_stats "${TIMES_GQ[@]}")
        
        printf "    WS Result: %.3f (+%.3f/-%.3f) ms\n" "$WS_MED" "$WS_EHI" "$WS_ELO"
        printf "    GQ Result: %.3f (+%.3f/-%.3f) ms\n" "$GQ_MED" "$GQ_EHI" "$GQ_ELO"
        
        # Write to CSV
        echo "$block_size,$grid_size,$grid_size,$max_tasks_per_block,$WS_MED,$WS_ELO,$WS_EHI,$GQ_MED,$GQ_ELO,$GQ_EHI" >> "$RESULTS_FILE"
        
        # Clean up binaries to save space
        rm -f "$BINARY_WS_PATH" "$BINARY_GQ_PATH"
    done
done

cd "$SCALING_DIR"

echo ""
echo "=== Results saved to $RESULTS_FILE ==="
echo "CSV contents:"
cat "$RESULTS_FILE"

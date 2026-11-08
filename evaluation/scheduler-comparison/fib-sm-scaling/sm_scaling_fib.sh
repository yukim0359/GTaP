#!/bin/bash

# Fibonacci SM scaling. Same three schedulers as thread_scaling_fib.sh.
# Green contexts provision 8, 16, 32, 64, 128 SMs. The full chip is a separate
# point with no partition. Grid size is BLOCKS_PER_SM * SM count.

BENCHMARK_NAME=fib
SCALING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
cd "$SCALING_DIR" || exit 1
source "$SCALING_DIR/../../common/benchmark.sh" || exit 1
PROJECT_ROOT=$(cd "$SCALING_DIR/../../.." && pwd) || exit 1
RUNTIME_DIR="$PROJECT_ROOT/runtime"
TMP_BIN_DIR="$SCALING_DIR/tmp_bin_${BENCHMARK_NAME}_sm"

CUDA_PATH="${CUDA_PATH:-${CUDA_HOME:-}}"
CUDA_ARCH="${CUDA_ARCH:-sm_90}"
CLANG_BIN="${CLANG_BIN:-$PROJECT_ROOT/clang-gtap/build/bin/clang}"
GTAP_CFLAGS="${GTAP_CFLAGS:-}"

if [ -z "$CUDA_PATH" ]; then
    echo "Error: set CUDA_PATH or CUDA_HOME to the CUDA installation root"
    exit 1
fi
if [ ! -x "$CLANG_BIN" ]; then
    echo "Error: clang not found at $CLANG_BIN"
    exit 1
fi

BLOCK_SIZES=(${BLOCK_SIZES:-32})
BLOCKS_PER_SM="${BLOCKS_PER_SM:-16}"
SM_COUNTS=(${SM_COUNTS:-8 16 32 64 128})
INCLUDE_FULL_CHIP="${INCLUDE_FULL_CHIP:-1}"
NUM_RUNS="${NUM_RUNS:-20}"
INPUT_N="${INPUT_N:-35}"
CONSTANT_TOTAL_THREADS_MAX_TASKS=$((4000 * 32 * 200000))

mkdir -p "$TMP_BIN_DIR"

# nvidia-smi has no SM-count field. Ask the driver when the full chip is a point.
DEVICE_SMS=""
if [ "$INCLUDE_FULL_CHIP" = "1" ]; then
    cat > "$TMP_BIN_DIR/query_sms.c" << 'EOF'
#include <cuda.h>
#include <stdio.h>
int main(void) {
    CUdevice dev;
    int sms = 0;
    if (cuInit(0) != CUDA_SUCCESS) return 1;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 1;
    if (cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev) != CUDA_SUCCESS) return 1;
    printf("%d\n", sms);
    return 0;
}
EOF
    "$CLANG_BIN" -O2 -I"$CUDA_PATH/include" \
        -L"$CUDA_PATH/lib64/stubs" -o "$TMP_BIN_DIR/query_sms" \
        "$TMP_BIN_DIR/query_sms.c" -lcuda
    DEVICE_SMS=$("$TMP_BIN_DIR/query_sms")
    case "$DEVICE_SMS" in
        ''|*[!0-9]*)
            echo "Error: failed to query device SM count"
            exit 1
            ;;
    esac
fi

RESULTS_FILE="${RESULTS_FILE:-$SCALING_DIR/${BENCHMARK_NAME}_sm_scaling_results.csv}"
echo "sm_count,block_size,blocks_per_sm,total_threads,grid_size,max_tasks_per_warp,observed_sms,ws_med,ws_err_low,ws_err_high,gq_med,gq_err_low,gq_err_high,chaselev_med,chaselev_err_low,chaselev_err_high" > "$RESULTS_FILE"

echo "=== ${BENCHMARK_NAME} SM Scaling ==="
echo "device_sms=$DEVICE_SMS  INPUT_N=$INPUT_N  BLOCKS_PER_SM=$BLOCKS_PER_SM"
echo "SM_COUNTS=${SM_COUNTS[*]}  INCLUDE_FULL_CHIP=$INCLUDE_FULL_CHIP  NUM_RUNS=$NUM_RUNS"
echo "=============================================================="

TIMES=()
OBSERVED_SMS=""

run_variant() {
    local bin=$1
    local grid_size=$2
    local block_size=$3
    local max_tasks_per_warp=$4
    TIMES=()
    local checked=0
    local i output prov obs time
    for i in $(seq 1 "$NUM_RUNS"); do
        if ! output=$(GTAP_GRID_SIZE="$grid_size" GTAP_BLOCK_SIZE="$block_size" GTAP_MAX_TASKS_PER_WARP="$max_tasks_per_warp" \
            "$bin" "$INPUT_N" 2>&1); then
            printf '%s\n' "$output"
            echo "ERROR: $bin failed"
            exit 1
        fi
        if [ "$checked" -eq 0 ]; then
            printf '%s\n' "$output" | grep -E '^(device_sms=|SM scaling:)'
            prov=$(printf '%s\n' "$output" | sed -n 's/^SM scaling: requested=[0-9]* provisioned=\([0-9]*\) observed=[0-9]* grid=[0-9]*/\1/p' | head -1)
            obs=$(printf '%s\n' "$output" | sed -n 's/^SM scaling: requested=[0-9]* provisioned=[0-9]* observed=\([0-9]*\) grid=[0-9]*/\1/p' | head -1)
            if [ "$prov" != "$EXPECTED_SMS" ] || [ "$obs" != "$EXPECTED_SMS" ]; then
                echo "ERROR: expected $EXPECTED_SMS SMs, provisioned=$prov observed=$obs"
                exit 1
            fi
            OBSERVED_SMS=$obs
            checked=1
        fi
        time=$(printf '%s\n' "$output" | sed -n 's/^Execution time: \([0-9.]*\) ms.*/\1/p' | head -1)
        if [ -z "$time" ]; then
            printf '%s\n' "$output"
            echo "ERROR: missing execution time from $bin"
            exit 1
        fi
        TIMES+=("$time")
    done
}

compile_variant() {
    local output=$1
    shift
    echo "    Compiling $(basename "$output")"
    if ! "$CLANG_BIN" "$@" -o "$output" "$SCALING_DIR/gtap_${BENCHMARK_NAME}.cu"; then
        echo "ERROR: compilation failed"
        exit 1
    fi
}

run_point() {
    local sm_count=$1
    local sm_define=$2
    local block_size=$3
    local grid_size=$((BLOCKS_PER_SM * sm_count))
    local total_threads=$((grid_size * block_size))
    local max_tasks_per_warp=$((CONSTANT_TOTAL_THREADS_MAX_TASKS / total_threads))
    EXPECTED_SMS=$sm_count

    if [ "$grid_size" -lt 1 ] || [ "$max_tasks_per_warp" -lt 1 ]; then
        echo "ERROR: invalid grid=$grid_size or max_tasks_per_warp=$max_tasks_per_warp"
        exit 1
    fi

    echo ""
    echo "=== SMs: $sm_count  block: $block_size  grid: $grid_size  tasks/warp: $max_tasks_per_warp ==="

    local common_flags="-O3 -x cuda --cuda-path=$CUDA_PATH --cuda-gpu-arch=$CUDA_ARCH"
    common_flags="$common_flags -Wall -Wextra -Xcuda-ptxas --warn-on-spills"
    common_flags="$common_flags -I$RUNTIME_DIR"
    common_flags="$common_flags -DGTAP_SM_COUNT=$sm_define"
    common_flags="$common_flags -DGTAP_GRID_SIZE=$grid_size"
    common_flags="$common_flags -DGTAP_BLOCK_SIZE=$block_size"
    common_flags="$common_flags -DGTAP_MAX_TASKS_PER_WARP=$max_tasks_per_warp"
    common_flags="$common_flags $GTAP_CFLAGS"
    local link_flags="-L$CUDA_PATH/lib64 -Wl,-rpath,$CUDA_PATH/lib64 -lcudart"
    link_flags="$link_flags -L$CUDA_PATH/lib64/stubs -lcuda"

    local tag="${block_size}_${grid_size}"
    local binary_ws="$TMP_BIN_DIR/${BENCHMARK_NAME}_ws_${tag}"
    local binary_gq="$TMP_BIN_DIR/${BENCHMARK_NAME}_gq_${tag}"
    local binary_cl="$TMP_BIN_DIR/${BENCHMARK_NAME}_cl_${tag}"

    compile_variant "$binary_ws" $common_flags $link_flags
    compile_variant "$binary_gq" $common_flags -DGQ $link_flags
    compile_variant "$binary_cl" $common_flags -DCHASELEV $link_flags

    echo "    Running WS $NUM_RUNS times..."
    run_variant "$binary_ws" "$grid_size" "$block_size" "$max_tasks_per_warp"
    local times_ws=("${TIMES[@]}")
    echo "    Running GQ $NUM_RUNS times..."
    run_variant "$binary_gq" "$grid_size" "$block_size" "$max_tasks_per_warp"
    local times_gq=("${TIMES[@]}")
    echo "    Running CHASELEV $NUM_RUNS times..."
    run_variant "$binary_cl" "$grid_size" "$block_size" "$max_tasks_per_warp"
    local times_cl=("${TIMES[@]}")

    local ws_med ws_elo ws_ehi gq_med gq_elo gq_ehi cl_med cl_elo cl_ehi
    read ws_med _ _ ws_elo ws_ehi < <(benchmark_stats_from_times "${times_ws[@]}")
    read gq_med _ _ gq_elo gq_ehi < <(benchmark_stats_from_times "${times_gq[@]}")
    read cl_med _ _ cl_elo cl_ehi < <(benchmark_stats_from_times "${times_cl[@]}")

    printf "    WS Result: %.3f (+%.3f/-%.3f) ms\n" "$ws_med" "$ws_ehi" "$ws_elo"
    printf "    GQ Result: %.3f (+%.3f/-%.3f) ms\n" "$gq_med" "$gq_ehi" "$gq_elo"
    printf "    CHASELEV Result: %.3f (+%.3f/-%.3f) ms\n" "$cl_med" "$cl_ehi" "$cl_elo"

    echo "$sm_count,$block_size,$BLOCKS_PER_SM,$total_threads,$grid_size,$max_tasks_per_warp,$OBSERVED_SMS,$ws_med,$ws_elo,$ws_ehi,$gq_med,$gq_elo,$gq_ehi,$cl_med,$cl_elo,$cl_ehi" >> "$RESULTS_FILE"
    rm -f "$binary_ws" "$binary_gq" "$binary_cl"
}

for block_size in "${BLOCK_SIZES[@]}"; do
    for sm_count in "${SM_COUNTS[@]}"; do
        run_point "$sm_count" "$sm_count" "$block_size"
    done
    if [ "$INCLUDE_FULL_CHIP" = "1" ]; then
        run_point "$DEVICE_SMS" 0 "$block_size"
    fi
done

echo ""
echo "=== Results saved to $RESULTS_FILE ==="
cat "$RESULTS_FILE"

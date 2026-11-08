#!/bin/bash

BENCHMARK_NAME=fib

# Run from gtap/evaluation/scheduler-comparison/fib-thread-scaling
SCALING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
cd "$SCALING_DIR" || exit 1
source "$SCALING_DIR/../../common/benchmark.sh" || exit 1
TMP_BIN_DIR="$SCALING_DIR/tmp_bin_${BENCHMARK_NAME}"

mkdir -p "$TMP_BIN_DIR"

# Block sizes to test (override: BLOCK_SIZES="32 64")
BLOCK_SIZES=(${BLOCK_SIZES:-32})

# Total threads: 2^10 to 2^17 (resident ceiling for t/b=32 on H100: ~135k)
# Total threads = gridDim × blockDim
TOTAL_THREADS_EXPONENTS=(10 11 12 13 14 15 16 17)

# Number of runs (matching compare_fib.sh)
NUM_RUNS=20

### fib-specific settings (override: INPUT_N=40) ###
INPUT_N="${INPUT_N:-35}"

echo "=== ${BENCHMARK_NAME} Thread Scaling Analysis ==="
echo "INPUT_N=$INPUT_N  BLOCK_SIZES=${BLOCK_SIZES[*]}"
echo "=============================================================="

# Results file
RESULTS_FILE="$SCALING_DIR/${BENCHMARK_NAME}_thread_scaling_results.csv"
echo "block_size,total_threads,grid_size,max_tasks_per_warp,ws_med,ws_err_low,ws_err_high,gq_med,gq_err_low,gq_err_high,chaselev_med,chaselev_err_low,chaselev_err_high" > "$RESULTS_FILE"

# Constant: total_threads * MAX_TASKS_PER_WARP = 1024 * 96 * 150000
CONSTANT_TOTAL_THREADS_MAX_TASKS=$((4000 * 32 * 200000))

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

        echo "    Compiling WS / GQ / CHASELEV variants..."
        if ! make -C "$SCALING_DIR" -B gtap gtap_gq gtap_chaselev \
            OUT_DIR="$TMP_BIN_DIR" \
            GTAP_GRID_SIZE="$grid_size" \
            GTAP_BLOCK_SIZE="$block_size" \
            GTAP_MAX_TASKS_PER_WARP="$max_tasks_per_warp"; then
            echo "    ERROR: compilation failed"
            continue
        fi

        BINARY_WS_PATH="$TMP_BIN_DIR/gtap_${BENCHMARK_NAME}"
        BINARY_GQ_PATH="$TMP_BIN_DIR/gtap_${BENCHMARK_NAME}_gq"
        BINARY_CL_PATH="$TMP_BIN_DIR/gtap_${BENCHMARK_NAME}_chaselev"

        echo "    Running WS variant $NUM_RUNS times..."
        read WS_MED _ _ WS_ELO WS_EHI < <(
            benchmark_run_stats "$NUM_RUNS" benchmark_parse_execution_ms \
                env GTAP_GRID_SIZE="$grid_size" GTAP_BLOCK_SIZE="$block_size" \
                    GTAP_MAX_TASKS_PER_WARP="$max_tasks_per_warp" \
                "$BINARY_WS_PATH" "$INPUT_N"
        )

        echo "    Running GQ variant $NUM_RUNS times..."
        read GQ_MED _ _ GQ_ELO GQ_EHI < <(
            benchmark_run_stats "$NUM_RUNS" benchmark_parse_execution_ms \
                env GTAP_GRID_SIZE="$grid_size" GTAP_BLOCK_SIZE="$block_size" \
                    GTAP_MAX_TASKS_PER_WARP="$max_tasks_per_warp" \
                "$BINARY_GQ_PATH" "$INPUT_N"
        )

        echo "    Running CHASELEV variant $NUM_RUNS times..."
        read CL_MED _ _ CL_ELO CL_EHI < <(
            benchmark_run_stats "$NUM_RUNS" benchmark_parse_execution_ms \
                env GTAP_GRID_SIZE="$grid_size" GTAP_BLOCK_SIZE="$block_size" \
                    GTAP_MAX_TASKS_PER_WARP="$max_tasks_per_warp" \
                "$BINARY_CL_PATH" "$INPUT_N"
        )

        if [ "$WS_MED" = "0" ] || [ "$GQ_MED" = "0" ] || [ "$CL_MED" = "0" ]; then
            echo "    ERROR: Not enough timing results (need at least 5)"
            rm -f "$BINARY_WS_PATH" "$BINARY_GQ_PATH" "$BINARY_CL_PATH"
            continue
        fi

        printf "    WS Result: %.3f (+%.3f/-%.3f) ms\n" "$WS_MED" "$WS_EHI" "$WS_ELO"
        printf "    GQ Result: %.3f (+%.3f/-%.3f) ms\n" "$GQ_MED" "$GQ_EHI" "$GQ_ELO"
        printf "    CHASELEV Result: %.3f (+%.3f/-%.3f) ms\n" "$CL_MED" "$CL_EHI" "$CL_ELO"

        # Write to CSV
        echo "$block_size,$total_threads,$grid_size,$max_tasks_per_warp,$WS_MED,$WS_ELO,$WS_EHI,$GQ_MED,$GQ_ELO,$GQ_EHI,$CL_MED,$CL_ELO,$CL_EHI" >> "$RESULTS_FILE"

        # Clean up binaries to save space
        rm -f "$BINARY_WS_PATH" "$BINARY_GQ_PATH" "$BINARY_CL_PATH"
    done
done

cd "$SCALING_DIR" || exit 1

echo ""
echo "=== Results saved to $RESULTS_FILE ==="
echo "CSV contents:"
cat "$RESULTS_FILE"

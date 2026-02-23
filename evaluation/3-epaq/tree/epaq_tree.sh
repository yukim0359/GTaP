#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -W group_list=gc64
#PBS -j oe

BENCHMARK_NAME=tree

cd "$PBS_O_WORKDIR"

COMPARE_DIR=$(pwd)
PROJECT_ROOT=$(cd "$COMPARE_DIR" && pwd)
BIN_DIR="$PROJECT_ROOT/bin"

# Test values for tree height
HEIGHTS=(10 11 12 13 14 15 16 17 18 19 20)

# Fixed parameters for testing
MEM_OPS=512
COMPUTE_ITERS=512

# Number of runs for averaging
NUM_RUNS=20

# Create results file
RESULTS_FILE="epaq_performance_results_$BENCHMARK_NAME.csv"
echo "height,1queue_med,1queue_err_low,1queue_err_high,2queue_med,2queue_err_low,2queue_err_high,Speedup_1queue/2queue" > $RESULTS_FILE

# Pretty header (fixed width columns)
printf "%6s | %25s | %25s | %10s\n" "height" "1queue (ms)" "2queue (ms)" "Speedup"
printf "%6s-+-%25s-+-%25s-+-%10s\n" "------" "-------------------------" "-------------------------" "----------"


run_stats() {
    local program=$1
    local height=$2
    local compute_iters=$3
    local mem_ops=$4
    local grep_pattern=$5
    local times=()

    for i in $(seq 1 $NUM_RUNS); do
        local output
        output=$($program $height $compute_iters $mem_ops 2>&1)

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

for height in "${HEIGHTS[@]}"; do
    # --- 1-queue ---
    QUEUE1_MED=0 QUEUE1_ELO=0 QUEUE1_EHI=0
    if [ -x "$BIN_DIR/${BENCHMARK_NAME}_queue_1" ]; then
        read QUEUE1_MED _ _ QUEUE1_ELO QUEUE1_EHI < <(run_stats "$BIN_DIR/${BENCHMARK_NAME}_queue_1" "$height" "$COMPUTE_ITERS" "$MEM_OPS" "Execution time")
    fi

    # --- 2-queue ---
    QUEUE2_MED=0 QUEUE2_ELO=0 QUEUE2_EHI=0
    if [ -x "$BIN_DIR/${BENCHMARK_NAME}_queue_2" ]; then
        read QUEUE2_MED _ _ QUEUE2_ELO QUEUE2_EHI < <(run_stats "$BIN_DIR/${BENCHMARK_NAME}_queue_2" "$height" "$COMPUTE_ITERS" "$MEM_OPS" "Execution time")
    fi

    # Speedup (1-queue / 2-queue) based on medians
    SPEEDUP=0
    if [ "$QUEUE1_MED" != "0" ] && [ "$QUEUE2_MED" != "0" ]; then
        SPEEDUP=$(echo "scale=6; $QUEUE1_MED / $QUEUE2_MED" | bc -l)
    fi

    # Print aligned row
    printf "%6d | " "$height"
    fmt_med_iqr "$QUEUE1_MED" "$QUEUE1_ELO" "$QUEUE1_EHI"; printf " | "
    fmt_med_iqr "$QUEUE2_MED"  "$QUEUE2_ELO"  "$QUEUE2_EHI";  printf " | "
    if [ "$SPEEDUP" = "0" ]; then
        printf "%10.2f\n" "N/A"
    else
        printf "%10.2f\n" "$SPEEDUP"
    fi

    # CSV
    echo "$height,$QUEUE1_MED,$QUEUE1_ELO,$QUEUE1_EHI,$QUEUE2_MED,$QUEUE2_ELO,$QUEUE2_EHI,$SPEEDUP" >> "$RESULTS_FILE"
done

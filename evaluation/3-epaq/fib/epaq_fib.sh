#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -W group_list=gc64
#PBS -j oe

BENCHMARK_NAME=fib

cd "$PBS_O_WORKDIR"

COMPARE_DIR=$(pwd)
PROJECT_ROOT=$(cd "$COMPARE_DIR" && pwd)
BIN_DIR="$PROJECT_ROOT/bin"

# Test values for cutoff depth
CUTOFF_DEPTHS=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)

# Fixed n value for testing
N_VALUE=40

# Number of runs for averaging
NUM_RUNS=20

# Create results file
RESULTS_FILE="epaq_performance_results_$BENCHMARK_NAME.csv"
echo "cutoff,1queue_med,1queue_err_low,1queue_err_high,3queue_med,3queue_err_low,3queue_err_high,Speedup_1queue/3queue" > $RESULTS_FILE

# Pretty header (fixed width columns)
printf "%6s | %25s | %25s | %10s\n" "cutoff" "1queue (ms)" "3queue (ms)" "Speedup"
printf "%6s-+-%25s-+-%25s-+-%10s\n" "------" "-------------------------" "-------------------------" "----------"


run_stats() {
    local program=$1
    local n=$2
    local cutoff=$3
    local grep_pattern=$4
    local times=()

    for i in $(seq 1 $NUM_RUNS); do
        local output
        output=$($program $n $cutoff 2>&1)

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

for cutoff in "${CUTOFF_DEPTHS[@]}"; do
    # --- 1-queue ---
    QUEUE1_MED=0 QUEUE1_ELO=0 QUEUE1_EHI=0
    if [ -x "$BIN_DIR/${BENCHMARK_NAME}_queue_1" ]; then
        read QUEUE1_MED _ _ QUEUE1_ELO QUEUE1_EHI < <(run_stats "$BIN_DIR/${BENCHMARK_NAME}_queue_1" "$N_VALUE" "$cutoff" "Execution time")
    fi

    # --- 3-queue ---
    QUEUE3_MED=0 QUEUE3_ELO=0 QUEUE3_EHI=0
    if [ -x "$BIN_DIR/${BENCHMARK_NAME}_queue_3" ]; then
        read QUEUE3_MED _ _ QUEUE3_ELO QUEUE3_EHI < <(run_stats "$BIN_DIR/${BENCHMARK_NAME}_queue_3" "$N_VALUE" "$cutoff" "Execution time")
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
        printf "%10.2f\n" "N/A"
    else
        printf "%10.2f\n" "$SPEEDUP"
    fi

    # CSV
    echo "$cutoff,$QUEUE1_MED,$QUEUE1_ELO,$QUEUE1_EHI,$QUEUE3_MED,$QUEUE3_ELO,$QUEUE3_EHI,$SPEEDUP" >> "$RESULTS_FILE"
done

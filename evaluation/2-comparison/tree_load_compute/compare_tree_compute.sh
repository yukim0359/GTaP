#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -W group_list=gc64
#PBS -j oe

BENCHMARK_NAME=tree_load_compute

cd "$PBS_O_WORKDIR"

COMPARE_DIR=$(pwd)
PROJECT_ROOT=$(cd "$COMPARE_DIR" && pwd)
BIN_DIR="$PROJECT_ROOT/bin"

if [ ! -d "$BIN_DIR" ]; then
    echo "Error: Please submit this script from gtap/evaluation/2-comparison/$BENCHMARK_NAME"
    exit 1
fi

export OMP_STACKSIZE=10M

# Vary compute iterations (compute_iters)
COMPUTE_VALUES=(64 128 256 512 1024 2048 4096 8192 16384 32768)
DEPTH=20
MEMORY=256

# Number of runs
NUM_RUNS=20

# Results CSV (median + IQR error bars)
RESULTS_FILE="$COMPARE_DIR/${BENCHMARK_NAME}_compute_results.csv"
echo "compute_iters,GTAP_block_med,GTAP_block_err_low,GTAP_block_err_high,GTAP_thread_med,GTAP_thread_err_low,GTAP_thread_err_high,OMP_med,OMP_err_low,OMP_err_high" > "$RESULTS_FILE"

# Pretty header (fixed width columns)
printf "%12s | %25s | %25s | %25s\n" "compute_iters" "GTAP_block (ms)" "GTAP_thread (ms)" "OpenMP (ms)"
printf "%12s-+-%25s-+-%25s-+-%25s\n" "------------" "-------------------------" "-------------------------" "-------------------------"

run_stats() {
    local program=$1
    local depth=$2
    local compute=$3
    local memory=$4
    local grep_pattern=$5
    local times=()

    for i in $(seq 1 $NUM_RUNS); do
        local output
        output=$($program $depth $compute $memory 2>&1)

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

for compute in "${COMPUTE_VALUES[@]}"; do
    # --- GTAP_BLOCK ---
    GTAP_BLOCK_MED=0 GTAP_BLOCK_ELO=0 GTAP_BLOCK_EHI=0
    if [ -x "$BIN_DIR/gtap_block_$BENCHMARK_NAME" ]; then
        read GTAP_BLOCK_MED _ _ GTAP_BLOCK_ELO GTAP_BLOCK_EHI < <(run_stats "$BIN_DIR/gtap_block_$BENCHMARK_NAME" "$DEPTH" "$compute" "$MEMORY" "Execution time")
    fi

    # --- GTAP_THREAD ---
    GTAP_THREAD_MED=0 GTAP_THREAD_ELO=0 GTAP_THREAD_EHI=0
    if [ -x "$BIN_DIR/gtap_thread_$BENCHMARK_NAME" ]; then
        read GTAP_THREAD_MED _ _ GTAP_THREAD_ELO GTAP_THREAD_EHI < <(run_stats "$BIN_DIR/gtap_thread_$BENCHMARK_NAME" "$DEPTH" "$compute" "$MEMORY" "Execution time")
    fi

    # --- OpenMP ---
    OMP_MED=0 OMP_ELO=0 OMP_EHI=0
    if [ -x "$BIN_DIR/omp_$BENCHMARK_NAME" ]; then
        read OMP_MED _ _ OMP_ELO OMP_EHI < <(run_stats "$BIN_DIR/omp_$BENCHMARK_NAME" "$DEPTH" "$compute" "$MEMORY" "Execution time")
    fi

    # Print aligned row
    printf "%12d | " "$compute"
    fmt_med_iqr "$GTAP_BLOCK_MED" "$GTAP_BLOCK_ELO" "$GTAP_BLOCK_EHI"; printf " | "
    fmt_med_iqr "$GTAP_THREAD_MED" "$GTAP_THREAD_ELO" "$GTAP_THREAD_EHI"; printf " | "
    fmt_med_iqr "$OMP_MED"  "$OMP_ELO"  "$OMP_EHI";  printf "\n"

    # CSV
    echo "$compute,$GTAP_BLOCK_MED,$GTAP_BLOCK_ELO,$GTAP_BLOCK_EHI,$GTAP_THREAD_MED,$GTAP_THREAD_ELO,$GTAP_THREAD_EHI,$OMP_MED,$OMP_ELO,$OMP_EHI" >> "$RESULTS_FILE"
done


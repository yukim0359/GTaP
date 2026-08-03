#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -W group_list=gc64
#PBS -j oe

# Depth sweep: thread / block / block-cutoff (K=1 binary tree).
BENCHMARK_NAME=binary_tree

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PBS_O_WORKDIR:-$SCRIPT_DIR}"

COMPARE_DIR=$(pwd)
BIN_DIR="$COMPARE_DIR/bin"
mkdir -p "$BIN_DIR"

echo "Building GTaP binaries (force rebuild for updated runtime) ..."
make -C "$COMPARE_DIR" -B gtap_thread gtap_block gtap_block_cutoff

DEPTH_VALUES=(12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28)
COMPUTE=${COMPUTE:-2048}
MEMORY=${MEMORY:-0}
NUM_RUNS=${NUM_RUNS:-20}

RESULTS_FILE="$COMPARE_DIR/${BENCHMARK_NAME}_performance_results.csv"
echo "n,GTAP_thread_med,GTAP_thread_err_low,GTAP_thread_err_high,GTAP_block_med,GTAP_block_err_low,GTAP_block_err_high,GTAP_block_cutoff_med,GTAP_block_cutoff_err_low,GTAP_block_cutoff_err_high" > "$RESULTS_FILE"

printf "%6s | %25s | %25s | %25s\n" "n" "thread (ms)" "block (ms)" "block cutoff (ms)"
printf "%6s-+-%25s-+-%25s-+-%25s\n" \
    "------" "-------------------------" "-------------------------" "-------------------------"

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

    IFS=$'\n' sorted=($(printf "%s\n" "${times[@]}" | sort -n))
    unset IFS
    m=${#sorted[@]}

    local median
    if (( m % 2 == 1 )); then
        median=${sorted[$((m/2))]}
    else
        median=$(echo "scale=6; (${sorted[$((m/2-1))]} + ${sorted[$((m/2))]}) / 2" | bc -l)
    fi

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

for depth in "${DEPTH_VALUES[@]}"; do
    TH_MED=0 TH_ELO=0 TH_EHI=0
    if [ -x "$BIN_DIR/gtap_thread_$BENCHMARK_NAME" ]; then
        read TH_MED _ _ TH_ELO TH_EHI < <(run_stats "$BIN_DIR/gtap_thread_$BENCHMARK_NAME" "$depth" "$COMPUTE" "$MEMORY" "Execution time")
    fi

    BLK_MED=0 BLK_ELO=0 BLK_EHI=0
    if [ -x "$BIN_DIR/gtap_block_$BENCHMARK_NAME" ]; then
        read BLK_MED _ _ BLK_ELO BLK_EHI < <(run_stats "$BIN_DIR/gtap_block_$BENCHMARK_NAME" "$depth" "$COMPUTE" "$MEMORY" "Execution time")
    fi

    CUT_MED=0 CUT_ELO=0 CUT_EHI=0
    if [ -x "$BIN_DIR/gtap_block_${BENCHMARK_NAME}_cutoff" ]; then
        read CUT_MED _ _ CUT_ELO CUT_EHI < <(run_stats "$BIN_DIR/gtap_block_${BENCHMARK_NAME}_cutoff" "$depth" "$COMPUTE" "$MEMORY" "Execution time")
    fi

    printf "%6d | " "$depth"
    fmt_med_iqr "$TH_MED" "$TH_ELO" "$TH_EHI"; printf " | "
    fmt_med_iqr "$BLK_MED" "$BLK_ELO" "$BLK_EHI"; printf " | "
    fmt_med_iqr "$CUT_MED" "$CUT_ELO" "$CUT_EHI"; printf "\n"

    echo "$depth,$TH_MED,$TH_ELO,$TH_EHI,$BLK_MED,$BLK_ELO,$BLK_EHI,$CUT_MED,$CUT_ELO,$CUT_EHI" >> "$RESULTS_FILE"
done

echo "Wrote $RESULTS_FILE"
cd "$COMPARE_DIR" && python3 plot_performance_tree.py

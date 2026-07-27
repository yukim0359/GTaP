#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -W group_list=gc64
#PBS -j oe

# compute_iters sweep for hetero_tree (4 impls): thread wo/DAQ, block, block-cutoff.
# Usage:
#   HETERO_TREE_K=2 qsub -v HETERO_TREE_K compare_hetero_compute.sh
#   HETERO_TREE_K=4 bash compare_hetero_compute.sh
# Writes: hetero_tree_compute_results_K${HETERO_TREE_K}.csv

BENCHMARK_NAME=hetero_tree

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PBS_O_WORKDIR:-$SCRIPT_DIR}"

COMPARE_DIR=$(pwd)
BIN_DIR="$COMPARE_DIR/bin"
mkdir -p "$BIN_DIR"

HETERO_TREE_K=${HETERO_TREE_K:-2}
DEPTH=${DEPTH:-25}
NUM_RUNS=${NUM_RUNS:-20}

echo "Building hetero_tree binaries (K=${HETERO_TREE_K}) ..."
make -C "$COMPARE_DIR" -B all HETERO_TREE_K="$HETERO_TREE_K"

# compute_iters sweep: D fixed.
COMPUTE_VALUES=(64 128 256 512 1024 2048 4096 8192 16384 32768)

RESULTS_FILE="$COMPARE_DIR/${BENCHMARK_NAME}_compute_results_K${HETERO_TREE_K}.csv"
echo "compute_iters,GTAP_thread_med,GTAP_thread_err_low,GTAP_thread_err_high,GTAP_thread_daq_med,GTAP_thread_daq_err_low,GTAP_thread_daq_err_high,GTAP_block_med,GTAP_block_err_low,GTAP_block_err_high,GTAP_block_cutoff_med,GTAP_block_cutoff_err_low,GTAP_block_cutoff_err_high" > "$RESULTS_FILE"

echo "K=${HETERO_TREE_K} DEPTH=${DEPTH} NUM_RUNS=${NUM_RUNS}"
printf "%12s | %25s | %25s | %25s | %25s\n" \
    "compute_iters" "thread wo (ms)" "thread DAQ (ms)" "block (ms)" "block cutoff (ms)"
printf "%12s-+-%25s-+-%25s-+-%25s-+-%25s\n" \
    "------------" "-------------------------" "-------------------------" \
    "-------------------------" "-------------------------"

run_stats() {
    local program=$1
    local depth=$2
    local compute=$3
    local grep_pattern=$4
    local times=()

    for i in $(seq 1 $NUM_RUNS); do
        local output
        output=$($program $depth $compute 2>&1)

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

for compute in "${COMPUTE_VALUES[@]}"; do
    TH_MED=0 TH_ELO=0 TH_EHI=0
    if [ -x "$BIN_DIR/gtap_thread_hetero_tree" ]; then
        read TH_MED _ _ TH_ELO TH_EHI < <(run_stats "$BIN_DIR/gtap_thread_hetero_tree" "$DEPTH" "$compute" "Execution time")
    fi

    DAQ_MED=0 DAQ_ELO=0 DAQ_EHI=0
    if [ -x "$BIN_DIR/gtap_thread_hetero_tree_daq" ]; then
        read DAQ_MED _ _ DAQ_ELO DAQ_EHI < <(run_stats "$BIN_DIR/gtap_thread_hetero_tree_daq" "$DEPTH" "$compute" "Execution time")
    fi

    BLK_MED=0 BLK_ELO=0 BLK_EHI=0
    if [ -x "$BIN_DIR/gtap_block_hetero_tree" ]; then
        read BLK_MED _ _ BLK_ELO BLK_EHI < <(run_stats "$BIN_DIR/gtap_block_hetero_tree" "$DEPTH" "$compute" "Execution time")
    fi

    CUT_MED=0 CUT_ELO=0 CUT_EHI=0
    if [ -x "$BIN_DIR/gtap_block_hetero_tree_cutoff" ]; then
        read CUT_MED _ _ CUT_ELO CUT_EHI < <(run_stats "$BIN_DIR/gtap_block_hetero_tree_cutoff" "$DEPTH" "$compute" "Execution time")
    fi

    printf "%12d | " "$compute"
    fmt_med_iqr "$TH_MED" "$TH_ELO" "$TH_EHI"; printf " | "
    fmt_med_iqr "$DAQ_MED" "$DAQ_ELO" "$DAQ_EHI"; printf " | "
    fmt_med_iqr "$BLK_MED" "$BLK_ELO" "$BLK_EHI"; printf " | "
    fmt_med_iqr "$CUT_MED" "$CUT_ELO" "$CUT_EHI"; printf "\n"

    echo "$compute,$TH_MED,$TH_ELO,$TH_EHI,$DAQ_MED,$DAQ_ELO,$DAQ_EHI,$BLK_MED,$BLK_ELO,$BLK_EHI,$CUT_MED,$CUT_ELO,$CUT_EHI" >> "$RESULTS_FILE"
done

echo "Wrote $RESULTS_FILE"

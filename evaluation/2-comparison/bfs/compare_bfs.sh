#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=01:30:00
#PBS -W group_list=gc64
#PBS -j oe

BENCHMARK_NAME=bfs

cd "$PBS_O_WORKDIR"

COMPARE_DIR=$(pwd)
PROJECT_ROOT=$(cd "$COMPARE_DIR" && cd ../../../../ && pwd)
BIN_DIR="$COMPARE_DIR/bin"
DATASETS_DIR="$PROJECT_ROOT/ATOS/datasets"

if [ ! -d "$BIN_DIR" ]; then
    echo "Error: Please submit this script from gtap/evaluation/2-comparison/$BENCHMARK_NAME"
    exit 1
fi

# 4 datasets from run_bfs.batch (non-commented ones)
declare -A DATASETS=(
    ["soc-LiveJournal1"]="$DATASETS_DIR/soc-LiveJournal1/soc-LiveJournal1_di.csr"
    ["hollywood-2009"]="$DATASETS_DIR/hollywood-2009/hollywood-2009_ud.csr"
    ["indochina-2004"]="$DATASETS_DIR/indochina-2004/indochina-2004_di.csr"
    ["road_usa"]="$DATASETS_DIR/road_usa/road_usa_ud.csr"
)

# Source vertex for each dataset (default is 0, indochina-2004 uses 40)
declare -A SOURCE_VERTICES=(
    ["soc-LiveJournal1"]=0
    ["hollywood-2009"]=0
    ["indochina-2004"]=40
    ["road_usa"]=0
)

# Number of runs
NUM_RUNS=20

# Results CSV
RESULTS_FILE="$COMPARE_DIR/${BENCHMARK_NAME}_performance_results.csv"
echo "Dataset,GTAP_THREAD_avg,GTAP_BLOCK_avg,Speedup(BLOCK/THREAD)" > "$RESULTS_FILE"

# Pretty header (fixed width columns)
printf "%20s | %25s | %25s | %10s\n" "Dataset" "GTAP_THREAD (ms)" "GTAP_BLOCK (ms)" "Speedup"
printf "%20s-+-%25s-+-%25s-+-%10s\n" "--------------------" "-------------------------" "-------------------------" "----------"

run_average() {
    local program=$1
    local dataset_path=$2
    local source=$3
    local times=()
    local sum=0
    local count=0

    for i in $(seq 1 $NUM_RUNS); do
        local output
        output=$($program "$dataset_path" "$source" 2>&1)

        local time
        time=$(echo "$output" | grep "Execution time:" | sed -n 's/.*Execution time: \([0-9.]*\) ms.*/\1/p' | head -n 1)

        if [ -n "$time" ]; then
            times+=("$time")
            sum=$(echo "scale=6; $sum + $time" | bc -l)
            count=$((count + 1))
        fi
    done

    if [ "$count" -eq 0 ]; then
        echo "0"
        return
    fi

    local avg
    avg=$(echo "scale=6; $sum / $count" | bc -l)
    echo "$avg"
}

for dataset_name in "${!DATASETS[@]}"; do
    dataset_path="${DATASETS[$dataset_name]}"
    source_vertex="${SOURCE_VERTICES[$dataset_name]}"
    
    if [ ! -f "$dataset_path" ]; then
        echo "Warning: Dataset file not found: $dataset_path"
        continue
    fi

    # --- GTAP_THREAD ---
    GTAP_THREAD_AVG=0
    if [ -x "$BIN_DIR/gtap_thread_$BENCHMARK_NAME" ]; then
        GTAP_THREAD_AVG=$(run_average "$BIN_DIR/gtap_thread_$BENCHMARK_NAME" "$dataset_path" "$source_vertex")
    fi

    # --- GTAP_BLOCK ---
    GTAP_BLOCK_AVG=0
    if [ -x "$BIN_DIR/gtap_block_$BENCHMARK_NAME" ]; then
        GTAP_BLOCK_AVG=$(run_average "$BIN_DIR/gtap_block_$BENCHMARK_NAME" "$dataset_path" "$source_vertex")
    fi

    # Speedup (BLOCK / THREAD) based on averages
    SPEEDUP=0
    if [ "$GTAP_THREAD_AVG" != "0" ] && [ "$GTAP_BLOCK_AVG" != "0" ]; then
        SPEEDUP=$(echo "scale=6; $GTAP_BLOCK_AVG / $GTAP_THREAD_AVG" | bc -l)
    fi

    # Print aligned row
    printf "%20s | " "$dataset_name"
    if [ "$GTAP_THREAD_AVG" = "0" ]; then
        printf "%25s | " "N/A"
    else
        printf "%25.3f | " "$GTAP_THREAD_AVG"
    fi
    if [ "$GTAP_BLOCK_AVG" = "0" ]; then
        printf "%25s | " "N/A"
    else
        printf "%25.3f | " "$GTAP_BLOCK_AVG"
    fi
    if [ "$SPEEDUP" = "0" ]; then
        printf "%10s\n" "N/A"
    else
        printf "%10.3f\n" "$SPEEDUP"
    fi

    # CSV
    echo "$dataset_name,$GTAP_THREAD_AVG,$GTAP_BLOCK_AVG,$SPEEDUP" >> "$RESULTS_FILE"
done

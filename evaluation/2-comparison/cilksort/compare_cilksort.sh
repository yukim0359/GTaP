#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -W group_list=gc64
#PBS -j oe

BENCHMARK_NAME=cilksort

cd "$PBS_O_WORKDIR"

COMPARE_DIR=$(pwd)
PROJECT_ROOT=$(cd "$COMPARE_DIR" && pwd)
BIN_DIR="$PROJECT_ROOT/bin"
UTIL_DIR="$PROJECT_ROOT/../../util"
TMP_DIR="$COMPARE_DIR/tmp"

if [ ! -d "$BIN_DIR" ]; then
    echo "Error: Please submit this script from gtap/evaluation/2-comparison/$BENCHMARK_NAME"
    exit 1
fi

mkdir -p "$TMP_DIR"

SIZES=(100000 200000 500000 1000000 2000000 5000000 10000000 20000000 50000000 100000000)

NUM_RUNS=20

# Results CSV (median + IQR error bars)
RESULTS_FILE="$COMPARE_DIR/${BENCHMARK_NAME}_performance_results.csv"
echo "n,GTAP_med,GTAP_err_low,GTAP_err_high,OMP_med,OMP_err_low,OMP_err_high,SEQ_med,SEQ_err_low,SEQ_err_high,Speedup_med(OMP/GTAP)" > "$RESULTS_FILE"

# Pretty header (fixed width columns)
printf "%6s | %25s | %25s | %25s | %10s\n" "size" "GTAP (ms)" "OpenMP (ms)" "Seq (ms)" "Speedup"
printf "%6s-+-%25s-+-%25s-+-%25s-+-%10s\n" "------" "-------------------------" "-------------------------" "-------------------------" "----------"

run_stats() {
    local program=$1
    local n=$2
    local grep_pattern=$3
    local times=()

    for i in $(seq 1 $NUM_RUNS); do
        local output
        output=$($program $n 2>&1)

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

for size in "${SIZES[@]}"; do
    echo "  Generating test data..."
    cd "$UTIL_DIR"
    DATA_FILE="test_data_${size}.bin"
    ./gen_vector "$size" "$TMP_DIR/$DATA_FILE" > /dev/null 2>&1
    cd "$COMPARE_DIR"

    # --- GTAP ---
    GTAP_MED=0 GTAP_ELO=0 GTAP_EHI=0
    if [ -x "$BIN_DIR/gtap_$BENCHMARK_NAME" ]; then
        read GTAP_MED _ _ GTAP_ELO GTAP_EHI < <(run_stats "$BIN_DIR/gtap_$BENCHMARK_NAME" "$TMP_DIR/$DATA_FILE" "Execution time")
    fi

    # --- OpenMP ---
    OMP_MED=0 OMP_ELO=0 OMP_EHI=0
    if [ -x "$BIN_DIR/omp_$BENCHMARK_NAME" ]; then
        read OMP_MED _ _ OMP_ELO OMP_EHI < <(run_stats "$BIN_DIR/omp_$BENCHMARK_NAME" "$TMP_DIR/$DATA_FILE" "Execution time")
    fi

    # --- Sequential ---
    SEQ_MED=0 SEQ_ELO=0 SEQ_EHI=0
    if [ -x "$BIN_DIR/seq_$BENCHMARK_NAME" ]; then
        read SEQ_MED _ _ SEQ_ELO SEQ_EHI < <(run_stats "$BIN_DIR/seq_$BENCHMARK_NAME" "$TMP_DIR/$DATA_FILE" "Execution time")
    fi

    # Speedup (OMP / GTAP) based on medians
    SPEEDUP=0
    if [ "$GTAP_MED" != "0" ] && [ "$OMP_MED" != "0" ]; then
        SPEEDUP=$(echo "scale=6; $OMP_MED / $GTAP_MED" | bc -l)
    fi

    rm -f "$TMP_DIR/$DATA_FILE"

    # Print aligned row
    printf "%6d | " "$size"
    fmt_med_iqr "$GTAP_MED" "$GTAP_ELO" "$GTAP_EHI"; printf " | "
    fmt_med_iqr "$OMP_MED"  "$OMP_ELO"  "$OMP_EHI";  printf " | "
    fmt_med_iqr "$SEQ_MED"  "$SEQ_ELO"  "$SEQ_EHI";  printf " | "
    if [ "$SPEEDUP" = "0" ]; then
        printf "%10.2f\n" "N/A"
    else
        printf "%10.2f\n" "$SPEEDUP"
    fi

    # CSV
    echo "$size,$GTAP_MED,$GTAP_ELO,$GTAP_EHI,$OMP_MED,$OMP_ELO,$OMP_EHI,$SEQ_MED,$SEQ_ELO,$SEQ_EHI,$SPEEDUP" >> "$RESULTS_FILE"
done

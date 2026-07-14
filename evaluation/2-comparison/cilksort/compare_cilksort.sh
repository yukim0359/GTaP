#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=03:00:00
#PBS -W group_list=gc64
#PBS -j oe

BENCHMARK_NAME=cilksort

# Sort correctness checks (reference qsort + is_sorted) are expensive at large n.
# Disable for benchmark runs:
#   ./compare_cilksort.sh --no-validate
#   VALIDATE=0 ./compare_cilksort.sh
#   qsub -v VALIDATE=0 compare_cilksort.sh
VALIDATE="${VALIDATE:-1}"
for arg in "$@"; do
    case "$arg" in
        --no-validate|--skip-validate) VALIDATE=0 ;;
        --validate) VALIDATE=1 ;;
    esac
done
if [ "$VALIDATE" = "0" ]; then
    export CILKSORT_VALIDATE=0
    echo "Sort validation: disabled"
else
    unset CILKSORT_VALIDATE
    echo "Sort validation: enabled"
fi
VALIDATE_ARGS=()
if [ "$VALIDATE" = "0" ]; then
    VALIDATE_ARGS=(--no-validate)
fi

cd "$PBS_O_WORKDIR"

COMPARE_DIR=$(pwd)
PROJECT_ROOT=$(cd "$COMPARE_DIR/../../.." && pwd)
BIN_DIR="$COMPARE_DIR/bin"
UTIL_DIR="$PROJECT_ROOT/evaluation/util"
TMP_DIR="$COMPARE_DIR/tmp"

if [ ! -d "$BIN_DIR" ]; then
    echo "Error: Please submit this script from gtap/evaluation/2-comparison/$BENCHMARK_NAME"
    exit 1
fi

mkdir -p "$TMP_DIR"

SIZES=(100000 200000 500000 1000000 2000000 5000000 10000000 20000000 50000000 100000000 200000000 500000000 1000000000)

NUM_RUNS=20
# Set INCLUDE_SEQ=1 to benchmark CPU sequential (default: skip).
INCLUDE_SEQ="${INCLUDE_SEQ:-0}"

### cilksort-specific settings ###
export OMP_STACKSIZE=500M
export CILK_NWORKERS="${CILK_NWORKERS:-${OMP_NUM_THREADS:-$(nproc)}}"

# Dynasoar fork-join implementation
if [ -z "${DYNASOAR_ROOT:-}" ]; then
    for _candidate in \
        "$COMPARE_DIR/../../../../fine-grained-fork-join" \
        "$HOME/fine-grained-fork-join"; do
        if [ -d "$_candidate" ]; then
            DYNASOAR_ROOT=$(cd "$_candidate" && pwd)
            break
        fi
    done
fi
DYNASOAR_CILKSORT_BIN="${DYNASOAR_CILKSORT_BIN:-${DYNASOAR_ROOT:+$DYNASOAR_ROOT/bin/CilkSort}}"

ensure_dynasoar_bin() {
    if [ -z "${DYNASOAR_ROOT:-}" ] || [ ! -d "$DYNASOAR_ROOT" ]; then
        echo "Warning: fine-grained-fork-join not found (set DYNASOAR_ROOT); skipping Dynasoar runs" >&2
        return 1
    fi
    if [ -x "$DYNASOAR_CILKSORT_BIN" ]; then
        return 0
    fi
    echo "Building Dynasoar CilkSort in $DYNASOAR_ROOT ..."
    make -C "$DYNASOAR_ROOT" bin/CilkSort || {
        echo "Warning: Dynasoar CilkSort build failed; skipping Dynasoar runs" >&2
        return 1
    }
    [ -x "$DYNASOAR_CILKSORT_BIN" ]
}

# Results CSV (median + IQR error bars)
RESULTS_FILE="$COMPARE_DIR/${BENCHMARK_NAME}_performance_results.csv"
echo "n,GTAP_med,GTAP_err_low,GTAP_err_high,THRUST_med,THRUST_err_low,THRUST_err_high,OMP_med,OMP_err_low,OMP_err_high,CILK_med,CILK_err_low,CILK_err_high,DYNASOAR_med,DYNASOAR_err_low,DYNASOAR_err_high,SEQ_med,SEQ_err_low,SEQ_err_high,Speedup_med(OMP/GTAP),Speedup_med(CILK/GTAP),Speedup_med(DYNASOAR/GTAP),Speedup_med(GTAP/THRUST)" > "$RESULTS_FILE"

DYNASOAR_ENABLED=0
if ensure_dynasoar_bin; then
    DYNASOAR_ENABLED=1
    echo "Dynasoar CilkSort: $DYNASOAR_CILKSORT_BIN"
else
    echo "Dynasoar CilkSort: disabled"
fi

# Pretty header (fixed width columns)
if [ "$INCLUDE_SEQ" -eq 1 ]; then
    printf "%6s | %25s | %25s | %25s | %25s | %25s | %25s | %10s\n" \
        "size" "GTAP (ms)" "Thrust (ms)" "OpenMP (ms)" "Cilk (ms)" "Dynasoar (ms)" "Seq (ms)" "GTAP/Thrust"
    printf "%6s-+-%25s-+-%25s-+-%25s-+-%25s-+-%25s-+-%25s-+-%10s\n" \
        "------" "-------------------------" "-------------------------" "-------------------------" \
        "-------------------------" "-------------------------" "-------------------------" "----------"
else
    printf "%6s | %25s | %25s | %25s | %25s | %25s | %10s\n" \
        "size" "GTAP (ms)" "Thrust (ms)" "OpenMP (ms)" "Cilk (ms)" "Dynasoar (ms)" "GTAP/Thrust"
    printf "%6s-+-%25s-+-%25s-+-%25s-+-%25s-+-%25s-+-%10s\n" \
        "------" "-------------------------" "-------------------------" "-------------------------" \
        "-------------------------" "-------------------------" "----------"
fi

run_stats() {
    local program=$1
    local n=$2
    local grep_pattern=$3
    local times=()

    for i in $(seq 1 $NUM_RUNS); do
        local output
        output=$("$program" "${VALIDATE_ARGS[@]}" "$n" 2>&1)

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

run_stats_dynasoar() {
    local program=$1
    local data_file=$2
    local times=()

    for i in $(seq 1 $NUM_RUNS); do
        local output
        output=$("$program" "$data_file" 2>&1)

        local time
        time=$(echo "$output" | grep '^\[TIME\]' | sed -n 's/.*time = \([0-9.]*\).*/\1/p' | head -n 1)

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

    # --- Thrust (GPU standard sort) ---
    THRUST_MED=0 THRUST_ELO=0 THRUST_EHI=0
    if [ -x "$BIN_DIR/thrust_$BENCHMARK_NAME" ]; then
        read THRUST_MED _ _ THRUST_ELO THRUST_EHI < <(run_stats "$BIN_DIR/thrust_$BENCHMARK_NAME" "$TMP_DIR/$DATA_FILE" "Execution time")
    fi

    # --- OpenMP ---
    OMP_MED=0 OMP_ELO=0 OMP_EHI=0
    if [ -x "$BIN_DIR/omp_$BENCHMARK_NAME" ]; then
        read OMP_MED _ _ OMP_ELO OMP_EHI < <(run_stats "$BIN_DIR/omp_$BENCHMARK_NAME" "$TMP_DIR/$DATA_FILE" "Execution time")
    fi

    # --- Cilk ---
    CILK_MED=0 CILK_ELO=0 CILK_EHI=0
    if [ -x "$BIN_DIR/cilk_$BENCHMARK_NAME" ]; then
        read CILK_MED _ _ CILK_ELO CILK_EHI < <(run_stats "$BIN_DIR/cilk_$BENCHMARK_NAME" "$TMP_DIR/$DATA_FILE" "Execution time")
    fi

    # --- Dynasoar ---
    DYNASOAR_MED=0 DYNASOAR_ELO=0 DYNASOAR_EHI=0
    if [ "$DYNASOAR_ENABLED" = "1" ]; then
        read DYNASOAR_MED _ _ DYNASOAR_ELO DYNASOAR_EHI < <(run_stats_dynasoar "$DYNASOAR_CILKSORT_BIN" "$TMP_DIR/$DATA_FILE")
    fi

    # --- Sequential (optional) ---
    SEQ_MED=0 SEQ_ELO=0 SEQ_EHI=0
    if [ "$INCLUDE_SEQ" -eq 1 ] && [ -x "$BIN_DIR/seq_$BENCHMARK_NAME" ]; then
        read SEQ_MED _ _ SEQ_ELO SEQ_EHI < <(run_stats "$BIN_DIR/seq_$BENCHMARK_NAME" "$TMP_DIR/$DATA_FILE" "Execution time")
    fi

    # Speedup based on medians
    SPEEDUP=0
    CILK_SPEEDUP=0
    DYNASOAR_SPEEDUP=0
    THRUST_SPEEDUP=0
    if [ "$GTAP_MED" != "0" ] && [ "$OMP_MED" != "0" ]; then
        SPEEDUP=$(echo "scale=6; $OMP_MED / $GTAP_MED" | bc -l)
    fi
    if [ "$GTAP_MED" != "0" ] && [ "$CILK_MED" != "0" ]; then
        CILK_SPEEDUP=$(echo "scale=6; $CILK_MED / $GTAP_MED" | bc -l)
    fi
    if [ "$GTAP_MED" != "0" ] && [ "$DYNASOAR_MED" != "0" ]; then
        DYNASOAR_SPEEDUP=$(echo "scale=6; $DYNASOAR_MED / $GTAP_MED" | bc -l)
    fi
    if [ "$GTAP_MED" != "0" ] && [ "$THRUST_MED" != "0" ]; then
        THRUST_SPEEDUP=$(echo "scale=6; $GTAP_MED / $THRUST_MED" | bc -l)
    fi

    rm -f "$TMP_DIR/$DATA_FILE"

    # Print aligned row
    printf "%6d | " "$size"
    fmt_med_iqr "$GTAP_MED" "$GTAP_ELO" "$GTAP_EHI"; printf " | "
    fmt_med_iqr "$THRUST_MED" "$THRUST_ELO" "$THRUST_EHI"; printf " | "
    fmt_med_iqr "$OMP_MED"  "$OMP_ELO"  "$OMP_EHI";  printf " | "
    fmt_med_iqr "$CILK_MED" "$CILK_ELO" "$CILK_EHI"; printf " | "
    fmt_med_iqr "$DYNASOAR_MED" "$DYNASOAR_ELO" "$DYNASOAR_EHI"; printf " | "
    if [ "$INCLUDE_SEQ" -eq 1 ]; then
        fmt_med_iqr "$SEQ_MED" "$SEQ_ELO" "$SEQ_EHI"; printf " | "
    fi
    if [ "$THRUST_SPEEDUP" = "0" ]; then
        printf "%10s\n" "N/A"
    else
        printf "%10.2f\n" "$THRUST_SPEEDUP"
    fi

    # CSV
    echo "$size,$GTAP_MED,$GTAP_ELO,$GTAP_EHI,$THRUST_MED,$THRUST_ELO,$THRUST_EHI,$OMP_MED,$OMP_ELO,$OMP_EHI,$CILK_MED,$CILK_ELO,$CILK_EHI,$DYNASOAR_MED,$DYNASOAR_ELO,$DYNASOAR_EHI,$SEQ_MED,$SEQ_ELO,$SEQ_EHI,$SPEEDUP,$CILK_SPEEDUP,$DYNASOAR_SPEEDUP,$THRUST_SPEEDUP" >> "$RESULTS_FILE"
done

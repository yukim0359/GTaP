#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=00:10:00
#PBS -W group_list=gc64
#PBS -j oe

BENCHMARK_NAME=fib

cd "$PBS_O_WORKDIR"

COMPARE_DIR=$(pwd)
PROJECT_ROOT=$(cd "$COMPARE_DIR" && pwd)
BIN_DIR="$PROJECT_ROOT/bin"

if [ ! -d "$BIN_DIR" ]; then
    echo "Error: Please submit this script from gtap/evaluation/2-comparison/$BENCHMARK_NAME"
    exit 1
fi

N_VALUES=(2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40)

### fib-specific settings ###
export OMP_STACKSIZE=20M
# Match OpenMP thread count when not set explicitly.
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
DYNASOAR_FIB_TARGET="${DYNASOAR_FIB_TARGET:-FibOnlyOne}"
DYNASOAR_FIB_BIN="${DYNASOAR_FIB_BIN:-${DYNASOAR_ROOT:+$DYNASOAR_ROOT/bin/$DYNASOAR_FIB_TARGET}}"

ensure_dynasoar_bin() {
    if [ -z "${DYNASOAR_ROOT:-}" ] || [ ! -d "$DYNASOAR_ROOT" ]; then
        echo "Warning: fine-grained-fork-join not found (set DYNASOAR_ROOT); skipping Dynasoar runs" >&2
        return 1
    fi
    if [ -x "$DYNASOAR_FIB_BIN" ]; then
        return 0
    fi
    echo "Building Dynasoar $DYNASOAR_FIB_TARGET in $DYNASOAR_ROOT ..."
    make -C "$DYNASOAR_ROOT" "bin/$DYNASOAR_FIB_TARGET" || {
        echo "Warning: Dynasoar $DYNASOAR_FIB_TARGET build failed; skipping Dynasoar runs" >&2
        return 1
    }
    [ -x "$DYNASOAR_FIB_BIN" ]
}

# Number of runs
NUM_RUNS=20

# Results CSV (median + IQR error bars)
RESULTS_FILE="$COMPARE_DIR/${BENCHMARK_NAME}_performance_results.csv"
echo "n,GTAP_med,GTAP_err_low,GTAP_err_high,OMP_med,OMP_err_low,OMP_err_high,CILK_med,CILK_err_low,CILK_err_high,DYNASOAR_med,DYNASOAR_err_low,DYNASOAR_err_high,SEQ_med,SEQ_err_low,SEQ_err_high,Speedup_med(OMP/GTAP),Speedup_med(CILK/GTAP),Speedup_med(DYNASOAR/GTAP)" > "$RESULTS_FILE"

DYNASOAR_ENABLED=0
if ensure_dynasoar_bin; then
    DYNASOAR_ENABLED=1
    echo "Dynasoar Fib: $DYNASOAR_FIB_BIN"
else
    echo "Dynasoar Fib: disabled"
fi

# Pretty header (fixed width columns)
printf "%6s | %25s | %25s | %25s | %25s | %25s | %10s\n" "n" "GTAP (ms)" "OpenMP (ms)" "Cilk (ms)" "Dynasoar (ms)" "Seq (ms)" "OMP/GTAP"
printf "%6s-+-%25s-+-%25s-+-%25s-+-%25s-+-%25s-+-%10s\n" "------" "-------------------------" "-------------------------" "-------------------------" "-------------------------" "-------------------------" "----------"

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

run_stats_dynasoar() {
    local program=$1
    local n=$2
    local times=()

    for i in $(seq 1 $NUM_RUNS); do
        local output
        output=$("$program" "$n" 2>&1)

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

for n in "${N_VALUES[@]}"; do
    # --- GTAP ---
    GTAP_MED=0 GTAP_ELO=0 GTAP_EHI=0
    if [ -x "$BIN_DIR/gtap_$BENCHMARK_NAME" ]; then
        read GTAP_MED _ _ GTAP_ELO GTAP_EHI < <(run_stats "$BIN_DIR/gtap_$BENCHMARK_NAME" "$n" "Execution time")
    fi

    # --- OpenMP ---
    OMP_MED=0 OMP_ELO=0 OMP_EHI=0
    if [ -x "$BIN_DIR/omp_$BENCHMARK_NAME" ]; then
        read OMP_MED _ _ OMP_ELO OMP_EHI < <(run_stats "$BIN_DIR/omp_$BENCHMARK_NAME" "$n" "Execution time")
    fi

    # --- Cilk ---
    CILK_MED=0 CILK_ELO=0 CILK_EHI=0
    if [ -x "$BIN_DIR/cilk_$BENCHMARK_NAME" ]; then
        read CILK_MED _ _ CILK_ELO CILK_EHI < <(run_stats "$BIN_DIR/cilk_$BENCHMARK_NAME" "$n" "Execution time")
    fi

    # --- Dynasoar ---
    DYNASOAR_MED=0 DYNASOAR_ELO=0 DYNASOAR_EHI=0
    if [ "$DYNASOAR_ENABLED" = "1" ]; then
        read DYNASOAR_MED _ _ DYNASOAR_ELO DYNASOAR_EHI < <(run_stats_dynasoar "$DYNASOAR_FIB_BIN" "$n")
    fi

    # --- Sequential ---
    SEQ_MED=0 SEQ_ELO=0 SEQ_EHI=0
    if [ -x "$BIN_DIR/seq_$BENCHMARK_NAME" ]; then
        read SEQ_MED _ _ SEQ_ELO SEQ_EHI < <(run_stats "$BIN_DIR/seq_$BENCHMARK_NAME" "$n" "Execution time")
    fi

    # Speedup (OMP / GTAP, CILK / GTAP) based on medians
    SPEEDUP=0
    CILK_SPEEDUP=0
    DYNASOAR_SPEEDUP=0
    if [ "$GTAP_MED" != "0" ] && [ "$OMP_MED" != "0" ]; then
        SPEEDUP=$(echo "scale=6; $OMP_MED / $GTAP_MED" | bc -l)
    fi
    if [ "$GTAP_MED" != "0" ] && [ "$CILK_MED" != "0" ]; then
        CILK_SPEEDUP=$(echo "scale=6; $CILK_MED / $GTAP_MED" | bc -l)
    fi
    if [ "$GTAP_MED" != "0" ] && [ "$DYNASOAR_MED" != "0" ]; then
        DYNASOAR_SPEEDUP=$(echo "scale=6; $DYNASOAR_MED / $GTAP_MED" | bc -l)
    fi

    # Print aligned row
    printf "%6d | " "$n"
    fmt_med_iqr "$GTAP_MED" "$GTAP_ELO" "$GTAP_EHI"; printf " | "
    fmt_med_iqr "$OMP_MED"  "$OMP_ELO"  "$OMP_EHI";  printf " | "
    fmt_med_iqr "$CILK_MED" "$CILK_ELO" "$CILK_EHI"; printf " | "
    fmt_med_iqr "$DYNASOAR_MED" "$DYNASOAR_ELO" "$DYNASOAR_EHI"; printf " | "
    fmt_med_iqr "$SEQ_MED"  "$SEQ_ELO"  "$SEQ_EHI";  printf " | "
    if [ "$SPEEDUP" = "0" ]; then
        printf "%10s\n" "N/A"
    else
        printf "%10.2f\n" "$SPEEDUP"
    fi

    # CSV
    echo "$n,$GTAP_MED,$GTAP_ELO,$GTAP_EHI,$OMP_MED,$OMP_ELO,$OMP_EHI,$CILK_MED,$CILK_ELO,$CILK_EHI,$DYNASOAR_MED,$DYNASOAR_ELO,$DYNASOAR_EHI,$SEQ_MED,$SEQ_ELO,$SEQ_EHI,$SPEEDUP,$CILK_SPEEDUP,$DYNASOAR_SPEEDUP" >> "$RESULTS_FILE"
done

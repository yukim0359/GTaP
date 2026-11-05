#!/bin/bash

BENCHMARK_NAME=cilksort

# Sort checks compare the output to gen_vector's sidecar .sorted.bin
# (same as examples/cilksort). Timing is recorded before the check.
# Default is on. Disable with:
#   ./compare_cilksort.sh --no-validate
#   VALIDATE=0 ./compare_cilksort.sh
#   qsub -v VALIDATE=0 compare_cilksort.pbs
VALIDATE="${VALIDATE:-1}"
for arg in "$@"; do
    case "$arg" in
        --no-validate|--skip-validate) VALIDATE=0 ;;
        --validate) VALIDATE=1 ;;
    esac
done

COMPARE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
cd "$COMPARE_DIR" || exit 1
source "$COMPARE_DIR/../../../common/benchmark.sh" || exit 1
mkdir -p "$COMPARE_DIR/bin"
echo "Building cilksort binaries (force rebuild for updated runtime) ..."
make -C "$COMPARE_DIR" -B gtap omp cilk || exit 1

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

PROJECT_ROOT=$(cd "$COMPARE_DIR/../../.." && pwd) || exit 1
BIN_DIR="$COMPARE_DIR/bin"
UTIL_DIR="$PROJECT_ROOT/evaluation/util"
TMP_DIR="$COMPARE_DIR/tmp"
mkdir -p "$TMP_DIR"
make -C "$UTIL_DIR" gen_vector || exit 1

SIZES=(100000 200000 500000 1000000 2000000 5000000 10000000 20000000 50000000 100000000 200000000 500000000 1000000000)

NUM_RUNS=20

### cilksort-specific settings ###
export OMP_STACKSIZE=500M
export CILK_NWORKERS="${CILK_NWORKERS:-${OMP_NUM_THREADS:-$(nproc)}}"

# Dynasoar fork-join implementation
if [ -z "${DYNASOAR_ROOT:-}" ]; then
    for _candidate in \
        "$COMPARE_DIR/../../../../../fine-grained-fork-join" \
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
echo "n,GTAP_med,GTAP_err_low,GTAP_err_high,OMP_med,OMP_err_low,OMP_err_high,CILK_med,CILK_err_low,CILK_err_high,DYNASOAR_med,DYNASOAR_err_low,DYNASOAR_err_high,SEQ_med,SEQ_err_low,SEQ_err_high,Speedup_med(OMP/GTAP),Speedup_med(CILK/GTAP),Speedup_med(DYNASOAR/GTAP)" > "$RESULTS_FILE"

DYNASOAR_ENABLED=0
if ensure_dynasoar_bin; then
    DYNASOAR_ENABLED=1
    echo "Dynasoar CilkSort: $DYNASOAR_CILKSORT_BIN"
else
    echo "Dynasoar CilkSort: disabled"
fi

# Pretty header (fixed width columns)
printf "%6s | %25s | %25s | %25s | %25s | %10s\n" \
    "size" "GTAP (ms)" "OpenMP (ms)" "Cilk (ms)" "Dynasoar (ms)" "OMP/GTAP"
printf "%6s-+-%25s-+-%25s-+-%25s-+-%25s-+-%10s\n" \
    "------" "-------------------------" "-------------------------" "-------------------------" \
    "-------------------------" "----------"

for size in "${SIZES[@]}"; do
    echo "  Generating test data..."
    cd "$UTIL_DIR" || exit 1
    DATA_FILE="test_data_${size}.bin"
    REF_FILE="test_data_${size}.sorted.bin"
    ./gen_vector "$size" "$TMP_DIR/$DATA_FILE" || exit 1
    cd "$COMPARE_DIR" || exit 1
    if [ ! -f "$TMP_DIR/$REF_FILE" ]; then
        echo "Error: missing sorted reference $TMP_DIR/$REF_FILE" >&2
        exit 1
    fi
    RUN_ARGS=("${VALIDATE_ARGS[@]}" "$TMP_DIR/$DATA_FILE")
    if [ "$VALIDATE" = "1" ]; then
        RUN_ARGS+=("$TMP_DIR/$REF_FILE")
    fi

    # --- GTAP ---
    GTAP_MED=0 GTAP_ELO=0 GTAP_EHI=0
    if [ -x "$BIN_DIR/gtap_$BENCHMARK_NAME" ]; then
        read GTAP_MED _ _ GTAP_ELO GTAP_EHI < <(
            benchmark_run_stats "$NUM_RUNS" benchmark_parse_execution_ms \
                "$BIN_DIR/gtap_$BENCHMARK_NAME" "${RUN_ARGS[@]}"
        )
    fi

    # --- OpenMP ---
    OMP_MED=0 OMP_ELO=0 OMP_EHI=0
    if [ -x "$BIN_DIR/omp_$BENCHMARK_NAME" ]; then
        read OMP_MED _ _ OMP_ELO OMP_EHI < <(
            benchmark_run_stats "$NUM_RUNS" benchmark_parse_execution_ms \
                "$BIN_DIR/omp_$BENCHMARK_NAME" "${RUN_ARGS[@]}"
        )
    fi

    # --- Cilk ---
    CILK_MED=0 CILK_ELO=0 CILK_EHI=0
    if [ -x "$BIN_DIR/cilk_$BENCHMARK_NAME" ]; then
        read CILK_MED _ _ CILK_ELO CILK_EHI < <(
            benchmark_run_stats "$NUM_RUNS" benchmark_parse_execution_ms \
                "$BIN_DIR/cilk_$BENCHMARK_NAME" "${RUN_ARGS[@]}"
        )
    fi

    # --- Dynasoar ---
    DYNASOAR_MED=0 DYNASOAR_ELO=0 DYNASOAR_EHI=0
    if [ "$DYNASOAR_ENABLED" = "1" ]; then
        read DYNASOAR_MED _ _ DYNASOAR_ELO DYNASOAR_EHI < <(
            benchmark_run_stats "$NUM_RUNS" benchmark_parse_dynasoar_ms \
                "$DYNASOAR_CILKSORT_BIN" "$TMP_DIR/$DATA_FILE"
        )
    fi

    SEQ_MED=0 SEQ_ELO=0 SEQ_EHI=0

    # Speedup based on medians
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

    rm -f "$TMP_DIR/$DATA_FILE" "$TMP_DIR/$REF_FILE"

    # Print aligned row
    printf "%6d | " "$size"
    benchmark_fmt_med_iqr "$GTAP_MED" "$GTAP_ELO" "$GTAP_EHI"; printf " | "
    benchmark_fmt_med_iqr "$OMP_MED"  "$OMP_ELO"  "$OMP_EHI";  printf " | "
    benchmark_fmt_med_iqr "$CILK_MED" "$CILK_ELO" "$CILK_EHI"; printf " | "
    benchmark_fmt_med_iqr "$DYNASOAR_MED" "$DYNASOAR_ELO" "$DYNASOAR_EHI"; printf " | "
    if [ "$SPEEDUP" = "0" ]; then
        printf "%10s\n" "N/A"
    else
        printf "%10.2f\n" "$SPEEDUP"
    fi

    # CSV
    echo "$size,$GTAP_MED,$GTAP_ELO,$GTAP_EHI,$OMP_MED,$OMP_ELO,$OMP_EHI,$CILK_MED,$CILK_ELO,$CILK_EHI,$DYNASOAR_MED,$DYNASOAR_ELO,$DYNASOAR_EHI,$SEQ_MED,$SEQ_ELO,$SEQ_EHI,$SPEEDUP,$CILK_SPEEDUP,$DYNASOAR_SPEEDUP" >> "$RESULTS_FILE"
done

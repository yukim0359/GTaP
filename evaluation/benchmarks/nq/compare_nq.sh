#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=08:00:00
#PBS -W group_list=gc64
#PBS -j oe

BENCHMARK_NAME=nq

cd "$PBS_O_WORKDIR"

COMPARE_DIR=$(pwd)
BIN_DIR="$COMPARE_DIR/bin"

if [ ! -d "$COMPARE_DIR" ]; then
    echo "Error: Please submit this script from gtap/evaluation/benchmarks/$BENCHMARK_NAME"
    exit 1
fi

# GTaP, Kiuchi (Dynasoar), OpenMP, Cilk — n=1..17 (n=18 omitted for now)
N_VALUES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17)

### nq-specific settings ###
export CILK_NWORKERS="${CILK_NWORKERS:-${OMP_NUM_THREADS:-$(nproc)}}"
CUTOFF="${CUTOFF:-7}"

# Dynasoar fork-join implementation (Kiuchi et al.)
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
DYNASOAR_NQ_BIN="${DYNASOAR_NQ_BIN:-${DYNASOAR_ROOT:+$DYNASOAR_ROOT/bin/NQCutoff}}"
DYNASOAR_NQ_TARGET="${DYNASOAR_NQ_TARGET:-NQCutoff}"
DYNASOAR_NQ_CUTOFF="${DYNASOAR_NQ_CUTOFF:-$CUTOFF}"

ensure_dynasoar_bin() {
    if [ -z "${DYNASOAR_ROOT:-}" ] || [ ! -d "$DYNASOAR_ROOT" ]; then
        echo "Warning: fine-grained-fork-join not found (set DYNASOAR_ROOT); skipping Kiuchi runs" >&2
        return 1
    fi
    if [ -x "$DYNASOAR_NQ_BIN" ]; then
        return 0
    fi
    echo "Building Dynasoar $DYNASOAR_NQ_TARGET in $DYNASOAR_ROOT ..."
    make -C "$DYNASOAR_ROOT" "bin/$DYNASOAR_NQ_TARGET" || {
        echo "Warning: Dynasoar $DYNASOAR_NQ_TARGET build failed; skipping Kiuchi runs" >&2
        return 1
    }
    [ -x "$DYNASOAR_NQ_BIN" ]
}

export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME:-/work/opt/local/aarch64/cores/nvidia/25.9/Linux_aarch64/25.9/cuda}}"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"

NUM_RUNS=20
NUM_RUNS_LARGE="${NUM_RUNS_LARGE:-10}"

RESULTS_FILE="$COMPARE_DIR/${BENCHMARK_NAME}_performance_results.csv"
echo "n,GTAP_med,GTAP_err_low,GTAP_err_high,OMP_med,OMP_err_low,OMP_err_high,CILK_med,CILK_err_low,CILK_err_high,DYNASOAR_med,DYNASOAR_err_low,DYNASOAR_err_high,DYNASOAR_cutoff,Speedup_med(OMP/GTAP),Speedup_med(CILK/GTAP),Speedup_med(DYNASOAR/GTAP)" > "$RESULTS_FILE"

echo "Building GTaP / OpenMP / Cilk binaries ..."
make -C "$COMPARE_DIR" gtap omp cilk

DYNASOAR_ENABLED=0
if ensure_dynasoar_bin; then
    DYNASOAR_ENABLED=1
    echo "Kiuchi (Dynasoar) NQ: $DYNASOAR_NQ_BIN (cutoff=$DYNASOAR_NQ_CUTOFF)"
else
    echo "Kiuchi (Dynasoar) NQ: disabled"
fi

printf "%6s | %25s | %25s | %25s | %25s | %10s\n" \
    "n" "GTAP (ms)" "OpenMP (ms)" "Cilk (ms)" "Kiuchi (ms)" "OMP/GTAP"
printf "%6s-+-%25s-+-%25s-+-%25s-+-%25s-+-%10s\n" \
    "------" "-------------------------" "-------------------------" \
    "-------------------------" "-------------------------" "----------"

run_stats() {
    local program=$1
    local n=$2
    local grep_pattern=$3
    local num_runs=${4:-$NUM_RUNS}
    local times=()

    for i in $(seq 1 "$num_runs"); do
        local output
        output=$($program $n $CUTOFF 2>&1)

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

run_stats_dynasoar() {
    local program=$1
    local n=$2
    local num_runs=${3:-$NUM_RUNS}
    local times=()

    for i in $(seq 1 "$num_runs"); do
        local output
        output=$("$program" "$n" 0 "$DYNASOAR_NQ_CUTOFF" 2>&1)

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

runs_for_n() {
    local n=$1
    if (( n >= 17 )); then
        echo "$NUM_RUNS_LARGE"
    else
        echo "$NUM_RUNS"
    fi
}

measure_and_record() {
    local n=$1
    local num_runs
    num_runs=$(runs_for_n "$n")

    GTAP_MED=0 GTAP_ELO=0 GTAP_EHI=0
    if [ -x "$BIN_DIR/gtap_$BENCHMARK_NAME" ]; then
        read GTAP_MED _ _ GTAP_ELO GTAP_EHI < <(run_stats "$BIN_DIR/gtap_$BENCHMARK_NAME" "$n" "Execution time" "$num_runs")
    fi

    OMP_MED=0 OMP_ELO=0 OMP_EHI=0
    if [ -x "$BIN_DIR/omp_$BENCHMARK_NAME" ]; then
        read OMP_MED _ _ OMP_ELO OMP_EHI < <(run_stats "$BIN_DIR/omp_$BENCHMARK_NAME" "$n" "Execution time" "$num_runs")
    fi

    CILK_MED=0 CILK_ELO=0 CILK_EHI=0
    if [ -x "$BIN_DIR/cilk_$BENCHMARK_NAME" ]; then
        read CILK_MED _ _ CILK_ELO CILK_EHI < <(run_stats "$BIN_DIR/cilk_$BENCHMARK_NAME" "$n" "Execution time" "$num_runs")
    fi

    DYNASOAR_MED=0 DYNASOAR_ELO=0 DYNASOAR_EHI=0
    if [ "$DYNASOAR_ENABLED" = "1" ]; then
        read DYNASOAR_MED _ _ DYNASOAR_ELO DYNASOAR_EHI < <(run_stats_dynasoar "$DYNASOAR_NQ_BIN" "$n" "$num_runs")
    fi

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

    printf "%6d | " "$n"
    fmt_med_iqr "$GTAP_MED" "$GTAP_ELO" "$GTAP_EHI"; printf " | "
    fmt_med_iqr "$OMP_MED"  "$OMP_ELO"  "$OMP_EHI";  printf " | "
    fmt_med_iqr "$CILK_MED" "$CILK_ELO" "$CILK_EHI"; printf " | "
    fmt_med_iqr "$DYNASOAR_MED" "$DYNASOAR_ELO" "$DYNASOAR_EHI"; printf " | "
    if [ "$SPEEDUP" = "0" ]; then
        printf "%10s\n" "N/A"
    else
        printf "%10.2f\n" "$SPEEDUP"
    fi

    echo "$n,$GTAP_MED,$GTAP_ELO,$GTAP_EHI,$OMP_MED,$OMP_ELO,$OMP_EHI,$CILK_MED,$CILK_ELO,$CILK_EHI,$DYNASOAR_MED,$DYNASOAR_ELO,$DYNASOAR_EHI,$DYNASOAR_NQ_CUTOFF,$SPEEDUP,$CILK_SPEEDUP,$DYNASOAR_SPEEDUP" >> "$RESULTS_FILE"
}

for n in "${N_VALUES[@]}"; do
    measure_and_record "$n"
done

echo ""
echo "Results saved to: $RESULTS_FILE"

#!/usr/bin/env bash
#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -W group_list=gc64
#PBS -j oe

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
GTAP_ROOT=$(cd "$EVAL_DIR/../.." && pwd)
NCU_BIN=${NCU_BIN:-ncu}
OUT_DIR=${OUT_DIR:-"$SCRIPT_DIR"}
RESULTS_CSV=${RESULTS_CSV:-"$OUT_DIR/warp_efficiency_summary.csv"}

METRICS=${METRICS:-smsp__thread_inst_executed_per_inst_executed.ratio,smsp__inst_executed.sum,smsp__thread_inst_executed.sum}
BENCHES=${BENCHES:-probe fib nq}
PROBE_ITERS=${PROBE_ITERS:-256}
PROBE_REPEATS=${PROBE_REPEATS:-128}
PROBE_PARTIAL_COUNT=${PROBE_PARTIAL_COUNT:-20}
FIB_N=${FIB_N:-40}
FIB_CUTOFFS=${FIB_CUTOFFS:-"8 10 12 14"}
NQ_N=${NQ_N:-16}
NQ_CUTOFFS=${NQ_CUTOFFS:-"5 7 9"}
CILKSORT_SIZE=${CILKSORT_SIZE:-1000000}
CILKSORT_CUTOFFS=${CILKSORT_CUTOFFS:-"64 256 1024"}
NCU_EXTRA_ARGS=${NCU_EXTRA_ARGS:-}

mkdir -p "$OUT_DIR/raw"
printf "benchmark,variant,queues,input,cutoff,kernel_name,metric_name,metric_unit,metric_value,raw_csv\n" > "$RESULTS_CSV"

append_metrics() {
    local benchmark=$1
    local variant=$2
    local queues=$3
    local input=$4
    local cutoff=$5
    local raw_csv=$6

    python3 "$SCRIPT_DIR/extract_ncu_metrics.py" \
        --benchmark "$benchmark" \
        --variant "$variant" \
        --queues "$queues" \
        --input "$input" \
        --cutoff "$cutoff" \
        --raw-csv "$raw_csv" \
        >> "$RESULTS_CSV"
}

profile_one() {
    local benchmark=$1
    local variant=$2
    local queues=$3
    local input=$4
    local cutoff=$5
    shift 5
    local binary=$1
    shift

    local raw_csv="$OUT_DIR/raw/${benchmark}_${variant}_input${input}_cutoff${cutoff}.csv"
    printf "[ncu] %-14s %-15s input=%s cutoff=%s -> %s\n" \
        "$benchmark" "$variant" "$input" "$cutoff" "$raw_csv"

    "$NCU_BIN" \
        --target-processes all \
        --csv \
        --page raw \
        --metrics "$METRICS" \
        $NCU_EXTRA_ARGS \
        "$binary" "$@" \
        > "$raw_csv"

    append_metrics "$benchmark" "$variant" "$queues" "$input" "$cutoff" "$raw_csv"
}

build_bench() {
    local benchmark=$1
    make -C "$EVAL_DIR/$benchmark" all
}

run_probe() {
    local probe_dir="$GTAP_ROOT/tmp/dispatch_lockstep_probe"
    make -C "$probe_dir" all
    local bin="$probe_dir/bin/dispatch_lockstep_probe"
    profile_one dispatch_probe null_uniform 1 "$PROBE_ITERS" na \
        "$bin" "$PROBE_ITERS" "$PROBE_REPEATS" "$PROBE_PARTIAL_COUNT" null-uniform
    profile_one dispatch_probe state_divergent 1 "$PROBE_ITERS" na \
        "$bin" "$PROBE_ITERS" "$PROBE_REPEATS" "$PROBE_PARTIAL_COUNT" state-divergent
}

run_fib() {
    build_bench fib
    local bin_dir="$EVAL_DIR/fib/bin"
    for cutoff in $FIB_CUTOFFS; do
        profile_one fib queue_1 1 "$FIB_N" "$cutoff" "$bin_dir/fib_queue_1" "$FIB_N" "$cutoff"
        profile_one fib queue_3 3 "$FIB_N" "$cutoff" "$bin_dir/fib_queue_3" "$FIB_N" "$cutoff"
    done
}

run_nq() {
    build_bench nq
    local bin_dir="$EVAL_DIR/nq/bin"
    for cutoff in $NQ_CUTOFFS; do
        profile_one nq queue_1 1 "$NQ_N" "$cutoff" "$bin_dir/nq_queue_1" "$NQ_N" "$cutoff"
        profile_one nq queue_2 2 "$NQ_N" "$cutoff" "$bin_dir/nq_queue_2" "$NQ_N" "$cutoff"
    done
}

run_cilksort() {
    local bench_dir="$EVAL_DIR/cilksort"
    local util_dir="$EVAL_DIR/../util"
    local tmp_dir="$bench_dir/tmp"
    local data_file="$tmp_dir/ncu_data_${CILKSORT_SIZE}.bin"
    mkdir -p "$tmp_dir"
    make -C "$util_dir" gen_vector
    if [ ! -f "$data_file" ]; then
        "$util_dir/gen_vector" "$CILKSORT_SIZE" "$data_file" >/dev/null
    fi

    local bin_dir="$bench_dir/bin"
    for cutoff in $CILKSORT_CUTOFFS; do
        make -C "$bench_dir" all \
            GTAP_CFLAGS="-DTASK_SPAWN_CUTOFF_SORT=$cutoff -DTASK_SPAWN_CUTOFF_MERGE=$cutoff"
        profile_one cilksort queue_1 1 "$CILKSORT_SIZE" "$cutoff" \
            "$bin_dir/cilksort_queue_1" "$data_file"
        profile_one cilksort queue_3 3 "$CILKSORT_SIZE" "$cutoff" \
            "$bin_dir/cilksort_queue_3" "$data_file"
    done
}

for bench in $BENCHES; do
    case "$bench" in
        probe) run_probe ;;
        fib) run_fib ;;
        nq) run_nq ;;
        cilksort) run_cilksort ;;
        *) echo "Unknown benchmark '$bench' in BENCHES='$BENCHES'" >&2; exit 2 ;;
    esac
done

echo "Wrote $RESULTS_CSV"

#!/bin/bash
# Front door for paper performance remeasurement.
# submit exports the current environment to the job with qsub -V.
#
#   ./eval.sh list
#   ./eval.sh run fib
#   ./eval.sh submit fib
#   ./eval.sh run paper
#   ./eval.sh submit paper
#   ./eval.sh submit paper --dry-run
#   ./eval.sh plot fmm-theta
#   ./eval.sh plot paper

set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
DRY_RUN=0

usage() {
    cat <<EOF
Usage: $0 list
       $0 run <target> [args...]
       $0 run paper
       $0 submit <target>
       $0 submit paper [--dry-run]
       $0 plot <target>
       $0 help

  list      show targets
  run       bash the experiment in this shell (unset leftover PBS_O_WORKDIR)
            extra args are forwarded only for a single target
  submit    qsub -V the matching .pbs from its directory
  plot      rebuild a paper figure from existing CSVs

Targets: tree-depth tree-compute hetero fib nq cilksort
         daq-fib daq-nq daq-cilksort worker-fib
         fmm-theta fmm-stack kclique-orientation kclique-pivot
         paper

Plots:   tree hetero micro daq worker-fib
         fmm-theta fmm-stack kclique paper
EOF
}

reject_extra() {
    if [ $# -ne 0 ]; then
        echo "unexpected argument: $1" >&2
        usage >&2
        exit 1
    fi
}

# Submit order. afterok: tree-compute, fmm-stack, kclique-pivot.
PAPER_TARGETS=(
    tree-depth
    tree-compute
    hetero
    fib
    nq
    cilksort
    daq-fib
    daq-nq
    daq-cilksort
    worker-fib
    fmm-theta
    fmm-stack
    kclique-orientation
    kclique-pivot
)

PLOT_TARGETS=(
    tree
    hetero
    micro
    daq
    worker-fib
    fmm-theta
    fmm-stack
    kclique
)

norm() {
    printf '%s' "$1" | tr '_' '-'
}

# Sets TARGET_DIR TARGET_RUN TARGET_SUBMIT
resolve() {
    local name
    name="$(norm "$1")"
    case "$name" in
        tree-depth)
            TARGET_DIR="$EVAL_DIR/mode-comparison/binary_tree"
            TARGET_RUN=compare_tree_depth.sh
            TARGET_SUBMIT=compare_tree_depth.pbs
            ;;
        tree-compute)
            TARGET_DIR="$EVAL_DIR/mode-comparison/binary_tree"
            TARGET_RUN=compare_tree_compute.sh
            TARGET_SUBMIT=compare_tree_compute.pbs
            ;;
        hetero)
            TARGET_DIR="$EVAL_DIR/mode-comparison/hetero_tree"
            TARGET_RUN=compare_hetero_compute_k24.sh
            TARGET_SUBMIT=compare_hetero_compute_k24.pbs
            ;;
        fib)
            TARGET_DIR="$EVAL_DIR/microbenchmarks/runtime-comparison/fib"
            TARGET_RUN=compare_fib.sh
            TARGET_SUBMIT=compare_fib.pbs
            ;;
        nq)
            TARGET_DIR="$EVAL_DIR/microbenchmarks/runtime-comparison/nq"
            TARGET_RUN=compare_nq.sh
            TARGET_SUBMIT=compare_nq.pbs
            ;;
        cilksort)
            TARGET_DIR="$EVAL_DIR/microbenchmarks/runtime-comparison/cilksort"
            TARGET_RUN=compare_cilksort.sh
            TARGET_SUBMIT=compare_cilksort.pbs
            ;;
        daq-fib)
            TARGET_DIR="$EVAL_DIR/microbenchmarks/daq/fib"
            TARGET_RUN=daq_fib.sh
            TARGET_SUBMIT=daq_fib.pbs
            ;;
        daq-nq)
            TARGET_DIR="$EVAL_DIR/microbenchmarks/daq/nq"
            TARGET_RUN=daq_nq.sh
            TARGET_SUBMIT=daq_nq.pbs
            ;;
        daq-cilksort)
            TARGET_DIR="$EVAL_DIR/microbenchmarks/daq/cilksort"
            TARGET_RUN=daq_cilksort.sh
            TARGET_SUBMIT=daq_cilksort.pbs
            ;;
        worker-fib)
            TARGET_DIR="$EVAL_DIR/scheduler-comparison/fib"
            TARGET_RUN=worker_scaling_fib.sh
            TARGET_SUBMIT=worker_scaling_fib.pbs
            ;;
        fmm-theta)
            TARGET_DIR="$EVAL_DIR/applications/fmm"
            TARGET_RUN=scripts/dtt_theta/run_compare_fmm_dtt_theta_rebuild.sh
            TARGET_SUBMIT=scripts/dtt_theta/run_compare_fmm_dtt_theta_rebuild.pbs
            ;;
        fmm-stack)
            TARGET_DIR="$EVAL_DIR/applications/fmm"
            TARGET_RUN=scripts/stack/run_fmm_3way_stack_paper.sh
            TARGET_SUBMIT=scripts/stack/run_fmm_3way_stack_paper.pbs
            ;;
        kclique-orientation)
            TARGET_DIR="$EVAL_DIR/applications/k_clique"
            TARGET_RUN=scripts/k579/run_orientation_k579_e2e.sh
            TARGET_SUBMIT=scripts/k579/run_orientation_k579_e2e.pbs
            ;;
        kclique-pivot)
            TARGET_DIR="$EVAL_DIR/applications/k_clique"
            TARGET_RUN=scripts/k579/run_pivot_k579_e2e.sh
            TARGET_SUBMIT=scripts/k579/run_pivot_k579_e2e.pbs
            ;;
        *)
            echo "unknown target: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
}

# Sets PLOT_KIND PLOT_PATH
# PLOT_KIND is python3 or exec.
resolve_plot() {
    local name
    name="$(norm "$1")"
    case "$name" in
        tree|tree-depth|tree-compute)
            PLOT_KIND=python3
            PLOT_PATH="$EVAL_DIR/scripts/plot/plot_binary_tree_normalized_sweeps_paper.py"
            ;;
        hetero)
            PLOT_KIND=python3
            PLOT_PATH="$EVAL_DIR/scripts/plot/plot_hetero_tree_compute_normalized_paper.py"
            ;;
        micro|fib|nq|cilksort)
            PLOT_KIND=python3
            PLOT_PATH="$EVAL_DIR/scripts/plot/plot_performance_row_paper.py"
            ;;
        daq|daq-fib|daq-nq|daq-cilksort)
            PLOT_KIND=python3
            PLOT_PATH="$EVAL_DIR/scripts/plot/plot_daq_ratio_row_paper.py"
            ;;
        worker-fib)
            PLOT_KIND=python3
            PLOT_PATH="$EVAL_DIR/scripts/plot/plot_fib_worker_scaling_paper.py"
            ;;
        fmm-theta)
            PLOT_KIND=python3
            PLOT_PATH="$EVAL_DIR/scripts/plot/plot_fmm_dtt_theta_row_paper.py"
            ;;
        fmm-stack)
            PLOT_KIND=python3
            PLOT_PATH="$EVAL_DIR/scripts/plot/plot_fmm_3way_stack_paper.py"
            ;;
        kclique|kclique-orientation|kclique-pivot)
            PLOT_KIND=exec
            PLOT_PATH="$EVAL_DIR/applications/k_clique/scripts/k579/summarize_k579_e2e_combined.sh"
            ;;
        *)
            echo "unknown plot target: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
}

need_cuda() {
    if [ -z "${CUDA_PATH:-}" ] && [ -z "${CUDA_HOME:-}" ]; then
        echo "set CUDA_PATH or CUDA_HOME in this shell (submit copies it with qsub -V)" >&2
        exit 1
    fi
}

cmd_list() {
    local name
    echo "run / submit"
    for name in "${PAPER_TARGETS[@]}"; do
        resolve "$name"
        printf '  %-22s %s\n' "$name" "${TARGET_DIR#$EVAL_DIR/}/$TARGET_SUBMIT"
    done
    echo
    echo "  paper                  all of the above (submit uses afterok)"
    echo
    echo "plot"
    for name in "${PLOT_TARGETS[@]}"; do
        resolve_plot "$name"
        printf '  %-22s %s\n' "$name" "${PLOT_PATH#$EVAL_DIR/}"
    done
    printf '  %-22s %s\n' paper "all of the above"
}

cmd_run() {
    local name="${1:-}"
    if [ -z "$name" ]; then
        echo "usage: $0 run <target> [args...]" >&2
        exit 1
    fi
    shift
    name="$(norm "$name")"
    if [ "$name" = paper ]; then
        reject_extra "$@"
        need_cuda
        make -C "$EVAL_DIR/util" gen_vector || exit 1
        local t
        for t in "${PAPER_TARGETS[@]}"; do
            echo "=== run $t ==="
            cmd_run "$t"
        done
        return
    fi
    need_cuda
    resolve "$name"
    if [ ! -f "$TARGET_DIR/$TARGET_RUN" ]; then
        echo "missing: $TARGET_DIR/$TARGET_RUN" >&2
        exit 1
    fi
    (
        cd "$TARGET_DIR" || exit 1
        unset PBS_O_WORKDIR
        exec bash "./$TARGET_RUN" "$@"
    )
}

# Predecessor for paper afterok (shared bin/ / KCGPU). Empty if none.
paper_afterok() {
    case "$1" in
        tree-compute) echo tree-depth ;;
        fmm-stack) echo fmm-theta ;;
        kclique-pivot) echo kclique-orientation ;;
    esac
}

# name<TAB>id lines. Avoids Bash 4 associative arrays so help/list/plot
# work on macOS /bin/bash 3.2.
JOB_IDS=""

set_job_id() {
    JOB_IDS="${JOB_IDS}${1}"$'\t'"${2}"$'\n'
}

get_job_id() {
    printf '%s' "$JOB_IDS" | awk -F '\t' -v n="$1" '$1==n {print $2; exit}'
}

submit_one() {
    local name=$1
    local afterok="${2:-}"
    local depend=""
    resolve "$name"
    if [ ! -f "$TARGET_DIR/$TARGET_SUBMIT" ]; then
        echo "missing: $TARGET_DIR/$TARGET_SUBMIT" >&2
        exit 1
    fi
    if [ -n "$afterok" ]; then
        depend="$(get_job_id "$afterok")"
        if [ -z "$depend" ]; then
            echo "missing job id for dependency $afterok (needed by $name)" >&2
            exit 1
        fi
    fi
    if [ "${DRY_RUN:-0}" -eq 1 ]; then
        if [ -n "$depend" ]; then
            printf '(cd %q && qsub -V -W %q %q)\n' \
                "$TARGET_DIR" "depend=afterok:$depend" "$TARGET_SUBMIT"
        else
            printf '(cd %q && qsub -V %q)\n' "$TARGET_DIR" "$TARGET_SUBMIT"
        fi
        set_job_id "$name" "<$name>"
        return
    fi
    local job_id
    if [ -n "$depend" ]; then
        job_id="$(cd "$TARGET_DIR" && qsub -V -W "depend=afterok:$depend" "$TARGET_SUBMIT")"
    else
        job_id="$(cd "$TARGET_DIR" && qsub -V "$TARGET_SUBMIT")"
    fi
    job_id="$(printf '%s' "$job_id" | tail -n 1 | tr -d '[:space:]')"
    if [ -z "$job_id" ]; then
        echo "qsub produced no job id for $name" >&2
        exit 1
    fi
    set_job_id "$name" "$job_id"
    echo "$job_id"
    if [ -n "${JOBS_FILE:-}" ]; then
        if [ -n "$afterok" ]; then
            echo "$name $job_id afterok:$afterok:$depend" >> "$JOBS_FILE"
        else
            echo "$name $job_id" >> "$JOBS_FILE"
        fi
    fi
}

cmd_submit_paper() {
    local t missing=0
    DRY_RUN=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --dry-run|--list|-n) DRY_RUN=1 ;;
            --help|-h)
                echo "usage: $0 submit paper [--dry-run]"
                exit 0
                ;;
            *)
                echo "unknown argument: $1" >&2
                echo "usage: $0 submit paper [--dry-run]" >&2
                exit 1
                ;;
        esac
        shift
    done
    for t in "${PAPER_TARGETS[@]}"; do
        resolve "$t"
        if [ ! -f "$TARGET_DIR/$TARGET_SUBMIT" ]; then
            echo "missing: $TARGET_DIR/$TARGET_SUBMIT" >&2
            missing=1
        fi
    done
    if [ ! -f "$EVAL_DIR/util/Makefile" ]; then
        echo "missing: $EVAL_DIR/util/Makefile" >&2
        missing=1
    fi
    if [ "$missing" -ne 0 ]; then
        exit 1
    fi
    printf 'make -C %q gen_vector\n' "$EVAL_DIR/util"
    JOBS_FILE=""
    JOB_IDS=""
    if [ "$DRY_RUN" -eq 0 ]; then
        need_cuda
        make -C "$EVAL_DIR/util" gen_vector || exit 1
        JOBS_FILE="$EVAL_DIR/eval.jobs.$(date '+%Y%m%d-%H%M%S')"
        echo "# $(date '+%Y-%m-%d %H:%M:%S') paper remeasurement jobs" > "$JOBS_FILE"
    fi
    for t in "${PAPER_TARGETS[@]}"; do
        submit_one "$t" "$(paper_afterok "$t")"
    done
    if [ "$DRY_RUN" -eq 0 ]; then
        echo "Wrote $JOBS_FILE"
        echo "After k-clique finishes:"
        echo "  $0 plot kclique"
    fi
}

cmd_submit() {
    local name="${1:-}"
    if [ -z "$name" ]; then
        echo "usage: $0 submit <target>" >&2
        echo "       $0 submit paper [--dry-run]" >&2
        exit 1
    fi
    shift
    name="$(norm "$name")"
    if [ "$name" = paper ]; then
        cmd_submit_paper "$@"
        return
    fi
    reject_extra "$@"
    need_cuda
    JOBS_FILE=""
    JOB_IDS=""
    submit_one "$name"
}

run_plot() {
    resolve_plot "$1"
    if [ ! -e "$PLOT_PATH" ]; then
        echo "missing: $PLOT_PATH" >&2
        exit 1
    fi
    if [ "$PLOT_KIND" = python3 ]; then
        python3 "$PLOT_PATH"
    else
        "$PLOT_PATH"
    fi
}

cmd_plot() {
    local name="${1:-}"
    if [ -z "$name" ]; then
        echo "usage: $0 plot <target>" >&2
        exit 1
    fi
    shift
    reject_extra "$@"
    name="$(norm "$name")"
    if [ "$name" = paper ]; then
        local t
        for t in "${PLOT_TARGETS[@]}"; do
            echo "=== plot $t ==="
            run_plot "$t"
        done
        return
    fi
    run_plot "$name"
}

cmd="${1:-}"
if [ $# -gt 0 ]; then
    shift
fi
case "$cmd" in
    list|-l)
        reject_extra "$@"
        cmd_list
        ;;
    run) cmd_run "$@" ;;
    submit) cmd_submit "$@" ;;
    plot) cmd_plot "$@" ;;
    help|-h|--help|"")
        reject_extra "$@"
        usage
        ;;
    *)
        echo "unknown command: $cmd" >&2
        usage >&2
        exit 1
        ;;
esac

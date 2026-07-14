#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
K_CLIQUE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
GTAP_ROOT=$(cd "$K_CLIQUE_DIR/../../.." && pwd)
WORK_ROOT=$(cd "$GTAP_ROOT/.." && pwd)
KCGPU_DIR=${KCGPU_DIR:-"$WORK_ROOT/KCGPU"}
GRAPH_DIR=${GRAPH_DIR:-"$KCGPU_DIR/graphs"}

RESULTS_FILE=${RESULTS_FILE:-"$K_CLIQUE_DIR/legacy/gtap_kcgpu_k4_8_graph_tuned_results.csv"}
LOG_DIR=${LOG_DIR:-"$K_CLIQUE_DIR/legacy/gtap_kcgpu_k4_8_graph_tuned_logs"}

K_VALUES=${K_VALUES:-"4 5 6 7 8"}
REPEATS=${REPEATS:-1}
VALIDATE=${VALIDATE:-0}
DEVICE=${DEVICE:-0}

RUN_GTAP=${RUN_GTAP:-1}
RUN_KCGPU=${RUN_KCGPU:-1}
BUILD_GTAP=${BUILD_GTAP:-1}
BUILD_KCGPU=${BUILD_KCGPU:-1}

GTAP_VARIANT=${GTAP_VARIANT:-orientation}
if [ "$GTAP_VARIANT" = "pivot" ]; then
    KCGPU_ORIENT=${KCGPU_ORIENT:-degen}
    GTAP_ORIENT=${GTAP_ORIENT:-degen}
    KCGPU_Q=${KCGPU_Q:-p1b}
    KCGPU_VARIANT_SET=${KCGPU_VARIANT_SET:-pivot_binary}
else
    KCGPU_ORIENT=${KCGPU_ORIENT:-degree}
    GTAP_ORIENT=${GTAP_ORIENT:-degree}
    KCGPU_Q=${KCGPU_Q:-o8b}
    KCGPU_VARIANT_SET=${KCGPU_VARIANT_SET:-oriented_binary}
fi
export GTAP_ORIENT
KCGPU_PROCESS=${KCGPU_PROCESS:-edge}
KCGPU_ELEMENT=${KCGPU_ELEMENT:-bw}
KCGPU_VARIANTS=${KCGPU_VARIANTS:-}
# full: use KCGPU_VARIANT_SET (o1b..o8b / p1b..p16b).
# k579_min: one KCGPU baseline per graph×k from prior k579 best results.
KCGPU_VARIANT_MODE=${KCGPU_VARIANT_MODE:-full}
KCGPU_ALLOC=${KCGPU_ALLOC:-gpu}
KCGPU_SMALL_GRAPH=${KCGPU_SMALL_GRAPH:-0}
KCGPU_SORT=${KCGPU_SORT:-1}
KCGPU_CUDA_ARCH=${KCGPU_CUDA_ARCH:-90}
KCGPU_FORCE_REBUILD=${KCGPU_FORCE_REBUILD:-0}
KCGPU_PROFILE=${KCGPU_PROFILE:-0}
GTAP_PROFILE=${GTAP_PROFILE:-0}

DBLP_GRAPH=${DBLP_GRAPH:-"$GRAPH_DIR/com-DBLP/com-DBLP.mtx"}
SKITTER_GRAPH=${SKITTER_GRAPH:-"$GRAPH_DIR/as-Skitter/as-Skitter.mtx"}
ORKUT_GRAPH=${ORKUT_GRAPH:-"$GRAPH_DIR/com-Orkut/com-Orkut.mtx"}
GRAPHS=${GRAPHS:-"DBLP as-Skitter Orkut"}

GTAP_CONFIG=${GTAP_CONFIG:-"$K_CLIQUE_DIR/data/gtap_graph_k_config.csv"}
GTAP_CONFIG_LOADER=${GTAP_CONFIG_LOADER:-"$SCRIPT_DIR/load_gtap_graph_k_config.py"}

if [ -n "${CUDA_PATH:-}" ]; then
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
fi

case "$GTAP_VARIANT" in
    orientation)
        GTAP_TARGET=gtap_orientation
        GTAP_BIN="$K_CLIQUE_DIR/bin/gtap_orientation"
        ;;
    pivot)
        GTAP_TARGET=gtap_pivot
        GTAP_BIN="$K_CLIQUE_DIR/bin/gtap_pivot"
        ;;
    *)
        echo "Unknown GTAP_VARIANT=$GTAP_VARIANT (expected orientation or pivot)" >&2
        exit 1
        ;;
esac

mkdir -p "$LOG_DIR"

if [ "$BUILD_KCGPU" = "1" ] && [ "$RUN_KCGPU" = "1" ]; then
    if [ "$KCGPU_FORCE_REBUILD" = "1" ] || [ ! -x "$KCGPU_DIR/build/exe/src/main.cu.exe" ]; then
        echo "Building KCGPU: CUDA_ARCH=$KCGPU_CUDA_ARCH KCGPU_PROFILE=$KCGPU_PROFILE"
        make -C "$KCGPU_DIR" CUDA_ARCH="$KCGPU_CUDA_ARCH" KCGPU_PROFILE="$KCGPU_PROFILE" -B >/dev/null
    elif [ "$KCGPU_PROFILE" = "0" ] &&
         strings "$KCGPU_DIR/build/exe/src/main.cu.exe" 2>/dev/null | grep -q "KCGPU profile exported"; then
        echo "Rebuilding KCGPU without profile (existing binary was built with KCGPU_PROFILE=1)"
        make -C "$KCGPU_DIR" CUDA_ARCH="$KCGPU_CUDA_ARCH" KCGPU_PROFILE=0 -B >/dev/null
    fi
fi

flag_enabled() {
    case "${1:-0}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

resolve_graph_spec() {
    local name=$1
    case "$name" in
        DBLP|com-DBLP)
            printf "DBLP:%s\n" "$DBLP_GRAPH"
            ;;
        Skitter|as-Skitter)
            printf "as-Skitter:%s\n" "$SKITTER_GRAPH"
            ;;
        Orkut|com-Orkut)
            printf "Orkut:%s\n" "$ORKUT_GRAPH"
            ;;
        LiveJournal|com-LiveJournal)
            printf "LiveJournal:%s\n" "$GRAPH_DIR/com-LiveJournal/com-LiveJournal.mtx"
            ;;
        Facebook|ego-Facebook)
            printf "Facebook:%s\n" "$GRAPH_DIR/ego-Facebook/ego-Facebook.mtx"
            ;;
        /*|*.mtx)
            local base
            base=$(basename "$name" .mtx)
            printf "%s:%s\n" "$base" "$name"
            ;;
        *)
            local candidate="$GRAPH_DIR/$name/$name.mtx"
            printf "%s:%s\n" "$name" "$candidate"
            ;;
    esac
}

graph_k_config_values() {
    local graph_name=$1
    local k=$2
    python3 "$GTAP_CONFIG_LOADER" -c "$GTAP_CONFIG" --values "$graph_name" "$k"
}

build_gtap_for_case() {
    local graph_name=$1
    local k=$2

    if [ "$BUILD_GTAP" != "1" ] || [ "$RUN_GTAP" != "1" ]; then
        return
    fi

    local -a make_args=()
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        make_args+=("$line")
    done < <(python3 "$GTAP_CONFIG_LOADER" -c "$GTAP_CONFIG" --make-args "$GTAP_VARIANT" "$graph_name" "$k")

    echo "Rebuilding $GTAP_TARGET for $graph_name k=$k: GTAP_PROFILE=$GTAP_PROFILE ${make_args[*]}"
    make -C "$K_CLIQUE_DIR" -B "$GTAP_TARGET" GTAP_PROFILE="$GTAP_PROFILE" "${make_args[@]}" >/dev/null
}

parse_gtap_count() {
    sed -n 's/GTaP count: *\([0-9]*\).*/\1/p' | tail -n 1
}

parse_gtap_count_e2e_ms() {
    sed -n 's/GTaP count e2e time: *\([0-9.]*\) ms.*/\1/p' | tail -n 1
}

parse_gtap_initialize_ms() {
    sed -n 's/GTaP gtap_initialize time: *\([0-9.]*\) ms.*/\1/p' | tail -n 1
}

parse_gtap_count_phase_ms() {
    local count_phase
    count_phase=$(sed -n 's/GTaP count phase time: *\([0-9.]*\) ms.*/\1/p' | tail -n 1)
    if [ -n "$count_phase" ]; then
        printf "%s\n" "$count_phase"
        return
    fi
    sed -n 's/GTaP execution time: *\([0-9.]*\) ms.*/\1/p' | tail -n 1
}

parse_gtap_kernel_ms() {
    sed -n 's/GTaP kernel time: *\([0-9.]*\) ms.*/\1/p' | tail -n 1
}

parse_gtap_preprocess_transfer_ms() {
    local preprocess
    preprocess=$(sed -n 's/GTaP preprocess time: *\([0-9.]*\) ms.*/\1/p' | tail -n 1)
    if [ -n "$preprocess" ]; then
        printf "%s\n" "$preprocess"
        return
    fi
    sed -n 's/GTaP preprocess+transfer time: *\([0-9.]*\) ms.*/\1/p' | tail -n 1
}

parse_gtap_transfer_ms() {
    sed -n 's/GTaP transfer time: *\([0-9.]*\) ms.*/\1/p' | tail -n 1
}

parse_gtap_orient_ms() {
    sed -n 's/GTaP orient time: *\([0-9.]*\) ms.*/\1/p' | tail -n 1
}

parse_gtap_status() {
    sed -n 's/Validation: *\(.*\)/\1/p' | tail -n 1
}

parse_kcgpu_count() {
    sed -n 's/.*Counter = *\([0-9,]*\).*/\1/p' | tail -n 1 | tr -d ','
}

parse_kcgpu_ms() {
    local count_e2e
    count_e2e=$(sed -n 's/.*count end-to-end time *\([0-9.]*\) s.*/\1/p' | tail -n 1)
    if [ -n "$count_e2e" ]; then
        awk -v t="$count_e2e" 'BEGIN { if (t != "") printf "%.3f", t * 1000.0 }'
        return
    fi
    local count_phase
    count_phase=$(sed -n 's/.*count time *\([0-9.]*\) s.*/\1/p' | tail -n 1)
    if [ -n "$count_phase" ]; then
        awk -v t="$count_phase" 'BEGIN { if (t != "") printf "%.3f", t * 1000.0 }'
    fi
}

expand_kcgpu_variant_set() {
    case "$KCGPU_VARIANT_SET" in
        single)
            printf '%s' "$KCGPU_PROCESS:$KCGPU_Q"
            ;;
        oriented_binary)
            local variants=""
            for process in edge; do
                for part in 1 2 4 8; do
                    variants="${variants:+$variants }$process:o${part}b"
                done
            done
            printf '%s' "$variants"
            ;;
        oriented_all)
            local variants=""
            for process in node edge; do
                for part in 1 2 4 8 16 32; do
                    variants="${variants:+$variants }$process:o${part}b"
                    variants="${variants:+$variants }$process:o${part}n"
                done
            done
            printf '%s' "$variants"
            ;;
        pivot_binary)
            local variants=""
            for process in edge; do
                for part in 1 2 4 8 16; do
                    variants="${variants:+$variants }$process:p${part}b"
                done
            done
            printf '%s' "$variants"
            ;;
        *)
            echo "Unsupported KCGPU_VARIANT_SET=$KCGPU_VARIANT_SET" >&2
            exit 1
            ;;
    esac
}

resolve_kcgpu_variants() {
    local graph_name=$1
    local k=$2

    if [ -n "$KCGPU_VARIANTS" ]; then
        printf '%s' "$KCGPU_VARIANTS"
        return
    fi

    case "$KCGPU_VARIANT_MODE" in
        k579_min)
            if [ "$GTAP_VARIANT" = "pivot" ]; then
                printf '%s' "edge:p1b"
                return
            fi
            case "$k" in
                9)
                    printf '%s' "edge:o1b"
                    ;;
                7)
                    printf '%s' "edge:o1b edge:o2b"
                    ;;
                5)
                    printf '%s' "edge:o4b"
                    ;;
                *)
                    expand_kcgpu_variant_set
                    ;;
            esac
            ;;
        full|*)
            expand_kcgpu_variant_set
            ;;
    esac
}

if [ -z "$KCGPU_VARIANTS" ] && [ "$KCGPU_VARIANT_MODE" = "full" ]; then
    KCGPU_VARIANTS=$(expand_kcgpu_variant_set)
fi

csv_header="graph,k,repeat,gtap_variant,gtap_heavy,gtap_second_heavy,gtap_third_heavy,gtap_orientation_max_tasks_per_warp,gtap_pivot_max_tasks_per_warp,gtap_count,gtap_count_e2e_ms,gtap_initialize_ms,gtap_count_phase_ms,gtap_kernel_ms,gtap_preprocess_transfer_ms,gtap_status,kcgpu_orient,kcgpu_process,kcgpu_element,kcgpu_q,kcgpu_alloc,kcgpu_small_graph,kcgpu_sort,kcgpu_count,kcgpu_ms,match,gtap_count_e2e_ms_over_kcgpu_ms,gtap_count_phase_ms_over_kcgpu_ms,gtap_kernel_ms_over_kcgpu_ms,gtap_exit,kcgpu_exit,graph_path"
printf "%s\n" "$csv_header" > "$RESULTS_FILE"

for graph_name_arg in $GRAPHS; do
    graph_spec=$(resolve_graph_spec "$graph_name_arg")
    graph_name=${graph_spec%%:*}
    graph_path=${graph_spec#*:}

    if [ ! -f "$graph_path" ]; then
        echo "Missing graph: $graph_name -> $graph_path" >&2
        continue
    fi

    for k in $K_VALUES; do
        read -r gtap_heavy gtap_second_heavy gtap_third_heavy gtap_orientation_max_tasks_per_warp gtap_pivot_max_tasks_per_warp < <(
            graph_k_config_values "$graph_name" "$k"
        )
        case_kcgpu_variants=$(resolve_kcgpu_variants "$graph_name" "$k")
        build_gtap_for_case "$graph_name" "$k"
        for repeat in $(seq 1 "$REPEATS"); do
            echo "== graph=$graph_name k=$k repeat=$repeat =="

            gtap_count=""
            gtap_count_e2e_ms=""
            gtap_initialize_ms=""
            gtap_count_phase_ms=""
            gtap_kernel_ms=""
            gtap_preprocess_transfer_ms=""
            gtap_status=""
            gtap_exit=""

            if [ "$RUN_GTAP" = "1" ]; then
                gtap_log="$LOG_DIR/${graph_name}_k${k}_r${repeat}_gtap_${GTAP_VARIANT}.log"
                set +e
                gtap_out=$("$GTAP_BIN" "$graph_path" "$k" 0 "$VALIDATE" 2>&1)
                gtap_exit=$?
                set -e
                printf "%s\n" "$gtap_out" | tee "$gtap_log"
                gtap_count=$(printf "%s\n" "$gtap_out" | parse_gtap_count)
                gtap_count_e2e_ms=$(printf "%s\n" "$gtap_out" | parse_gtap_count_e2e_ms)
                gtap_initialize_ms=$(printf "%s\n" "$gtap_out" | parse_gtap_initialize_ms)
                gtap_count_phase_ms=$(printf "%s\n" "$gtap_out" | parse_gtap_count_phase_ms)
                gtap_kernel_ms=$(printf "%s\n" "$gtap_out" | parse_gtap_kernel_ms)
                gtap_preprocess_transfer_ms=$(printf "%s\n" "$gtap_out" | parse_gtap_preprocess_transfer_ms)
                gtap_status=$(printf "%s\n" "$gtap_out" | parse_gtap_status)
            fi

            if [ "$RUN_KCGPU" = "1" ]; then
                for variant in $case_kcgpu_variants; do
                    kcgpu_process=${variant%%:*}
                    kcgpu_q=${variant#*:}
                    if [ "$kcgpu_process" = "$variant" ] || [ -z "$kcgpu_process" ] || [ -z "$kcgpu_q" ]; then
                        echo "Invalid KCGPU variant '$variant'. Use process:q, e.g. edge:o8b." >&2
                        exit 1
                    fi

                    kcgpu_log="$LOG_DIR/${graph_name}_k${k}_r${repeat}_kcgpu_${kcgpu_process}_${kcgpu_q}.log"
                    kcgpu_args=(
                        ./build/exe/src/main.cu.exe
                        -g "$graph_path"
                        -d "$DEVICE"
                        -m kc
                        -o "$KCGPU_ORIENT"
                        -a "$KCGPU_ALLOC"
                        -k "$k"
                        -p "$kcgpu_process"
                        -e "$KCGPU_ELEMENT"
                        -q "$kcgpu_q"
                    )
                    if flag_enabled "$KCGPU_SMALL_GRAPH"; then
                        kcgpu_args+=(-w 1)
                    fi
                    if flag_enabled "$KCGPU_SORT"; then
                        kcgpu_args+=(-s 1)
                    fi

                    set +e
                    kcgpu_out=$(cd "$KCGPU_DIR" && "${kcgpu_args[@]}" 2>&1)
                    kcgpu_exit=$?
                    set -e
                    printf "%s\n" "$kcgpu_out" | tee "$kcgpu_log"

                    kcgpu_count=$(printf "%s\n" "$kcgpu_out" | parse_kcgpu_count)
                    kcgpu_ms=$(printf "%s\n" "$kcgpu_out" | parse_kcgpu_ms)

                    match=""
                    count_e2e_ratio=""
                    count_phase_ratio=""
                    kernel_ratio=""
                    if [ -n "$gtap_count" ] && [ -n "$kcgpu_count" ]; then
                        if [ "$gtap_count" = "$kcgpu_count" ]; then
                            match="YES"
                        else
                            match="NO"
                        fi
                    fi
                    if [ -n "$gtap_count_e2e_ms" ] && [ -n "$kcgpu_ms" ]; then
                        count_e2e_ratio=$(awk -v g="$gtap_count_e2e_ms" -v kcgpu="$kcgpu_ms" 'BEGIN { if (kcgpu > 0) printf "%.6f", g / kcgpu }')
                    fi
                    if [ -n "$gtap_count_phase_ms" ] && [ -n "$kcgpu_ms" ]; then
                        count_phase_ratio=$(awk -v g="$gtap_count_phase_ms" -v kcgpu="$kcgpu_ms" 'BEGIN { if (kcgpu > 0) printf "%.6f", g / kcgpu }')
                    fi
                    if [ -n "$gtap_kernel_ms" ] && [ -n "$kcgpu_ms" ]; then
                        kernel_ratio=$(awk -v g="$gtap_kernel_ms" -v kcgpu="$kcgpu_ms" 'BEGIN { if (kcgpu > 0) printf "%.6f", g / kcgpu }')
                    fi

                    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
                        "$graph_name" "$k" "$repeat" "$GTAP_VARIANT" \
                        "$gtap_heavy" "$gtap_second_heavy" "$gtap_third_heavy" \
                        "$gtap_orientation_max_tasks_per_warp" "$gtap_pivot_max_tasks_per_warp" \
                        "$gtap_count" "$gtap_count_e2e_ms" "$gtap_initialize_ms" "$gtap_count_phase_ms" "$gtap_kernel_ms" "$gtap_preprocess_transfer_ms" "$gtap_status" \
                        "$KCGPU_ORIENT" "$kcgpu_process" "$KCGPU_ELEMENT" "$kcgpu_q" "$KCGPU_ALLOC" "$KCGPU_SMALL_GRAPH" "$KCGPU_SORT" \
                        "$kcgpu_count" "$kcgpu_ms" "$match" "$count_e2e_ratio" "$count_phase_ratio" "$kernel_ratio" "$gtap_exit" "$kcgpu_exit" "$graph_path" >> "$RESULTS_FILE"
                done
            else
                printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,,,,,,,,,,,,,,,,,%s\n" \
                    "$graph_name" "$k" "$repeat" "$GTAP_VARIANT" \
                    "$gtap_heavy" "$gtap_second_heavy" "$gtap_third_heavy" \
                    "$gtap_orientation_max_tasks_per_warp" "$gtap_pivot_max_tasks_per_warp" \
                    "$gtap_count" "$gtap_count_e2e_ms" "$gtap_initialize_ms" "$gtap_count_phase_ms" "$gtap_kernel_ms" "$gtap_preprocess_transfer_ms" "$gtap_status" \
                    "$graph_path" >> "$RESULTS_FILE"
            fi
        done
    done
done

echo "Wrote $RESULTS_FILE"

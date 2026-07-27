#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=01:30:00
#PBS -W group_list=gc64
#PBS -j oe

set -u

BENCHMARK_NAME=bfs

SCRIPT_SOURCE="${BASH_SOURCE[0]}"
if [[ "$SCRIPT_SOURCE" != /* ]]; then
    SCRIPT_SOURCE="$(pwd)/$SCRIPT_SOURCE"
fi
SCRIPT_DIR=$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)

if [ -n "${PBS_O_WORKDIR:-}" ]; then
    cd "$PBS_O_WORKDIR" || exit 1
fi

COMPARE_DIR="$SCRIPT_DIR"
PROJECT_ROOT=$(cd "$COMPARE_DIR" && cd ../../../../ && pwd)
BIN_DIR="$COMPARE_DIR/bin"
ATOS_TESTS_DIR="$PROJECT_ROOT/ATOS/single-GPU/tests"
DATASETS_DIR="$PROJECT_ROOT/ATOS/datasets"

if [ ! -d "$BIN_DIR" ]; then
    echo "Error: GTAP binary directory not found: $BIN_DIR"
    echo "Please run 'make gtap_thread' in $COMPARE_DIR first."
    exit 1
fi

if [ ! -d "$ATOS_TESTS_DIR" ]; then
    echo "Error: ATOS tests directory not found: $ATOS_TESTS_DIR"
    exit 1
fi

declare -A DATASETS=(
    ["soc-LiveJournal1"]="$DATASETS_DIR/soc-LiveJournal1/soc-LiveJournal1_di.csr"
    ["hollywood-2009"]="$DATASETS_DIR/hollywood-2009/hollywood-2009_ud.csr"
    ["indochina-2004"]="$DATASETS_DIR/indochina-2004/indochina-2004_di.csr"
    ["road_usa"]="$DATASETS_DIR/road_usa/road_usa_ud.csr"
)

declare -A SOURCE_VERTICES=(
    ["soc-LiveJournal1"]=0
    ["hollywood-2009"]=0
    ["indochina-2004"]=40
    ["road_usa"]=0
)

NUM_RUNS=${NUM_RUNS:-20}
ATOS_ROUNDS=${ATOS_ROUNDS:-20}

RESULTS_FILE="$COMPARE_DIR/${BENCHMARK_NAME}_atos_performance_results.csv"
DETAILS_FILE="$COMPARE_DIR/${BENCHMARK_NAME}_atos_variant_results.csv"
echo "Dataset,GTAP_THREAD_avg_ms,GTAP_THREAD_errors,GTAP_THREAD_correct,ATOS_best_variant,ATOS_best_avg_ms,ATOS_best_error,ATOS_best_correct,Speedup(ATOS/GTAP_THREAD)" > "$RESULTS_FILE"
echo "Dataset,ATOS_variant,ATOS_avg_ms,ATOS_error,ATOS_correct,Command" > "$DETAILS_FILE"

printf "%20s | %18s | %9s | %7s | %22s | %12s | %10s | %7s | %14s\n" \
    "Dataset" "GTAP_THREAD" "GTAP_err" "GTAP_OK" "ATOS_best" "ATOS_ms" "ATOS_err" "ATOS_OK" "ATOS/THREAD"
printf "%20s-+-%18s-+-%9s-+-%7s-+-%22s-+-%12s-+-%10s-+-%7s-+-%14s\n" \
    "--------------------" "------------------" "---------" "-------" "----------------------" "------------" "----------" "-------" "--------------"

atos_candidates() {
    local dataset_name=$1

    case "$dataset_name" in
        soc-LiveJournal1)
            printf "option2_q8\t./bfs_32 -f ../../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr -o 2 -i 100 -rounds %s -q 8\n" "$ATOS_ROUNDS"
            printf "option3_q1\t./bfs_32 -f ../../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr -o 3 -q 1 -i 50 -rounds %s\n" "$ATOS_ROUNDS"
            printf "cta\t./bfs_cta_32 -f ../../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr -i 50 -rounds %s\n" "$ATOS_ROUNDS"
            printf "option4_q1\t./bfs_32 -f ../../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr -o 4 -q 1 -i 10 -rounds %s\n" "$ATOS_ROUNDS"
            printf "cta_discrete\t./bfs_cta_discrete_32 -f ../../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr -i 10 -rounds %s\n" "$ATOS_ROUNDS"
            ;;
        hollywood-2009)
            printf "option2_q4\t./bfs_32 -f ../../datasets/hollywood-2009/hollywood-2009_ud.csr -o 2 -i 100 -rounds %s -q 4\n" "$ATOS_ROUNDS"
            printf "option3_q1\t./bfs_32 -f ../../datasets/hollywood-2009/hollywood-2009_ud.csr -o 3 -q 1 -i 50 -rounds %s\n" "$ATOS_ROUNDS"
            printf "cta\t./bfs_cta_32 -f ../../datasets/hollywood-2009/hollywood-2009_ud.csr -i 50 -rounds %s\n" "$ATOS_ROUNDS"
            printf "option4_q1\t./bfs_32 -f ../../datasets/hollywood-2009/hollywood-2009_ud.csr -o 4 -q 1 -i 10 -rounds %s\n" "$ATOS_ROUNDS"
            printf "cta_discrete\t./bfs_cta_discrete_32 -f ../../datasets/hollywood-2009/hollywood-2009_ud.csr -i 10 -rounds %s\n" "$ATOS_ROUNDS"
            ;;
        indochina-2004)
            printf "option2_q4_i200\t./bfs_128 -f ../../datasets/indochina-2004/indochina-2004_di.csr -r 40 -i 200 -o 2 -rounds %s -q 4\n" "$ATOS_ROUNDS"
            printf "option2_q4_i1000\t./bfs_128 -f ../../datasets/indochina-2004/indochina-2004_di.csr -r 40 -i 1000 -o 2 -rounds %s -q 4\n" "$ATOS_ROUNDS"
            printf "option3_q1\t./bfs_128 -f ../../datasets/indochina-2004/indochina-2004_di.csr -o 3 -q 1 -i 70 -r 40 -rounds %s\n" "$ATOS_ROUNDS"
            printf "cta\t./bfs_cta_128 -f ../../datasets/indochina-2004/indochina-2004_di.csr -i 70 -r 40 -rounds %s\n" "$ATOS_ROUNDS"
            printf "option4_q1\t./bfs_64 -f ../../datasets/indochina-2004/indochina-2004_di.csr -o 4 -q 1 -i 10 -r 40 -rounds %s\n" "$ATOS_ROUNDS"
            printf "cta_discrete\t./bfs_cta_discrete_64 -f ../../datasets/indochina-2004/indochina-2004_di.csr -i 10 -r 40 -rounds %s\n" "$ATOS_ROUNDS"
            ;;
        road_usa)
            printf "option2_q8\t./bfs_32 -f ../../datasets/road_usa/road_usa_ud.csr -i 85000 -o 2 -rounds %s -q 8\n" "$ATOS_ROUNDS"
            printf "option3_q1\t./bfs_32 -f ../../datasets/road_usa/road_usa_ud.csr -o 3 -q 1 -i 20000 -rounds %s\n" "$ATOS_ROUNDS"
            printf "cta\t./bfs_cta_32 -f ../../datasets/road_usa/road_usa_ud.csr -i 15000 -rounds %s\n" "$ATOS_ROUNDS"
            printf "option4_q1\t./bfs_64 -f ../../datasets/road_usa/road_usa_ud.csr -o 4 -q 1 -i 10 -rounds %s\n" "$ATOS_ROUNDS"
            printf "cta_discrete\t./bfs_cta_discrete_64 -f ../../datasets/road_usa/road_usa_ud.csr -i 10 -rounds %s\n" "$ATOS_ROUNDS"
            ;;
    esac
}

run_gtap_average() {
    local program=$1
    local dataset_path=$2
    local source=$3
    local sum=0
    local count=0
    local total_errors=0
    local validation_count=0

    for _ in $(seq 1 "$NUM_RUNS"); do
        local output
        output=$("$program" "$dataset_path" "$source" 2>&1)

        local time
        time=$(echo "$output" | sed -n 's/.*Execution time: \([0-9.]*\) ms.*/\1/p' | head -n 1)

        if [ -n "$time" ]; then
            sum=$(echo "scale=6; $sum + $time" | bc -l)
            count=$((count + 1))
        fi

        local errors
        errors=$(echo "$output" | sed -n 's/[[:space:]]*Errors found:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | tail -n 1)
        if [ -n "$errors" ]; then
            total_errors=$((total_errors + errors))
            validation_count=$((validation_count + 1))
        fi
    done

    if [ "$count" -eq 0 ]; then
        echo "0 -1 UNKNOWN"
        return
    fi

    local avg
    avg=$(echo "scale=6; $sum / $count" | bc -l)

    local correct="UNKNOWN"
    if [ "$validation_count" -eq "$count" ]; then
        if [ "$total_errors" -eq 0 ]; then
            correct="YES"
        else
            correct="NO"
        fi
    fi

    echo "$avg $total_errors $correct"
}

run_atos_average() {
    local command=$1
    local output
    local time
    local error

    output=$(cd "$ATOS_TESTS_DIR" && eval "$command" 2>&1)
    time=$(echo "$output" | sed -n 's/.*Ave\. Time:[[:space:]]*\([0-9.]*\).*/\1/p' | tail -n 1)

    if [ -z "$time" ]; then
        time=$(echo "$output" | sed -n 's/.*Time:[[:space:]]*\([0-9.]*\).*/\1/p' | awk '{sum += $1; count += 1} END {if (count > 0) printf "%.6f", sum / count}')
    fi

    if [ -z "$time" ]; then
        echo "0 UNKNOWN UNKNOWN"
        return
    fi

    error=$(echo "$output" | sed -n 's/.*ERROR between CPU and GPU implimentation:[[:space:]]*\([-0-9.]*\).*/\1/p' | tail -n 1)
    if [ -z "$error" ]; then
        echo "$time UNKNOWN UNKNOWN"
        return
    fi

    local correct="NO"
    if [ "$error" = "0" ] || [ "$error" = "0.0" ] || [ "$error" = "0.00" ] || [ "$error" = "0.000000" ]; then
        correct="YES"
    fi

    echo "$time $error $correct"
}

speedup() {
    local numerator=$1
    local denominator=$2

    if [ "$numerator" = "0" ] || [ "$denominator" = "0" ]; then
        echo "0"
        return
    fi

    echo "scale=6; $numerator / $denominator" | bc -l
}

format_cell() {
    local value=$1

    if [ "$value" = "0" ]; then
        printf "%s" "N/A"
    else
        printf "%.3f" "$value"
    fi
}

for dataset_name in soc-LiveJournal1 hollywood-2009 indochina-2004 road_usa; do
    dataset_path="${DATASETS[$dataset_name]}"
    source_vertex="${SOURCE_VERTICES[$dataset_name]}"

    if [ ! -f "$dataset_path" ]; then
        echo "Warning: Dataset file not found: $dataset_path"
        continue
    fi

    GTAP_THREAD_AVG=0
    GTAP_THREAD_ERRORS=-1
    GTAP_THREAD_CORRECT=UNKNOWN
    if [ -x "$BIN_DIR/gtap_thread_$BENCHMARK_NAME" ]; then
        read GTAP_THREAD_AVG GTAP_THREAD_ERRORS GTAP_THREAD_CORRECT < <(run_gtap_average "$BIN_DIR/gtap_thread_$BENCHMARK_NAME" "$dataset_path" "$source_vertex")
    else
        echo "Warning: GTAP thread binary not executable: $BIN_DIR/gtap_thread_$BENCHMARK_NAME"
    fi

    ATOS_AVG=0
    ATOS_ERROR=UNKNOWN
    ATOS_CORRECT=UNKNOWN
    ATOS_VARIANT=N/A

    while IFS=$'\t' read -r candidate_variant candidate_command; do
        [ -n "$candidate_variant" ] || continue

        candidate_bin=$(echo "$candidate_command" | awk '{print $1}')
        candidate_avg=0
        candidate_error=UNKNOWN
        candidate_correct=UNKNOWN

        if [ -x "$ATOS_TESTS_DIR/${candidate_bin#./}" ]; then
            read candidate_avg candidate_error candidate_correct < <(run_atos_average "$candidate_command")
        else
            echo "Warning: ATOS binary not executable: $ATOS_TESTS_DIR/${candidate_bin#./}"
        fi

        echo "$dataset_name,$candidate_variant,$candidate_avg,$candidate_error,$candidate_correct,\"$candidate_command\"" >> "$DETAILS_FILE"

        if [ "$candidate_correct" = "YES" ]; then
            if [ "$ATOS_CORRECT" != "YES" ] || [ "$(echo "$candidate_avg < $ATOS_AVG" | bc -l)" -eq 1 ]; then
                ATOS_AVG="$candidate_avg"
                ATOS_ERROR="$candidate_error"
                ATOS_CORRECT="$candidate_correct"
                ATOS_VARIANT="$candidate_variant"
            fi
        fi
    done < <(atos_candidates "$dataset_name")

    SPEEDUP_ATOS_THREAD=0
    if [ "$GTAP_THREAD_CORRECT" = "YES" ] && [ "$ATOS_CORRECT" = "YES" ]; then
        SPEEDUP_ATOS_THREAD=$(speedup "$ATOS_AVG" "$GTAP_THREAD_AVG")
    fi

    printf "%20s | %18s | %9s | %7s | %22s | %12s | %10s | %7s | %14s\n" \
        "$dataset_name" \
        "$(format_cell "$GTAP_THREAD_AVG")" \
        "$GTAP_THREAD_ERRORS" \
        "$GTAP_THREAD_CORRECT" \
        "$ATOS_VARIANT" \
        "$(format_cell "$ATOS_AVG")" \
        "$ATOS_ERROR" \
        "$ATOS_CORRECT" \
        "$(format_cell "$SPEEDUP_ATOS_THREAD")"

    echo "$dataset_name,$GTAP_THREAD_AVG,$GTAP_THREAD_ERRORS,$GTAP_THREAD_CORRECT,$ATOS_VARIANT,$ATOS_AVG,$ATOS_ERROR,$ATOS_CORRECT,$SPEEDUP_ATOS_THREAD" >> "$RESULTS_FILE"
done

echo "Results written to $RESULTS_FILE"
echo "ATOS variant details written to $DETAILS_FILE"

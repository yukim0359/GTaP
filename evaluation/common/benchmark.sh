# Shared run / parse / median-IQR helpers for evaluation compare scripts.
# Source from a compare_*.sh; keep experiment knobs and CSV layout there.

BENCHMARK_MIN_SAMPLES="${BENCHMARK_MIN_SAMPLES:-5}"

benchmark_parse_execution_ms() {
    printf '%s\n' "$1" | grep "Execution time" \
        | sed -n 's/.*: \([0-9.]*\) ms.*/\1/p' | head -n 1
}

benchmark_parse_dynasoar_ms() {
    printf '%s\n' "$1" | grep '^\[TIME\]' \
        | sed -n 's/.*time = \([0-9.]*\).*/\1/p' | head -n 1
}

benchmark_stats_from_times() {
    local times=("$@")
    local m=${#times[@]}
    if [ "$m" -lt "$BENCHMARK_MIN_SAMPLES" ]; then
        echo "0 0 0 0 0"
        return
    fi

    local sorted=()
    local value
    while IFS= read -r value; do
        sorted+=("$value")
    done < <(printf '%s\n' "${times[@]}" | LC_ALL=C sort -n)
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

# Usage: benchmark_run_stats NUM_RUNS PARSER cmd [args...]
# PARSER is a function name that takes the command's combined stdout/stderr.
# Non-zero exits and unparseable output are dropped before aggregation.
benchmark_run_stats() {
    local num_runs=$1
    local parser=$2
    shift 2
    local times=()
    local i output elapsed rc
    local dropped=0

    for i in $(seq 1 "$num_runs"); do
        rc=0
        output=$("$@" 2>&1) || rc=$?
        if [ "$rc" -ne 0 ]; then
            echo "benchmark: drop $i/$num_runs exit=$rc: $*" >&2
            printf '%s\n' "$output" >&2
            dropped=$((dropped + 1))
            continue
        fi
        elapsed=$("$parser" "$output")
        if [ -n "$elapsed" ]; then
            times+=("$elapsed")
        else
            echo "benchmark: drop $i/$num_runs: no parseable time: $*" >&2
            printf '%s\n' "$output" >&2
            dropped=$((dropped + 1))
        fi
    done

    if [ "$dropped" -gt 0 ]; then
        echo "benchmark: used ${#times[@]}/$num_runs samples (dropped $dropped)" >&2
    fi

    benchmark_stats_from_times "${times[@]}"
}

benchmark_fmt_med_iqr() {
    local med=$1 elo=$2 ehi=$3
    local s
    if [ "$med" = "0" ]; then
        s="N/A"
    else
        s=$(printf "%.3f(+%.3f/%.3f)" "$med" "$ehi" "$elo")
    fi
    printf "%25s" "$s"
}

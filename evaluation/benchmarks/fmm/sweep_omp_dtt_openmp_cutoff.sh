#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CUTOFF_VALUES="${CUTOFF_VALUES:-512,1024,2048,4096,8192,16384,32768}"
DEPTH_VALUES="${DEPTH_VALUES:-5}"
N_BODIES="${N_BODIES:-40000000}"
THETA="${THETA:-0.3}"
NUM_RUNS="${NUM_RUNS:-3}"
TIMEOUT_SEC="${TIMEOUT_SEC:-0}"
BUILD_TIMEOUT_SEC="${BUILD_TIMEOUT_SEC:-0}"
RESULTS_CSV="${RESULTS_CSV:-$SCRIPT_DIR/fmm_omp_dtt_openmp_cutoff_sweep.csv}"
RAW_CSV="${RAW_CSV:-${RESULTS_CSV%.csv}_raw.csv}"
KEEP_LOGS="${KEEP_LOGS:-0}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/sweep_logs}"
MAKE_TARGET="${MAKE_TARGET:-omp_dtt_fmm}"
BIN_PATH="${BIN_PATH:-$SCRIPT_DIR/bin/omp_dtt_fmm}"

usage() {
  cat <<EOF
Usage: $0 [options]

Sweep OpenMP host DTT task cutoff for the 3D FMM fixed-cap host-DTT pipeline.
Records DTT core traversal (OpenMP parallel traversal), not list construction or H2D.

Options:
  --cutoffs "512,1024,2048"   Comma-separated FMM3D_HOST_DTT_TASK_MIN_N values
  --depths "4,5,6"            Comma-separated FMM3D_HOST_DTT_TASK_DEPTH values
  --n 40000000                Number of bodies
  --theta 0.3                 Theta value
  --runs 3                    Repetitions per configuration
  --timeout-sec 600           Per-run timeout in seconds; 0 disables timeout
  --build-timeout-sec 300     Per-build timeout in seconds; 0 disables timeout
  --out /path/summary.csv     Summary CSV path
  --raw-out /path/raw.csv     Raw per-run CSV path
  --keep-logs                 Keep per-run stdout/stderr logs
  --log-dir /path/logs        Log directory when --keep-logs is set

Useful env vars:
  OMP_NUM_THREADS=64
  FMM3D_DIRECT_SAMPLE_N=1000
  CUTOFF_VALUES, DEPTH_VALUES, N_BODIES, THETA, NUM_RUNS,
  TIMEOUT_SEC, BUILD_TIMEOUT_SEC, RESULTS_CSV, RAW_CSV, KEEP_LOGS, LOG_DIR
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cutoffs|--values) CUTOFF_VALUES="$2"; shift 2 ;;
    --depths) DEPTH_VALUES="$2"; shift 2 ;;
    --n) N_BODIES="$2"; shift 2 ;;
    --theta) THETA="$2"; shift 2 ;;
    --runs) NUM_RUNS="$2"; shift 2 ;;
    --timeout-sec|--timeout) TIMEOUT_SEC="$2"; shift 2 ;;
    --build-timeout-sec|--build-timeout) BUILD_TIMEOUT_SEC="$2"; shift 2 ;;
    --out) RESULTS_CSV="$2"; shift 2 ;;
    --raw-out) RAW_CSV="$2"; shift 2 ;;
    --keep-logs) KEEP_LOGS=1; shift ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

trim() {
  local x="$1"
  x="${x#"${x%%[![:space:]]*}"}"
  x="${x%"${x##*[![:space:]]}"}"
  printf "%s" "$x"
}

extract_ms() {
  local label="$1"
  sed -n "s/^${label}: \([0-9.][0-9.]*\) ms.*/\1/p" | head -n 1
}

# Match compare_fmm_dtt_theta.sh: primary metric is DTT core traversal (OpenMP parallel h_dtt_fixed).
extract_dtt_ms() {
  local text="$1"
  local value

  value="$(printf '%s\n' "$text" | sed -n 's/.*DTT traversal core: \([0-9.][0-9.]*\).*/\1/p' | head -n 1)"
  if [ -n "$value" ]; then
    printf '%s\n' "$value"
    return 0
  fi

  value="$(printf '%s\n' "$text" | sed -n 's/^DTT core traversal: \([0-9.][0-9.]*\) ms.*/\1/p' | head -n 1)"
  if [ -n "$value" ]; then
    printf '%s\n' "$value"
    return 0
  fi

  value="$(printf '%s\n' "$text" | sed -n 's/^DTT list construction (steady-state): \([0-9.][0-9.]*\) ms.*/\1/p' | head -n 1)"
  if [ -n "$value" ]; then
    printf '%s\n' "$value"
    return 0
  fi

  value="$(printf '%s\n' "$text" | sed -n 's/^DTT list construction + GPU handoff: \([0-9.][0-9.]*\) ms.*/\1/p' | head -n 1)"
  if [ -n "$value" ]; then
    printf '%s\n' "$value"
    return 0
  fi

  value="$(printf '%s\n' "$text" | sed -n 's/^DTT total (steady-state construction): \([0-9.][0-9.]*\) ms.*/\1/p' | head -n 1)"
  if [ -n "$value" ]; then
    printf '%s\n' "$value"
    return 0
  fi

  value="$(printf '%s\n' "$text" | sed -n 's/^DTT total (steady-state construction + GPU handoff): \([0-9.][0-9.]*\) ms.*/\1/p' | head -n 1)"
  if [ -n "$value" ]; then
    printf '%s\n' "$value"
    return 0
  fi

  value="$(printf '%s\n' "$text" | sed -n 's/^DTT total: \([0-9.][0-9.]*\) ms.*/\1/p' | head -n 1)"
  if [ -n "$value" ]; then
    printf '%s\n' "$value"
    return 0
  fi

  value="$(printf '%s\n' "$text" | extract_ms "DTT core only")"
  if [ -n "$value" ]; then
    printf '%s\n' "$value"
    return 0
  fi

  return 0
}

extract_field() {
  local pattern="$1"
  sed -n "$pattern" | head -n 1
}

stats_csv() {
  python - "$@" <<'PY'
import sys
vals = sorted(float(x) for x in sys.argv[1:] if x and x != "nan")
if not vals:
    print("nan,nan,nan,nan")
    raise SystemExit
n = len(vals)
mean = sum(vals) / n
median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
print(f"{median:.6f},{mean:.6f},{vals[0]:.6f},{vals[-1]:.6f}")
PY
}

nan_stats_csv() {
  printf "nan,nan,nan,nan"
}

run_build_with_timeout() {
  if [[ "$BUILD_TIMEOUT_SEC" == "0" ]]; then
    make "$@"
  else
    timeout "$BUILD_TIMEOUT_SEC" make "$@"
  fi
}

run_with_timeout() {
  if [[ "$TIMEOUT_SEC" == "0" ]]; then
    "$BIN_PATH" "$N_BODIES" "$THETA" 2>&1
  else
    timeout "$TIMEOUT_SEC" "$BIN_PATH" "$N_BODIES" "$THETA" 2>&1
  fi
}

IFS=',' read -r -a CUTOFFS <<< "$CUTOFF_VALUES"
IFS=',' read -r -a DEPTHS <<< "$DEPTH_VALUES"

if [[ "$KEEP_LOGS" == "1" ]]; then
  mkdir -p "$LOG_DIR"
fi

echo "depth,cutoff,n,theta,runs,omp_num_threads,direct_sample_n,dtt_median_ms,dtt_mean_ms,dtt_min_ms,dtt_max_ms,execution_median_ms,execution_mean_ms,execution_min_ms,execution_max_ms,sample_direct_median_ms,sample_direct_mean_ms,sample_direct_min_ms,sample_direct_max_ms" > "$RESULTS_CSV"
echo "depth,cutoff,n,theta,run,omp_num_threads,direct_sample_n,dtt_core_ms,execution_ms,sample_direct_ms,phi_rel_l2,acc_rel_l2,phi_rel_linf,acc_rel_linf,phi_p95,acc_p95,phi_p99,acc_p99,m2l_total,p2p_total,max_m2l,max_p2p,overflow,phi_checksum,acc_checksum,status" > "$RAW_CSV"

echo "Sweep settings:"
echo "  target      : $MAKE_TARGET"
echo "  binary      : $BIN_PATH"
echo "  depths      : $DEPTH_VALUES"
echo "  cutoffs     : $CUTOFF_VALUES"
echo "  n           : $N_BODIES"
echo "  theta       : $THETA"
echo "  runs        : $NUM_RUNS"
echo "  omp threads : ${OMP_NUM_THREADS:-}"
echo "  direct samp : ${FMM3D_DIRECT_SAMPLE_N:-0}"
echo "  timeout     : $TIMEOUT_SEC sec"
echo "  build tout  : $BUILD_TIMEOUT_SEC sec"
echo "  summary csv : $RESULTS_CSV"
echo "  raw csv     : $RAW_CSV"
echo

for DEPTH in "${DEPTHS[@]}"; do
  DEPTH="$(trim "$DEPTH")"
  [[ -z "$DEPTH" ]] && continue

  for CUTOFF in "${CUTOFFS[@]}"; do
    CUTOFF="$(trim "$CUTOFF")"
    [[ -z "$CUTOFF" ]] && continue

    echo "=== depth=$DEPTH cutoff=$CUTOFF ==="
    build_log="/tmp/fmm_omp_dtt_build_d${DEPTH}_cutoff${CUTOFF}.log"
    set +e
    run_build_with_timeout -B "$MAKE_TARGET" \
      FMM3D_HOST_DTT_TASK_DEPTH="$DEPTH" \
      FMM3D_HOST_DTT_TASK_MIN_N="$CUTOFF" \
      >"$build_log" 2>&1
    build_rc=$?
    set -e

    if [[ "$build_rc" -ne 0 ]]; then
      if [[ "$BUILD_TIMEOUT_SEC" != "0" && "$build_rc" -eq 124 ]]; then
        status="build_timeout"
      else
        status="build_failed_${build_rc}"
      fi
      echo "  $status; see: $build_log"
      for run_id in $(seq 1 "$NUM_RUNS"); do
        echo "$DEPTH,$CUTOFF,$N_BODIES,$THETA,$run_id,${OMP_NUM_THREADS:-},${FMM3D_DIRECT_SAMPLE_N:-0},nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,,,,,,,,$status" >> "$RAW_CSV"
      done
      nan_stats="$(nan_stats_csv)"
      echo "$DEPTH,$CUTOFF,$N_BODIES,$THETA,$NUM_RUNS,${OMP_NUM_THREADS:-},${FMM3D_DIRECT_SAMPLE_N:-0},$nan_stats,$nan_stats,$nan_stats" >> "$RESULTS_CSV"
      echo
      continue
    fi

    dtt_times=()
    execution_times=()
    sample_times=()
    ok_runs=0

    for run_id in $(seq 1 "$NUM_RUNS"); do
      set +e
      output="$(run_with_timeout)"
      run_rc=$?
      set -e

      status="ok"
      if [[ "$run_rc" -ne 0 ]]; then
        if [[ "$TIMEOUT_SEC" != "0" && "$run_rc" -eq 124 ]]; then
          status="timeout"
        else
          status="run_failed_${run_rc}"
        fi
      fi

      if [[ "$KEEP_LOGS" == "1" ]]; then
        log_file="$LOG_DIR/omp_dtt_d${DEPTH}_cutoff${CUTOFF}_run${run_id}.log"
        printf "%s\n" "$output" > "$log_file"
      fi

      if [[ "$status" != "ok" ]]; then
        echo "$DEPTH,$CUTOFF,$N_BODIES,$THETA,$run_id,${OMP_NUM_THREADS:-},${FMM3D_DIRECT_SAMPLE_N:-0},nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,,,,,,,,$status" >> "$RAW_CSV"
        echo "  run $run_id/$NUM_RUNS: $status"
        continue
      fi

      dtt_core="$(extract_dtt_ms "$output")"
      execution="$(printf "%s\n" "$output" | extract_ms "Execution time")"
      sample_direct="$(printf "%s\n" "$output" | sed -n 's/^Sample direct check: .*sample_direct_time=\([0-9.][0-9.]*\) ms.*/\1/p' | head -n 1)"
      [[ -z "$sample_direct" ]] && sample_direct="nan"

      sample_metrics="$(printf "%s\n" "$output" | sed -n 's/^Sample direct check: .*phi_rel_l2=\([^ ]*\)  acc_rel_l2=\([^ ]*\)  phi_rel_linf=\([^ ]*\)  acc_rel_linf=\([^ ]*\)  phi_p95=\([^ ]*\)  acc_p95=\([^ ]*\)  phi_p99=\([^ ]*\)  acc_p99=\([^ ]*\).*/\1,\2,\3,\4,\5,\6,\7,\8/p' | head -n 1)"
      if [[ -z "$sample_metrics" ]]; then
        sample_metrics="nan,nan,nan,nan,nan,nan,nan,nan"
      fi

      interactions="$(printf "%s\n" "$output" | extract_field 's/^Interactions: M2L=\([0-9][0-9]*\)  P2P=\([0-9][0-9]*\).*/\1,\2/p')"
      [[ -z "$interactions" ]] && interactions=","
      m2l_total="${interactions%,*}"
      p2p_total="${interactions#*,}"

      fixed_metrics="$(printf "%s\n" "$output" | extract_field 's/^DTT fixed-cap metrics: .*max_m2l=\([0-9][0-9]*\)  max_p2p=\([0-9][0-9]*\)  overflow=\([0-9][0-9]*\).*/\1,\2,\3/p')"
      [[ -z "$fixed_metrics" ]] && fixed_metrics=",,"
      max_m2l="${fixed_metrics%%,*}"
      rest="${fixed_metrics#*,}"
      max_p2p="${rest%%,*}"
      overflow="${rest#*,}"

      checksums="$(printf "%s\n" "$output" | extract_field 's/^Validation: phi_checksum=\([^ ]*\)  acc_checksum=\([^ ]*\).*/\1,\2/p')"
      [[ -z "$checksums" ]] && checksums=","
      phi_checksum="${checksums%,*}"
      acc_checksum="${checksums#*,}"

      if [[ -z "$dtt_core" || -z "$execution" ]]; then
        status="parse_failed"
        echo "$DEPTH,$CUTOFF,$N_BODIES,$THETA,$run_id,${OMP_NUM_THREADS:-},${FMM3D_DIRECT_SAMPLE_N:-0},nan,nan,nan,$sample_metrics,$m2l_total,$p2p_total,$max_m2l,$max_p2p,$overflow,$phi_checksum,$acc_checksum,$status" >> "$RAW_CSV"
        echo "  run $run_id/$NUM_RUNS: parse_failed"
        continue
      fi

      dtt_times+=("$dtt_core")
      execution_times+=("$execution")
      sample_times+=("$sample_direct")
      ok_runs=$((ok_runs + 1))

      echo "$DEPTH,$CUTOFF,$N_BODIES,$THETA,$run_id,${OMP_NUM_THREADS:-},${FMM3D_DIRECT_SAMPLE_N:-0},$dtt_core,$execution,$sample_direct,$sample_metrics,$m2l_total,$p2p_total,$max_m2l,$max_p2p,$overflow,$phi_checksum,$acc_checksum,$status" >> "$RAW_CSV"
      echo "  run $run_id/$NUM_RUNS: dtt=${dtt_core} ms, exec=${execution} ms, sample=${sample_direct} ms"
    done

    if [[ "$ok_runs" -eq 0 ]]; then
      dtt_stats="$(nan_stats_csv)"
      execution_stats="$(nan_stats_csv)"
      sample_stats="$(nan_stats_csv)"
    else
      dtt_stats="$(stats_csv "${dtt_times[@]}")"
      execution_stats="$(stats_csv "${execution_times[@]}")"
      sample_stats="$(stats_csv "${sample_times[@]}")"
    fi

    echo "$DEPTH,$CUTOFF,$N_BODIES,$THETA,$NUM_RUNS,${OMP_NUM_THREADS:-},${FMM3D_DIRECT_SAMPLE_N:-0},$dtt_stats,$execution_stats,$sample_stats" >> "$RESULTS_CSV"
    IFS=',' read -r dtt_med _ _ _ <<< "$dtt_stats"
    IFS=',' read -r exec_med _ _ _ <<< "$execution_stats"
    echo "  summary: ok_runs=${ok_runs}/${NUM_RUNS}, dtt_median=${dtt_med} ms, exec_median=${exec_med} ms"
    echo
  done
done

echo "Done."
echo "Summary: $RESULTS_CSV"
echo "Raw runs: $RAW_CSV"

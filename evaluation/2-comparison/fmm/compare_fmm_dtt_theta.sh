#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -W group_list=gc64
#PBS -j oe
#
# Compare DTT core traversal time vs theta (fixed N) across three FMM pipelines.
# Uses "DTT core traversal" (GTaP/worklist kernel; Host OMP parallel h_dtt_fixed only).
# Host OMP steady-state construction + H2D are separate lines in bin output / CSV breakdown cols.
#   - GTaP task traversal      (bin/N<N>/theta<THETA>/gtap_dtt_fmm)
#   - GPU worklist traversal   (bin/N<N>/theta<THETA>/worklist_dtt_fmm)
#   - Host OpenMP traversal    (bin/N<N>/theta<THETA>/omp_dtt_fmm)
#
# Build staged bins first (same layout as plot_fmm_dtt_stack.sh):
#   ./build_fmm_dtt_theta_bins.sh 0.3 10000000
#   # or for all thetas: ./plot_fmm_dtt_stack.sh --rebuild
# Falls back to bin/theta<THETA>/ then bin/ if staged dirs are missing.
#
# Local run:
#   ./compare_fmm_dtt_theta.sh
#   ./compare_fmm_dtt_theta.sh --plot-only   # skip measurement if CSV exists
#
# Plot (combined absolute + GTaP-normalized per N):
#   python3 plot_performance_fmm_dtt_theta.py
#   -> img/fmm_dtt_theta_combined_N{N}.pdf
#
# Useful env overrides:
#   N_VALUES="10000000,50000000"
#   THETA_VALUES="0.2,0.25,0.3,0.35,0.4,0.5"
#   NUM_RUNS=20  OMP_NUM_THREADS=48  TIMEOUT_SEC=900
# Host OMP defaults: source omp_host_dtt_env.sh (72 threads, bind=close, places=cores)
#
# Note: GTaP DTT at N=50M and small theta needs gtap_dtt_fmm rebuilt with
# FMM3D_DTT_TASK_MIN_N=32 (see Makefile). If it still fails, try 50M only for
# theta>=0.3 or raise GTAP_MAX_TASKS_PER_WARP / FMM3D_CUDA_STACK_SIZE.

set -euo pipefail

# Under PBS, BASH_SOURCE points at the spool copy of this script, so prefer
# PBS_O_WORKDIR (the directory from which qsub was run).
if [ -n "${PBS_O_WORKDIR:-}" ]; then
  COMPARE_DIR="$PBS_O_WORKDIR"
else
  COMPARE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$COMPARE_DIR"

plot_only=0
for arg in "$@"; do
  case "$arg" in
    --plot-only)
      plot_only=1
      ;;
    --help|-h)
      cat <<EOF
Usage: $0 [--plot-only]

Compare DTT core traversal time vs theta (GTaP / worklist / Host OpenMP).
Writes: fmm_dtt_theta_results.csv
Plots:  python3 plot_performance_fmm_dtt_theta.py

  $0              Run measurement sweep (default)
  $0 --plot-only  Replot from existing fmm_dtt_theta_results.csv only

Env: N_VALUES, THETA_VALUES, NUM_RUNS, RESULTS_FILE, OMP_NUM_THREADS, ...
EOF
      exit 0
      ;;
  esac
done

if [ "$plot_only" -eq 1 ]; then
  RESULTS_FILE="${RESULTS_FILE:-$COMPARE_DIR/fmm_dtt_theta_results.csv}"
  if [ ! -f "$RESULTS_FILE" ]; then
    echo "Error: missing $RESULTS_FILE (run ./compare_fmm_dtt_theta.sh first)" >&2
    exit 1
  fi
  echo "Plot only from: $RESULTS_FILE"
  exec python3 "$COMPARE_DIR/plot_performance_fmm_dtt_theta.py"
fi

# shellcheck source=omp_host_dtt_env.sh
source "$COMPARE_DIR/omp_host_dtt_env.sh"

BIN_GTAP=""
BIN_WORKLIST=""
BIN_OMP=""
BIN_BINDIR=""

theta_label() {
  python3 -c "t=format(float('${1}'), 'f').rstrip('0'); t=t[:-1] if t.endswith('.') else t; print(t.replace('.', ''))"
}

bindir_for_theta() {
  local n=$1 theta=$2
  local theta_lbl staged_n staged_legacy
  theta_lbl="$(theta_label "$theta")"
  staged_n="$COMPARE_DIR/bin/N${n}/theta${theta_lbl}"
  staged_legacy="$COMPARE_DIR/bin/theta${theta_lbl}"
  if [ -x "$staged_n/gtap_dtt_fmm" ]; then
    echo "$staged_n"
  elif [ -x "$staged_legacy/gtap_dtt_fmm" ]; then
    echo "$staged_legacy"
  else
    echo "$COMPARE_DIR/bin"
  fi
}

resolve_bins() {
  local n=$1 theta=$2
  BIN_BINDIR="$(bindir_for_theta "$n" "$theta")"
  BIN_GTAP="$BIN_BINDIR/gtap_dtt_fmm"
  BIN_WORKLIST="$BIN_BINDIR/worklist_dtt_fmm"
  BIN_OMP="$BIN_BINDIR/omp_dtt_fmm"
}

require_bins() {
  local n=$1 theta=$2
  local missing=0
  for bin in "$BIN_GTAP" "$BIN_WORKLIST" "$BIN_OMP"; do
    if [ ! -x "$bin" ]; then
      echo "Error: missing executable: $bin" >&2
      missing=1
    fi
  done
  if [ "$missing" -ne 0 ]; then
    echo "Build with: $COMPARE_DIR/build_fmm_dtt_theta_bins.sh ${theta} ${n}" >&2
    exit 1
  fi
}

NUM_RUNS="${NUM_RUNS:-20}"
TIMEOUT_SEC="${TIMEOUT_SEC:-0}"

if [ -n "${N_VALUES:-}" ]; then
  IFS=',' read -r -a N_SIZES <<< "$N_VALUES"
else
  N_SIZES=(10000000 50000000)
fi

if [ -n "${THETA_VALUES:-}" ]; then
  IFS=',' read -r -a THETAS <<< "$THETA_VALUES"
else
  THETAS=(0.2 0.3 0.4 0.5)
fi

RESULTS_FILE="${RESULTS_FILE:-$COMPARE_DIR/fmm_dtt_theta_results.csv}"
echo "n,theta,nodes,leaves,max_m2l,max_p2p,max_leaf,at_cap_pct,wl_pair_cap,wl_iterations,wl_max_frontier,wl_overflow,GTAP_med,GTAP_err_low,GTAP_err_high,WORKLIST_med,WORKLIST_err_low,WORKLIST_err_high,OMP_med,OMP_err_low,OMP_err_high,OMP_reset_med,OMP_traverse_med,OMP_stats_med,OMP_h2d_counts_med,OMP_h2d_lists_med,Ratio_OMP/GTAP,Ratio_WORKLIST/GTAP" > "$RESULTS_FILE"

printf "FMM DTT theta sweep (runs=%s, N=%s, theta=%s)\n" \
  "$NUM_RUNS" "${N_VALUES:-${N_SIZES[*]}}" "${THETA_VALUES:-${THETAS[*]}}"
printf "OMP env: OMP_NUM_THREADS=%s OMP_PROC_BIND=%s OMP_PLACES=%s\n" \
  "${OMP_NUM_THREADS:-}" "${OMP_PROC_BIND:-}" "${OMP_PLACES:-}"
printf "%10s | %6s | %10s | %10s | %8s | %8s | %6s | %6s | %10s | %22s | %22s | %22s\n" \
  "N" "theta" "Nodes" "Leaves" "max_m2l" "max_p2p" "mxleaf" "cap%%" \
  "WL maxF" "GTaP DTT" "Worklist" "Host OMP"
printf "%10s-+-%6s-+-%10s-+-%10s-+-%8s-+-%8s-+-%6s-+-%6s-+-%10s-+-%22s-+-%22s-+-%22s\n" \
  "----------" "------" "----------" "----------" "--------" "--------" "------" "------" \
  "----------" "----------------------" "----------------------" "----------------------"

extract_field() {
  local text="$1"
  local pattern="$2"
  printf "%s\n" "$text" | sed -n "$pattern" | head -n 1
}

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

  value="$(printf '%s\n' "$text" | sed -n 's/.*DTT list construction, steady-state: \([0-9.][0-9.]*\).*/\1/p' | head -n 1)"
  if [ -n "$value" ]; then
    printf '%s\n' "$value"
    return 0
  fi

  value="$(printf '%s\n' "$text" | sed -n 's/^DTT list construction (steady-state): \([0-9.][0-9.]*\) ms.*/\1/p' | head -n 1)"
  if [ -n "$value" ]; then
    printf '%s\n' "$value"
    return 0
  fi

  value="$(printf '%s\n' "$text" | sed -n 's/^DTT total (steady-state construction): \([0-9.][0-9.]*\) ms.*/\1/p' | head -n 1)"
  if [ -n "$value" ]; then
    printf '%s\n' "$value"
    return 0
  fi

  value="$(printf '%s\n' "$text" | sed -n 's/^DTT list construction + GPU handoff: \([0-9.][0-9.]*\) ms.*/\1/p' | head -n 1)"
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

  value="$(printf '%s\n' "$text" | sed -n 's/.*DTT lists (worklist): \([0-9.][0-9.]*\).*/\1/p' | head -n 1)"
  if [ -n "$value" ]; then
    printf '%s\n' "$value"
    return 0
  fi

  value="$(printf '%s\n' "$text" | sed -n 's/.*DTT lists (GTaP task traversal): \([0-9.][0-9.]*\).*/\1/p' | head -n 1)"
  if [ -n "$value" ]; then
    printf '%s\n' "$value"
    return 0
  fi

  value="$(printf '%s\n' "$text" | sed -n 's/.*DTT lists (construction + GPU handoff): \([0-9.][0-9.]*\).*/\1/p' | head -n 1)"
  if [ -n "$value" ]; then
    printf '%s\n' "$value"
    return 0
  fi

  value="$(printf '%s\n' "$text" | sed -n 's/.*DTT lists (host traversal): \([0-9.][0-9.]*\).*/\1/p' | head -n 1)"
  if [ -n "$value" ]; then
    printf '%s\n' "$value"
    return 0
  fi

  return 0
}

log_run_failure() {
  local label=$1
  local n=$2
  local theta=$3
  local rc=$4
  local output=$5
  local reason

  reason="$(printf '%s\n' "$output" | grep -E 'overflow|CUDA error|Error' | head -n 1 || true)"
  if [ -z "$reason" ]; then
    reason="rc=$rc (no DTT total in output)"
  fi
  printf '[warn] %s N=%s theta=%s: %s\n' "$label" "$n" "$theta" "$reason" >&2
}

run_program() {
  local program=$1
  local n=$2
  local theta=$3
  if [ "$TIMEOUT_SEC" = "0" ]; then
    "$program" "$n" "$theta" 2>&1
  else
    timeout "$TIMEOUT_SEC" "$program" "$n" "$theta" 2>&1
  fi
}

extract_worklist_metrics() {
  local text="$1"
  local details
  details="$(printf '%s\n' "$text" | sed -n 's/^DTT worklist metrics: pair_cap=\([0-9][0-9]*\)  iterations=\([0-9][0-9]*\)  max_frontier=\([0-9][0-9]*\)  overflow=\([0-9][0-9]*\).*/\1 \2 \3 \4/p' | head -n 1)"
  if [ -n "$details" ]; then
    printf '%s\n' "$details"
    return 0
  fi
  printf '%s\n' "$text" | sed -n 's/^DTT worklist metrics (until overflow): pair_cap=\([0-9][0-9]*\)  iterations=\([0-9][0-9]*\)  max_frontier=\([0-9][0-9]*\)  overflow=\([0-9][0-9]*\).*/\1 \2 \3 \4/p' | head -n 1
}

run_stats() {
  local program=$1
  local label=$2
  local n=$3
  local theta=$4
  local times=()
  local warned=0

  for _ in $(seq 1 "$NUM_RUNS"); do
    local output dtt rc
    set +e
    output="$(run_program "$program" "$n" "$theta")"
    rc=$?
    set -e

    dtt="$(extract_dtt_ms "$output")"
    if [ -n "$dtt" ]; then
      times+=("$dtt")
    elif [ "$warned" -eq 0 ]; then
      log_run_failure "$label" "$n" "$theta" "$rc" "$output"
      warned=1
    fi
  done

  local m=${#times[@]}
  if [ "$m" -lt 5 ]; then
    echo "0 0 0 0 0"
    return
  fi

  local -a sorted
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

extract_host_breakdown() {
  local text="$1"
  local details
  details="$(printf '%s\n' "$text" | sed -n 's/^DTT host breakdown (steady-state): reset_counts=\([0-9.][0-9.]*\)  traverse=\([0-9.][0-9.]*\)  stats=\([0-9.][0-9.]*\)  h2d_counts=\([0-9.][0-9.]*\)  h2d_lists=\([0-9.][0-9.]*\).*/\1 \2 \3 \4 \5/p' | head -n 1)"
  if [ -n "$details" ]; then
    printf '%s\n' "$details"
    return 0
  fi
  printf '%s\n' "$text" | sed -n 's/^DTT host breakdown: alloc_init=\([0-9.][0-9.]*\)  traverse=\([0-9.][0-9.]*\)  stats=\([0-9.][0-9.]*\)  h2d_counts=\([0-9.][0-9.]*\)  h2d_lists=\([0-9.][0-9.]*\).*/\1 \2 \3 \4 \5/p' | head -n 1
}

median_from_array() {
  local -a values=("$@")
  local m=${#values[@]}
  if [ "$m" -lt 5 ]; then
    echo "0"
    return
  fi

  local -a sorted
  IFS=$'\n' sorted=($(printf "%s\n" "${values[@]}" | sort -n))
  unset IFS
  m=${#sorted[@]}
  if (( m % 2 == 1 )); then
    echo "${sorted[$((m/2))]}"
  else
    echo "$(echo "scale=6; (${sorted[$((m/2-1))]} + ${sorted[$((m/2))]}) / 2" | bc -l)"
  fi
}

run_stats_omp_details() {
  local program=$1
  local n=$2
  local theta=$3
  local times=()
  local alloc_times=()
  local trav_times=()
  local stats_times=()
  local h2dc_times=()
  local h2dl_times=()
  local warned=0

  for _ in $(seq 1 "$NUM_RUNS"); do
    local output dtt rc
    set +e
    output="$(run_program "$program" "$n" "$theta")"
    rc=$?
    set -e

    dtt="$(extract_dtt_ms "$output")"
    if [ -n "$dtt" ]; then
      times+=("$dtt")
      local details
      details="$(extract_host_breakdown "$output")"
      if [ -n "$details" ]; then
        local a t s c l
        read -r a t s c l <<< "$details"
        alloc_times+=("$a")
        trav_times+=("$t")
        stats_times+=("$s")
        h2dc_times+=("$c")
        h2dl_times+=("$l")
      fi
    elif [ "$warned" -eq 0 ]; then
      log_run_failure "OpenMP" "$n" "$theta" "$rc" "$output"
      warned=1
    fi
  done

  local m=${#times[@]}
  if [ "$m" -lt 5 ]; then
    echo "0 0 0 0 0 0 0 0 0 0"
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

  local alloc_med trav_med stats_med h2dc_med h2dl_med
  alloc_med="$(median_from_array "${alloc_times[@]}")"
  trav_med="$(median_from_array "${trav_times[@]}")"
  stats_med="$(median_from_array "${stats_times[@]}")"
  h2dc_med="$(median_from_array "${h2dc_times[@]}")"
  h2dl_med="$(median_from_array "${h2dl_times[@]}")"

  echo "$median $q1 $q3 $err_low $err_high $alloc_med $trav_med $stats_med $h2dc_med $h2dl_med"
}

probe_meta() {
  local n=$1
  local theta=$2
  local output

  PROBE_NODES=""
  PROBE_LEAVES=""
  PROBE_MAX_M2L=""
  PROBE_MAX_P2P=""
  PROBE_MAX_LEAF=""
  PROBE_AT_CAP_PCT=""
  PROBE_WL_PAIR_CAP=""
  PROBE_WL_ITERATIONS=""
  PROBE_WL_MAX_FRONTIER=""
  PROBE_WL_OVERFLOW=""

  set +e
  output="$(run_program "$BIN_GTAP" "$n" "$theta")"
  set -e

  PROBE_NODES="$(extract_field "$output" 's/.*Nodes=\([0-9][0-9]*\).*/\1/p')"
  PROBE_LEAVES="$(extract_field "$output" 's/.*Leaves=\([0-9][0-9]*\).*/\1/p')"
  PROBE_MAX_M2L="$(extract_field "$output" 's/.*max_m2l=\([0-9][0-9]*\).*/\1/p')"
  PROBE_MAX_P2P="$(extract_field "$output" 's/.*max_p2p=\([0-9][0-9]*\).*/\1/p')"
  PROBE_MAX_LEAF="$(extract_field "$output" 's/.*max_leaf=\([0-9][0-9]*\).*/\1/p')"
  PROBE_AT_CAP_PCT="$(extract_field "$output" 's/.*Leaf depth: at_cap=[0-9]*\/[0-9]* (\([0-9.][0-9.]*\)%).*/\1/p')"

  set +e
  output="$(run_program "$BIN_WORKLIST" "$n" "$theta")"
  set -e
  local wl_metrics
  wl_metrics="$(extract_worklist_metrics "$output")"
  if [ -n "$wl_metrics" ]; then
    read -r PROBE_WL_PAIR_CAP PROBE_WL_ITERATIONS PROBE_WL_MAX_FRONTIER PROBE_WL_OVERFLOW <<< "$wl_metrics"
  fi
}

fmt_med_iqr() {
  local med=$1 elo=$2 ehi=$3
  if [ "$med" = "0" ]; then
    printf "%22s" "N/A"
  else
    printf "%22s" "$(printf "%.3f(+%.3f/%.3f)" "$med" "$ehi" "$elo")"
  fi
}

fmt_int_or_na() {
  local value=$1
  local width=$2
  if [ -z "$value" ]; then
    printf "%${width}s" "N/A"
  else
    printf "%${width}d" "$value"
  fi
}

fmt_float_or_na() {
  local value=$1
  local width=$2
  if [ -z "$value" ]; then
    printf "%${width}s" "N/A"
  else
    printf "%${width}.1f" "$value"
  fi
}

for n in "${N_SIZES[@]}"; do
  n="$(echo "$n" | tr -d '[:space:]')"
  [ -z "$n" ] && continue

  for theta in "${THETAS[@]}"; do
    theta="$(echo "$theta" | tr -d '[:space:]')"
    [ -z "$theta" ] && continue

    resolve_bins "$n" "$theta"
    require_bins "$n" "$theta"
    printf "Bins for N=%s theta=%s: %s\n" "$n" "$theta" "$BIN_BINDIR" >&2

    probe_meta "$n" "$theta"

    GTAP_MED=0 GTAP_ELO=0 GTAP_EHI=0
    read GTAP_MED _ _ GTAP_ELO GTAP_EHI < <(run_stats "$BIN_GTAP" "GTaP" "$n" "$theta")

    WORKLIST_MED=0 WORKLIST_ELO=0 WORKLIST_EHI=0
    read WORKLIST_MED _ _ WORKLIST_ELO WORKLIST_EHI < <(run_stats "$BIN_WORKLIST" "worklist" "$n" "$theta")

    OMP_MED=0 OMP_ELO=0 OMP_EHI=0
    OMP_ALLOC_MED=0 OMP_TRAVERSE_MED=0 OMP_STATS_MED=0 OMP_H2D_COUNTS_MED=0 OMP_H2D_LISTS_MED=0
    read OMP_MED _ _ OMP_ELO OMP_EHI OMP_ALLOC_MED OMP_TRAVERSE_MED OMP_STATS_MED OMP_H2D_COUNTS_MED OMP_H2D_LISTS_MED \
      < <(run_stats_omp_details "$BIN_OMP" "$n" "$theta")

    RATIO_OMP=0
    RATIO_WL=0
    if [ "$GTAP_MED" != "0" ] && [ "$OMP_MED" != "0" ]; then
      RATIO_OMP=$(echo "scale=6; $OMP_MED / $GTAP_MED" | bc -l)
    fi
    if [ "$GTAP_MED" != "0" ] && [ "$WORKLIST_MED" != "0" ]; then
      RATIO_WL=$(echo "scale=6; $WORKLIST_MED / $GTAP_MED" | bc -l)
    fi

    printf "%10d | %6s | " "$n" "$theta"
    fmt_int_or_na "$PROBE_NODES" 10; printf " | "
    fmt_int_or_na "$PROBE_LEAVES" 10; printf " | "
    fmt_int_or_na "$PROBE_MAX_M2L" 8; printf " | "
    fmt_int_or_na "$PROBE_MAX_P2P" 8; printf " | "
    fmt_int_or_na "$PROBE_MAX_LEAF" 6; printf " | "
    fmt_float_or_na "$PROBE_AT_CAP_PCT" 6; printf " | "
    fmt_int_or_na "$PROBE_WL_MAX_FRONTIER" 10; printf " | "
    fmt_med_iqr "$GTAP_MED" "$GTAP_ELO" "$GTAP_EHI"; printf " | "
    fmt_med_iqr "$WORKLIST_MED" "$WORKLIST_ELO" "$WORKLIST_EHI"; printf " | "
    fmt_med_iqr "$OMP_MED" "$OMP_ELO" "$OMP_EHI"; printf "\n"

    echo "$n,$theta,$PROBE_NODES,$PROBE_LEAVES,$PROBE_MAX_M2L,$PROBE_MAX_P2P,$PROBE_MAX_LEAF,$PROBE_AT_CAP_PCT,$PROBE_WL_PAIR_CAP,$PROBE_WL_ITERATIONS,$PROBE_WL_MAX_FRONTIER,$PROBE_WL_OVERFLOW,$GTAP_MED,$GTAP_ELO,$GTAP_EHI,$WORKLIST_MED,$WORKLIST_ELO,$WORKLIST_EHI,$OMP_MED,$OMP_ELO,$OMP_EHI,$OMP_ALLOC_MED,$OMP_TRAVERSE_MED,$OMP_STATS_MED,$OMP_H2D_COUNTS_MED,$OMP_H2D_LISTS_MED,$RATIO_OMP,$RATIO_WL" >> "$RESULTS_FILE"
  done
  echo
done

echo "Wrote: $RESULTS_FILE"
echo "Plot with: python3 $COMPARE_DIR/plot_performance_fmm_dtt_theta.py"

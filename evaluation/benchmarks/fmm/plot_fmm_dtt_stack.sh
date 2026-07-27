#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
# shellcheck source=omp_host_dtt_env.sh
source "$script_dir/omp_host_dtt_env.sh"

theta_label() {
  python3 -c "t=format(float('${1}'), 'f').rstrip('0'); t=t[:-1] if t.endswith('.') else t; print(t.replace('.', ''))"
}

stack_paper_title() {
  python3 - "$1" "$2" <<'PY'
import sys

def format_paper_n(n: int) -> str:
    if n >= 1_000_000 and n % 1_000_000 == 0:
        exp = 0
        value = n
        while value % 10 == 0 and value >= 10:
            value //= 10
            exp += 1
        return rf"$N={value}\times10^{{{exp}}}$"
    return f"N={n:,}"

def format_paper_theta(theta: float) -> str:
    return rf"$\theta={format(theta, 'g')}$"

n = int(sys.argv[1])
theta = float(sys.argv[2])
print(f"End-to-end FMM3D runtime breakdown ({format_paper_n(n)}, {format_paper_theta(theta)})")
PY
}

run_one() {
  local n=$1 theta=$2 bindir=$3 out=$4
  shift 4
  local num_runs="${NUM_RUNS:-20}"
  local csv_out="csv/fmm_3way_stack_N${n}_theta$(theta_label "$theta")"
  local title
  title="$(stack_paper_title "$n" "$theta")"
  python3 plot_phase_stack.py \
    --n "$n" \
    --theta "$theta" \
    --runs "$num_runs" \
    --bin "$bindir/gtap_dtt_fmm" \
    --label "GTaP DTT" \
    --bin "$bindir/worklist_dtt_fmm" \
    --label "GPU worklist DTT" \
    --bin "$bindir/omp_dtt_fmm" \
    --label "Host OpenMP DTT" \
    --out "$out" \
    --csv-out "$csv_out" \
    --title "$title" \
    --include-init \
    "$@"
}

replot_one() {
  local n=$1 theta=$2 out=$3
  shift 3
  local theta_lbl csv_in title
  theta_lbl="$(theta_label "$theta")"
  csv_in="csv/fmm_3way_stack_N${n}_theta${theta_lbl}_summary.csv"
  if [[ ! -f "$csv_in" ]]; then
    echo "Error: missing $csv_in (run measurement first without --plot-only)" >&2
    exit 1
  fi
  title="$(stack_paper_title "$n" "$theta")"
  python3 plot_phase_stack.py \
    --csv-in "$csv_in" \
    --no-run \
    --include-init \
    --title "$title" \
    --out "$out" \
    "$@"
}

plot_paper_figure() {
  local n=$1
  shift
  local thetas=("$@")
  local out="img/fmm_e2e_breakdown_paper_N${n}.png"
  local caption_out="img/fmm_e2e_breakdown_paper_N${n}_caption.txt"
  local -a panel_csv=()
  local -a panel_tag=()
  local idx=0
  for theta in "${thetas[@]}"; do
    local theta_lbl csv_in
    theta_lbl="$(theta_label "$theta")"
    csv_in="csv/fmm_3way_stack_N${n}_theta${theta_lbl}_summary.csv"
    if [[ ! -f "$csv_in" ]]; then
      echo "Error: missing $csv_in" >&2
      exit 1
    fi
    panel_csv+=("$csv_in")
    if [[ "$idx" -eq 0 ]]; then
      panel_tag+=("(a) DTT-heavy regime")
    else
      panel_tag+=("(b) DTT-light regime")
    fi
    idx=$((idx + 1))
  done
  local -a panel_csv_args=()
  local -a panel_tag_args=()
  for csv in "${panel_csv[@]}"; do
    panel_csv_args+=(--panel-csv "$csv")
  done
  for tag in "${panel_tag[@]}"; do
    panel_tag_args+=(--panel-tag "$tag")
  done
  python3 plot_phase_stack.py \
    --paper \
    --no-run \
    "${panel_csv_args[@]}" \
    "${panel_tag_args[@]}" \
    --out "$out" \
    --caption-out "$caption_out" \
    "${extra_args[@]}"
}

plot_paper_controlled() {
  local n=$1
  shift
  local thetas=("$@")
  local out="img/fmm_e2e_breakdown_paper_N${n}_controlled.png"
  local caption_out="img/fmm_e2e_breakdown_paper_N${n}_controlled_caption.txt"
  local -a panel_csv=()
  local -a panel_tag=()
  local idx=0
  for theta in "${thetas[@]}"; do
    local theta_lbl csv_in
    theta_lbl="$(theta_label "$theta")"
    csv_in="csv/fmm_3way_stack_N${n}_theta${theta_lbl}_summary.csv"
    if [[ ! -f "$csv_in" ]]; then
      echo "Error: missing $csv_in" >&2
      exit 1
    fi
    panel_csv+=("$csv_in")
    if [[ "$idx" -eq 0 ]]; then
      panel_tag+=("(a) DTT-heavy regime")
    else
      panel_tag+=("(b) DTT-light regime")
    fi
    idx=$((idx + 1))
  done
  local -a panel_csv_args=()
  local -a panel_tag_args=()
  for csv in "${panel_csv[@]}"; do
    panel_csv_args+=(--panel-csv "$csv")
  done
  for tag in "${panel_tag[@]}"; do
    panel_tag_args+=(--panel-tag "$tag")
  done
  python3 plot_phase_stack.py \
    --paper \
    --controlled \
    --no-run \
    "${panel_csv_args[@]}" \
    "${panel_tag_args[@]}" \
    --out "$out" \
    --caption-out "$caption_out" \
    "${extra_args[@]}"
}

usage() {
  cat <<EOF
Usage: $0 [N] [--rebuild] [THETA] [OUT] [extra plot_phase_stack.py args...]
       $0 n=10000000,50000000 [--rebuild] theta=0.5,0.4,0.3,0.2 [extra plot_phase_stack.py args...]
       $0 --paper n=10000000 thetas=0.2,0.5
       $0 --paper-controlled n=10000000 thetas=0.2,0.5
       $0 --plot-only n=10000000 theta=0.2
       $0 --ns 10000000,50000000 [--thetas 0.2,0.3,0.4,0.5]

Default: N=50000000, thetas 0.2, 0.3, 0.4, 0.5, NUM_RUNS=20 (per-phase average; bar sum = average execution time).
Host DTT OpenMP defaults from omp_host_dtt_env.sh (OMP_NUM_THREADS=72, bind=close, places=cores).
Override before run: OMP_NUM_THREADS=48 $0 ...

Outputs:
  img/fmm_3way_stack_N<N>_theta<THETA_LABEL>.png
  csv/fmm_3way_stack_N<N>_theta<THETA_LABEL>_summary.csv  (aggregated rows for replot)
  csv/fmm_3way_stack_N<N>_theta<THETA_LABEL>_runs.csv     (per-run raw rows)

Replot from saved CSV (--plot-only / --no-run on this script, not forwarded to Python):
  $0 --plot-only n=10000000 theta=0.2

Paper figure (horizontal bars, collapsed phases, multi-panel):
  $0 --paper n=10000000 thetas=0.2,0.5
  $0 --paper-controlled n=10000000 thetas=0.2,0.5

With --rebuild, runs build_fmm_dtt_theta_bins.sh per theta first (compile-time caps from
compare_fmm_dtt_theta.sh probes, see fmm_dtt_theta_build_defs.sh) and uses:
  bin/N<N>/theta<THETA_LABEL>/{gtap,worklist,omp}_dtt_fmm

Without --rebuild, uses bin/N<N>/theta<THETA_LABEL>/ if present, then
bin/theta<THETA_LABEL>/, else falls back to bin/.
EOF
}

n_values=(10000000 50000000)
rebuild=0
plot_only=0
paper_mode=0
paper_controlled=0
thetas=(0.2 0.3 0.4 0.5)
custom_out=""
extra_args=()

parse_csv_values() {
  local raw=$1
  local -n out_ref=$2
  local -a tmp_values=()
  IFS=',' read -r -a tmp_values <<< "$raw"
  out_ref=()
  for v in "${tmp_values[@]}"; do
    v="${v//[[:space:]]/}"
    if [[ -n "$v" ]]; then
      out_ref+=("$v")
    fi
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild)
      rebuild=1
      shift
      ;;
    --plot-only|--no-run)
      plot_only=1
      shift
      ;;
    --paper)
      paper_mode=1
      shift
      ;;
    --paper-controlled)
      paper_mode=1
      paper_controlled=1
      shift
      ;;
    --ns|--n-values)
      shift
      parse_csv_values "${1:-}" n_values
      shift
      ;;
    --thetas)
      shift
      parse_csv_values "${1:-}" thetas
      shift
      ;;
    n=*|N=*)
      val="${1#*=}"
      parse_csv_values "$val" n_values
      shift
      ;;
    theta=*)
      val="${1#theta=}"
      parse_csv_values "$val" thetas
      shift
      ;;
    thetas=*)
      val="${1#thetas=}"
      parse_csv_values "$val" thetas
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        n_values=("$1")
      elif [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        thetas=("$1")
      elif [[ "$1" == --* ]]; then
        extra_args+=("$1")
      elif [[ -z "$custom_out" && "$1" == *.png ]]; then
        custom_out=$1
      else
        extra_args+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ "${#n_values[@]}" -eq 0 ]]; then
  echo "Error: empty N list" >&2
  exit 1
fi
if [[ "${#thetas[@]}" -eq 0 ]]; then
  echo "Error: empty theta list" >&2
  exit 1
fi

if [[ "$paper_mode" -eq 1 ]]; then
  if [[ "${#n_values[@]}" -ne 1 ]]; then
    echo "Error: --paper requires exactly one N (e.g. n=10000000)" >&2
    exit 1
  fi
  n="${n_values[0]}"
  if [[ "$paper_controlled" -eq 1 ]]; then
    plot_paper_controlled "$n" "${thetas[@]}"
  else
    plot_paper_figure "$n" "${thetas[@]}"
  fi
  exit 0
fi

if [[ -n "$custom_out" && ! ( "${#n_values[@]}" -eq 1 && "${#thetas[@]}" -eq 1 ) ]]; then
  echo "Error: custom OUT can only be used with a single N and a single theta" >&2
  exit 1
fi

bindir_for_theta() {
  local n=$1 theta=$2
  local theta_lbl
  theta_lbl="$(theta_label "$theta")"
  local staged_n="bin/N${n}/theta${theta_lbl}"
  local staged_legacy="bin/theta${theta_lbl}"
  if [[ -x "$staged_n/gtap_dtt_fmm" ]]; then
    echo "$staged_n"
  elif [[ -x "$staged_legacy/gtap_dtt_fmm" ]]; then
    echo "$staged_legacy"
  else
    echo "bin"
  fi
}

for n in "${n_values[@]}"; do
  for theta in "${thetas[@]}"; do
    theta_lbl="$(theta_label "$theta")"

    if [[ -n "$custom_out" ]]; then
      out="$custom_out"
    else
      out="img/fmm_3way_stack_N${n}_theta${theta_lbl}.png"
    fi

    if [[ "$plot_only" -eq 1 ]]; then
      csv_in="csv/fmm_3way_stack_N${n}_theta${theta_lbl}_summary.csv"
      echo "Replotting N=${n} theta=${theta} from ${csv_in} -> ${out}"
      replot_one "$n" "$theta" "$out" "${extra_args[@]}"
      continue
    fi

    if [[ "$rebuild" -eq 1 ]]; then
      echo "=== Rebuilding theta=${theta} (N=${n}) ==="
      ./build_fmm_dtt_theta_bins.sh "$theta" "$n"
    fi

    bindir="$(bindir_for_theta "$n" "$theta")"
    if [[ ! -x "$bindir/gtap_dtt_fmm" || ! -x "$bindir/worklist_dtt_fmm" || ! -x "$bindir/omp_dtt_fmm" ]]; then
      echo "Error: missing binaries under $bindir for N=${n} theta=${theta}" >&2
      echo "Run: ./build_fmm_dtt_theta_bins.sh ${theta} ${n}" >&2
      echo "  or: $0 ${n} --rebuild ${theta}" >&2
      exit 1
    fi

    echo "Plotting N=${n} theta=${theta} (bins from ${bindir}) -> ${out}"
    run_one "$n" "$theta" "$bindir" "$out" "${extra_args[@]}"
  done
done

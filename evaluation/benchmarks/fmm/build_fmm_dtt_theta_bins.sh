#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# shellcheck source=fmm_dtt_theta_build_defs.sh
source "$script_dir/fmm_dtt_theta_build_defs.sh"

theta_label() {
  python3 -c "t=format(float('${1}'), 'f').rstrip('0'); t=t[:-1] if t.endswith('.') else t; print(t.replace('.', ''))"
}

usage() {
  cat <<EOF
Usage: $0 THETA [N]

Build gtap_dtt_fmm, worklist_dtt_fmm, omp_dtt_fmm with compile-time caps matched to
compare_fmm_dtt_theta.sh probe (max_m2l / max_p2p) for THETA/N, output directly under:
  bin/N<N>/theta<THETA_LABEL>/

Example:
  $0 0.2 50000000
  $0 0.5 50000000
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

theta=$1
n=${2:-50000000}
theta_lbl="$(theta_label "$theta")"
dest="bin/N${n}/theta${theta_lbl}"
dest_abs="$script_dir/$dest"

fmm_dtt_apply_build_defs "$theta" "$n"

echo "Building FMM DTT binaries for theta=${theta} N=${n} -> ${dest}/"
fmm_dtt_print_build_defs | sed 's/^/  /'

# These binaries encode the theta-specific caps as compile-time defines. Force the
# relink so changed environment overrides are not skipped by make's timestamps.
mkdir -p "$dest"
make -B OUT_DIR="$dest_abs" \
  GTAP_GRID_SIZE="${GTAP_GRID_SIZE}" \
  GTAP_MAX_TASKS_PER_WARP="${GTAP_MAX_TASKS_PER_WARP}" \
  FMM3D_DTT_M2L_CAP="${FMM3D_DTT_M2L_CAP}" \
  FMM3D_DTT_P2P_CAP="${FMM3D_DTT_P2P_CAP}" \
  FMM3D_WORKLIST_PAIR_CAP_FACTOR="${FMM3D_WORKLIST_PAIR_CAP_FACTOR}" \
  gtap_dtt_fmm worklist_dtt_fmm omp_dtt_fmm

echo "Built:"
echo "  $dest/gtap_dtt_fmm"
echo "  $dest/worklist_dtt_fmm"
echo "  $dest/omp_dtt_fmm"

#!/usr/bin/env bash
# Compile-time caps for FMM DTT binaries, derived from compare_fmm_dtt_theta.sh probes.
# Reference sweep: N=10M/50M, theta=0.2,0.25,0.3,0.35,0.4,0.5, cap%%=0.0.
#
# Usage (source only):
#   source fmm_dtt_theta_build_defs.sh
#   fmm_dtt_apply_build_defs 0.2 50000000

fmm_dtt_next_pow2() {
  local v=$1
  local p=1
  while [ "$p" -lt "$v" ]; do
    p=$((p * 2))
  done
  echo "$p"
}

# Sets FMM3D_* and GTAP_* make overrides for the given theta and N.
fmm_dtt_apply_build_defs() {
  local theta=$1
  local n=${2:-50000000}

  local max_m2l max_p2p default_gtap_max_tasks_per_warp
  local default_worklist_pair_cap_factor
  if [ "$n" -eq 10000000 ]; then
    case "$theta" in
      0.2)
        max_m2l=1752
        max_p2p=784
        default_gtap_max_tasks_per_warp=50000
        default_worklist_pair_cap_factor=1024
        ;;
      0.3)
        max_m2l=626
        max_p2p=246
        default_gtap_max_tasks_per_warp=20000
        default_worklist_pair_cap_factor=256
        ;;
      0.4)
        max_m2l=274
        max_p2p=131
        default_gtap_max_tasks_per_warp=8000
        default_worklist_pair_cap_factor=128
        ;;
      0.5)
        max_m2l=160
        max_p2p=65
        default_gtap_max_tasks_per_warp=5000
        default_worklist_pair_cap_factor=64
        ;;
      *)
        echo "fmm_dtt_apply_build_defs: unknown theta=$theta for N=$n" >&2
        return 1
        ;;
    esac
  elif [ "$n" -eq 50000000 ]; then
    case "$theta" in
      0.2)
        max_m2l=1519
        max_p2p=485
        default_gtap_max_tasks_per_warp=200000
        default_worklist_pair_cap_factor=1024
        ;;
      0.3)
        max_m2l=483
        max_p2p=171
        default_gtap_max_tasks_per_warp=100000
        default_worklist_pair_cap_factor=256
        ;;
      0.4)
        max_m2l=182
        max_p2p=81
        default_gtap_max_tasks_per_warp=50000
        default_worklist_pair_cap_factor=128
        ;;
      0.5)
        max_m2l=119
        max_p2p=27
        default_gtap_max_tasks_per_warp=20000
        default_worklist_pair_cap_factor=64
        ;;
      *)
        echo "fmm_dtt_apply_build_defs: unknown theta=$theta for N=$n" >&2
        return 1
        ;;
    esac
  else
    echo "fmm_dtt_apply_build_defs: unknown N=$n (supported: 10000000, 50000000)" >&2
    return 1
  fi

  export FMM3D_DTT_M2L_CAP
  export FMM3D_DTT_P2P_CAP
  FMM3D_DTT_M2L_CAP="$(fmm_dtt_next_pow2 "$max_m2l")"
  FMM3D_DTT_P2P_CAP="$(fmm_dtt_next_pow2 "$max_p2p")"

  export GTAP_MAX_TASKS_PER_WARP="${GTAP_MAX_TASKS_PER_WARP:-$default_gtap_max_tasks_per_warp}"
  export FMM3D_WORKLIST_PAIR_CAP_FACTOR="${FMM3D_WORKLIST_PAIR_CAP_FACTOR:-$default_worklist_pair_cap_factor}"
}


fmm_dtt_print_build_defs() {
  echo "  FMM3D_DTT_M2L_CAP=${FMM3D_DTT_M2L_CAP}"
  echo "  FMM3D_DTT_P2P_CAP=${FMM3D_DTT_P2P_CAP}"
  echo "  GTAP_MAX_TASKS_PER_WARP=${GTAP_MAX_TASKS_PER_WARP}"
  echo "  FMM3D_WORKLIST_PAIR_CAP_FACTOR=${FMM3D_WORKLIST_PAIR_CAP_FACTOR}"
}

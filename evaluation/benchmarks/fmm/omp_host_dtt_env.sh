#!/usr/bin/env bash
# OpenMP runtime tuning for Host DTT (omp_dtt_fmm).
# Source this before stack plots / compare scripts. Override in the shell first, e.g.:
#   OMP_NUM_THREADS=48 ./plot_fmm_dtt_stack.sh n=10000000 theta=0.3

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-72}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

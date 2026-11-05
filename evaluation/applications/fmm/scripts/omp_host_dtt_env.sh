#!/usr/bin/env bash
# OpenMP runtime tuning for Host DTT (omp_dtt_fmm).
# Source this before Host DTT compare scripts. Override in the shell first, e.g.:
#   OMP_NUM_THREADS=48 ./scripts/stack/run_fmm_3way_stack_paper.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-72}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

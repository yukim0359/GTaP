#pragma once

#include "fmm_types.cuh"

// ---------- Device globals ----------

__device__ TreeNode *g_tree;
__device__ const Real *g_bx;
__device__ const Real *g_by;
__device__ const Real *g_bz;
__device__ const Real *g_bq;
__device__ Real *g_phi;
__device__ Real *g_ax;
__device__ Real *g_ay;
__device__ Real *g_az;
__device__ int g_n;
__device__ int g_ncells;
__device__ int g_nleaf;
__device__ const int *g_leaf_idx;
__device__ const int *g_body_cell;
__device__ int *g_dtt_m2l_count;
__device__ int *g_dtt_p2p_count;
__device__ int *g_dtt_m2l_src;
__device__ int *g_dtt_p2p_src;
__device__ int *g_dtt_overflow;
__device__ Real g_dtt_theta;

#if FMM3D_PARTITIONED_FUSED_DTT
__device__ const int *g_child_lo;
__device__ const int *g_child_hi;
__device__ int g_dtt_nspawn;
#endif

#if FMM_CONST_IN_CONSTANT
__constant__ Real g_m2l_cnm_re[M2L_CNM_SIZE];
__constant__ Real g_m2l_cnm_im[M2L_CNM_SIZE];
__constant__ Real g_sph_prefactor[EXA_NSPH];
#else
__device__ Real g_m2l_cnm_re[M2L_CNM_SIZE];
__device__ Real g_m2l_cnm_im[M2L_CNM_SIZE];
__device__ Real g_sph_prefactor[EXA_NSPH];
#endif

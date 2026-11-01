#pragma once

#include "../device_symbols.cuh"
#include "../fmm_types.cuh"
#if FMM3D_GTAP_FUSED
#include "../numerics/device_pairs.cuh"
#endif

#include <algorithm>
#include <cuda_runtime.h>
#include <vector>

__device__ __forceinline__ size_t d_dtt_m2l_slot(int ci, int pos) {
  return static_cast<size_t>(ci) * static_cast<size_t>(DTT_M2L_CAP) +
         static_cast<size_t>(pos);
}

__device__ __forceinline__ size_t d_dtt_p2p_slot(int ci, int pos) {
  return static_cast<size_t>(ci) * static_cast<size_t>(DTT_P2P_CAP) +
         static_cast<size_t>(pos);
}

__device__ __forceinline__ bool d_dtt_well_separated(const TreeNode &Ci, const TreeNode &Cj) {
  const Real dx = Ci.cx - Cj.cx;
  const Real dy = Ci.cy - Cj.cy;
  const Real dz = Ci.cz - Cj.cz;
  const Real d2 = dx * dx + dy * dy + dz * dz;
  const Real sr = Ci.r + Cj.r;
  return (g_dtt_theta * g_dtt_theta * d2 > sr * sr * (Real(1) - Real(1e-3)));
}

__device__ __forceinline__ bool d_dtt_split_ci(const TreeNode &Ci, const TreeNode &Cj) {
  if (Cj.is_leaf) return true;
  if (Ci.is_leaf) return false;
  return Ci.r >= Cj.r;
}

static inline bool h_dtt_well_separated(const TreeNode &Ci, const TreeNode &Cj, Real theta) {
  const Real dx = Ci.cx - Cj.cx;
  const Real dy = Ci.cy - Cj.cy;
  const Real dz = Ci.cz - Cj.cz;
  const Real d2 = dx * dx + dy * dy + dz * dz;
  const Real sr = Ci.r + Cj.r;
  return (theta * theta * d2 > sr * sr * (Real(1) - Real(1e-3)));
}

static inline bool h_dtt_split_ci(const TreeNode &Ci, const TreeNode &Cj) {
  if (Cj.is_leaf) return true;
  if (Ci.is_leaf) return false;
  return Ci.r >= Cj.r;
}

__device__ __forceinline__ void d_append_m2l(int ci, int cj) {
#if FMM3D_GTAP_FUSED
  atomicAdd(&g_dtt_m2l_count[ci], 1);
  d_apply_m2l_pair(ci, cj);
#else
  int pos = atomicAdd(&g_dtt_m2l_count[ci], 1);
  if (pos < DTT_M2L_CAP) {
    g_dtt_m2l_src[d_dtt_m2l_slot(ci, pos)] = cj;
  } else if (g_dtt_overflow) {
    atomicExch(g_dtt_overflow, 1);
  }
#endif
}

__device__ __forceinline__ void d_append_p2p(int ci, int cj) {
#if FMM3D_GTAP_FUSED
  atomicAdd(&g_dtt_p2p_count[ci], 1);
  d_apply_p2p_pair(ci, cj);
#else
  int pos = atomicAdd(&g_dtt_p2p_count[ci], 1);
  if (pos < DTT_P2P_CAP) {
    g_dtt_p2p_src[d_dtt_p2p_slot(ci, pos)] = cj;
  } else if (g_dtt_overflow) {
    atomicExch(g_dtt_overflow, 1);
  }
#endif
}

static cudaError_t h_copy_dtt_counts_from_device(
    int *d_m2l_count, int *d_p2p_count, int *d_dtt_overflow,
    int ncells,
    int &out_overflow,
    int &max_m2l_count, int &max_p2p_count,
    long long &m2l_total_ll, long long &p2p_total_ll) {
  cudaError_t err = cudaMemcpy(&out_overflow, d_dtt_overflow, sizeof(int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return err;
  std::vector<int> h_m2l_count(ncells), h_p2p_count(ncells);
  err = cudaMemcpy(h_m2l_count.data(), d_m2l_count, sizeof(int) * ncells, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return err;
  err = cudaMemcpy(h_p2p_count.data(), d_p2p_count, sizeof(int) * ncells, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return err;

  m2l_total_ll = 0;
  p2p_total_ll = 0;
  max_m2l_count = 0;
  max_p2p_count = 0;
  for (int ci = 0; ci < ncells; ++ci) {
    m2l_total_ll += h_m2l_count[ci];
    p2p_total_ll += h_p2p_count[ci];
    max_m2l_count = std::max(max_m2l_count, h_m2l_count[ci]);
    max_p2p_count = std::max(max_p2p_count, h_p2p_count[ci]);
  }
  return cudaSuccess;
}

#pragma once

#include "../device_symbols.cuh"
#include "../fmm_types.cuh"
#include "../numerics/device_sph.cuh"
#include "gtap_thread.cuh"

// ---------- Device dual-tree traversal ----------

__device__ __forceinline__ size_t d_dtt_m2l_slot(int ci, int pos) {
  return static_cast<size_t>(ci) * static_cast<size_t>(DTT_M2L_CAP) +
         static_cast<size_t>(pos);
}

__device__ __forceinline__ size_t d_dtt_p2p_slot(int ci, int pos) {
  return static_cast<size_t>(ci) * static_cast<size_t>(DTT_P2P_CAP) +
         static_cast<size_t>(pos);
}

__device__ __forceinline__ void d_atomic_add_cx(Cx *dst, Cx v) {
  atomicAdd(&dst->re, v.re);
  atomicAdd(&dst->im, v.im);
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

__device__ void d_apply_m2l_pair(int ci, int cj) {
#if FMM3D_GTAP_ATOMIC_FUSED
  atomicAdd(&g_dtt_m2l_count[ci], 1);
#endif
  TreeNode &Ci = g_tree[ci];
  const TreeNode &Cj = g_tree[cj];
  Cx Lacc[EXA_NTERM];
  for (int s = 0; s < EXA_NTERM; ++s) Lacc[s] = cx_make(Real(0));
  Cx Ynm2[EXA_NSPH];
  Real dx = Ci.cx - Cj.cx;
  Real dy = Ci.cy - Cj.cy;
  Real dz = Ci.cz - Cj.cz;
  Real rho, alpha, beta;
  d_cart2sph(dx, dy, dz, rho, alpha, beta);
  d_eval_local(rho, alpha, beta, Ynm2);
  for (int j = 0; j < EXA_P; ++j) {
    for (int k = 0; k <= j; ++k) {
      const int jks = tri_idx(j, k);
      Cx L = cx_make(Real(0));
      for (int n = 0; n < EXA_P; ++n) {
        for (int m = -n; m < 0; ++m) {
          const int nms = tri_idx(n, -m);
          const int jnkm = sph_idx(j + n, m - k);
          const int jk = j * j + j + k;
          const int nm = n * n + n + m;
          const Cx c = d_m2l_cnm(jk * EXA_P * EXA_P + nm);
          L = cx_add(L, cx_mul(cx_mul(cx_conj(Cj.M[nms]), c), Ynm2[jnkm]));
        }
        for (int m = 0; m <= n; ++m) {
          const int nms = tri_idx(n, m);
          const int jnkm = sph_idx(j + n, m - k);
          const int jk = j * j + j + k;
          const int nm = n * n + n + m;
          const Cx c = d_m2l_cnm(jk * EXA_P * EXA_P + nm);
          L = cx_add(L, cx_mul(cx_mul(Cj.M[nms], c), Ynm2[jnkm]));
        }
      }
      Lacc[jks] = cx_add(Lacc[jks], L);
    }
  }
#if FMM3D_GTAP_ATOMIC_FUSED
  for (int s = 0; s < EXA_NTERM; ++s) d_atomic_add_cx(&Ci.L[s], Lacc[s]);
#else
  for (int s = 0; s < EXA_NTERM; ++s) Ci.L[s] = cx_add(Ci.L[s], Lacc[s]);
#endif
}

__device__ void d_apply_p2p_pair(int ci, int cj) {
#if FMM3D_GTAP_ATOMIC_FUSED
  atomicAdd(&g_dtt_p2p_count[ci], 1);
#endif
  const TreeNode &Ci = g_tree[ci];
  const TreeNode &Cj = g_tree[cj];
  for (int ii = Ci.ibody; ii < Ci.ibody + Ci.nbody; ++ii) {
    const Real xi = g_bx[ii];
    const Real yi = g_by[ii];
    const Real zi = g_bz[ii];
    Real phi = Real(0);
    Real ax = Real(0);
    Real ay = Real(0);
    Real az = Real(0);
    for (int jj = Cj.ibody; jj < Cj.ibody + Cj.nbody; ++jj) {
      if (ii == jj) continue;
      Real dx = xi - g_bx[jj];
      Real dy = yi - g_by[jj];
      Real dz = zi - g_bz[jj];
      Real r2 = dx * dx + dy * dy + dz * dz + H_EPS2;
      Real inv_r = real_rsqrt(r2);
      Real q_over_r = g_bq[jj] * inv_r;
      phi += q_over_r;
      Real inv_r3_q = q_over_r / r2;
      ax -= dx * inv_r3_q;
      ay -= dy * inv_r3_q;
      az -= dz * inv_r3_q;
    }
#if FMM3D_GTAP_ATOMIC_FUSED
    atomicAdd(&g_phi[ii], phi);
    atomicAdd(&g_ax[ii], ax);
    atomicAdd(&g_ay[ii], ay);
    atomicAdd(&g_az[ii], az);
#else
    g_phi[ii] += phi;
    g_ax[ii] += ax;
    g_ay[ii] += ay;
    g_az[ii] += az;
#endif
  }
}

__device__ __forceinline__ void d_append_m2l(int ci, int cj) {
#if FMM3D_GTAP_FUSED
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

#if FMM3D_PARTITIONED_FUSED_DTT
// exafmm-beta splitCell small paths: sequential dualTreeTraversal on one side's children.
__device__ void d_traverse_pair_seq(int ci, int cj) {
  TreeNode Ci = g_tree[ci];
  if (ci == cj) {
    if (Ci.is_leaf) {
      d_apply_p2p_pair(ci, cj);
    } else {
      for (int oi = 0; oi < 8; ++oi) {
        int child_i = d_child(Ci, oi);
        if (child_i < 0) continue;
        for (int oj = 0; oj < 8; ++oj) {
          int child_j = d_child(Ci, oj);
          if (child_j >= 0) d_traverse_pair_seq(child_i, child_j);
        }
      }
    }
    return;
  }

  if (d_dtt_well_separated(Ci, g_tree[cj])) {
    d_apply_m2l_pair(ci, cj);
    return;
  }
  if (Ci.is_leaf && g_tree[cj].is_leaf) {
    d_apply_p2p_pair(ci, cj);
    return;
  }

  TreeNode Cj = g_tree[cj];
  if (Cj.is_leaf) {
    for (int oct = 0; oct < 8; ++oct) {
      int child_i = d_child(Ci, oct);
      if (child_i >= 0) d_traverse_pair_seq(child_i, cj);
    }
    return;
  }
  if (Ci.is_leaf) {
    for (int oct = 0; oct < 8; ++oct) {
      int child_j = d_child(Cj, oct);
      if (child_j >= 0) d_traverse_pair_seq(ci, child_j);
    }
    return;
  }

  if (Ci.r >= Cj.r) {
    for (int oct = 0; oct < 8; ++oct) {
      int child_i = d_child(Ci, oct);
      if (child_i >= 0) d_traverse_pair_seq(child_i, cj);
    }
  } else {
    for (int oct = 0; oct < 8; ++oct) {
      int child_j = d_child(Cj, oct);
      if (child_j >= 0) d_traverse_pair_seq(ci, child_j);
    }
  }
}

// mode == 0: dualTreeTraversal(ci,cj); mode == 1: TraverseRange(ci_lo..hi, cj_lo..hi)
#pragma gtap function
__device__ void d_exafmm_dtt(int ci, int cj, int ci_lo, int ci_hi, int cj_lo, int cj_hi,
                             int mode) {
  if (mode == 0) {
    TreeNode Ci = g_tree[ci];
    if (ci == cj) {
      if (Ci.is_leaf) {
        d_apply_p2p_pair(ci, cj);
        return;
      }
      if (Ci.nbody >= g_dtt_nspawn) {
        int ci_sub_lo = g_child_lo[ci];
        int ci_sub_hi = g_child_hi[ci];
        #pragma gtap task
        d_exafmm_dtt(0, 0, ci_sub_lo, ci_sub_hi, ci_sub_lo, ci_sub_hi, 1);
        #pragma gtap taskwait
        return;
      }
      for (int oi = 0; oi < 8; ++oi) {
        int child_i = d_child(Ci, oi);
        if (child_i < 0) continue;
        for (int oj = 0; oj < 8; ++oj) {
          int child_j = d_child(Ci, oj);
          if (child_j >= 0) d_traverse_pair_seq(child_i, child_j);
        }
      }
      return;
    }

    if (d_dtt_well_separated(Ci, g_tree[cj])) {
      d_apply_m2l_pair(ci, cj);
      return;
    }
    if (Ci.is_leaf && g_tree[cj].is_leaf) {
      d_apply_p2p_pair(ci, cj);
      return;
    }

    TreeNode Cj = g_tree[cj];
    if (Cj.is_leaf) {
      for (int oct = 0; oct < 8; ++oct) {
        int child_i = d_child(Ci, oct);
        if (child_i >= 0) d_traverse_pair_seq(child_i, cj);
      }
      return;
    }
    if (Ci.is_leaf) {
      for (int oct = 0; oct < 8; ++oct) {
        int child_j = d_child(Cj, oct);
        if (child_j >= 0) d_traverse_pair_seq(ci, child_j);
      }
      return;
    }

    if (Ci.nbody + Cj.nbody >= g_dtt_nspawn) {
      int ci_sub_lo = g_child_lo[ci];
      int ci_sub_hi = g_child_hi[ci];
      int cj_sub_lo = g_child_lo[cj];
      int cj_sub_hi = g_child_hi[cj];
      #pragma gtap task
      d_exafmm_dtt(0, 0, ci_sub_lo, ci_sub_hi, cj_sub_lo, cj_sub_hi, 1);
      #pragma gtap taskwait
      return;
    }

    if (Ci.r >= Cj.r) {
      for (int oct = 0; oct < 8; ++oct) {
        int child_i = d_child(Ci, oct);
        if (child_i >= 0) d_traverse_pair_seq(child_i, cj);
      }
    } else {
      for (int oct = 0; oct < 8; ++oct) {
        int child_j = d_child(Cj, oct);
        if (child_j >= 0) d_traverse_pair_seq(ci, child_j);
      }
    }
    return;
  }

  if (ci_hi <= ci_lo || cj_hi <= cj_lo) return;

  if (ci_hi - ci_lo == 1 || cj_hi - cj_lo == 1) {
    if (ci_lo == cj_lo && ci_hi == cj_hi) {
      d_traverse_pair_seq(ci_lo, cj_lo);
    } else {
      for (int i = ci_lo; i < ci_hi; ++i) {
        for (int j = cj_lo; j < cj_hi; ++j) {
          d_traverse_pair_seq(i, j);
        }
      }
    }
    return;
  }

  int ci_mid = ci_lo + (ci_hi - ci_lo) / 2;
  int cj_mid = cj_lo + (cj_hi - cj_lo) / 2;
  {
    #pragma gtap task
    d_exafmm_dtt(0, 0, ci_lo, ci_mid, cj_lo, cj_mid, 1);
    #pragma gtap task
    d_exafmm_dtt(0, 0, ci_mid, ci_hi, cj_mid, cj_hi, 1);
    #pragma gtap taskwait
  }
  {
    #pragma gtap task
    d_exafmm_dtt(0, 0, ci_lo, ci_mid, cj_mid, cj_hi, 1);
    #pragma gtap task
    d_exafmm_dtt(0, 0, ci_mid, ci_hi, cj_lo, cj_mid, 1);
    #pragma gtap taskwait
  }
}
#endif

__device__ void d_dtt_fill_seq(int ci, int cj) {
  const TreeNode &Ci = g_tree[ci];
  if (ci == cj) {
    if (Ci.is_leaf) {
      d_append_p2p(ci, cj);
    } else {
      for (int oi = 0; oi < 8; ++oi) {
        const int child_i = d_child(Ci, oi);
        if (child_i == -1) continue;
        for (int oj = 0; oj < 8; ++oj) {
          const int child_j = d_child(Ci, oj);
          if (child_j != -1) d_dtt_fill_seq(child_i, child_j);
        }
      }
    }
    return;
  }

  const TreeNode &Cj = g_tree[cj];
  if (d_dtt_well_separated(Ci, Cj)) {
    d_append_m2l(ci, cj);
    return;
  }
  if (Ci.is_leaf && Cj.is_leaf) {
    d_append_p2p(ci, cj);
    return;
  }

  {
    const bool split_ci = d_dtt_split_ci(Ci, Cj);
    for (int oct = 0; oct < 8; ++oct) {
      const int child_i = split_ci ? d_child(Ci, oct) : ci;
      const int child_j = split_ci ? cj : d_child(Cj, oct);
      if (split_ci ? (child_i == -1) : (child_j == -1)) continue;
      d_dtt_fill_seq(child_i, child_j);
    }
  }
}

#pragma gtap function
__device__ void d_dtt_fill(int ci, int cj, int depth) {
  const TreeNode &Ci = g_tree[ci];
  if (ci == cj) {
    if (Ci.is_leaf) {
      d_append_p2p(ci, cj);
    } else {
      for (int oi = 0; oi < 8; ++oi) {
        const int child_i = d_child(Ci, oi);
        if (child_i == -1) continue;
        const bool spawn = (depth < DTT_TASK_DEPTH) &&
                           (g_tree[child_i].nbody >= DTT_TASK_MIN_N);
        for (int oj = 0; oj < 8; ++oj) {
          const int child_j = d_child(Ci, oj);
          if (child_j == -1) continue;
          if (spawn) {
            #pragma gtap task
            d_dtt_fill(child_i, child_j, depth + 1);
          } else {
            d_dtt_fill_seq(child_i, child_j);
          }
        }
      }
      // #pragma gtap taskwait
    }
    return;
  }

  const TreeNode &Cj = g_tree[cj];
  if (d_dtt_well_separated(Ci, Cj)) {
    d_append_m2l(ci, cj);
    return;
  }
  if (Ci.is_leaf && Cj.is_leaf) {
    d_append_p2p(ci, cj);
    return;
  }

  {
    const bool split_ci = d_dtt_split_ci(Ci, Cj);
    for (int oct = 0; oct < 8; ++oct) {
      const int child_i = split_ci ? d_child(Ci, oct) : ci;
      const int child_j = split_ci ? cj : d_child(Cj, oct);
      if (split_ci ? (child_i == -1) : (child_j == -1)) continue;
      const int nb_lhs = split_ci ? g_tree[child_i].nbody : Ci.nbody;
      const int nb_rhs = split_ci ? Cj.nbody : g_tree[child_j].nbody;
      if (depth < DTT_TASK_DEPTH && nb_lhs + nb_rhs >= DTT_TASK_MIN_N) {
        #pragma gtap task
        d_dtt_fill(child_i, child_j, depth + 1);
      } else {
        d_dtt_fill_seq(child_i, child_j);
      }
    }
  }
  // #pragma gtap taskwait
}

__global__ void fmm3d_dtt_fill_kernel() {
  #pragma gtap entry
#if FMM3D_PARTITIONED_FUSED_DTT
  d_exafmm_dtt(0, 0, 0, 0, 0, 0, 0);
#else
  d_dtt_fill(0, 0, 0);
#endif
}

__device__ __forceinline__ void d_worklist_push(DttPair *out, int *out_count,
                                                int pair_cap, int *overflow,
                                                int ci, int cj) {
  int pos = atomicAdd(out_count, 1);
  if (pos < pair_cap) {
    out[pos] = DttPair{ci, cj};
  } else if (overflow) {
    atomicExch(overflow, 1);
  }
}

__device__ __forceinline__ void d_worklist_expand_one(int ci, int cj, DttPair *out,
                                                      int *out_count, int pair_cap,
                                                      int *overflow) {
  const TreeNode &Ci = g_tree[ci];
  if (ci == cj) {
    if (Ci.is_leaf) {
      d_append_p2p(ci, cj);
    } else {
      for (int oi = 0; oi < 8; ++oi) {
        const int child_i = d_child(Ci, oi);
        if (child_i == -1) continue;
        for (int oj = 0; oj < 8; ++oj) {
          const int child_j = d_child(Ci, oj);
          if (child_j != -1) {
            d_worklist_push(out, out_count, pair_cap, overflow, child_i, child_j);
          }
        }
      }
    }
    return;
  }

  const TreeNode &Cj = g_tree[cj];
  if (d_dtt_well_separated(Ci, Cj)) {
    d_append_m2l(ci, cj);
    return;
  }
  if (Ci.is_leaf && Cj.is_leaf) {
    d_append_p2p(ci, cj);
    return;
  }

  if (d_dtt_split_ci(Ci, Cj)) {
    for (int oct = 0; oct < 8; ++oct) {
      const int child_i = d_child(Ci, oct);
      if (child_i != -1) {
        d_worklist_push(out, out_count, pair_cap, overflow, child_i, cj);
      }
    }
  } else {
    for (int oct = 0; oct < 8; ++oct) {
      const int child_j = d_child(Cj, oct);
      if (child_j != -1) {
        d_worklist_push(out, out_count, pair_cap, overflow, ci, child_j);
      }
    }
  }
}

__global__ void fmm3d_worklist_expand_kernel(const DttPair *in, int in_count,
                                             DttPair *out, int *out_count,
                                             int pair_cap, int *overflow) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int idx = tid; idx < in_count; idx += stride) {
    d_worklist_expand_one(in[idx].ci, in[idx].cj, out, out_count, pair_cap, overflow);
  }
}

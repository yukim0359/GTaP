#pragma once

#include "device_symbols.cuh"
#include "host_sph.hpp"

#include <cmath>
#include <vector>

static cudaError_t h_upload_fmm_constant_tables() {
  const int P = EXA_P;
  std::vector<Real> pref(static_cast<size_t>(EXA_NSPH), Real(0));
  for (int n = 0; n < 2 * P; ++n) {
    for (int m = -n; m <= n; ++m) {
      pref[static_cast<size_t>(sph_idx(n, m))] = h_prefactor(n, m);
    }
  }
  cudaError_t err = cudaMemcpyToSymbol(g_sph_prefactor, pref.data(),
                                       sizeof(Real) * static_cast<size_t>(EXA_NSPH));
  if (err != cudaSuccess) return err;

  std::vector<Real> re(static_cast<size_t>(M2L_CNM_SIZE), Real(0));
  std::vector<Real> im(static_cast<size_t>(M2L_CNM_SIZE), Real(0));
  for (int j = 0, jk = 0, jknm = 0; j < P; ++j) {
    for (int k = -j; k <= j; ++k, ++jk) {
      for (int n = 0, nm = 0; n < P; ++n) {
        for (int m = -n; m <= n; ++m, ++nm, ++jknm) {
          const int ipow_e = std::abs(k - m) - std::abs(k) - std::abs(m);
          const Cx ipow = h_ipow(ipow_e);
          const Real scale = static_cast<Real>(odd_or_even(j)) *
                              h_anm(n, m) * h_anm(j, k) / h_anm(j + n, m - k) * EXA_EPS;
          const Cx val = cx_scale(ipow, scale);
          re[static_cast<size_t>(jknm)] = val.re;
          im[static_cast<size_t>(jknm)] = val.im;
        }
      }
    }
  }
  err = cudaMemcpyToSymbol(g_m2l_cnm_re, re.data(),
                           sizeof(Real) * static_cast<size_t>(M2L_CNM_SIZE));
  if (err != cudaSuccess) return err;
  return cudaMemcpyToSymbol(g_m2l_cnm_im, im.data(),
                            sizeof(Real) * static_cast<size_t>(M2L_CNM_SIZE));
}

// ---------- Device symbol binders ----------

static cudaError_t bind_ptrs(TreeNode *tree,
                             const Real *bx, const Real *by, const Real *bz,
                             const Real *bq, Real *phi,
                             Real *ax, Real *ay, Real *az,
                             const int *leaf_idx,
                             const int *body_cell) {
  cudaError_t s;
  s = cudaMemcpyToSymbol(g_tree, &tree, sizeof(tree)); if (s) return s;
  s = cudaMemcpyToSymbol(g_bx, &bx, sizeof(bx)); if (s) return s;
  s = cudaMemcpyToSymbol(g_by, &by, sizeof(by)); if (s) return s;
  s = cudaMemcpyToSymbol(g_bz, &bz, sizeof(bz)); if (s) return s;
  s = cudaMemcpyToSymbol(g_bq, &bq, sizeof(bq)); if (s) return s;
  s = cudaMemcpyToSymbol(g_phi, &phi, sizeof(phi)); if (s) return s;
  s = cudaMemcpyToSymbol(g_ax, &ax, sizeof(ax)); if (s) return s;
  s = cudaMemcpyToSymbol(g_ay, &ay, sizeof(ay)); if (s) return s;
  s = cudaMemcpyToSymbol(g_az, &az, sizeof(az)); if (s) return s;
  s = cudaMemcpyToSymbol(g_leaf_idx, &leaf_idx, sizeof(leaf_idx)); if (s) return s;
  s = cudaMemcpyToSymbol(g_body_cell, &body_cell, sizeof(body_cell)); if (s) return s;
  return cudaSuccess;
}

static cudaError_t bind_params(int n, int ncells, int nleaf) {
  cudaError_t s;
  s = cudaMemcpyToSymbol(g_n, &n, sizeof(n)); if (s) return s;
  s = cudaMemcpyToSymbol(g_ncells, &ncells, sizeof(ncells)); if (s) return s;
  s = cudaMemcpyToSymbol(g_nleaf, &nleaf, sizeof(nleaf)); if (s) return s;
  return cudaSuccess;
}

static cudaError_t bind_dtt_ptrs(int *m2l_count, int *p2p_count,
                                 int *m2l_src, int *p2p_src,
                                 int *overflow, Real theta) {
  cudaError_t s;
  s = cudaMemcpyToSymbol(g_dtt_m2l_count, &m2l_count, sizeof(m2l_count)); if (s) return s;
  s = cudaMemcpyToSymbol(g_dtt_p2p_count, &p2p_count, sizeof(p2p_count)); if (s) return s;
  s = cudaMemcpyToSymbol(g_dtt_m2l_src, &m2l_src, sizeof(m2l_src)); if (s) return s;
  s = cudaMemcpyToSymbol(g_dtt_p2p_src, &p2p_src, sizeof(p2p_src)); if (s) return s;
  s = cudaMemcpyToSymbol(g_dtt_overflow, &overflow, sizeof(overflow)); if (s) return s;
  s = cudaMemcpyToSymbol(g_dtt_theta, &theta, sizeof(theta)); if (s) return s;
  return cudaSuccess;
}

#if FMM3D_PARTITIONED_FUSED_DTT
static cudaError_t bind_partitioned_dtt_ptrs(const int *child_lo,
                                             const int *child_hi,
                                             int nspawn) {
  cudaError_t s;
  s = cudaMemcpyToSymbol(g_child_lo, &child_lo, sizeof(child_lo)); if (s) return s;
  s = cudaMemcpyToSymbol(g_child_hi, &child_hi, sizeof(child_hi)); if (s) return s;
  s = cudaMemcpyToSymbol(g_dtt_nspawn, &nspawn, sizeof(nspawn)); if (s) return s;
  return cudaSuccess;
}
#endif

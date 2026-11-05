#pragma once

#include "../device_symbols.cuh"
#include "../fmm_types.cuh"
#include "device_sph.cuh"

__device__ __forceinline__ void d_atomic_add_cx(Cx *dst, Cx v) {
  atomicAdd(&dst->re, v.re);
  atomicAdd(&dst->im, v.im);
}

__device__ void d_apply_m2l_pair(int ci, int cj) {
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
  for (int s = 0; s < EXA_NTERM; ++s) d_atomic_add_cx(&Ci.L[s], Lacc[s]);
}

__device__ void d_apply_p2p_pair(int ci, int cj) {
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
    atomicAdd(&g_phi[ii], phi);
    atomicAdd(&g_ax[ii], ax);
    atomicAdd(&g_ay[ii], ay);
    atomicAdd(&g_az[ii], az);
  }
}

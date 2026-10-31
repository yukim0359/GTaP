#pragma once

#include "../fmm_types.cuh"
#include "../host_sph.hpp"

#include <algorithm>
#include <vector>

// ---------- Host upward pass ----------

[[maybe_unused]] static void h_upward_pass(std::vector<TreeNode> &tree, const HostBodies &bodies) {
  for (int i = static_cast<int>(tree.size()) - 1; i >= 0; --i) {
    TreeNode &c = tree[i];
    for (int k = 0; k < EXA_NTERM; ++k) {
      c.M[k] = cx_make(Real(0));
      c.L[k] = cx_make(Real(0));
    }
    if (c.is_leaf) {
      Cx Ynm[EXA_P * EXA_P];
      for (int b = c.ibody; b < c.ibody + c.nbody; ++b) {
        const Real dx = bodies.x[b] - c.cx;
        const Real dy = bodies.y[b] - c.cy;
        const Real dz = bodies.z[b] - c.cz;
        Real rho, alpha, beta;
        h_cart2sph(dx, dy, dz, rho, alpha, beta);
        h_eval_multipole(rho, alpha, -beta, Ynm);
        for (int n = 0; n < EXA_P; ++n) {
          for (int m = 0; m <= n; ++m) {
            const int nm = sph_idx(n, m);
            const int nms = tri_idx(n, m);
            c.M[nms] = cx_add(c.M[nms], cx_scale(Ynm[nm], bodies.q[b]));
          }
        }
      }
    } else {
      Cx Ynm[EXA_P * EXA_P];
      for (int oct = 0; oct < 8; ++oct) {
        int ch = c.child[oct];
        if (ch == -1) continue;
        const TreeNode &cc = tree[ch];
        const Real dx = c.cx - cc.cx;
        const Real dy = c.cy - cc.cy;
        const Real dz = c.cz - cc.cz;
        Real rho, alpha, beta;
        h_cart2sph(dx, dy, dz, rho, alpha, beta);
        h_eval_multipole(rho, alpha, -beta, Ynm);
        for (int j = 0; j < EXA_P; ++j) {
          for (int k = 0; k <= j; ++k) {
            const int jks = tri_idx(j, k);
            Cx M = cx_make(Real(0));
            for (int n = 0; n <= j; ++n) {
              for (int m = -n; m <= std::min(k - 1, n); ++m) {
                if (j - n >= k - m) {
                  const int jnkms = tri_idx(j - n, k - m);
                  const int nm = sph_idx(n, m);
                  Cx term = cx_mul(cc.M[jnkms], h_ipow(m - std::abs(m)));
                  term = cx_mul(term, Ynm[nm]);
                  term = cx_scale(term, static_cast<Real>(odd_or_even(n)) *
                                        h_anm(n, m) * h_anm(j - n, k - m) /
                                        h_anm(j, k));
                  M = cx_add(M, term);
                }
              }
              for (int m = k; m <= n; ++m) {
                if (j - n >= m - k) {
                  const int jnkms = tri_idx(j - n, -k + m);
                  const int nm = sph_idx(n, m);
                  Cx term = cx_mul(cx_conj(cc.M[jnkms]), Ynm[nm]);
                  term = cx_scale(term, static_cast<Real>(odd_or_even(k + n + m)) *
                                        h_anm(n, m) * h_anm(j - n, k - m) /
                                        h_anm(j, k));
                  M = cx_add(M, term);
                }
              }
            }
            c.M[jks] = cx_add(c.M[jks], cx_scale(M, EXA_EPS));
          }
        }
      }
    }
  }
}

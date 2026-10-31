#pragma once

#include "../fmm_types.cuh"
#include "../host_sph.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

// ---------- Host L2L fallback ----------

[[maybe_unused]] static void h_downward_pass_devtree(std::vector<TreeNode> &dtree) {
  for (int ci = 1; ci < static_cast<int>(dtree.size()); ++ci) {
    int parent = dtree[ci].parent;
    if (parent < 0) continue;
    TreeNode &child = dtree[ci];
    const TreeNode &p = dtree[parent];
    const Real dx = child.cx - p.cx;
    const Real dy = child.cy - p.cy;
    const Real dz = child.cz - p.cz;
    Real rho, alpha, beta;
    Cx Ynm[EXA_P * EXA_P];
    h_cart2sph(dx, dy, dz, rho, alpha, beta);
    h_eval_multipole(rho, alpha, beta, Ynm);
    for (int j = 0; j < EXA_P; ++j) {
      for (int k = 0; k <= j; ++k) {
        const int jks = tri_idx(j, k);
        Cx L = cx_make(Real(0));
        for (int n = j; n < EXA_P; ++n) {
          for (int m = j + k - n; m < 0; ++m) {
            const int jnkm = sph_idx(n - j, m - k);
            const int nms = tri_idx(n, -m);
            Cx term = cx_mul(cx_conj(p.L[nms]), Ynm[jnkm]);
            term = cx_scale(term, static_cast<Real>(odd_or_even(k)) *
                                  h_anm(n - j, m - k) * h_anm(j, k) /
                                  h_anm(n, -m));
            L = cx_add(L, term);
          }
          for (int m = 0; m <= n; ++m) {
            if (n - j >= std::abs(m - k)) {
              const int jnkm = sph_idx(n - j, m - k);
              const int nms = tri_idx(n, m);
              Cx term = cx_mul(p.L[nms], h_ipow(m - k - std::abs(m - k)));
              term = cx_mul(term, Ynm[jnkm]);
              term = cx_scale(term, h_anm(n - j, m - k) * h_anm(j, k) /
                                    h_anm(n, m));
              L = cx_add(L, term);
            }
          }
        }
        child.L[jks] = cx_add(child.L[jks], cx_scale(L, EXA_EPS));
      }
    }
  }
}

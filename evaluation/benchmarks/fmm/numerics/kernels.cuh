#pragma once

#include "../device_symbols.cuh"
#include "../fmm_types.cuh"
#include "device_sph.cuh"

// ---------- Device FMM kernels ----------

__global__ void fmm3d_m2l_kernel(const int *target_cells, int ntarget) {
  const int bi = blockIdx.x;
  if (bi >= ntarget) return;
  const int ci = target_cells ? target_cells[bi] : bi;
  if (ci >= g_ncells) return;
  TreeNode &Ci = g_tree[ci];
  const size_t base =
      static_cast<size_t>(ci) * static_cast<size_t>(DTT_M2L_CAP);
  int nsrc = g_dtt_m2l_count ? g_dtt_m2l_count[ci] : 0;
  if (nsrc > DTT_M2L_CAP) nsrc = DTT_M2L_CAP;

  Cx Lacc[EXA_NTERM];
  for (int s = 0; s < EXA_NTERM; ++s) Lacc[s] = cx_make(Real(0));
  Cx Ynm2[EXA_NSPH];
  for (int p = threadIdx.x; p < nsrc; p += blockDim.x) {
    const TreeNode &Cj = g_tree[g_dtt_m2l_src[base + static_cast<size_t>(p)]];
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
  }

  extern __shared__ Real smem[];
  Real *sre = smem;
  Real *sim = smem + EXA_NTERM * blockDim.x;
  for (int s = 0; s < EXA_NTERM; ++s) {
    sre[s * blockDim.x + threadIdx.x] = Lacc[s].re;
    sim[s * blockDim.x + threadIdx.x] = Lacc[s].im;
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (static_cast<int>(threadIdx.x) < offset) {
      for (int s = 0; s < EXA_NTERM; ++s) {
        sre[s * blockDim.x + threadIdx.x] += sre[s * blockDim.x + threadIdx.x + offset];
        sim[s * blockDim.x + threadIdx.x] += sim[s * blockDim.x + threadIdx.x + offset];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (int s = 0; s < EXA_NTERM; ++s) {
      Ci.L[s] = cx_add(Ci.L[s], cx_make(sre[s * blockDim.x], sim[s * blockDim.x]));
    }
  }
}

__global__ void fmm3d_p2m_leaf_kernel(const int *leaf_cells, int nleaf) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int li = tid; li < nleaf; li += stride) {
    TreeNode &c = g_tree[leaf_cells[li]];
    for (int s = 0; s < EXA_NTERM; ++s) c.M[s] = cx_make(Real(0));
    Cx Ynm[EXA_P * EXA_P];
    for (int b = c.ibody; b < c.ibody + c.nbody; ++b) {
      const Real dx = g_bx[b] - c.cx;
      const Real dy = g_by[b] - c.cy;
      const Real dz = g_bz[b] - c.cz;
      Real rho, alpha, beta;
      d_cart2sph(dx, dy, dz, rho, alpha, beta);
      d_eval_multipole(rho, alpha, -beta, Ynm);
      for (int n = 0; n < EXA_P; ++n) {
        for (int m = 0; m <= n; ++m) {
          const int nm = sph_idx(n, m);
          const int nms = tri_idx(n, m);
          c.M[nms] = cx_add(c.M[nms], cx_scale(Ynm[nm], g_bq[b]));
        }
      }
    }
  }
}

__global__ void fmm3d_m2m_level_kernel(const int *cells, int count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int idx = tid; idx < count; idx += stride) {
    TreeNode &c = g_tree[cells[idx]];
    if (c.is_leaf) continue;
    for (int s = 0; s < EXA_NTERM; ++s) c.M[s] = cx_make(Real(0));
    Cx Ynm[EXA_P * EXA_P];
    for (int oct = 0; oct < 8; ++oct) {
      const int ch = d_child(c, oct);
      if (ch == -1) continue;
      const TreeNode &cc = g_tree[ch];
      const Real dx = c.cx - cc.cx;
      const Real dy = c.cy - cc.cy;
      const Real dz = c.cz - cc.cz;
      Real rho, alpha, beta;
      d_cart2sph(dx, dy, dz, rho, alpha, beta);
      d_eval_multipole(rho, alpha, -beta, Ynm);
      for (int j = 0; j < EXA_P; ++j) {
        for (int k = 0; k <= j; ++k) {
          const int jks = tri_idx(j, k);
          Cx M = cx_make(Real(0));
          for (int n = 0; n <= j; ++n) {
            const int mend = ((k - 1) < n) ? (k - 1) : n;
            for (int m = -n; m <= mend; ++m) {
              if (j - n >= k - m) {
                const int jnkms = tri_idx(j - n, k - m);
                const int nm = sph_idx(n, m);
                Cx term = cx_mul(cc.M[jnkms], d_ipow(m - d_abs_i(m)));
                term = cx_mul(term, Ynm[nm]);
                term = cx_scale(term, static_cast<Real>(odd_or_even(n)) *
                                      d_anm(n, m) * d_anm(j - n, k - m) /
                                      d_anm(j, k));
                M = cx_add(M, term);
              }
            }
            for (int m = k; m <= n; ++m) {
              if (j - n >= m - k) {
                const int jnkms = tri_idx(j - n, -k + m);
                const int nm = sph_idx(n, m);
                Cx term = cx_mul(cx_conj(cc.M[jnkms]), Ynm[nm]);
                term = cx_scale(term, static_cast<Real>(odd_or_even(k + n + m)) *
                                      d_anm(n, m) * d_anm(j - n, k - m) /
                                      d_anm(j, k));
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

__global__ void fmm3d_l2l_level_kernel(const int *cells, int count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int idx = tid; idx < count; idx += stride) {
    const int ci = cells[idx];
    TreeNode &child = g_tree[ci];
    const int parent = child.parent;
    if (parent < 0) continue;
    const TreeNode &p = g_tree[parent];
    const Real dx = child.cx - p.cx;
    const Real dy = child.cy - p.cy;
    const Real dz = child.cz - p.cz;
    Real rho, alpha, beta;
    Cx Ynm[EXA_P * EXA_P];
    d_cart2sph(dx, dy, dz, rho, alpha, beta);
    d_eval_multipole(rho, alpha, beta, Ynm);
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
                                  d_anm(n - j, m - k) * d_anm(j, k) /
                                  d_anm(n, -m));
            L = cx_add(L, term);
          }
          for (int m = 0; m <= n; ++m) {
            if (n - j >= d_abs_i(m - k)) {
              const int jnkm = sph_idx(n - j, m - k);
              const int nms = tri_idx(n, m);
              Cx term = cx_mul(p.L[nms], d_ipow(m - k - d_abs_i(m - k)));
              term = cx_mul(term, Ynm[jnkm]);
              term = cx_scale(term, d_anm(n - j, m - k) * d_anm(j, k) /
                                    d_anm(n, m));
              L = cx_add(L, term);
            }
          }
        }
        child.L[jks] = cx_add(child.L[jks], cx_scale(L, EXA_EPS));
      }
    }
  }
}

__device__ void d_sph2cart(Real r, Real theta, Real phi,
                           Real sr, Real st, Real sp,
                           Real &cx, Real &cy, Real &cz) {
  Real sin_theta = real_sin(theta);
  if (real_abs(sin_theta) < Real(1.0e-7)) {
    sin_theta = real_copysign(Real(1.0e-7), sin_theta == Real(0) ? Real(1) : sin_theta);
  }
  cx = real_sin(theta) * real_cos(phi) * sr +
       real_cos(theta) * real_cos(phi) / r * st -
       real_sin(phi) / r / sin_theta * sp;
  cy = real_sin(theta) * real_sin(phi) * sr +
       real_cos(theta) * real_sin(phi) / r * st +
       real_cos(phi) / r / sin_theta * sp;
  cz = real_cos(theta) * sr - real_sin(theta) / r * st;
}

__device__ void d_eval_local_trg(const TreeNode &Ci, Real x, Real y, Real z,
                                 Real &phi, Real &ax, Real &ay, Real &az) {
  Cx Ynm[EXA_P * EXA_P], YnmTheta[EXA_P * EXA_P];
  Real rho, alpha, beta;
  d_cart2sph(x - Ci.cx + EXA_EPS, y - Ci.cy + EXA_EPS,
             z - Ci.cz + EXA_EPS, rho, alpha, beta);
  d_eval_multipole_theta(rho, alpha, beta, Ynm, YnmTheta);
  phi = Real(0);
  Real sr = Real(0);
  Real st = Real(0);
  Real sp = Real(0);
  for (int n = 0; n < EXA_P; ++n) {
    int nm = sph_idx(n, 0);
    int nms = tri_idx(n, 0);
    Cx term = cx_mul(Ci.L[nms], Ynm[nm]);
    phi += term.re;
    sr += term.re / rho * static_cast<Real>(n);
    st += cx_mul(Ci.L[nms], YnmTheta[nm]).re;
    for (int m = 1; m <= n; ++m) {
      nm = sph_idx(n, m);
      nms = tri_idx(n, m);
      term = cx_mul(Ci.L[nms], Ynm[nm]);
      phi += Real(2) * term.re;
      sr += Real(2) * term.re / rho * static_cast<Real>(n);
      st += Real(2) * cx_mul(Ci.L[nms], YnmTheta[nm]).re;
      sp += Real(2) * cx_mul(term, cx_make(Real(0), Real(1))).re * static_cast<Real>(m);
    }
  }
  d_sph2cart(rho, alpha, beta, sr, st, sp, ax, ay, az);
}

__global__ void fmm3d_eval_kernel() {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < g_n; i += stride) {
    const int ci = g_body_cell[i];
    const TreeNode &Ci = g_tree[ci];
#if !FMM3D_GTAP_FUSED
    const size_t ps =
        static_cast<size_t>(ci) * static_cast<size_t>(DTT_P2P_CAP);
    int pcount = g_dtt_p2p_count ? g_dtt_p2p_count[ci] : 0;
    if (pcount > DTT_P2P_CAP) pcount = DTT_P2P_CAP;
#endif
    const Real xi = g_bx[i];
    const Real yi = g_by[i];
    const Real zi = g_bz[i];
    Real local_phi, local_ax, local_ay, local_az;
    d_eval_local_trg(Ci, xi, yi, zi, local_phi, local_ax, local_ay, local_az);
#if FMM3D_GTAP_FUSED
    Real phi = g_phi[i] + local_phi;
    Real ax = g_ax[i] + local_ax;
    Real ay = g_ay[i] + local_ay;
    Real az = g_az[i] + local_az;
#else
    Real phi = local_phi;
    Real ax = local_ax;
    Real ay = local_ay;
    Real az = local_az;
    for (int p = 0; p < pcount; ++p) {
      const int cj = g_dtt_p2p_src[ps + static_cast<size_t>(p)];
      const TreeNode &Cj = g_tree[cj];
      for (int j = Cj.ibody; j < Cj.ibody + Cj.nbody; ++j) {
        if (j == i) continue;
        Real dx = xi - g_bx[j];
        Real dy = yi - g_by[j];
        Real dz = zi - g_bz[j];
        Real r2 = dx * dx + dy * dy + dz * dz + H_EPS2;
        Real inv_r = real_rsqrt(r2);
        Real q_over_r = g_bq[j] * inv_r;
        phi += q_over_r;
        Real inv_r3_q = q_over_r / r2;
        ax -= dx * inv_r3_q;
        ay -= dy * inv_r3_q;
        az -= dz * inv_r3_q;
      }
    }
#endif
    g_phi[i] = phi;
    g_ax[i] = ax;
    g_ay[i] = ay;
    g_az[i] = az;
  }
}

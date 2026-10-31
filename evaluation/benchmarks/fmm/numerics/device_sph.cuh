#pragma once

#include "../device_symbols.cuh"
#include "../fmm_types.cuh"

__device__ __forceinline__ int d_child(const TreeNode &c, int oct) {
  return c.child[oct];
}

__device__ __forceinline__ int d_abs_i(int x) { return x < 0 ? -x : x; }

// ---------- Device spherical-harmonic helpers ----------

__device__ __forceinline__ Real d_factorial_eps(int n) {
  Real v = EXA_EPS;
  for (int i = 1; i <= n; ++i) v *= static_cast<Real>(i);
  return v;
}

__device__ __forceinline__ Real d_anm(int n, int m) {
  return static_cast<Real>(odd_or_even(n)) *
         real_rsqrt(d_factorial_eps(n - m) * d_factorial_eps(n + m));
}

__device__ __forceinline__ Real d_prefactor(int n, int m) {
  return g_sph_prefactor[sph_idx(n, m)];
}

__device__ __forceinline__ Cx d_ipow(int e) {
  int r = e % 4;
  if (r < 0) r += 4;
  if (r == 0) return cx_make(Real(1), Real(0));
  if (r == 1) return cx_make(Real(0), Real(1));
  if (r == 2) return cx_make(-Real(1), Real(0));
  return cx_make(Real(0), -Real(1));
}

__device__ __forceinline__ Cx d_m2l_cnm(int jknm) {
  return cx_make(g_m2l_cnm_re[jknm], g_m2l_cnm_im[jknm]);
}

__device__ void d_cart2sph(Real x, Real y, Real z,
                           Real &r, Real &theta, Real &phi) {
  r = real_sqrt(x * x + y * y + z * z);
  theta = (r == Real(0)) ? Real(0) : real_acos(z / r);
  phi = real_atan2(y, x);
}

__device__ void d_eval_multipole(Real rho, Real alpha, Real beta, Cx *Ynm) {
  Real x = real_cos(alpha);
  Real y = real_sin(alpha);
  Real fact = Real(1);
  Real pn = Real(1);
  Real rhom = Real(1);
  for (int m = 0; m < EXA_P; ++m) {
    Cx eim = cx_make(real_cos(static_cast<Real>(m) * beta),
                     real_sin(static_cast<Real>(m) * beta));
    Real p = pn;
    int npn = m * m + 2 * m;
    int nmn = m * m;
    Ynm[npn] = cx_scale(eim, rhom * p * d_prefactor(m, m));
    Ynm[nmn] = cx_conj(Ynm[npn]);
    Real p1 = p;
    p = x * (2 * m + 1) * p1;
    rhom *= rho;
    Real rhon = rhom;
    for (int n = m + 1; n < EXA_P; ++n) {
      int npm = sph_idx(n, m);
      int nmm = sph_idx(n, -m);
      Ynm[npm] = cx_scale(eim, rhon * p * d_prefactor(n, m));
      Ynm[nmm] = cx_conj(Ynm[npm]);
      Real p2 = p1;
      p1 = p;
      p = (x * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);
      rhon *= rho;
    }
    pn = -pn * fact * y;
    fact += Real(2);
  }
}

__device__ void d_eval_multipole_theta(Real rho, Real alpha, Real beta,
                                       Cx *Ynm, Cx *YnmTheta) {
  Real x = real_cos(alpha);
  Real y = real_sin(alpha);
  if (real_abs(y) < Real(1.0e-7)) y = real_copysign(Real(1.0e-7), y == Real(0) ? Real(1) : y);
  Real fact = Real(1);
  Real pn = Real(1);
  Real rhom = Real(1);
  for (int m = 0; m < EXA_P; ++m) {
    Cx eim = cx_make(real_cos(static_cast<Real>(m) * beta),
                     real_sin(static_cast<Real>(m) * beta));
    Real p = pn;
    int npn = m * m + 2 * m;
    int nmn = m * m;
    Ynm[npn] = cx_scale(eim, rhom * p * d_prefactor(m, m));
    Ynm[nmn] = cx_conj(Ynm[npn]);
    Real p1 = p;
    p = x * (2 * m + 1) * p1;
    YnmTheta[npn] =
        cx_scale(eim, rhom * (p - (m + 1) * x * p1) / y * d_prefactor(m, m));
    YnmTheta[nmn] = cx_conj(YnmTheta[npn]);
    rhom *= rho;
    Real rhon = rhom;
    for (int n = m + 1; n < EXA_P; ++n) {
      int npm = sph_idx(n, m);
      int nmm = sph_idx(n, -m);
      Ynm[npm] = cx_scale(eim, rhon * p * d_prefactor(n, m));
      Ynm[nmm] = cx_conj(Ynm[npm]);
      Real p2 = p1;
      p1 = p;
      p = (x * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);
      YnmTheta[npm] =
          cx_scale(eim, rhon * ((n - m + 1) * p - (n + 1) * x * p1) /
                            y * d_prefactor(n, m));
      YnmTheta[nmm] = cx_conj(YnmTheta[npm]);
      rhon *= rho;
    }
    pn = -pn * fact * y;
    fact += Real(2);
  }
}

__device__ void d_eval_local(Real rho, Real alpha, Real beta, Cx *Ynm2) {
  Real x = real_cos(alpha);
  Real y = real_sin(alpha);
  Real fact = Real(1);
  Real pn = Real(1);
  Real rhom = Real(1) / rho;
  for (int m = 0; m < 2 * EXA_P; ++m) {
    Cx eim = cx_make(real_cos(static_cast<Real>(m) * beta),
                     real_sin(static_cast<Real>(m) * beta));
    Real p = pn;
    int npn = m * m + 2 * m;
    int nmn = m * m;
    Ynm2[npn] = cx_scale(eim, rhom * p * d_prefactor(m, m));
    Ynm2[nmn] = cx_conj(Ynm2[npn]);
    Real p1 = p;
    p = x * (2 * m + 1) * p1;
    rhom /= rho;
    Real rhon = rhom;
    for (int n = m + 1; n < 2 * EXA_P; ++n) {
      int npm = sph_idx(n, m);
      int nmm = sph_idx(n, -m);
      Ynm2[npm] = cx_scale(eim, rhon * p * d_prefactor(n, m));
      Ynm2[nmm] = cx_conj(Ynm2[npm]);
      Real p2 = p1;
      p1 = p;
      p = (x * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);
      rhon /= rho;
    }
    pn = -pn * fact * y;
    fact += Real(2);
  }
}

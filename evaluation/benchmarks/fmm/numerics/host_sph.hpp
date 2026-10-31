#pragma once

#include "../fmm_types.cuh"

#include <cmath>

// ---------- Host spherical-harmonic helpers ----------

static inline Real h_factorial_eps(int n) {
  Real v = EXA_EPS;
  for (int i = 1; i <= n; ++i) v *= static_cast<Real>(i);
  return v;
}

static inline Real h_factorial_one(int n) {
  Real v = Real(1);
  for (int i = 1; i <= n; ++i) v *= static_cast<Real>(i);
  return v;
}

static inline Real h_anm(int n, int m) {
  return static_cast<Real>(odd_or_even(n)) /
         std::sqrt(h_factorial_eps(n - m) * h_factorial_eps(n + m));
}

static inline Real h_prefactor(int n, int m) {
  const int am = std::abs(m);
  return std::sqrt(h_factorial_one(n - am) / h_factorial_one(n + am));
}

static inline Cx h_ipow(int e) {
  int r = e % 4;
  if (r < 0) r += 4;
  if (r == 0) return cx_make(Real(1), Real(0));
  if (r == 1) return cx_make(Real(0), Real(1));
  if (r == 2) return cx_make(-Real(1), Real(0));
  return cx_make(Real(0), -Real(1));
}

static inline void h_cart2sph(Real x, Real y, Real z,
                              Real &r, Real &theta, Real &phi) {
  r = std::sqrt(x * x + y * y + z * z);
  theta = (r == Real(0)) ? Real(0) : std::acos(z / r);
  phi = std::atan2(y, x);
}

static void h_eval_multipole(Real rho, Real alpha, Real beta, Cx *Ynm) {
  const Real x = std::cos(alpha);
  const Real y = std::sin(alpha);
  Real fact = Real(1);
  Real pn = Real(1);
  Real rhom = Real(1);
  for (int m = 0; m < EXA_P; ++m) {
    Cx eim = cx_make(std::cos(static_cast<Real>(m) * beta),
                     std::sin(static_cast<Real>(m) * beta));
    Real p = pn;
    int npn = m * m + 2 * m;
    int nmn = m * m;
    Ynm[npn] = cx_scale(eim, rhom * p * h_prefactor(m, m));
    Ynm[nmn] = cx_conj(Ynm[npn]);
    Real p1 = p;
    p = x * (2 * m + 1) * p1;
    rhom *= rho;
    Real rhon = rhom;
    for (int n = m + 1; n < EXA_P; ++n) {
      int npm = sph_idx(n, m);
      int nmm = sph_idx(n, -m);
      Ynm[npm] = cx_scale(eim, rhon * p * h_prefactor(n, m));
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

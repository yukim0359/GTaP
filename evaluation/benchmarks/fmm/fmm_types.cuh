#pragma once

#include "config_defaults.cuh"

#include <memory>
#include <vector>

// ---------- Types ----------

struct Cx {
  Real re;
  Real im;
};

__host__ __device__ static inline Cx cx_make(Real re, Real im = Real(0)) { return Cx{re, im}; }
__host__ __device__ static inline Cx cx_add(Cx a, Cx b) { return Cx{a.re + b.re, a.im + b.im}; }
__host__ __device__ static inline Cx cx_mul(Cx a, Cx b) {
  return Cx{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
__host__ __device__ static inline Cx cx_scale(Cx a, Real s) { return Cx{a.re * s, a.im * s}; }
__host__ __device__ static inline Cx cx_conj(Cx a) { return Cx{a.re, -a.im}; }

__host__ __device__ static inline int odd_or_even(int n) { return (n & 1) ? -1 : 1; }
__host__ __device__ static inline int sph_idx(int n, int m) { return n * n + n + m; }
__host__ __device__ static inline int tri_idx(int n, int m) { return n * (n + 1) / 2 + m; }

struct HostBodies {
  std::vector<Real> x, y, z, q;
  explicit HostBodies(int n = 0) : x(n), y(n), z(n), q(n) {}
  int size() const { return static_cast<int>(x.size()); }
};

struct DeviceBodies {
  Real *x = nullptr;
  Real *y = nullptr;
  Real *z = nullptr;
  Real *q = nullptr;
};

struct TreeNode {
  Real cx, cy, cz, r;
  int parent;
  int child[8];
  int ibody, nbody;
  int is_leaf;
  Cx M[EXA_NTERM];
  Cx L[EXA_NTERM];
};

struct HostFixedLists {
  std::vector<int> m2l_count;
  std::vector<int> p2p_count;
  std::unique_ptr<int[]> m2l_src;
  std::unique_ptr<int[]> p2p_src;
  int overflow = 0;
  int max_m2l = 0;
  int max_p2p = 0;
  long long total_m2l = 0;
  long long total_p2p = 0;
};

struct HostDttBreakdown {
  double ms_reset_counts = 0.0;
  double ms_traverse = 0.0;
  double ms_stats = 0.0;
  double ms_h2d_counts = 0.0;
  double ms_h2d_lists = 0.0;
  double ms_buffer_alloc = 0.0;
};

struct GpuTreeBuildResult {
  std::vector<TreeNode> tree;
  DeviceBodies sorted_bodies;
};

struct DttPair {
  int ci;
  int cj;
};

static inline void h_zero_multipole_coeffs(TreeNode &node) {
  for (int k = 0; k < EXA_NTERM; ++k) {
    node.M[k] = cx_make(Real(0));
    node.L[k] = cx_make(Real(0));
  }
}

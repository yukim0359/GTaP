// 3D Laplace FMM-style benchmark for GTaP.
//
// The numerical kernel is still compact, but the data path now uses a small
// fixed-order spherical-harmonic Laplace expansion following exafmm-beta's
// kernels/laplace.h indexing and translation formulas, while preserving the
// DTT comparison axis.

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#ifndef FMM3D_SERIAL_HOST_DTT
#define FMM3D_SERIAL_HOST_DTT 0
#endif

#ifndef FMM3D_WORKLIST_DTT
#define FMM3D_WORKLIST_DTT 0
#endif

// #define GTAP_ENABLE_PROFILING
#include "gtap_thread.cuh"
#include "fmm_real.cuh"

#define CUDA_CHECK(stmt)                                                        \
  do {                                                                          \
    cudaError_t _e = (stmt);                                                    \
    if (_e != cudaSuccess) {                                                    \
      std::printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                  cudaGetErrorString(_e));                                      \
      return 1;                                                                 \
    }                                                                           \
  } while (0)

// ---------- Tunable parameters ----------

#ifndef FMM3D_ENABLE_GTAP_DTT
#define FMM3D_ENABLE_GTAP_DTT 1
#endif
#ifndef FMM3D_DTT_TASK_DEPTH
#define FMM3D_DTT_TASK_DEPTH 5
#endif
#ifndef FMM3D_DTT_TASK_MIN_N
#define FMM3D_DTT_TASK_MIN_N 4096
#endif
#ifndef FMM3D_HOST_DTT_TASK_DEPTH
#define FMM3D_HOST_DTT_TASK_DEPTH 5
#endif
#ifndef FMM3D_HOST_DTT_TASK_MIN_N
#define FMM3D_HOST_DTT_TASK_MIN_N 4096
#endif
#ifndef FMM3D_DTT_M2L_CAP
#define FMM3D_DTT_M2L_CAP 1000
#endif
#ifndef FMM3D_DTT_P2P_CAP
#define FMM3D_DTT_P2P_CAP 500
#endif
#ifndef FMM3D_NCRIT
#define FMM3D_NCRIT 32
#endif
#ifndef FMM3D_SET_CUDA_STACK_LIMIT
#define FMM3D_SET_CUDA_STACK_LIMIT 0
#endif
#ifndef FMM3D_CUDA_STACK_SIZE
#define FMM3D_CUDA_STACK_SIZE 8192
#endif
#ifndef FMM3D_P
#define FMM3D_P 3
#endif
#ifndef FMM3D_FUSED_DTT
#define FMM3D_FUSED_DTT 0
#endif
#ifndef FMM3D_PARTITIONED_FUSED_DTT
#define FMM3D_PARTITIONED_FUSED_DTT 0
#endif
#if FMM3D_PARTITIONED_FUSED_DTT && !FMM3D_FUSED_DTT
#error "FMM3D_PARTITIONED_FUSED_DTT requires FMM3D_FUSED_DTT=1"
#endif
#ifndef FMM3D_DTT_NSPAWN
#define FMM3D_DTT_NSPAWN 10000
#endif
#ifndef FMM3D_WORKLIST_PAIR_CAP_FACTOR
#define FMM3D_WORKLIST_PAIR_CAP_FACTOR 64
#endif
#ifndef FMM3D_M2L_BLOCK_SIZE
#define FMM3D_M2L_BLOCK_SIZE 64
#endif
#ifndef FMM3D_GPU_L2L
#define FMM3D_GPU_L2L 1
#endif
#ifndef FMM3D_GPU_UPWARD
#define FMM3D_GPU_UPWARD 1
#endif
#ifndef FMM3D_GPU_TREE_BUILD
#define FMM3D_GPU_TREE_BUILD 1
#endif
#ifndef FMM3D_M2L_COMPACT_TARGETS
#define FMM3D_M2L_COMPACT_TARGETS 0
#endif
#ifndef FMM3D_MORTON_BITS
#define FMM3D_MORTON_BITS 21
#endif
#ifndef FMM3D_DIRECT_VALIDATE_N
#define FMM3D_DIRECT_VALIDATE_N 4096
#endif
#ifndef FMM3D_DIRECT_SAMPLE_N
#define FMM3D_DIRECT_SAMPLE_N 0
#endif
#ifndef FMM3D_DIRECT_SAMPLE_SEED
#define FMM3D_DIRECT_SAMPLE_SEED 12345
#endif
#ifndef FMM3D_DIRECT_SAMPLE_BLOCK_SIZE
#define FMM3D_DIRECT_SAMPLE_BLOCK_SIZE 256
#endif
#ifndef FMM3D_BODY_SEED
#define FMM3D_BODY_SEED 42
#endif
#ifndef FMM3D_BODY_SEED_PER_N
#define FMM3D_BODY_SEED_PER_N 1
#endif
#if FMM3D_P < 1
#error "FMM3D_P must be at least 1"
#endif
#if FMM3D_DIRECT_SAMPLE_BLOCK_SIZE < 32 || \
    (FMM3D_DIRECT_SAMPLE_BLOCK_SIZE & (FMM3D_DIRECT_SAMPLE_BLOCK_SIZE - 1)) != 0
#error "FMM3D_DIRECT_SAMPLE_BLOCK_SIZE must be a power of two and at least 32"
#endif
#define FMM3D_NTERM (FMM3D_P * (FMM3D_P + 1) / 2)

static constexpr int NCRIT = FMM3D_NCRIT;
static constexpr int MAX_DEPTH = 24;
static constexpr int EXA_P = FMM3D_P;
static constexpr int EXA_NTERM = FMM3D_NTERM;
static constexpr int EXA_NSPH = 4 * EXA_P * EXA_P;
static constexpr int M2L_CNM_SIZE = EXA_P * EXA_P * EXA_P * EXA_P;
static constexpr int FMM_CONST_BYTES =
    M2L_CNM_SIZE * 2 * static_cast<int>(sizeof(Real)) +
    EXA_NSPH * static_cast<int>(sizeof(Real));
static constexpr bool FMM_CONST_IN_CONSTANT = FMM_CONST_BYTES <= 65536;
static constexpr int DTT_TASK_DEPTH = FMM3D_DTT_TASK_DEPTH;
static constexpr int DTT_TASK_MIN_N = FMM3D_DTT_TASK_MIN_N;
static constexpr int HOST_DTT_TASK_DEPTH = FMM3D_HOST_DTT_TASK_DEPTH;
static constexpr int HOST_DTT_TASK_MIN_N = FMM3D_HOST_DTT_TASK_MIN_N;
static constexpr int DTT_M2L_CAP = FMM3D_DTT_M2L_CAP;
static constexpr int DTT_P2P_CAP = FMM3D_DTT_P2P_CAP;
[[maybe_unused]] static constexpr int WORKLIST_PAIR_CAP_FACTOR = FMM3D_WORKLIST_PAIR_CAP_FACTOR;
static constexpr int M2L_BLOCK_SIZE = FMM3D_M2L_BLOCK_SIZE;
static constexpr int MORTON_BITS = FMM3D_MORTON_BITS;
static constexpr int MORTON_BITS_MAX = 21;  // uint64 Morton key uses 3 bits per level
static_assert(FMM3D_MORTON_BITS >= 1 && FMM3D_MORTON_BITS <= MORTON_BITS_MAX,
              "FMM3D_MORTON_BITS must be in [1, 21] (64-bit Morton key limit)");
static constexpr int DIRECT_VALIDATE_N = FMM3D_DIRECT_VALIDATE_N;
static constexpr int DIRECT_SAMPLE_N_DEFAULT = FMM3D_DIRECT_SAMPLE_N;
static constexpr int DIRECT_SAMPLE_SEED = FMM3D_DIRECT_SAMPLE_SEED;
static constexpr int DIRECT_SAMPLE_BLOCK_SIZE = FMM3D_DIRECT_SAMPLE_BLOCK_SIZE;
static constexpr int DTT_NSPAWN = FMM3D_DTT_NSPAWN;

#define FMM3D_GTAP_FUSED (FMM3D_ENABLE_GTAP_DTT && FMM3D_FUSED_DTT)
#define FMM3D_GTAP_ATOMIC_FUSED (FMM3D_GTAP_FUSED && !FMM3D_PARTITIONED_FUSED_DTT)

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

// ---------- Morton sort kernels ----------

__host__ __device__ static inline uint64_t hdev_morton_key(Real x, Real y, Real z,
                                                           int bits) {
  const uint32_t scale = 1u << bits;
  uint32_t ix = static_cast<uint32_t>(real_fmin(real_fmax(x, Real(0)), Real(0.99999994)) * scale);
  uint32_t iy = static_cast<uint32_t>(real_fmin(real_fmax(y, Real(0)), Real(0.99999994)) * scale);
  uint32_t iz = static_cast<uint32_t>(real_fmin(real_fmax(z, Real(0)), Real(0.99999994)) * scale);
  if (ix >= scale) ix = scale - 1;
  if (iy >= scale) iy = scale - 1;
  if (iz >= scale) iz = scale - 1;
  uint64_t key = 0;
  for (int level = 0; level < bits; ++level) {
    const int bit = bits - 1 - level;
    const uint64_t oct =
        ((static_cast<uint64_t>((ix >> bit) & 1u)) |
         (static_cast<uint64_t>((iy >> bit) & 1u) << 1) |
         (static_cast<uint64_t>((iz >> bit) & 1u) << 2));
    key = (key << 3) | oct;
  }
  return key;
}

__global__ void fmm3d_morton_key_kernel(const Real *x, const Real *y, const Real *z,
                                        uint64_t *keys,
                                        int *indices, int n, int bits) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < n; i += stride) {
    keys[i] = hdev_morton_key(x[i], y[i], z[i], bits);
    indices[i] = i;
  }
}

__global__ void fmm3d_reorder_bodies_kernel(const Real *in_x, const Real *in_y,
                                            const Real *in_z, const Real *in_q,
                                            const int *sorted_indices,
                                            Real *out_x, Real *out_y,
                                            Real *out_z, Real *out_q,
                                            int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < n; i += stride) {
    const int src = sorted_indices[i];
    out_x[i] = in_x[src];
    out_y[i] = in_y[src];
    out_z[i] = in_z[src];
    out_q[i] = in_q[src];
  }
}

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

static void h_print_leaf_depth_stats(const std::vector<TreeNode> &tree,
                                     const std::vector<int> &cell_depth,
                                     const std::vector<int> &leaf_idx,
                                     int n_bodies) {
  const int nleaf = static_cast<int>(leaf_idx.size());
  const int morton_cap = std::min(MORTON_BITS, MAX_DEPTH);
  if (nleaf == 0) {
    std::printf("Tree depth: no leaves (morton_cap=%d)\n", morton_cap);
    return;
  }

  int max_all_depth = 0;
  for (int d : cell_depth) max_all_depth = std::max(max_all_depth, d);

  int max_leaf_depth = 0;
  int leaves_at_cap = 0;
  int leaves_nbody_gt_ncrit = 0;
  std::vector<int> hist(static_cast<size_t>(morton_cap) + 1u, 0);
  for (int li = 0; li < nleaf; ++li) {
    const int ci = leaf_idx[li];
    const int d = cell_depth[ci];
    max_leaf_depth = std::max(max_leaf_depth, d);
    if (d >= 0 && d <= morton_cap) hist[static_cast<size_t>(d)]++;
    if (d >= morton_cap) leaves_at_cap++;
    if (tree[ci].nbody > NCRIT) leaves_nbody_gt_ncrit++;
  }

  std::printf("Tree depth: max_all=%d max_leaf=%d morton_cap=%d (MORTON_BITS=%d)\n",
              max_all_depth, max_leaf_depth, morton_cap, MORTON_BITS);
  std::printf("Leaf depth: at_cap=%d/%d (%.1f%%)  nbody>NCRIT=%d/%d (%.1f%%)  avg_nbody/leaf=%.1f\n",
              leaves_at_cap, nleaf, 100.0 * static_cast<double>(leaves_at_cap) / nleaf,
              leaves_nbody_gt_ncrit, nleaf,
              100.0 * static_cast<double>(leaves_nbody_gt_ncrit) / nleaf,
              static_cast<double>(n_bodies) / nleaf);
  std::printf("Leaf depth histogram:\n");
  for (int d = 0; d <= morton_cap; ++d) {
    if (hist[static_cast<size_t>(d)] == 0) continue;
    std::printf("  d=%2d: %8d (%5.1f%%)\n", d, hist[static_cast<size_t>(d)],
                100.0 * static_cast<double>(hist[static_cast<size_t>(d)]) / nleaf);
  }
}

#if FMM3D_PARTITIONED_FUSED_DTT
static void h_build_child_ranges(const std::vector<TreeNode> &tree,
                                 std::vector<int> &child_lo,
                                 std::vector<int> &child_hi) {
  const int ncells = static_cast<int>(tree.size());
  child_lo.assign(ncells, 0);
  child_hi.assign(ncells, 0);
  for (int ci = 0; ci < ncells; ++ci) {
    if (tree[ci].is_leaf) continue;
    int lo = ncells;
    int hi = 0;
    for (int oct = 0; oct < 8; ++oct) {
      const int ch = tree[ci].child[oct];
      if (ch < 0) continue;
      lo = std::min(lo, ch);
      hi = std::max(hi, ch + 1);
    }
    if (hi > lo) {
      child_lo[ci] = lo;
      child_hi[ci] = hi;
    }
  }
}
#endif

struct GpuTreeBuildResult {
  std::vector<TreeNode> tree;
  DeviceBodies sorted_bodies;
};

static int h_get_env_int(const char *name, int fallback) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0') return fallback;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value) return fallback;
  if (parsed < 0) return 0;
  if (parsed > std::numeric_limits<int>::max()) return std::numeric_limits<int>::max();
  return static_cast<int>(parsed);
}

static inline void h_zero_multipole_coeffs(TreeNode &node) {
  for (int k = 0; k < EXA_NTERM; ++k) {
    node.M[k] = cx_make(Real(0));
    node.L[k] = cx_make(Real(0));
  }
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

// ---------- Host octree build ----------

static uint64_t h_body_rng_seed(int n) {
#if FMM3D_BODY_SEED_PER_N
  return static_cast<uint64_t>(FMM3D_BODY_SEED) + static_cast<uint64_t>(n);
#else
  return static_cast<uint64_t>(FMM3D_BODY_SEED);
#endif
}

static void h_generate_bodies(HostBodies &bodies, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<Real> pos(Real(0.05), Real(0.95));
  std::uniform_real_distribution<Real> mass(Real(0.5), Real(2));
  const int n = bodies.size();
  for (int i = 0; i < n; ++i) {
    bodies.x[i] = pos(rng);
    bodies.y[i] = pos(rng);
    bodies.z[i] = pos(rng);
    bodies.q[i] = mass(rng);
  }
}

static void h_make_child_node(std::vector<TreeNode> &tree, int parent_idx, int oct) {
  const TreeNode &p = tree[parent_idx];
  TreeNode c{};
  c.r = p.r * Real(0.5);
  c.cx = p.cx + ((oct & 1) ? c.r : -c.r);
  c.cy = p.cy + ((oct & 2) ? c.r : -c.r);
  c.cz = p.cz + ((oct & 4) ? c.r : -c.r);
  c.parent = parent_idx;
  for (int k = 0; k < 8; ++k) c.child[k] = -1;
  c.ibody = 0;
  c.nbody = 0;
  c.is_leaf = 1;
  h_zero_multipole_coeffs(c);
  tree.push_back(c);
}

static void h_subdivide(std::vector<TreeNode> &tree,
                        std::vector<int> &perm,
                        const HostBodies &bodies,
                        int node_idx, int lo, int hi, int depth) {
  const int n = hi - lo;
  if (n <= NCRIT || depth >= MAX_DEPTH) {
    tree[node_idx].is_leaf = 1;
    tree[node_idx].ibody = lo;
    tree[node_idx].nbody = n;
    return;
  }

  const Real cx = tree[node_idx].cx;
  const Real cy = tree[node_idx].cy;
  const Real cz = tree[node_idx].cz;
  int count[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = lo; i < hi; ++i) {
    const int pi = perm[i];
    int oct = int(bodies.x[pi] >= cx) | (int(bodies.y[pi] >= cy) << 1) |
              (int(bodies.z[pi] >= cz) << 2);
    count[oct]++;
  }

  int offs[8];
  {
    int run = lo;
    for (int oct = 0; oct < 8; ++oct) {
      offs[oct] = run;
      run += count[oct];
    }
  }
  std::vector<int> tmp(n);
  int local_off[8];
  for (int oct = 0; oct < 8; ++oct) local_off[oct] = offs[oct] - lo;
  for (int i = lo; i < hi; ++i) {
    const int pi = perm[i];
    int oct = int(bodies.x[pi] >= cx) | (int(bodies.y[pi] >= cy) << 1) |
              (int(bodies.z[pi] >= cz) << 2);
    tmp[local_off[oct]++] = perm[i];
  }
  for (int i = 0; i < n; ++i) perm[lo + i] = tmp[i];

  tree[node_idx].is_leaf = 0;
  int child_idx_local[8];
  for (int oct = 0; oct < 8; ++oct) child_idx_local[oct] = -1;
  for (int oct = 0; oct < 8; ++oct) {
    if (count[oct] == 0) {
      tree[node_idx].child[oct] = -1;
      continue;
    }
    int child_idx = -1;
    #pragma omp critical(fmm3d_tree_push_node)
    {
      h_make_child_node(tree, node_idx, oct);
      child_idx = static_cast<int>(tree.size()) - 1;
    }
    tree[node_idx].child[oct] = child_idx;
    child_idx_local[oct] = child_idx;
  }

  for (int oct = 0; oct < 8; ++oct) {
    if (child_idx_local[oct] == -1) continue;
    int cidx = child_idx_local[oct];
    int clo = offs[oct];
    int chi = offs[oct] + count[oct];
    #pragma omp task default(none) firstprivate(cidx, clo, chi, depth, oct) \
                     shared(tree, perm, bodies, count) \
                     if((depth < HOST_DTT_TASK_DEPTH) && (count[oct] >= HOST_DTT_TASK_MIN_N))
    {
      h_subdivide(tree, perm, bodies, cidx, clo, chi, depth + 1);
    }
  }
  #pragma omp taskwait
  tree[node_idx].ibody = lo;
  tree[node_idx].nbody = n;
}

[[maybe_unused]] static std::vector<TreeNode> h_build_tree(HostBodies &bodies_inout) {
  const int n = static_cast<int>(bodies_inout.size());
  std::vector<TreeNode> tree;
  tree.reserve(std::max(2, 2 * n));
  TreeNode root{};
  root.cx = Real(0.5);
  root.cy = Real(0.5);
  root.cz = Real(0.5);
  root.r = Real(0.5);
  root.parent = -1;
  for (int k = 0; k < 8; ++k) root.child[k] = -1;
  root.ibody = 0;
  root.nbody = n;
  root.is_leaf = 1;
  h_zero_multipole_coeffs(root);
  tree.push_back(root);

  std::vector<int> perm(n);
  for (int i = 0; i < n; ++i) perm[i] = i;
  #pragma omp parallel
  {
    #pragma omp single nowait
    h_subdivide(tree, perm, bodies_inout, 0, 0, n, 0);
  }

  HostBodies sorted(n);
  for (int i = 0; i < n; ++i) {
    const int src = perm[i];
    sorted.x[i] = bodies_inout.x[src];
    sorted.y[i] = bodies_inout.y[src];
    sorted.z[i] = bodies_inout.z[src];
    sorted.q[i] = bodies_inout.q[src];
  }
  bodies_inout = std::move(sorted);
  return tree;
}

static inline int h_morton_octant_at(uint64_t key, int depth, int bits) {
  const int shift = 3 * (bits - depth);
  return static_cast<int>((key >> shift) & 7ull);
}

static void h_subdivide_morton_sorted(std::vector<TreeNode> &tree,
                                      const std::vector<uint64_t> &keys,
                                      int node_idx, int lo, int hi,
                                      int depth, int bits) {
  const int n = hi - lo;
  TreeNode &node = tree[node_idx];
  node.ibody = lo;
  node.nbody = n;
  if (n <= NCRIT || depth >= bits || depth >= MAX_DEPTH) {
    node.is_leaf = 1;
    return;
  }

  int begin[8];
  int end[8];
  for (int oct = 0; oct < 8; ++oct) {
    begin[oct] = hi;
    end[oct] = hi;
  }
  int pos = lo;
  int nonempty = 0;
  while (pos < hi) {
    const int oct = h_morton_octant_at(keys[pos], depth + 1, bits);
    const int b = pos;
    while (pos < hi && h_morton_octant_at(keys[pos], depth + 1, bits) == oct) {
      ++pos;
    }
    begin[oct] = b;
    end[oct] = pos;
    ++nonempty;
  }

  if (nonempty <= 1 && depth + 1 >= bits) {
    node.is_leaf = 1;
    return;
  }

  node.is_leaf = 0;
  for (int oct = 0; oct < 8; ++oct) {
    node.child[oct] = -1;
    if (begin[oct] == hi) continue;
    h_make_child_node(tree, node_idx, oct);
    const int child_idx = static_cast<int>(tree.size()) - 1;
    tree[node_idx].child[oct] = child_idx;
  }
  for (int oct = 0; oct < 8; ++oct) {
    const int child_idx = tree[node_idx].child[oct];
    if (child_idx == -1) continue;
    h_subdivide_morton_sorted(tree, keys, child_idx, begin[oct], end[oct],
                              depth + 1, bits);
  }
}

[[maybe_unused]] static GpuTreeBuildResult h_build_tree_gpu_morton(
    HostBodies &bodies_inout, bool copy_sorted_bodies_to_host) {
  const int n = static_cast<int>(bodies_inout.size());
  if (MORTON_BITS <= 0 || MORTON_BITS > MORTON_BITS_MAX) {
    throw std::runtime_error(
        "FMM3D_MORTON_BITS must be in [1, 21] (3 bits/level in uint64 Morton key)");
  }

  Real *d_x = nullptr, *d_y = nullptr, *d_z = nullptr, *d_q = nullptr;
  Real *d_sorted_x = nullptr, *d_sorted_y = nullptr;
  Real *d_sorted_z = nullptr, *d_sorted_q = nullptr;
  uint64_t *d_keys = nullptr;
  int *d_indices = nullptr;
  cudaError_t err = cudaMalloc(&d_x, sizeof(Real) * n);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  err = cudaMalloc(&d_y, sizeof(Real) * n);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  err = cudaMalloc(&d_z, sizeof(Real) * n);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  err = cudaMalloc(&d_q, sizeof(Real) * n);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  err = cudaMalloc(&d_sorted_x, sizeof(Real) * n);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  err = cudaMalloc(&d_sorted_y, sizeof(Real) * n);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  err = cudaMalloc(&d_sorted_z, sizeof(Real) * n);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  err = cudaMalloc(&d_sorted_q, sizeof(Real) * n);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  err = cudaMalloc(&d_keys, sizeof(uint64_t) * n);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  err = cudaMalloc(&d_indices, sizeof(int) * n);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  err = cudaMemcpy(d_x, bodies_inout.x.data(), sizeof(Real) * n, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  err = cudaMemcpy(d_y, bodies_inout.y.data(), sizeof(Real) * n, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  err = cudaMemcpy(d_z, bodies_inout.z.data(), sizeof(Real) * n, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  err = cudaMemcpy(d_q, bodies_inout.q.data(), sizeof(Real) * n, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

  constexpr int KEY_BLOCK_SIZE = 256;
  int blocks = (n + KEY_BLOCK_SIZE - 1) / KEY_BLOCK_SIZE;
  blocks = std::max(1, blocks);
  fmm3d_morton_key_kernel<<<blocks, KEY_BLOCK_SIZE>>>(d_x, d_y, d_z, d_keys, d_indices,
                                                      n, MORTON_BITS);
  err = cudaGetLastError();
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

  thrust::device_ptr<uint64_t> keys_ptr(d_keys);
  thrust::device_ptr<int> idx_ptr(d_indices);
  thrust::sort_by_key(keys_ptr, keys_ptr + n, idx_ptr);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

  fmm3d_reorder_bodies_kernel<<<blocks, KEY_BLOCK_SIZE>>>(d_x, d_y, d_z, d_q,
                                                          d_indices, d_sorted_x,
                                                          d_sorted_y, d_sorted_z,
                                                          d_sorted_q, n);
  err = cudaGetLastError();
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

  std::vector<uint64_t> keys(n);
  err = cudaMemcpy(keys.data(), d_keys, sizeof(uint64_t) * n, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  if (copy_sorted_bodies_to_host) {
    err = cudaMemcpy(bodies_inout.x.data(), d_sorted_x, sizeof(Real) * n, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    err = cudaMemcpy(bodies_inout.y.data(), d_sorted_y, sizeof(Real) * n, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    err = cudaMemcpy(bodies_inout.z.data(), d_sorted_z, sizeof(Real) * n, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    err = cudaMemcpy(bodies_inout.q.data(), d_sorted_q, sizeof(Real) * n, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  }
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_q);
  cudaFree(d_keys);
  cudaFree(d_indices);

  std::vector<TreeNode> tree;
  tree.reserve(std::max(2, 2 * n));
  TreeNode root{};
  root.cx = Real(0.5);
  root.cy = Real(0.5);
  root.cz = Real(0.5);
  root.r = Real(0.5);
  root.parent = -1;
  for (int k = 0; k < 8; ++k) root.child[k] = -1;
  root.ibody = 0;
  root.nbody = n;
  root.is_leaf = 1;
  h_zero_multipole_coeffs(root);
  tree.push_back(root);
  h_subdivide_morton_sorted(tree, keys, 0, 0, n, 0,
                            std::min(MORTON_BITS, MAX_DEPTH));
  GpuTreeBuildResult result;
  result.tree = std::move(tree);
  result.sorted_bodies = DeviceBodies{d_sorted_x, d_sorted_y, d_sorted_z, d_sorted_q};
  return result;
}

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

// ---------- Host dual-tree traversal (fixed-cap lists) ----------

static inline void h_fixed_append_m2l(HostFixedLists &lists, int ci, int cj) {
#if FMM3D_SERIAL_HOST_DTT
  const int pos = lists.m2l_count[ci]++;
#else
  int pos;
  #pragma omp atomic capture
  pos = lists.m2l_count[ci]++;
#endif
  if (pos < DTT_M2L_CAP) {
    lists.m2l_src[static_cast<size_t>(ci) * DTT_M2L_CAP + pos] = cj;
  } else {
#if FMM3D_SERIAL_HOST_DTT
    lists.overflow = 1;
#else
    #pragma omp atomic write
    lists.overflow = 1;
#endif
  }
}

static inline void h_fixed_append_p2p(HostFixedLists &lists, int ci, int cj) {
#if FMM3D_SERIAL_HOST_DTT
  const int pos = lists.p2p_count[ci]++;
#else
  int pos;
  #pragma omp atomic capture
  pos = lists.p2p_count[ci]++;
#endif
  if (pos < DTT_P2P_CAP) {
    lists.p2p_src[static_cast<size_t>(ci) * DTT_P2P_CAP + pos] = cj;
  } else {
#if FMM3D_SERIAL_HOST_DTT
    lists.overflow = 1;
#else
    #pragma omp atomic write
    lists.overflow = 1;
#endif
  }
}

static void h_dtt_fixed(int ci, int cj,
                        const std::vector<TreeNode> &tree,
                        HostFixedLists &lists, Real theta, int depth) {
  const TreeNode &Ci = tree[ci];
  if (ci == cj) {
    if (Ci.is_leaf) {
      h_fixed_append_p2p(lists, ci, cj);
    } else {
      for (int oi = 0; oi < 8; ++oi) {
        const int child_i = Ci.child[oi];
        if (child_i == -1) continue;
#if !FMM3D_SERIAL_HOST_DTT
        const bool spawn = (depth < HOST_DTT_TASK_DEPTH) &&
                           (tree[child_i].nbody >= HOST_DTT_TASK_MIN_N);
#endif
        for (int oj = 0; oj < 8; ++oj) {
          const int child_j = Ci.child[oj];
          if (child_j == -1) continue;
#if FMM3D_SERIAL_HOST_DTT
          h_dtt_fixed(child_i, child_j, tree, lists, theta, depth + 1);
#else
          if (spawn) {
            #pragma omp task default(none) firstprivate(child_i, child_j, theta, depth) \
                               shared(tree, lists)
            {
              h_dtt_fixed(child_i, child_j, tree, lists, theta, depth + 1);
            }
          } else {
            h_dtt_fixed(child_i, child_j, tree, lists, theta, depth + 1);
          }
#endif
        }
      }
    }
    return;
  }

  const TreeNode &Cj = tree[cj];
  if (h_dtt_well_separated(Ci, Cj, theta)) {
    h_fixed_append_m2l(lists, ci, cj);
    return;
  }
  if (Ci.is_leaf && Cj.is_leaf) {
    h_fixed_append_p2p(lists, ci, cj);
    return;
  }

  {
    const bool split_ci = h_dtt_split_ci(Ci, Cj);
    for (int oct = 0; oct < 8; ++oct) {
      const int child_i = split_ci ? Ci.child[oct] : ci;
      const int child_j = split_ci ? cj : Cj.child[oct];
      if (split_ci ? (child_i == -1) : (child_j == -1)) continue;
      const int nb_lhs = split_ci ? tree[child_i].nbody : Ci.nbody;
      const int nb_rhs = split_ci ? Cj.nbody : tree[child_j].nbody;
#if FMM3D_SERIAL_HOST_DTT
      h_dtt_fixed(child_i, child_j, tree, lists, theta, depth + 1);
#else
      if (depth < HOST_DTT_TASK_DEPTH &&
          nb_lhs + nb_rhs >= HOST_DTT_TASK_MIN_N) {
        #pragma omp task default(none) firstprivate(child_i, child_j, theta, depth) \
                           shared(tree, lists)
        {
          h_dtt_fixed(child_i, child_j, tree, lists, theta, depth + 1);
        }
      } else {
        h_dtt_fixed(child_i, child_j, tree, lists, theta, depth + 1);
      }
#endif
    }
  }
}

[[maybe_unused]] static void h_prepare_fixed_lists(HostFixedLists &lists, int ncells) {
  lists.m2l_count.assign(ncells, 0);
  lists.p2p_count.assign(ncells, 0);
  const size_t m2l_size = static_cast<size_t>(ncells) * DTT_M2L_CAP;
  const size_t p2p_size = static_cast<size_t>(ncells) * DTT_P2P_CAP;
  lists.m2l_src.reset(new int[m2l_size]);
  lists.p2p_src.reset(new int[p2p_size]);
  lists.overflow = 0;
  lists.max_m2l = 0;
  lists.max_p2p = 0;
  lists.total_m2l = 0;
  lists.total_p2p = 0;
}

[[maybe_unused]] static void h_reset_fixed_lists(HostFixedLists &lists, int ncells) {
  std::fill(lists.m2l_count.begin(), lists.m2l_count.end(), 0);
  std::fill(lists.p2p_count.begin(), lists.p2p_count.end(), 0);
  lists.overflow = 0;
  lists.max_m2l = 0;
  lists.max_p2p = 0;
  lists.total_m2l = 0;
  lists.total_p2p = 0;
}

[[maybe_unused]] static void h_aggregate_fixed_list_stats(HostFixedLists &lists, int ncells) {
  for (int ci = 0; ci < ncells; ++ci) {
    lists.total_m2l += lists.m2l_count[ci];
    lists.total_p2p += lists.p2p_count[ci];
    lists.max_m2l = std::max(lists.max_m2l, lists.m2l_count[ci]);
    lists.max_p2p = std::max(lists.max_p2p, lists.p2p_count[ci]);
  }
}

[[maybe_unused]] static void h_run_host_dtt_traversal(
    const std::vector<TreeNode> &tree, Real theta,
    HostFixedLists &lists, HostDttBreakdown *breakdown = nullptr) {
  const int ncells = static_cast<int>(tree.size());
  const auto t_reset_start = std::chrono::high_resolution_clock::now();
  h_reset_fixed_lists(lists, ncells);
  const auto t_reset_end = std::chrono::high_resolution_clock::now();
  const auto t_traverse_start = std::chrono::high_resolution_clock::now();
#if FMM3D_SERIAL_HOST_DTT
  h_dtt_fixed(0, 0, tree, lists, theta, 0);
#else
  #pragma omp parallel
  {
    #pragma omp single nowait
    h_dtt_fixed(0, 0, tree, lists, theta, 0);
  }
#endif
  const auto t_traverse_end = std::chrono::high_resolution_clock::now();
  const auto t_stats_start = std::chrono::high_resolution_clock::now();
  h_aggregate_fixed_list_stats(lists, ncells);
  const auto t_stats_end = std::chrono::high_resolution_clock::now();
  if (breakdown) {
    breakdown->ms_reset_counts =
        std::chrono::duration<double, std::milli>(t_reset_end - t_reset_start).count();
    breakdown->ms_traverse =
        std::chrono::duration<double, std::milli>(t_traverse_end - t_traverse_start).count();
    breakdown->ms_stats =
        std::chrono::duration<double, std::milli>(t_stats_end - t_stats_start).count();
  }
}

#if 0  // Legacy one-shot helper; unused.
[[maybe_unused]] static HostFixedLists h_build_fixed_interaction_lists(
    const std::vector<TreeNode> &tree, Real theta,
    HostDttBreakdown *breakdown = nullptr) {
  const int ncells = static_cast<int>(tree.size());
  HostFixedLists lists;
  const auto t_alloc_start = std::chrono::high_resolution_clock::now();
  lists.m2l_count.assign(ncells, 0);
  lists.p2p_count.assign(ncells, 0);
  const size_t m2l_size = static_cast<size_t>(ncells) * DTT_M2L_CAP;
  const size_t p2p_size = static_cast<size_t>(ncells) * DTT_P2P_CAP;
  lists.m2l_src.reset(new int[m2l_size]);
  lists.p2p_src.reset(new int[p2p_size]);
  const auto t_alloc_end = std::chrono::high_resolution_clock::now();
  const auto t_traverse_start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel
  {
    #pragma omp single nowait
    h_dtt_fixed(0, 0, tree, lists, theta, 0);
  }
  const auto t_traverse_end = std::chrono::high_resolution_clock::now();
  const auto t_stats_start = std::chrono::high_resolution_clock::now();
  for (int ci = 0; ci < ncells; ++ci) {
    lists.total_m2l += lists.m2l_count[ci];
    lists.total_p2p += lists.p2p_count[ci];
    lists.max_m2l = std::max(lists.max_m2l, lists.m2l_count[ci]);
    lists.max_p2p = std::max(lists.max_p2p, lists.p2p_count[ci]);
  }
  const auto t_stats_end = std::chrono::high_resolution_clock::now();
  if (breakdown) {
    breakdown->ms_buffer_alloc =
        std::chrono::duration<double, std::milli>(t_alloc_end - t_alloc_start).count();
    breakdown->ms_traverse =
        std::chrono::duration<double, std::milli>(t_traverse_end - t_traverse_start).count();
    breakdown->ms_stats =
        std::chrono::duration<double, std::milli>(t_stats_end - t_stats_start).count();
  }
  return lists;
}
#endif

static cudaError_t h_copy_dtt_counts_from_device(
    int *d_m2l_count, int *d_p2p_count, int *d_dtt_overflow,
    int ncells,
    int &out_overflow,
    int &max_m2l_count, int &max_p2p_count,
    long long &m2l_total_ll, long long &p2p_total_ll,
    std::vector<int> &m2l_targets,
    int &m2l_target_count) {
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
  m2l_targets.clear();
  for (int ci = 0; ci < ncells; ++ci) {
    m2l_total_ll += h_m2l_count[ci];
    p2p_total_ll += h_p2p_count[ci];
    max_m2l_count = std::max(max_m2l_count, h_m2l_count[ci]);
    max_p2p_count = std::max(max_p2p_count, h_p2p_count[ci]);
#if FMM3D_M2L_COMPACT_TARGETS
    if (h_m2l_count[ci] > 0) m2l_targets.push_back(ci);
#endif
  }
#if FMM3D_M2L_COMPACT_TARGETS
  m2l_target_count = static_cast<int>(m2l_targets.size());
#else
  m2l_target_count = ncells;
#endif
  return cudaSuccess;
}

static double h_elapsed_ms(std::chrono::high_resolution_clock::time_point begin,
                           std::chrono::high_resolution_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

// ---------- Device globals ----------

__device__ TreeNode *g_tree;
__device__ const Real *g_bx;
__device__ const Real *g_by;
__device__ const Real *g_bz;
__device__ const Real *g_bq;
__device__ Real *g_phi;
__device__ Real *g_ax;
__device__ Real *g_ay;
__device__ Real *g_az;
__device__ int g_n;
__device__ int g_ncells;
__device__ int g_nleaf;
__device__ const int *g_leaf_idx;
__device__ const int *g_body_cell;
__device__ int *g_dtt_m2l_count;
__device__ int *g_dtt_p2p_count;
__device__ int *g_dtt_m2l_src;
__device__ int *g_dtt_p2p_src;
__device__ int *g_dtt_overflow;
__device__ Real g_dtt_theta;

#if FMM3D_PARTITIONED_FUSED_DTT
__device__ const int *g_child_lo;
__device__ const int *g_child_hi;
__device__ int g_dtt_nspawn;
#endif

#if FMM_CONST_IN_CONSTANT
__constant__ Real g_m2l_cnm_re[M2L_CNM_SIZE];
__constant__ Real g_m2l_cnm_im[M2L_CNM_SIZE];
__constant__ Real g_sph_prefactor[EXA_NSPH];
#else
__device__ Real g_m2l_cnm_re[M2L_CNM_SIZE];
__device__ Real g_m2l_cnm_im[M2L_CNM_SIZE];
__device__ Real g_sph_prefactor[EXA_NSPH];
#endif

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

struct DttPair {
  int ci;
  int cj;
};

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

__global__ void fmm3d_direct_sample_validation_kernel(
    const int *sample_idx, int nsample,
    double *phi_diff2, double *phi_norm2,
    double *acc_diff2, double *acc_norm2,
    double *phi_rel_abs, double *acc_rel_abs) {
  const int sid = blockIdx.x;
  if (sid >= nsample) return;
  const int i = sample_idx[sid];
  if (i < 0 || i >= g_n) return;

  const double xi = static_cast<double>(g_bx[i]);
  const double yi = static_cast<double>(g_by[i]);
  const double zi = static_cast<double>(g_bz[i]);
  double phi = 0.0;
  double ax = 0.0;
  double ay = 0.0;
  double az = 0.0;

  for (int j = threadIdx.x; j < g_n; j += blockDim.x) {
    if (j == i) continue;
    const double dx = xi - static_cast<double>(g_bx[j]);
    const double dy = yi - static_cast<double>(g_by[j]);
    const double dz = zi - static_cast<double>(g_bz[j]);
    const double r2 = dx * dx + dy * dy + dz * dz + static_cast<double>(H_EPS2);
    const double inv_r = 1.0 / sqrt(r2);
    const double q_over_r = static_cast<double>(g_bq[j]) * inv_r;
    phi += q_over_r;
    const double inv_r3_q = q_over_r / r2;
    ax -= dx * inv_r3_q;
    ay -= dy * inv_r3_q;
    az -= dz * inv_r3_q;
  }

  extern __shared__ double smem_direct_sample[];
  double *sphi = smem_direct_sample;
  double *sax = sphi + blockDim.x;
  double *say = sax + blockDim.x;
  double *saz = say + blockDim.x;
  sphi[threadIdx.x] = phi;
  sax[threadIdx.x] = ax;
  say[threadIdx.x] = ay;
  saz[threadIdx.x] = az;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (static_cast<int>(threadIdx.x) < offset) {
      sphi[threadIdx.x] += sphi[threadIdx.x + offset];
      sax[threadIdx.x] += sax[threadIdx.x + offset];
      say[threadIdx.x] += say[threadIdx.x + offset];
      saz[threadIdx.x] += saz[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const double ref_phi = sphi[0];
    const double ref_ax = sax[0];
    const double ref_ay = say[0];
    const double ref_az = saz[0];
    const double dphi = static_cast<double>(g_phi[i]) - ref_phi;
    const double dax = static_cast<double>(g_ax[i]) - ref_ax;
    const double day = static_cast<double>(g_ay[i]) - ref_ay;
    const double daz = static_cast<double>(g_az[i]) - ref_az;
    const double acc_ref2 = ref_ax * ref_ax + ref_ay * ref_ay + ref_az * ref_az;
    const double acc_err2 = dax * dax + day * day + daz * daz;
    phi_diff2[sid] = dphi * dphi;
    phi_norm2[sid] = ref_phi * ref_phi;
    acc_diff2[sid] = acc_err2;
    acc_norm2[sid] = acc_ref2;
    phi_rel_abs[sid] = fabs(dphi) / fmax(fabs(ref_phi), 1.0e-300);
    acc_rel_abs[sid] = sqrt(acc_err2) / fmax(sqrt(acc_ref2), 1.0e-300);
  }
}

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

// ---------- Main ----------

int main(int argc, char **argv) {
  // 1. Args, CUDA context, optional device stack limit
  if (argc < 2) {
    std::printf("Usage: %s <n> [theta]\n", argv[0]);
    return 1;
  }
  const int n = std::atoi(argv[1]);
  const Real theta = (argc >= 3) ? static_cast<Real>(std::atof(argv[2])) : Real(0.5);
  if (n <= 0) return 1;
  const int direct_sample_n =
      std::min(n, h_get_env_int("FMM3D_DIRECT_SAMPLE_N", DIRECT_SAMPLE_N_DEFAULT));

  CUDA_CHECK(cudaFree(0));
  CUDA_CHECK(h_upload_fmm_constant_tables());

#if FMM3D_SET_CUDA_STACK_LIMIT
  {
    size_t old_stack = 0;
    cudaError_t stack_err = cudaDeviceGetLimit(&old_stack, cudaLimitStackSize);
    if (stack_err != cudaSuccess) {
      std::printf("CUDA warning: cudaLimitStackSize get failed: %s\n",
                  cudaGetErrorString(stack_err));
      cudaGetLastError();
      old_stack = 0;
    }

    const size_t requested_stack = static_cast<size_t>(FMM3D_CUDA_STACK_SIZE);
    stack_err = cudaDeviceSetLimit(cudaLimitStackSize, requested_stack);
    if (stack_err != cudaSuccess) {
      std::printf("CUDA warning: cudaLimitStackSize not set: %s\n",
                  cudaGetErrorString(stack_err));
      cudaGetLastError();
    } else {
      size_t actual_stack = 0;
      stack_err = cudaDeviceGetLimit(&actual_stack, cudaLimitStackSize);
      if (stack_err == cudaSuccess) {
        std::printf("CUDA stack limit: old=%zu requested=%zu actual=%zu bytes\n",
                    old_stack, requested_stack, actual_stack);
      } else {
        std::printf("CUDA stack limit: requested=%zu bytes\n", requested_stack);
        cudaGetLastError();
      }
    }
  }
#endif

  double ms_tree_build = 0.0;
  double ms_upward = 0.0;
  double ms_gtap_init = 0.0;
  double ms_dtt = 0.0;
  [[maybe_unused]] double ms_dtt_fill_setup = 0.0;
  [[maybe_unused]] double ms_dtt_fill_kernel = 0.0;
  [[maybe_unused]] double ms_dtt_fill_post = 0.0;
  [[maybe_unused]] double ms_dtt_core = 0.0;
  [[maybe_unused]] double ms_dtt_construct = 0.0;
  [[maybe_unused]] double ms_dtt_construct_handoff = 0.0;
  double ms_dtt_init = 0.0;
  [[maybe_unused]] double ms_host_list_buffer_alloc = 0.0;
  [[maybe_unused]] double ms_worklist_pair_buffer_alloc = 0.0;
  [[maybe_unused]] HostDttBreakdown host_dtt_breakdown;
  double ms_metadata = 0.0;
  double ms_h2d_bind = 0.0;
  double ms_m2l = 0.0;
  double ms_l2l = 0.0;
  [[maybe_unused]] double ms_l2l_tree_d2h = 0.0;
  [[maybe_unused]] double ms_l2l_tree_h2d = 0.0;
  double ms_eval = 0.0;
  double ms_d2h = 0.0;
  double ms_execution = 0.0;
  double ms_sample_direct = 0.0;
  double ms_direct = 0.0;

  std::chrono::high_resolution_clock::time_point t_run_start =
      std::chrono::high_resolution_clock::now();

  // 2. Generate bodies, build octree (Morton+GPU sort or CPU subdivide)
  const std::chrono::high_resolution_clock::time_point t_tree_build_start =
      std::chrono::high_resolution_clock::now();
  HostBodies hb(n);
  const uint64_t body_seed = h_body_rng_seed(n);
  h_generate_bodies(hb, body_seed);
  std::vector<TreeNode> htree;
  DeviceBodies sorted_device_bodies;
  try {
#if FMM3D_GPU_TREE_BUILD
    const bool need_host_sorted_bodies =
        (!FMM3D_GPU_UPWARD) || (DIRECT_VALIDATE_N > 0 && n <= DIRECT_VALIDATE_N);
    GpuTreeBuildResult build_result =
        h_build_tree_gpu_morton(hb, need_host_sorted_bodies);
    htree = std::move(build_result.tree);
    sorted_device_bodies = build_result.sorted_bodies;
#else
    htree = h_build_tree(hb);
#endif
  } catch (const std::exception &e) {
    std::printf("Tree build error: %s\n", e.what());
    return 1;
  }
  {
    const std::chrono::high_resolution_clock::time_point t_tree_build_end =
        std::chrono::high_resolution_clock::now();
    ms_tree_build = h_elapsed_ms(t_tree_build_start, t_tree_build_end);
  }
#if !FMM3D_GPU_UPWARD
  {
    const std::chrono::high_resolution_clock::time_point t_upward_start =
        std::chrono::high_resolution_clock::now();
    h_upward_pass(htree, hb);
    const std::chrono::high_resolution_clock::time_point t_upward_end =
        std::chrono::high_resolution_clock::now();
    ms_upward = h_elapsed_ms(t_upward_start, t_upward_end);
  }
#endif

  // 3. Traversal metadata: leaf_idx, body_cell, depth, L2L/upward level lists
  const std::chrono::high_resolution_clock::time_point t_metadata_start = std::chrono::high_resolution_clock::now();
  const int ncells = static_cast<int>(htree.size());
  std::vector<int> leaf_idx;
  leaf_idx.reserve(std::max(1, ncells / 2));
  std::vector<int> body_cell(n, -1);
  std::vector<int> cell_depth(ncells, 0);
  int max_cell_depth = 0;

  for (int ci = 0; ci < ncells; ++ci) {
    const int parent = htree[ci].parent;
    if (parent >= 0) {
      cell_depth[ci] = cell_depth[parent] + 1;
      max_cell_depth = std::max(max_cell_depth, cell_depth[ci]);
    }
    if (htree[ci].is_leaf) {
      leaf_idx.push_back(ci);
      for (int k = 0; k < htree[ci].nbody; ++k) {
        body_cell[htree[ci].ibody + k] = ci;
      }
    }
  }

  std::vector<int> l2l_level_offsets(max_cell_depth + 2, 0);
  for (int ci = 1; ci < ncells; ++ci) {
    l2l_level_offsets[cell_depth[ci] + 1]++;
  }
  for (int d = 1; d <= max_cell_depth + 1; ++d) {
    l2l_level_offsets[d] += l2l_level_offsets[d - 1];
  }
  std::vector<int> l2l_cells(std::max(0, ncells - 1));
  {
    std::vector<int> cursor = l2l_level_offsets;
    for (int ci = 1; ci < ncells; ++ci) {
      l2l_cells[cursor[cell_depth[ci]]++] = ci;
    }
  }

  std::vector<int> upward_level_offsets(max_cell_depth + 2, 0);
  for (int ci = 0; ci < ncells; ++ci) {
    upward_level_offsets[cell_depth[ci] + 1]++;
  }
  for (int d = 1; d <= max_cell_depth + 1; ++d) {
    upward_level_offsets[d] += upward_level_offsets[d - 1];
  }
  std::vector<int> upward_cells(ncells);
  {
    std::vector<int> cursor = upward_level_offsets;
    for (int ci = 0; ci < ncells; ++ci) {
      upward_cells[cursor[cell_depth[ci]]++] = ci;
    }
  }

  const int nleaf = static_cast<int>(leaf_idx.size());
  std::printf("Problem setup: N=%d  Nodes=%d  Leaves=%d  theta=%.2f  body_seed=%llu  per_n_seed=%d\n",
              n, ncells, nleaf, theta,
              static_cast<unsigned long long>(body_seed),
              FMM3D_BODY_SEED_PER_N);
  h_print_leaf_depth_stats(htree, cell_depth, leaf_idx, n);
  {
    const std::chrono::high_resolution_clock::time_point t_metadata_end = std::chrono::high_resolution_clock::now();
    ms_metadata = h_elapsed_ms(t_metadata_start, t_metadata_end);
  }

  // 4. cudaMalloc, H2D, bind_params / bind_ptrs / bind_dtt_ptrs
  const std::chrono::high_resolution_clock::time_point t_h2d_bind_start = std::chrono::high_resolution_clock::now();
  TreeNode *d_tree = nullptr;
  Real *d_bx = sorted_device_bodies.x;
  Real *d_by = sorted_device_bodies.y;
  Real *d_bz = sorted_device_bodies.z;
  Real *d_bq = sorted_device_bodies.q;
  Real *d_phi = nullptr, *d_ax = nullptr, *d_ay = nullptr, *d_az = nullptr;
  int *d_leaf_idx = nullptr;
  int *d_body_cell = nullptr;
  int *d_l2l_cells = nullptr;
  int *d_upward_cells = nullptr;
  int *d_m2l_count = nullptr, *d_p2p_count = nullptr;
  int *d_m2l_src = nullptr, *d_p2p_src = nullptr;
  int *d_m2l_targets = nullptr;
  int *d_dtt_overflow = nullptr;
#if FMM3D_PARTITIONED_FUSED_DTT
  int *d_child_lo = nullptr;
  int *d_child_hi = nullptr;
#endif

  CUDA_CHECK(cudaMalloc(&d_tree, sizeof(TreeNode) * ncells));
  if (!d_bx) CUDA_CHECK(cudaMalloc(&d_bx, sizeof(Real) * n));
  if (!d_by) CUDA_CHECK(cudaMalloc(&d_by, sizeof(Real) * n));
  if (!d_bz) CUDA_CHECK(cudaMalloc(&d_bz, sizeof(Real) * n));
  if (!d_bq) CUDA_CHECK(cudaMalloc(&d_bq, sizeof(Real) * n));
  CUDA_CHECK(cudaMalloc(&d_phi, sizeof(Real) * n));
  CUDA_CHECK(cudaMalloc(&d_ax, sizeof(Real) * n));
  CUDA_CHECK(cudaMalloc(&d_ay, sizeof(Real) * n));
  CUDA_CHECK(cudaMalloc(&d_az, sizeof(Real) * n));
  CUDA_CHECK(cudaMalloc(&d_leaf_idx, sizeof(int) * std::max(1, nleaf)));
  CUDA_CHECK(cudaMalloc(&d_body_cell, sizeof(int) * n));
  CUDA_CHECK(cudaMalloc(&d_l2l_cells, sizeof(int) * std::max(1, ncells - 1)));
  CUDA_CHECK(cudaMalloc(&d_upward_cells, sizeof(int) * std::max(1, ncells)));
  CUDA_CHECK(cudaMalloc(&d_m2l_targets, sizeof(int) * std::max(1, ncells)));

  CUDA_CHECK(cudaMemcpy(d_tree, htree.data(), sizeof(TreeNode) * ncells, cudaMemcpyHostToDevice));
  if (!sorted_device_bodies.x) {
    CUDA_CHECK(cudaMemcpy(d_bx, hb.x.data(), sizeof(Real) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_by, hb.y.data(), sizeof(Real) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bz, hb.z.data(), sizeof(Real) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bq, hb.q.data(), sizeof(Real) * n, cudaMemcpyHostToDevice));
  }
  if (nleaf > 0) {
    CUDA_CHECK(cudaMemcpy(d_leaf_idx, leaf_idx.data(), sizeof(int) * nleaf, cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMemcpy(d_body_cell, body_cell.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
  if (ncells > 1) {
    CUDA_CHECK(cudaMemcpy(d_l2l_cells, l2l_cells.data(), sizeof(int) * (ncells - 1),
                          cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMemcpy(d_upward_cells, upward_cells.data(), sizeof(int) * ncells,
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(bind_params(n, ncells, nleaf));
  CUDA_CHECK(bind_ptrs(d_tree, d_bx, d_by, d_bz, d_bq, d_phi, d_ax, d_ay, d_az,
                       d_leaf_idx, d_body_cell));
  const std::chrono::high_resolution_clock::time_point t_h2d_bind_end =
      std::chrono::high_resolution_clock::now();
  ms_h2d_bind = h_elapsed_ms(t_h2d_bind_start, t_h2d_bind_end);

  // 5. GPU upward: P2M on leaves, M2M per depth
#if FMM3D_GPU_UPWARD
  {
    const std::chrono::high_resolution_clock::time_point t_upward_start =
        std::chrono::high_resolution_clock::now();
    constexpr int UPWARD_BLOCK_SIZE = 128;
    if (nleaf > 0) {
      int blocks = std::max(1, (nleaf + UPWARD_BLOCK_SIZE - 1) / UPWARD_BLOCK_SIZE);
      fmm3d_p2m_leaf_kernel<<<blocks, UPWARD_BLOCK_SIZE>>>(d_leaf_idx, nleaf);
      CUDA_CHECK(cudaGetLastError());
    }
    for (int depth = max_cell_depth; depth >= 0; --depth) {
      const int begin = upward_level_offsets[depth];
      const int end = upward_level_offsets[depth + 1];
      const int count = end - begin;
      if (count == 0) continue;
      int blocks = (count + UPWARD_BLOCK_SIZE - 1) / UPWARD_BLOCK_SIZE;
      blocks = std::max(1, blocks);
      fmm3d_m2m_level_kernel<<<blocks, UPWARD_BLOCK_SIZE>>>(d_upward_cells + begin, count);
      CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::chrono::high_resolution_clock::time_point t_upward_end =
        std::chrono::high_resolution_clock::now();
    ms_upward = h_elapsed_ms(t_upward_start, t_upward_end);
  }
#endif

  // 6. DTT: worklist / GTaP / CPU (OpenMP)
  int h_dtt_overflow = 0;
  int max_m2l_count = 0;
  int max_p2p_count = 0;
  long long m2l_total_ll = 0;
  long long p2p_total_ll = 0;
  int m2l_target_count = 0;
  std::vector<int> h_m2l_targets;
#if FMM3D_WORKLIST_DTT
  int worklist_iterations = 0;
  int worklist_max_frontier = 0;
  int worklist_pair_cap = 0;
  int worklist_overflow = 0;
  DttPair *d_pairs_a = nullptr;
  DttPair *d_pairs_b = nullptr;
  int *d_worklist_out_count = nullptr;
  int *d_worklist_overflow = nullptr;
  const size_t pair_cap_size =
      static_cast<size_t>(std::max(1, ncells)) *
      static_cast<size_t>(std::max(1, WORKLIST_PAIR_CAP_FACTOR));
  if (pair_cap_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    std::printf("GPU worklist DTT pair cap too large for int counter: ncells=%d factor=%d cap=%zu\n",
                ncells, WORKLIST_PAIR_CAP_FACTOR, pair_cap_size);
    return 1;
  }
  worklist_pair_cap = static_cast<int>(pair_cap_size);
#endif
#if !FMM3D_ENABLE_GTAP_DTT && !FMM3D_WORKLIST_DTT
  HostFixedLists host_fixed_lists;
#endif

  {
    const std::chrono::high_resolution_clock::time_point t_dtt_init_start =
        std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMalloc(&d_m2l_count, sizeof(int) * ncells));
    CUDA_CHECK(cudaMalloc(&d_p2p_count, sizeof(int) * ncells));
    CUDA_CHECK(cudaMalloc(&d_m2l_src, sizeof(int) * static_cast<size_t>(ncells) * DTT_M2L_CAP));
    CUDA_CHECK(cudaMalloc(&d_p2p_src, sizeof(int) * static_cast<size_t>(ncells) * DTT_P2P_CAP));
    CUDA_CHECK(cudaMalloc(&d_dtt_overflow, sizeof(int)));
#if FMM3D_PARTITIONED_FUSED_DTT
    {
      std::vector<int> h_child_lo;
      std::vector<int> h_child_hi;
      h_build_child_ranges(htree, h_child_lo, h_child_hi);
      CUDA_CHECK(cudaMalloc(&d_child_lo, sizeof(int) * std::max(1, ncells)));
      CUDA_CHECK(cudaMalloc(&d_child_hi, sizeof(int) * std::max(1, ncells)));
      if (ncells > 0) {
        CUDA_CHECK(cudaMemcpy(d_child_lo, h_child_lo.data(), sizeof(int) * ncells,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_child_hi, h_child_hi.data(), sizeof(int) * ncells,
                              cudaMemcpyHostToDevice));
      }
      CUDA_CHECK(bind_partitioned_dtt_ptrs(d_child_lo, d_child_hi, DTT_NSPAWN));
    }
#endif
    CUDA_CHECK(bind_dtt_ptrs(d_m2l_count, d_p2p_count, d_m2l_src, d_p2p_src, d_dtt_overflow, theta));
#if FMM3D_ENABLE_GTAP_DTT
    {
      const std::chrono::high_resolution_clock::time_point t_gtap_init_start =
          std::chrono::high_resolution_clock::now();
      const int max_tasks_per_warp = h_get_env_int(
          "GTAP_MAX_TASKS_PER_WARP", GTAP_BENCH_MAX_TASKS_PER_WARP);
      std::printf("GTaP config: grid=%d block=%d max_tasks_per_warp=%d "
                  "num_queues=%d\n",
                  GTAP_BENCH_GRID_SIZE, GTAP_BENCH_BLOCK_SIZE,
                  max_tasks_per_warp, GTAP_BENCH_NUM_QUEUES);
      gtap_thread_config config{
          .grid_size = GTAP_BENCH_GRID_SIZE,
          .block_size = GTAP_BENCH_BLOCK_SIZE,
          .max_tasks_per_warp = max_tasks_per_warp,
          .num_queues = GTAP_BENCH_NUM_QUEUES,
          .profile_capacity_per_warp =
              GTAP_BENCH_PROFILE_CAPACITY_PER_WARP,
      };
      size_t gtap_device_bytes = 0;
      CUDA_CHECK(gtap_initialize(config, &gtap_device_bytes));
      CUDA_CHECK(cudaDeviceSynchronize());
      std::printf("GTaP runtime allocation: %.2f GiB\n",
                  static_cast<double>(gtap_device_bytes) /
                      (1024.0 * 1024.0 * 1024.0));
      const std::chrono::high_resolution_clock::time_point t_gtap_init_end =
          std::chrono::high_resolution_clock::now();
      ms_gtap_init = h_elapsed_ms(t_gtap_init_start, t_gtap_init_end);
    }
#elif FMM3D_WORKLIST_DTT
    {
      const std::chrono::high_resolution_clock::time_point t_wl_alloc_start =
          std::chrono::high_resolution_clock::now();
      CUDA_CHECK(cudaMalloc(&d_pairs_a, sizeof(DttPair) * pair_cap_size));
      CUDA_CHECK(cudaMalloc(&d_pairs_b, sizeof(DttPair) * pair_cap_size));
      CUDA_CHECK(cudaMalloc(&d_worklist_out_count, sizeof(int)));
      CUDA_CHECK(cudaMalloc(&d_worklist_overflow, sizeof(int)));
      const std::chrono::high_resolution_clock::time_point t_wl_alloc_end =
          std::chrono::high_resolution_clock::now();
      ms_worklist_pair_buffer_alloc =
          h_elapsed_ms(t_wl_alloc_start, t_wl_alloc_end);
    }
#else
    {
      const std::chrono::high_resolution_clock::time_point t_buffer_alloc_start =
          std::chrono::high_resolution_clock::now();
      h_prepare_fixed_lists(host_fixed_lists, ncells);
      const std::chrono::high_resolution_clock::time_point t_buffer_alloc_end =
          std::chrono::high_resolution_clock::now();
      host_dtt_breakdown.ms_buffer_alloc =
          h_elapsed_ms(t_buffer_alloc_start, t_buffer_alloc_end);
      ms_host_list_buffer_alloc = host_dtt_breakdown.ms_buffer_alloc;
    }
#endif
    const std::chrono::high_resolution_clock::time_point t_dtt_init_end =
        std::chrono::high_resolution_clock::now();
    ms_dtt_init = h_elapsed_ms(t_dtt_init_start, t_dtt_init_end);
  }

#if FMM3D_ENABLE_GTAP_DTT
#if FMM3D_GTAP_FUSED
  CUDA_CHECK(cudaMemset(d_phi, 0, sizeof(Real) * n));
  CUDA_CHECK(cudaMemset(d_ax, 0, sizeof(Real) * n));
  CUDA_CHECK(cudaMemset(d_ay, 0, sizeof(Real) * n));
  CUDA_CHECK(cudaMemset(d_az, 0, sizeof(Real) * n));
#endif
  const std::chrono::high_resolution_clock::time_point t_dtt_total_start =
      std::chrono::high_resolution_clock::now();
#if !FMM3D_PARTITIONED_FUSED_DTT
  CUDA_CHECK(cudaMemset(d_m2l_count, 0, sizeof(int) * ncells));
  CUDA_CHECK(cudaMemset(d_p2p_count, 0, sizeof(int) * ncells));
  CUDA_CHECK(cudaMemset(d_dtt_overflow, 0, sizeof(int)));
#endif

  const std::chrono::high_resolution_clock::time_point t_dtt_kernel_start =
      std::chrono::high_resolution_clock::now();
  CUDA_CHECK(gtap_launch(fmm3d_dtt_fill_kernel));
  CUDA_CHECK(cudaGetLastError());
  gtap_report_cuda_error(cudaDeviceSynchronize());
  // CUDA_CHECK(cudaDeviceSynchronize());
  const std::chrono::high_resolution_clock::time_point t_dtt_kernel_end =
      std::chrono::high_resolution_clock::now();

#if !FMM3D_PARTITIONED_FUSED_DTT
  CUDA_CHECK(h_copy_dtt_counts_from_device(
      d_m2l_count, d_p2p_count, d_dtt_overflow, ncells,
      h_dtt_overflow, max_m2l_count, max_p2p_count,
      m2l_total_ll, p2p_total_ll, h_m2l_targets, m2l_target_count));
  if (h_dtt_overflow) {
    std::printf("3D DTT fixed-cap overflow: m2l_cap=%d max_m2l=%d  p2p_cap=%d max_p2p=%d\n",
                DTT_M2L_CAP, max_m2l_count, DTT_P2P_CAP, max_p2p_count);
    return 1;
  }
#endif
  const std::chrono::high_resolution_clock::time_point t_dtt_total_end =
      std::chrono::high_resolution_clock::now();
  ms_dtt_fill_setup = h_elapsed_ms(t_dtt_total_start, t_dtt_kernel_start);
  ms_dtt_fill_kernel = h_elapsed_ms(t_dtt_kernel_start, t_dtt_kernel_end);
  ms_dtt_fill_post = h_elapsed_ms(t_dtt_kernel_end, t_dtt_total_end);
  ms_dtt = h_elapsed_ms(t_dtt_total_start, t_dtt_total_end);
  ms_dtt_core = ms_dtt_fill_kernel;
  ms_dtt_construct = ms_dtt_fill_setup + ms_dtt_fill_kernel;
  ms_dtt_construct_handoff = ms_dtt_construct;
#elif FMM3D_WORKLIST_DTT
  const std::chrono::high_resolution_clock::time_point t_dtt_total_start =
      std::chrono::high_resolution_clock::now();
  CUDA_CHECK(cudaMemset(d_m2l_count, 0, sizeof(int) * ncells));
  CUDA_CHECK(cudaMemset(d_p2p_count, 0, sizeof(int) * ncells));
  CUDA_CHECK(cudaMemset(d_dtt_overflow, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_worklist_overflow, 0, sizeof(int)));
  {
    DttPair root{0, 0};
    CUDA_CHECK(cudaMemcpy(d_pairs_a, &root, sizeof(DttPair), cudaMemcpyHostToDevice));
  }

  const std::chrono::high_resolution_clock::time_point t_dtt_kernel_start =
      std::chrono::high_resolution_clock::now();
  DttPair *d_in_pairs = d_pairs_a;
  DttPair *d_out_pairs = d_pairs_b;
  int in_count = 1;
  worklist_max_frontier = 1;
  while (in_count > 0) {
    CUDA_CHECK(cudaMemset(d_worklist_out_count, 0, sizeof(int)));
    int blocks = std::max(
        1, (in_count + GTAP_BENCH_BLOCK_SIZE - 1) /
               GTAP_BENCH_BLOCK_SIZE);
    fmm3d_worklist_expand_kernel<<<blocks, GTAP_BENCH_BLOCK_SIZE>>>(
        d_in_pairs, in_count, d_out_pairs, d_worklist_out_count,
        worklist_pair_cap, d_worklist_overflow);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int next_count = 0;
    CUDA_CHECK(cudaMemcpy(&next_count, d_worklist_out_count, sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&worklist_overflow, d_worklist_overflow, sizeof(int),
                          cudaMemcpyDeviceToHost));
    ++worklist_iterations;
    worklist_max_frontier = std::max(worklist_max_frontier, next_count);
    if (worklist_overflow || next_count > worklist_pair_cap) {
      std::printf("GPU worklist DTT pair overflow: pair_cap=%d next_frontier=%d iter=%d\n",
                  worklist_pair_cap, next_count, worklist_iterations);
      return 1;
    }
    DttPair *tmp = d_in_pairs;
    d_in_pairs = d_out_pairs;
    d_out_pairs = tmp;
    in_count = next_count;
  }
  const std::chrono::high_resolution_clock::time_point t_dtt_kernel_end =
      std::chrono::high_resolution_clock::now();

  CUDA_CHECK(h_copy_dtt_counts_from_device(
      d_m2l_count, d_p2p_count, d_dtt_overflow, ncells,
      h_dtt_overflow, max_m2l_count, max_p2p_count,
      m2l_total_ll, p2p_total_ll, h_m2l_targets, m2l_target_count));
  if (h_dtt_overflow) {
    std::printf("3D worklist DTT fixed-cap overflow: m2l_cap=%d max_m2l=%d  p2p_cap=%d max_p2p=%d\n",
                DTT_M2L_CAP, max_m2l_count, DTT_P2P_CAP, max_p2p_count);
    std::printf("DTT worklist metrics (until overflow): pair_cap=%d  iterations=%d  max_frontier=%d  overflow=%d\n",
                worklist_pair_cap, worklist_iterations, worklist_max_frontier, worklist_overflow);
    return 1;
  }
  const std::chrono::high_resolution_clock::time_point t_dtt_total_end =
      std::chrono::high_resolution_clock::now();
  ms_dtt_fill_setup = h_elapsed_ms(t_dtt_total_start, t_dtt_kernel_start);
  ms_dtt_fill_kernel = h_elapsed_ms(t_dtt_kernel_start, t_dtt_kernel_end);
  ms_dtt_fill_post = h_elapsed_ms(t_dtt_kernel_end, t_dtt_total_end);
  ms_dtt = h_elapsed_ms(t_dtt_total_start, t_dtt_total_end);
  ms_dtt_core = ms_dtt_fill_kernel;
  ms_dtt_construct = ms_dtt_fill_setup + ms_dtt_fill_kernel;
  ms_dtt_construct_handoff = ms_dtt_construct;
  cudaFree(d_pairs_a);
  cudaFree(d_pairs_b);
  cudaFree(d_worklist_out_count);
  cudaFree(d_worklist_overflow);
#else
  const std::chrono::high_resolution_clock::time_point t_dtt_total_start =
      std::chrono::high_resolution_clock::now();
  h_run_host_dtt_traversal(htree, theta, host_fixed_lists, &host_dtt_breakdown);

  h_dtt_overflow = host_fixed_lists.overflow;
  max_m2l_count = host_fixed_lists.max_m2l;
  max_p2p_count = host_fixed_lists.max_p2p;
  m2l_total_ll = host_fixed_lists.total_m2l;
  p2p_total_ll = host_fixed_lists.total_p2p;
  h_m2l_targets.reserve(ncells);
  for (int ci = 0; ci < ncells; ++ci) {
#if FMM3D_M2L_COMPACT_TARGETS
    if (host_fixed_lists.m2l_count[ci] > 0) h_m2l_targets.push_back(ci);
#endif
  }
#if FMM3D_M2L_COMPACT_TARGETS
  m2l_target_count = static_cast<int>(h_m2l_targets.size());
#else
  m2l_target_count = ncells;
#endif
  if (h_dtt_overflow) {
    std::printf("Host 3D DTT fixed-cap overflow: m2l_cap=%d max_m2l=%d  p2p_cap=%d max_p2p=%d\n",
                DTT_M2L_CAP, max_m2l_count, DTT_P2P_CAP, max_p2p_count);
    return 1;
  }
  const std::chrono::high_resolution_clock::time_point t_h2d_counts_start =
      std::chrono::high_resolution_clock::now();
  CUDA_CHECK(cudaMemcpy(d_m2l_count, host_fixed_lists.m2l_count.data(),
                        sizeof(int) * ncells, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_p2p_count, host_fixed_lists.p2p_count.data(),
                        sizeof(int) * ncells, cudaMemcpyHostToDevice));
  const std::chrono::high_resolution_clock::time_point t_h2d_counts_end =
      std::chrono::high_resolution_clock::now();
  const std::chrono::high_resolution_clock::time_point t_h2d_lists_start =
      std::chrono::high_resolution_clock::now();
  CUDA_CHECK(cudaMemcpy(d_m2l_src, host_fixed_lists.m2l_src.get(),
                        sizeof(int) * static_cast<size_t>(ncells) * DTT_M2L_CAP,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_p2p_src, host_fixed_lists.p2p_src.get(),
                        sizeof(int) * static_cast<size_t>(ncells) * DTT_P2P_CAP,
                        cudaMemcpyHostToDevice));
  const std::chrono::high_resolution_clock::time_point t_h2d_lists_end =
      std::chrono::high_resolution_clock::now();
  host_dtt_breakdown.ms_h2d_counts = h_elapsed_ms(t_h2d_counts_start, t_h2d_counts_end);
  host_dtt_breakdown.ms_h2d_lists = h_elapsed_ms(t_h2d_lists_start, t_h2d_lists_end);
  const std::chrono::high_resolution_clock::time_point t_dtt_total_end =
      std::chrono::high_resolution_clock::now();
  ms_dtt_fill_setup = host_dtt_breakdown.ms_reset_counts;
  ms_dtt_fill_kernel = host_dtt_breakdown.ms_traverse;
  ms_dtt_fill_post = host_dtt_breakdown.ms_stats + host_dtt_breakdown.ms_h2d_counts +
                     host_dtt_breakdown.ms_h2d_lists;
  ms_dtt = h_elapsed_ms(t_dtt_total_start, t_dtt_total_end);
  ms_dtt_core = host_dtt_breakdown.ms_traverse;
  ms_dtt_construct = host_dtt_breakdown.ms_reset_counts + host_dtt_breakdown.ms_traverse +
                     host_dtt_breakdown.ms_stats;
  ms_dtt_construct_handoff = ms_dtt;
#endif

#if FMM3D_M2L_COMPACT_TARGETS
  if (m2l_target_count > 0) {
    CUDA_CHECK(cudaMemcpy(d_m2l_targets, h_m2l_targets.data(),
                          sizeof(int) * m2l_target_count, cudaMemcpyHostToDevice));
  }
#endif

#ifdef GTAP_ENABLE_PROFILING
  gtap_export_profile();
#endif

  // 7. GPU M2L: accumulate remote contributions into Ci.L
#if !FMM3D_GTAP_FUSED
  CUDA_CHECK(cudaMemset(d_phi, 0, sizeof(Real) * n));
  CUDA_CHECK(cudaMemset(d_ax, 0, sizeof(Real) * n));
  CUDA_CHECK(cudaMemset(d_ay, 0, sizeof(Real) * n));
  CUDA_CHECK(cudaMemset(d_az, 0, sizeof(Real) * n));
#endif
  cudaEvent_t ev_gpu_start, ev_gpu_end;
  CUDA_CHECK(cudaEventCreate(&ev_gpu_start));
  CUDA_CHECK(cudaEventCreate(&ev_gpu_end));

#if !FMM3D_GTAP_FUSED
  CUDA_CHECK(cudaEventRecord(ev_gpu_start));
  fmm3d_m2l_kernel<<<std::max(1, m2l_target_count), M2L_BLOCK_SIZE,
                      sizeof(Real) * 2 * EXA_NTERM * M2L_BLOCK_SIZE>>>(
      FMM3D_M2L_COMPACT_TARGETS ? d_m2l_targets : nullptr, m2l_target_count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(ev_gpu_end));
  CUDA_CHECK(cudaEventSynchronize(ev_gpu_end));
  {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_gpu_start, ev_gpu_end));
    ms_m2l = static_cast<double>(ms);
  }
#endif

  // 8. L2L: propagate parent L to children (GPU per depth or CPU)
  {
    const std::chrono::high_resolution_clock::time_point t_l2l_start =
        std::chrono::high_resolution_clock::now();
#if FMM3D_GPU_L2L
    for (int depth = 1; depth <= max_cell_depth; ++depth) {
      const int begin = l2l_level_offsets[depth];
      const int end = l2l_level_offsets[depth + 1];
      const int count = end - begin;
      if (count == 0) continue;
      constexpr int L2L_BLOCK_SIZE = 128;
      int blocks = std::max(1, (count + L2L_BLOCK_SIZE - 1) / L2L_BLOCK_SIZE);
      fmm3d_l2l_level_kernel<<<blocks, L2L_BLOCK_SIZE>>>(d_l2l_cells + begin, count);
      CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::chrono::high_resolution_clock::time_point t_l2l_end =
        std::chrono::high_resolution_clock::now();
    ms_l2l = h_elapsed_ms(t_l2l_start, t_l2l_end);
#else
    CUDA_CHECK(cudaMemcpy(htree.data(), d_tree, sizeof(TreeNode) * ncells, cudaMemcpyDeviceToHost));
    const std::chrono::high_resolution_clock::time_point t_l2l_d2h_end = std::chrono::high_resolution_clock::now();
    h_downward_pass_devtree(htree);
    const std::chrono::high_resolution_clock::time_point t_l2l_cpu_end = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_tree, htree.data(), sizeof(TreeNode) * ncells, cudaMemcpyHostToDevice));
    const std::chrono::high_resolution_clock::time_point t_l2l_h2d_end = std::chrono::high_resolution_clock::now();
    ms_l2l_tree_d2h = h_elapsed_ms(t_l2l_start, t_l2l_d2h_end);
    ms_l2l = h_elapsed_ms(t_l2l_d2h_end, t_l2l_cpu_end);
    ms_l2l_tree_h2d = h_elapsed_ms(t_l2l_cpu_end, t_l2l_h2d_end);
#endif
  }

  // 9. GPU eval: L2P (local expansion) + P2P (direct pairs) per particle
  CUDA_CHECK(cudaEventRecord(ev_gpu_start));
  fmm3d_eval_kernel<<<GTAP_BENCH_GRID_SIZE, GTAP_BENCH_BLOCK_SIZE>>>();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(ev_gpu_end));
  CUDA_CHECK(cudaEventSynchronize(ev_gpu_end));
  {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_gpu_start, ev_gpu_end));
    ms_eval = static_cast<double>(ms);
  }

  // 10. D2H results, checksum, optional direct-sum validation
  std::vector<Real> hphi(n), hax(n), hay(n), haz(n);
  const std::chrono::high_resolution_clock::time_point t_d2h_start = std::chrono::high_resolution_clock::now();
  CUDA_CHECK(cudaMemcpy(hphi.data(), d_phi, sizeof(Real) * n, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hax.data(), d_ax, sizeof(Real) * n, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hay.data(), d_ay, sizeof(Real) * n, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(haz.data(), d_az, sizeof(Real) * n, cudaMemcpyDeviceToHost));
  const std::chrono::high_resolution_clock::time_point t_d2h_end = std::chrono::high_resolution_clock::now();
  ms_d2h = h_elapsed_ms(t_d2h_start, t_d2h_end);
  ms_execution = h_elapsed_ms(t_run_start, t_d2h_end);

  double checksum = 0.0;
  double acc_checksum = 0.0;
  int nonzero_phi = 0;
  int nonzero_acc = 0;
  for (int i = 0; i < n; ++i) {
    checksum += hphi[i];
    acc_checksum += static_cast<double>(hax[i]) + hay[i] + haz[i];
    if (hphi[i] != Real(0)) nonzero_phi++;
    if (hax[i] != Real(0) || hay[i] != Real(0) || haz[i] != Real(0)) nonzero_acc++;
  }

  bool sample_direct_validated = false;
  double sample_phi_rel_l2 = 0.0;
  double sample_acc_rel_l2 = 0.0;
  double sample_phi_rel_linf = 0.0;
  double sample_acc_rel_linf = 0.0;
  double sample_phi_rel_p95 = 0.0;
  double sample_acc_rel_p95 = 0.0;
  double sample_phi_rel_p99 = 0.0;
  double sample_acc_rel_p99 = 0.0;
  if (direct_sample_n > 0) {
    const std::chrono::high_resolution_clock::time_point t_sample_direct_start =
        std::chrono::high_resolution_clock::now();
    std::vector<int> sample_idx;
    sample_idx.reserve(direct_sample_n);
    if (direct_sample_n == n) {
      for (int i = 0; i < n; ++i) sample_idx.push_back(i);
    } else {
      std::mt19937_64 rng(static_cast<uint64_t>(DIRECT_SAMPLE_SEED));
      std::uniform_int_distribution<int> dist(0, n - 1);
      while (static_cast<int>(sample_idx.size()) < direct_sample_n) {
        sample_idx.push_back(dist(rng));
        std::sort(sample_idx.begin(), sample_idx.end());
        sample_idx.erase(std::unique(sample_idx.begin(), sample_idx.end()),
                         sample_idx.end());
      }
    }

    int *d_sample_idx = nullptr;
    double *d_phi_diff2 = nullptr, *d_phi_norm2 = nullptr;
    double *d_acc_diff2 = nullptr, *d_acc_norm2 = nullptr;
    double *d_phi_rel_abs = nullptr, *d_acc_rel_abs = nullptr;
    const size_t sample_bytes = sizeof(double) * sample_idx.size();
    CUDA_CHECK(cudaMalloc(&d_sample_idx, sizeof(int) * sample_idx.size()));
    CUDA_CHECK(cudaMalloc(&d_phi_diff2, sample_bytes));
    CUDA_CHECK(cudaMalloc(&d_phi_norm2, sample_bytes));
    CUDA_CHECK(cudaMalloc(&d_acc_diff2, sample_bytes));
    CUDA_CHECK(cudaMalloc(&d_acc_norm2, sample_bytes));
    CUDA_CHECK(cudaMalloc(&d_phi_rel_abs, sample_bytes));
    CUDA_CHECK(cudaMalloc(&d_acc_rel_abs, sample_bytes));
    CUDA_CHECK(cudaMemcpy(d_sample_idx, sample_idx.data(),
                          sizeof(int) * sample_idx.size(), cudaMemcpyHostToDevice));
    fmm3d_direct_sample_validation_kernel<<<static_cast<int>(sample_idx.size()),
                                            DIRECT_SAMPLE_BLOCK_SIZE,
                                            sizeof(double) * 4 * DIRECT_SAMPLE_BLOCK_SIZE>>>(
        d_sample_idx, static_cast<int>(sample_idx.size()),
        d_phi_diff2, d_phi_norm2, d_acc_diff2, d_acc_norm2,
        d_phi_rel_abs, d_acc_rel_abs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> phi_diff2(sample_idx.size()), phi_norm2(sample_idx.size());
    std::vector<double> acc_diff2(sample_idx.size()), acc_norm2(sample_idx.size());
    std::vector<double> phi_rel_abs(sample_idx.size()), acc_rel_abs(sample_idx.size());
    CUDA_CHECK(cudaMemcpy(phi_diff2.data(), d_phi_diff2, sample_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(phi_norm2.data(), d_phi_norm2, sample_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(acc_diff2.data(), d_acc_diff2, sample_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(acc_norm2.data(), d_acc_norm2, sample_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(phi_rel_abs.data(), d_phi_rel_abs, sample_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(acc_rel_abs.data(), d_acc_rel_abs, sample_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_sample_idx);
    cudaFree(d_phi_diff2);
    cudaFree(d_phi_norm2);
    cudaFree(d_acc_diff2);
    cudaFree(d_acc_norm2);
    cudaFree(d_phi_rel_abs);
    cudaFree(d_acc_rel_abs);

    double phi_diff2_sum = 0.0, phi_norm2_sum = 0.0;
    double acc_diff2_sum = 0.0, acc_norm2_sum = 0.0;
    for (size_t i = 0; i < sample_idx.size(); ++i) {
      phi_diff2_sum += phi_diff2[i];
      phi_norm2_sum += phi_norm2[i];
      acc_diff2_sum += acc_diff2[i];
      acc_norm2_sum += acc_norm2[i];
      sample_phi_rel_linf = std::max(sample_phi_rel_linf, phi_rel_abs[i]);
      sample_acc_rel_linf = std::max(sample_acc_rel_linf, acc_rel_abs[i]);
    }
    std::sort(phi_rel_abs.begin(), phi_rel_abs.end());
    std::sort(acc_rel_abs.begin(), acc_rel_abs.end());
    const auto percentile = [](const std::vector<double> &v, double p) {
      if (v.empty()) return 0.0;
      const size_t idx = std::min(v.size() - 1,
                                  static_cast<size_t>(std::ceil(p * v.size())) - 1);
      return v[idx];
    };
    sample_phi_rel_l2 = std::sqrt(phi_diff2_sum / std::max(phi_norm2_sum, 1.0e-300));
    sample_acc_rel_l2 = std::sqrt(acc_diff2_sum / std::max(acc_norm2_sum, 1.0e-300));
    sample_phi_rel_p95 = percentile(phi_rel_abs, 0.95);
    sample_acc_rel_p95 = percentile(acc_rel_abs, 0.95);
    sample_phi_rel_p99 = percentile(phi_rel_abs, 0.99);
    sample_acc_rel_p99 = percentile(acc_rel_abs, 0.99);
    const std::chrono::high_resolution_clock::time_point t_sample_direct_end =
        std::chrono::high_resolution_clock::now();
    ms_sample_direct = h_elapsed_ms(t_sample_direct_start, t_sample_direct_end);
    sample_direct_validated = true;
  }

  bool direct_validated = false;
  double phi_rel_l2 = 0.0;
  double acc_rel_l2 = 0.0;
  if (DIRECT_VALIDATE_N > 0 && n <= DIRECT_VALIDATE_N) {
    const std::chrono::high_resolution_clock::time_point t_direct_start =
        std::chrono::high_resolution_clock::now();
    double phi_diff2 = 0.0;
    double phi_norm2 = 0.0;
    double acc_diff2 = 0.0;
    double acc_norm2 = 0.0;
    #pragma omp parallel for reduction(+:phi_diff2,phi_norm2,acc_diff2,acc_norm2)
    for (int i = 0; i < n; ++i) {
      const double xi = hb.x[i];
      const double yi = hb.y[i];
      const double zi = hb.z[i];
      double phi_ref = 0.0;
      double ax_ref = 0.0;
      double ay_ref = 0.0;
      double az_ref = 0.0;
      for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        const double dx = xi - static_cast<double>(hb.x[j]);
        const double dy = yi - static_cast<double>(hb.y[j]);
        const double dz = zi - static_cast<double>(hb.z[j]);
        const double r2 = dx * dx + dy * dy + dz * dz + static_cast<double>(H_EPS2);
        const double inv_r = 1.0 / std::sqrt(r2);
        const double q_over_r = static_cast<double>(hb.q[j]) * inv_r;
        phi_ref += q_over_r;
        const double inv_r3_q = q_over_r / r2;
        ax_ref -= dx * inv_r3_q;
        ay_ref -= dy * inv_r3_q;
        az_ref -= dz * inv_r3_q;
      }
      const double dphi = static_cast<double>(hphi[i]) - phi_ref;
      const double dax = static_cast<double>(hax[i]) - ax_ref;
      const double day = static_cast<double>(hay[i]) - ay_ref;
      const double daz = static_cast<double>(haz[i]) - az_ref;
      phi_diff2 += dphi * dphi;
      phi_norm2 += phi_ref * phi_ref;
      acc_diff2 += dax * dax + day * day + daz * daz;
      acc_norm2 += ax_ref * ax_ref + ay_ref * ay_ref + az_ref * az_ref;
    }
    const std::chrono::high_resolution_clock::time_point t_direct_end =
        std::chrono::high_resolution_clock::now();
    ms_direct = h_elapsed_ms(t_direct_start, t_direct_end);
    phi_rel_l2 = std::sqrt(phi_diff2 / std::max(phi_norm2, 1.0e-300));
    acc_rel_l2 = std::sqrt(acc_diff2 / std::max(acc_norm2, 1.0e-300));
    direct_validated = true;
  }

  // 11. Print run summary and pipeline timeline
  std::printf("=== exa-style FMM3D run summary ===\n");
  std::printf("Problem: N=%d  Nodes=%d  Leaves=%d  theta=%.2f  NCRIT=%d  kernel=Laplace exa-style P=%d NTERM=%d  precision=%s\n",
              n, ncells, nleaf, theta, NCRIT, EXA_P, EXA_NTERM,
              FMM3D_SINGLE ? "single" : "double");
  std::printf("M2L Cnm constant table: %d entries, %.1f KiB (re+im, P^4=%d)\n",
              M2L_CNM_SIZE,
              static_cast<double>(M2L_CNM_SIZE) * 2.0 * sizeof(Real) / 1024.0,
              EXA_P * EXA_P * EXA_P * EXA_P);
  std::printf("Sph prefactor constant table: %d entries, %.1f KiB (4*P^2=%d)\n",
              EXA_NSPH,
              static_cast<double>(EXA_NSPH) * sizeof(Real) / 1024.0,
              4 * EXA_P * EXA_P);
  std::printf("FMM constant memory total: %.1f KiB / 64 KiB\n",
              static_cast<double>(FMM_CONST_BYTES) / 1024.0);
#if FMM3D_PARTITIONED_FUSED_DTT
  std::printf("Interactions: (partitioned fused DTT; list counters not updated during traversal)\n");
#else
  std::printf("Interactions: M2L=%lld  P2P=%lld\n", m2l_total_ll, p2p_total_ll);
#endif
  std::printf("Validation: phi_checksum=%.9e  acc_checksum=%.9e  nonzero_phi=%d/%d  nonzero_acc=%d/%d\n",
              checksum, acc_checksum, nonzero_phi, n, nonzero_acc, n);
  if (direct_validated) {
    std::printf("Direct check: N=%d  phi_rel_l2=%.9e  acc_rel_l2=%.9e  direct_time=%.3f ms\n",
                n, phi_rel_l2, acc_rel_l2, ms_direct);
  } else {
    std::printf("Direct check: skipped (N=%d > FMM3D_DIRECT_VALIDATE_N=%d)\n",
                n, DIRECT_VALIDATE_N);
  }
  if (sample_direct_validated) {
    std::printf("Sample direct check: samples=%d  phi_rel_l2=%.9e  acc_rel_l2=%.9e  "
                "phi_rel_linf=%.9e  acc_rel_linf=%.9e  "
                "phi_p95=%.9e  acc_p95=%.9e  phi_p99=%.9e  acc_p99=%.9e  "
                "sample_direct_time=%.3f ms\n",
                direct_sample_n, sample_phi_rel_l2, sample_acc_rel_l2,
                sample_phi_rel_linf, sample_acc_rel_linf,
                sample_phi_rel_p95, sample_acc_rel_p95,
                sample_phi_rel_p99, sample_acc_rel_p99, ms_sample_direct);
  } else {
    std::printf("Sample direct check: disabled (FMM3D_DIRECT_SAMPLE_N=0)\n");
  }

  std::printf("\n=== Pipeline timeline (ms) ===\n");
  // Chronological order (shared across GTaP / worklist / Host OMP for stack plots).
#if FMM3D_GPU_TREE_BUILD
  std::printf("  [GPU+CPU]     build adaptive tree, Morton sort assisted: %.3f\n", ms_tree_build);
#else
  std::printf("  [CPU]         build adaptive tree: %.3f\n", ms_tree_build);
#endif
#if !FMM3D_GPU_UPWARD
  std::printf("  [CPU]         upward pass (P2M/M2M): %.3f\n", ms_upward);
#endif
  std::printf("  [CPU]         build traversal metadata lists: %.3f\n", ms_metadata);
  std::printf("  [H2D+bind]    upload static arrays and bind symbols: %.3f\n", ms_h2d_bind);
#if FMM3D_GPU_UPWARD
  std::printf("  [GPU]         upward pass (P2M/M2M), level-by-level: %.3f\n", ms_upward);
#endif
  std::printf("  [DTT init]    fixed-cap buffers + runtime setup: %.3f\n", ms_dtt_init);
#if FMM3D_WORKLIST_DTT
  std::printf("  [GPU]         DTT worklist setup: %.3f\n", ms_dtt_fill_setup);
  std::printf("  [GPU]         DTT traversal core: %.3f\n", ms_dtt_core);
  std::printf("  [GPU]         DTT post/check: %.3f\n", ms_dtt_fill_post);
#elif FMM3D_ENABLE_GTAP_DTT
  std::printf("  [GPU]         DTT setup: %.3f\n", ms_dtt_fill_setup);
#if FMM3D_PARTITIONED_FUSED_DTT
  std::printf("  [GPU/GTaP]    DTT exafmm-style range split + fused M2L/P2P: %.3f\n",
              ms_dtt_construct_handoff);
#elif FMM3D_GTAP_FUSED
  std::printf("  [GPU/GTaP]    DTT traversal + fused M2L/P2P: %.3f\n", ms_dtt_construct_handoff);
#else
  std::printf("  [GPU/GTaP]    DTT traversal core: %.3f\n", ms_dtt_fill_kernel);
#endif
  std::printf("  [GPU]         DTT post/check: %.3f\n", ms_dtt_fill_post);
#else
#if FMM3D_SERIAL_HOST_DTT
  std::printf("  [CPU/Serial]  DTT reset/counts: %.3f\n",
              host_dtt_breakdown.ms_reset_counts);
  std::printf("  [CPU/Serial]  DTT traversal core: %.3f\n", ms_dtt_core);
  std::printf("  [CPU/Serial]  DTT list stats: %.3f\n", host_dtt_breakdown.ms_stats);
#else
  std::printf("  [CPU/OpenMP]  DTT reset/counts: %.3f\n",
              host_dtt_breakdown.ms_reset_counts);
  std::printf("  [CPU/OpenMP]  DTT traversal core: %.3f\n", ms_dtt_core);
  std::printf("  [CPU/OpenMP]  DTT list stats: %.3f\n", host_dtt_breakdown.ms_stats);
#endif
  std::printf("  [H2D]         DTT list counts: %.3f\n",
              host_dtt_breakdown.ms_h2d_counts);
  std::printf("  [H2D]         DTT list payloads: %.3f\n",
              host_dtt_breakdown.ms_h2d_lists);
#endif
  std::printf("  [GPU]         M2L kernel: %.3f\n", ms_m2l);
#if FMM3D_GPU_L2L
  std::printf("  [GPU]         L2L downward pass, level-by-level: %.3f\n", ms_l2l);
#else
  std::printf("  [D2H]         copy tree for L2L: %.3f\n", ms_l2l_tree_d2h);
  std::printf("  [CPU]         L2L downward pass: %.3f\n", ms_l2l);
  std::printf("  [H2D]         copy tree after L2L: %.3f\n", ms_l2l_tree_h2d);
#endif
  std::printf("  [GPU]         L2P + P2P eval kernel: %.3f\n", ms_eval);
  std::printf("  [D2H]         copy phi/acc result arrays: %.3f\n", ms_d2h);
  std::printf("\n=== DTT comparison focus ===\n");
  std::printf("DTT core traversal: %.3f ms\n", ms_dtt_core);
  std::printf("DTT list construction (steady-state): %.3f ms\n", ms_dtt_construct);
#if !FMM3D_WORKLIST_DTT && !FMM3D_ENABLE_GTAP_DTT
  std::printf("DTT GPU handoff (H2D lists): %.3f ms\n",
              host_dtt_breakdown.ms_h2d_counts + host_dtt_breakdown.ms_h2d_lists);
#endif
  std::printf("DTT list construction + GPU handoff: %.3f ms\n",
              ms_dtt_construct_handoff);
  std::printf("DTT init: %.3f ms\n", ms_dtt_init);
  std::printf("DTT + init: %.3f ms\n", ms_dtt_construct_handoff + ms_dtt_init);
#if FMM3D_ENABLE_GTAP_DTT
  std::printf("  (GTaP runtime subset: %.3f ms)\n", ms_gtap_init);
#elif FMM3D_WORKLIST_DTT
  std::printf("  (worklist pair-buffer subset: %.3f ms)\n", ms_worklist_pair_buffer_alloc);
#else
  std::printf("  (host list-buffer subset: %.3f ms)\n", ms_host_list_buffer_alloc);
#endif
  std::printf("DTT fixed-cap metrics: m2l_cap=%d  p2p_cap=%d  max_m2l=%d  max_p2p=%d  overflow=%d\n",
              DTT_M2L_CAP, DTT_P2P_CAP, max_m2l_count, max_p2p_count, h_dtt_overflow);
#if FMM3D_M2L_COMPACT_TARGETS
  std::printf("M2L target cells: %d/%d\n", m2l_target_count, ncells);
#endif
#if FMM3D_WORKLIST_DTT
  std::printf("DTT worklist metrics: pair_cap=%d  iterations=%d  max_frontier=%d  overflow=%d\n",
              worklist_pair_cap, worklist_iterations, worklist_max_frontier, worklist_overflow);
  std::printf("DTT mode: GPU worklist fixed-cap lists (level-synchronous)\n");
#elif FMM3D_ENABLE_GTAP_DTT
#if FMM3D_PARTITIONED_FUSED_DTT
  std::printf("DTT mode: GTaP partitioned fused traversal (exafmm-style dualTreeTraversal + child-range TraverseRange; no atomics on L/phi)\n");
#elif FMM3D_GTAP_FUSED
  std::printf("DTT mode: GTaP fused traversal (M2L/P2P during DTT; M2L kernel skipped)\n");
#else
  std::printf("DTT mode: GTaP task traversal, GPU fixed-cap lists\n");
#endif
#else
#if FMM3D_SERIAL_HOST_DTT
  std::printf("DTT mode: CPU serial host fixed-cap lists, then H2D to GPU\n");
#else
  std::printf("DTT mode: CPU/OpenMP host fixed-cap lists, then H2D to GPU\n");
#endif
#endif
  std::printf("\n=== Totals ===\n");
  std::printf("Execution time: %.3f ms\n", ms_execution);

  // 12. Free device memory, GTaP finalize
  cudaFree(d_tree);
  cudaFree(d_bx);
  cudaFree(d_by);
  cudaFree(d_bz);
  cudaFree(d_bq);
  cudaFree(d_phi);
  cudaFree(d_ax);
  cudaFree(d_ay);
  cudaFree(d_az);
  cudaFree(d_leaf_idx);
  cudaFree(d_body_cell);
  cudaFree(d_l2l_cells);
  cudaFree(d_upward_cells);
  cudaFree(d_m2l_count);
  cudaFree(d_p2p_count);
  cudaFree(d_m2l_src);
  cudaFree(d_p2p_src);
  cudaFree(d_m2l_targets);
  cudaFree(d_dtt_overflow);
#if FMM3D_PARTITIONED_FUSED_DTT
  cudaFree(d_child_lo);
  cudaFree(d_child_hi);
#endif
  cudaEventDestroy(ev_gpu_start);
  cudaEventDestroy(ev_gpu_end);
#if FMM3D_ENABLE_GTAP_DTT
  gtap_finalize();
#endif
  return 0;
}

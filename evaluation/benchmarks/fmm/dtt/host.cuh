#pragma once

#include "../fmm_types.cuh"

#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <vector>

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

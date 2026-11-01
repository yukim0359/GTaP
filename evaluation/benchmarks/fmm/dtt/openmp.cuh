#pragma once

#include "../host_cuda.hpp"
#include "../host_types.hpp"
#include "common.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

static inline void h_fixed_append_m2l(HostFixedLists &lists, int ci, int cj) {
  int pos;
  #pragma omp atomic capture
  pos = lists.m2l_count[ci]++;
  if (pos < DTT_M2L_CAP) {
    lists.m2l_src[static_cast<size_t>(ci) * DTT_M2L_CAP + pos] = cj;
  } else {
    #pragma omp atomic write
    lists.overflow = 1;
  }
}

static inline void h_fixed_append_p2p(HostFixedLists &lists, int ci, int cj) {
  int pos;
  #pragma omp atomic capture
  pos = lists.p2p_count[ci]++;
  if (pos < DTT_P2P_CAP) {
    lists.p2p_src[static_cast<size_t>(ci) * DTT_P2P_CAP + pos] = cj;
  } else {
    #pragma omp atomic write
    lists.overflow = 1;
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
        const bool spawn = (depth < HOST_DTT_TASK_DEPTH) &&
                           (tree[child_i].nbody >= HOST_DTT_TASK_MIN_N);
        for (int oj = 0; oj < 8; ++oj) {
          const int child_j = Ci.child[oj];
          if (child_j == -1) continue;
          if (spawn) {
            #pragma omp task default(none) firstprivate(child_i, child_j, theta, depth) \
                               shared(tree, lists)
            {
              h_dtt_fixed(child_i, child_j, tree, lists, theta, depth + 1);
            }
          } else {
            h_dtt_fixed(child_i, child_j, tree, lists, theta, depth + 1);
          }
        }
      }
    }
    return;
  }

  const TreeNode &Cj = tree[cj];
  if (dtt_well_separated(Ci, Cj, theta)) {
    h_fixed_append_m2l(lists, ci, cj);
    return;
  }
  if (Ci.is_leaf && Cj.is_leaf) {
    h_fixed_append_p2p(lists, ci, cj);
    return;
  }

  {
    const bool split_ci = dtt_split_ci(Ci, Cj);
    for (int oct = 0; oct < 8; ++oct) {
      const int child_i = split_ci ? Ci.child[oct] : ci;
      const int child_j = split_ci ? cj : Cj.child[oct];
      if (split_ci ? (child_i == -1) : (child_j == -1)) continue;
      const int nb_lhs = split_ci ? tree[child_i].nbody : Ci.nbody;
      const int nb_rhs = split_ci ? Cj.nbody : tree[child_j].nbody;
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
    }
  }
}

static void h_prepare_fixed_lists(HostFixedLists &lists, int ncells) {
  lists.m2l_count.assign(ncells, 0);
  lists.p2p_count.assign(ncells, 0);
  const size_t m2l_size = static_cast<size_t>(ncells) * DTT_M2L_CAP;
  const size_t p2p_size = static_cast<size_t>(ncells) * DTT_P2P_CAP;
  lists.m2l_src.reset(new int[m2l_size]);
  lists.p2p_src.reset(new int[p2p_size]);
  lists.overflow = 0;
}

static void h_reset_fixed_lists(HostFixedLists &lists) {
  std::fill(lists.m2l_count.begin(), lists.m2l_count.end(), 0);
  std::fill(lists.p2p_count.begin(), lists.p2p_count.end(), 0);
  lists.overflow = 0;
}

static void h_aggregate_fixed_list_stats(FmmDttState &dtt) {
  const HostFixedLists &lists = dtt.host_fixed_lists;
  const int ncells = static_cast<int>(lists.m2l_count.size());
  dtt.m2l_total_ll = 0;
  dtt.p2p_total_ll = 0;
  dtt.max_m2l_count = 0;
  dtt.max_p2p_count = 0;
  for (int ci = 0; ci < ncells; ++ci) {
    dtt.m2l_total_ll += lists.m2l_count[ci];
    dtt.p2p_total_ll += lists.p2p_count[ci];
    dtt.max_m2l_count = std::max(dtt.max_m2l_count, lists.m2l_count[ci]);
    dtt.max_p2p_count = std::max(dtt.max_p2p_count, lists.p2p_count[ci]);
  }
}

static void h_run_host_dtt_traversal(const std::vector<TreeNode> &tree, Real theta,
                                    FmmDttState &dtt) {
  HostFixedLists &lists = dtt.host_fixed_lists;
  HostDttBreakdown &breakdown = dtt.host_dtt_breakdown;
  const auto t_reset_start = std::chrono::high_resolution_clock::now();
  h_reset_fixed_lists(lists);
  const auto t_reset_end = std::chrono::high_resolution_clock::now();
  const auto t_traverse_start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel
  {
    #pragma omp single nowait
    h_dtt_fixed(0, 0, tree, lists, theta, 0);
  }
  const auto t_traverse_end = std::chrono::high_resolution_clock::now();
  const auto t_stats_start = std::chrono::high_resolution_clock::now();
  h_aggregate_fixed_list_stats(dtt);
  const auto t_stats_end = std::chrono::high_resolution_clock::now();
  breakdown.ms_reset_counts =
      std::chrono::duration<double, std::milli>(t_reset_end - t_reset_start).count();
  breakdown.ms_traverse =
      std::chrono::duration<double, std::milli>(t_traverse_end - t_traverse_start).count();
  breakdown.ms_stats =
      std::chrono::duration<double, std::milli>(t_stats_end - t_stats_start).count();
}

static bool init_dtt_openmp(FmmRun &run) {
  const int ncells = run.host.ncells;
  const std::chrono::high_resolution_clock::time_point t_buffer_alloc_start =
      std::chrono::high_resolution_clock::now();
  h_prepare_fixed_lists(run.dtt.host_fixed_lists, ncells);
  const std::chrono::high_resolution_clock::time_point t_buffer_alloc_end =
      std::chrono::high_resolution_clock::now();
  run.dtt.host_dtt_breakdown.ms_buffer_alloc =
      h_elapsed_ms(t_buffer_alloc_start, t_buffer_alloc_end);
  run.timings.ms_host_list_buffer_alloc = run.dtt.host_dtt_breakdown.ms_buffer_alloc;
  return true;
}

static bool run_dtt_openmp(FmmRun &run) {
  const int ncells = run.host.ncells;
  FmmDeviceBuffers &d = run.device;
  FmmDttState &dtt = run.dtt;
  FmmTimings &t = run.timings;

  const std::chrono::high_resolution_clock::time_point t_dtt_total_start =
      std::chrono::high_resolution_clock::now();
  h_run_host_dtt_traversal(run.host.htree, run.options.theta, dtt);
  dtt.overflow = dtt.host_fixed_lists.overflow;
  if (dtt.overflow) {
    std::printf("Host 3D DTT fixed-cap overflow: m2l_cap=%d max_m2l=%d  p2p_cap=%d max_p2p=%d\n",
                DTT_M2L_CAP, dtt.max_m2l_count, DTT_P2P_CAP, dtt.max_p2p_count);
    return false;
  }
  const std::chrono::high_resolution_clock::time_point t_h2d_counts_start =
      std::chrono::high_resolution_clock::now();
  CUDA_CHECK(cudaMemcpy(d.d_m2l_count, dtt.host_fixed_lists.m2l_count.data(),
                        sizeof(int) * ncells, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.d_p2p_count, dtt.host_fixed_lists.p2p_count.data(),
                        sizeof(int) * ncells, cudaMemcpyHostToDevice));
  const std::chrono::high_resolution_clock::time_point t_h2d_counts_end =
      std::chrono::high_resolution_clock::now();
  const std::chrono::high_resolution_clock::time_point t_h2d_lists_start =
      std::chrono::high_resolution_clock::now();
  CUDA_CHECK(cudaMemcpy(d.d_m2l_src, dtt.host_fixed_lists.m2l_src.get(),
                        sizeof(int) * static_cast<size_t>(ncells) * DTT_M2L_CAP,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.d_p2p_src, dtt.host_fixed_lists.p2p_src.get(),
                        sizeof(int) * static_cast<size_t>(ncells) * DTT_P2P_CAP,
                        cudaMemcpyHostToDevice));
  const std::chrono::high_resolution_clock::time_point t_h2d_lists_end =
      std::chrono::high_resolution_clock::now();
  dtt.host_dtt_breakdown.ms_h2d_counts = h_elapsed_ms(t_h2d_counts_start, t_h2d_counts_end);
  dtt.host_dtt_breakdown.ms_h2d_lists = h_elapsed_ms(t_h2d_lists_start, t_h2d_lists_end);
  const std::chrono::high_resolution_clock::time_point t_dtt_total_end =
      std::chrono::high_resolution_clock::now();
  t.ms_dtt_fill_setup = dtt.host_dtt_breakdown.ms_reset_counts;
  t.ms_dtt_fill_post = dtt.host_dtt_breakdown.ms_stats + dtt.host_dtt_breakdown.ms_h2d_counts +
                     dtt.host_dtt_breakdown.ms_h2d_lists;
  t.ms_dtt_core = dtt.host_dtt_breakdown.ms_traverse;
  t.ms_dtt_construct = dtt.host_dtt_breakdown.ms_reset_counts + dtt.host_dtt_breakdown.ms_traverse +
                     dtt.host_dtt_breakdown.ms_stats;
  t.ms_dtt_construct_handoff = h_elapsed_ms(t_dtt_total_start, t_dtt_total_end);

  return true;
}

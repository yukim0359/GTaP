#pragma once

#include "../host_cuda.hpp"
#include "../host_timing.hpp"
#include "../host_types.hpp"
#include "host.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>

static bool init_dtt_host(FmmRun &run) {
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

static bool run_dtt_host(FmmRun &run) {
  const int ncells = run.host.ncells;
  FmmDeviceBuffers &d = run.device;
  FmmDttState &dtt = run.dtt;
  FmmTimings &t = run.timings;

  const std::chrono::high_resolution_clock::time_point t_dtt_total_start =
      std::chrono::high_resolution_clock::now();
  h_run_host_dtt_traversal(run.host.htree, run.options.theta, dtt.host_fixed_lists, &dtt.host_dtt_breakdown);

  dtt.overflow = dtt.host_fixed_lists.overflow;
  dtt.max_m2l_count = dtt.host_fixed_lists.max_m2l;
  dtt.max_p2p_count = dtt.host_fixed_lists.max_p2p;
  dtt.m2l_total_ll = dtt.host_fixed_lists.total_m2l;
  dtt.p2p_total_ll = dtt.host_fixed_lists.total_p2p;
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
  t.ms_dtt_fill_kernel = dtt.host_dtt_breakdown.ms_traverse;
  t.ms_dtt_fill_post = dtt.host_dtt_breakdown.ms_stats + dtt.host_dtt_breakdown.ms_h2d_counts +
                     dtt.host_dtt_breakdown.ms_h2d_lists;
  t.ms_dtt = h_elapsed_ms(t_dtt_total_start, t_dtt_total_end);
  t.ms_dtt_core = dtt.host_dtt_breakdown.ms_traverse;
  t.ms_dtt_construct = dtt.host_dtt_breakdown.ms_reset_counts + dtt.host_dtt_breakdown.ms_traverse +
                     dtt.host_dtt_breakdown.ms_stats;
  t.ms_dtt_construct_handoff = t.ms_dtt;

  return true;
}

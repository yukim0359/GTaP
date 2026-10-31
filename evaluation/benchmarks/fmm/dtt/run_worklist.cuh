#pragma once

#ifndef FMM3D_WORKLIST_DTT
#define FMM3D_WORKLIST_DTT 1
#endif

#include "../host_cuda.hpp"
#include "../host_timing.hpp"
#include "../host_types.hpp"
#include "device.cuh"
#include "host.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>

static bool prepare_dtt_worklist(FmmRun &run) {
  FmmDeviceBuffers &d = run.device;
  const int ncells = run.host.ncells;
  d.pair_cap_size =
      static_cast<size_t>(std::max(1, ncells)) *
      static_cast<size_t>(std::max(1, WORKLIST_PAIR_CAP_FACTOR));
  if (d.pair_cap_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    std::printf("GPU worklist DTT pair cap too large for int counter: ncells=%d factor=%d cap=%zu\n",
                ncells, WORKLIST_PAIR_CAP_FACTOR, d.pair_cap_size);
    return false;
  }
  d.worklist_pair_cap = static_cast<int>(d.pair_cap_size);
  return true;
}

static bool init_dtt_worklist(FmmRun &run) {
  FmmDeviceBuffers &d = run.device;
  const std::chrono::high_resolution_clock::time_point t_wl_alloc_start =
      std::chrono::high_resolution_clock::now();
  CUDA_CHECK(cudaMalloc(&d.d_pairs_a, sizeof(DttPair) * d.pair_cap_size));
  CUDA_CHECK(cudaMalloc(&d.d_pairs_b, sizeof(DttPair) * d.pair_cap_size));
  CUDA_CHECK(cudaMalloc(&d.d_worklist_out_count, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d.d_worklist_overflow, sizeof(int)));
  const std::chrono::high_resolution_clock::time_point t_wl_alloc_end =
      std::chrono::high_resolution_clock::now();
  run.timings.ms_worklist_pair_buffer_alloc =
      h_elapsed_ms(t_wl_alloc_start, t_wl_alloc_end);
  return true;
}

static bool run_dtt_worklist(FmmRun &run) {
  const int ncells = run.host.ncells;
  FmmDeviceBuffers &d = run.device;
  FmmDttState &dtt = run.dtt;
  FmmTimings &t = run.timings;

  const std::chrono::high_resolution_clock::time_point t_dtt_total_start =
      std::chrono::high_resolution_clock::now();
  CUDA_CHECK(cudaMemset(d.d_m2l_count, 0, sizeof(int) * ncells));
  CUDA_CHECK(cudaMemset(d.d_p2p_count, 0, sizeof(int) * ncells));
  CUDA_CHECK(cudaMemset(d.d_dtt_overflow, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d.d_worklist_overflow, 0, sizeof(int)));
  {
    DttPair root{0, 0};
    CUDA_CHECK(cudaMemcpy(d.d_pairs_a, &root, sizeof(DttPair), cudaMemcpyHostToDevice));
  }

  const std::chrono::high_resolution_clock::time_point t_dtt_kernel_start =
      std::chrono::high_resolution_clock::now();
  DttPair *d_in_pairs = d.d_pairs_a;
  DttPair *d_out_pairs = d.d_pairs_b;
  int in_count = 1;
  dtt.worklist_max_frontier = 1;
  while (in_count > 0) {
    CUDA_CHECK(cudaMemset(d.d_worklist_out_count, 0, sizeof(int)));
    int blocks = std::max(
        1, (in_count + GTAP_BLOCK_SIZE - 1) /
               GTAP_BLOCK_SIZE);
    fmm3d_worklist_expand_kernel<<<blocks, GTAP_BLOCK_SIZE>>>(
        d_in_pairs, in_count, d_out_pairs, d.d_worklist_out_count,
        d.worklist_pair_cap, d.d_worklist_overflow);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int next_count = 0;
    CUDA_CHECK(cudaMemcpy(&next_count, d.d_worklist_out_count, sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&dtt.worklist_overflow, d.d_worklist_overflow, sizeof(int),
                          cudaMemcpyDeviceToHost));
    ++dtt.worklist_iterations;
    dtt.worklist_max_frontier = std::max(dtt.worklist_max_frontier, next_count);
    if (dtt.worklist_overflow || next_count > d.worklist_pair_cap) {
      std::printf("GPU worklist DTT pair overflow: pair_cap=%d next_frontier=%d iter=%d\n",
                  d.worklist_pair_cap, next_count, dtt.worklist_iterations);
      return false;
    }
    DttPair *tmp = d_in_pairs;
    d_in_pairs = d_out_pairs;
    d_out_pairs = tmp;
    in_count = next_count;
  }
  const std::chrono::high_resolution_clock::time_point t_dtt_kernel_end =
      std::chrono::high_resolution_clock::now();

  CUDA_CHECK(h_copy_dtt_counts_from_device(
      d.d_m2l_count, d.d_p2p_count, d.d_dtt_overflow, ncells,
      dtt.overflow, dtt.max_m2l_count, dtt.max_p2p_count,
      dtt.m2l_total_ll, dtt.p2p_total_ll));
  if (dtt.overflow) {
    std::printf("3D worklist DTT fixed-cap overflow: m2l_cap=%d max_m2l=%d  p2p_cap=%d max_p2p=%d\n",
                DTT_M2L_CAP, dtt.max_m2l_count, DTT_P2P_CAP, dtt.max_p2p_count);
    std::printf("DTT worklist metrics (until overflow): pair_cap=%d  iterations=%d  max_frontier=%d  overflow=%d\n",
                d.worklist_pair_cap, dtt.worklist_iterations, dtt.worklist_max_frontier, dtt.worklist_overflow);
    return false;
  }
  const std::chrono::high_resolution_clock::time_point t_dtt_total_end =
      std::chrono::high_resolution_clock::now();
  t.ms_dtt_fill_setup = h_elapsed_ms(t_dtt_total_start, t_dtt_kernel_start);
  t.ms_dtt_fill_kernel = h_elapsed_ms(t_dtt_kernel_start, t_dtt_kernel_end);
  t.ms_dtt_fill_post = h_elapsed_ms(t_dtt_kernel_end, t_dtt_total_end);
  t.ms_dtt = h_elapsed_ms(t_dtt_total_start, t_dtt_total_end);
  t.ms_dtt_core = t.ms_dtt_fill_kernel;
  t.ms_dtt_construct = t.ms_dtt_fill_setup + t.ms_dtt_fill_kernel;
  t.ms_dtt_construct_handoff = t.ms_dtt_construct;
  cudaFree(d.d_pairs_a);
  cudaFree(d.d_pairs_b);
  cudaFree(d.d_worklist_out_count);
  cudaFree(d.d_worklist_overflow);
  d.d_pairs_a = nullptr;
  d.d_pairs_b = nullptr;
  d.d_worklist_out_count = nullptr;
  d.d_worklist_overflow = nullptr;

  return true;
}

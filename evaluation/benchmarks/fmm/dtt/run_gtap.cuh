#pragma once

#include "../host_cuda.hpp"
#include "../host_timing.hpp"
#include "../host_types.hpp"
#include "device.cuh"
#include "host.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include "gtap_thread.cuh"

static int run_dtt_gtap(FmmRun &run) {
  [[maybe_unused]] const int ncells = run.host.ncells;
  FmmDeviceBuffers &d = run.device;
  [[maybe_unused]] FmmDttStats &dtt = run.dtt;
  FmmTimings &t = run.timings;

#if FMM3D_GTAP_FUSED
  CUDA_CHECK(cudaMemset(d.d_phi, 0, sizeof(Real) * run.options.n));
  CUDA_CHECK(cudaMemset(d.d_ax, 0, sizeof(Real) * run.options.n));
  CUDA_CHECK(cudaMemset(d.d_ay, 0, sizeof(Real) * run.options.n));
  CUDA_CHECK(cudaMemset(d.d_az, 0, sizeof(Real) * run.options.n));
#endif
  const std::chrono::high_resolution_clock::time_point t_dtt_total_start =
      std::chrono::high_resolution_clock::now();
#if !FMM3D_PARTITIONED_FUSED_DTT
  CUDA_CHECK(cudaMemset(d.d_m2l_count, 0, sizeof(int) * ncells));
  CUDA_CHECK(cudaMemset(d.d_p2p_count, 0, sizeof(int) * ncells));
  CUDA_CHECK(cudaMemset(d.d_dtt_overflow, 0, sizeof(int)));
#endif

  const std::chrono::high_resolution_clock::time_point t_dtt_kernel_start =
      std::chrono::high_resolution_clock::now();
  CUDA_CHECK(gtap_launch(fmm3d_dtt_fill_kernel));
  CUDA_CHECK(cudaGetLastError());
  gtap_synchronize();
  const std::chrono::high_resolution_clock::time_point t_dtt_kernel_end =
      std::chrono::high_resolution_clock::now();

#if !FMM3D_PARTITIONED_FUSED_DTT
  CUDA_CHECK(h_copy_dtt_counts_from_device(
      d.d_m2l_count, d.d_p2p_count, d.d_dtt_overflow, ncells,
      dtt.overflow, dtt.max_m2l_count, dtt.max_p2p_count,
      dtt.m2l_total_ll, dtt.p2p_total_ll, dtt.h_m2l_targets, dtt.m2l_target_count));
  if (dtt.overflow) {
    std::printf("3D DTT fixed-cap overflow: m2l_cap=%d max_m2l=%d  p2p_cap=%d max_p2p=%d\n",
                DTT_M2L_CAP, dtt.max_m2l_count, DTT_P2P_CAP, dtt.max_p2p_count);
    return 1;
  }
#endif
  const std::chrono::high_resolution_clock::time_point t_dtt_total_end =
      std::chrono::high_resolution_clock::now();
  t.ms_dtt_fill_setup = h_elapsed_ms(t_dtt_total_start, t_dtt_kernel_start);
  t.ms_dtt_fill_kernel = h_elapsed_ms(t_dtt_kernel_start, t_dtt_kernel_end);
  t.ms_dtt_fill_post = h_elapsed_ms(t_dtt_kernel_end, t_dtt_total_end);
  t.ms_dtt = h_elapsed_ms(t_dtt_total_start, t_dtt_total_end);
  t.ms_dtt_core = t.ms_dtt_fill_kernel;
  t.ms_dtt_construct = t.ms_dtt_fill_setup + t.ms_dtt_fill_kernel;
  t.ms_dtt_construct_handoff = t.ms_dtt_construct;

  return 0;
}

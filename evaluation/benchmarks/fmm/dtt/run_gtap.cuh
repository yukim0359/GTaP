#pragma once

#include "../host_cuda.hpp"
#include "../host_options.hpp"
#include "../host_timing.hpp"
#include "../host_types.hpp"
#include "device.cuh"
#include "host.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include "gtap_thread.cuh"

static bool init_dtt_gtap(FmmRun &run) {
  const std::chrono::high_resolution_clock::time_point t_gtap_init_start =
      std::chrono::high_resolution_clock::now();
  gtap_thread_config config{
      .grid_size = GTAP_GRID_SIZE,
      .block_size = GTAP_BLOCK_SIZE,
      .max_tasks_per_warp = GTAP_MAX_TASKS_PER_WARP,
      .num_queues = 1,
      .profile_capacity_per_warp = GTAP_PROFILE_CAPACITY,
  };
  if (!apply_launch_env(config.grid_size, config.block_size, config.max_tasks_per_warp)) {
    return false;
  }
  std::printf("GTaP config: grid=%d block=%d max_tasks_per_warp=%d "
              "num_queues=%d\n",
              config.grid_size, config.block_size,
              config.max_tasks_per_warp, config.num_queues);
  size_t gtap_device_bytes = 0;
  CUDA_CHECK(gtap_initialize(config, &gtap_device_bytes));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::printf("GTaP runtime allocation: %.2f GiB\n",
              static_cast<double>(gtap_device_bytes) /
                  (1024.0 * 1024.0 * 1024.0));
  const std::chrono::high_resolution_clock::time_point t_gtap_init_end =
      std::chrono::high_resolution_clock::now();
  run.timings.ms_gtap_init = h_elapsed_ms(t_gtap_init_start, t_gtap_init_end);
  return true;
}

static bool run_dtt_gtap(FmmRun &run) {
  [[maybe_unused]] const int ncells = run.host.ncells;
  FmmDeviceBuffers &d = run.device;
  [[maybe_unused]] FmmDttState &dtt = run.dtt;
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
  if (gtap_synchronize() != cudaSuccess) return false;
  const std::chrono::high_resolution_clock::time_point t_dtt_kernel_end =
      std::chrono::high_resolution_clock::now();

#if !FMM3D_PARTITIONED_FUSED_DTT
  CUDA_CHECK(h_copy_dtt_counts_from_device(
      d.d_m2l_count, d.d_p2p_count, d.d_dtt_overflow, ncells,
      dtt.overflow, dtt.max_m2l_count, dtt.max_p2p_count,
      dtt.m2l_total_ll, dtt.p2p_total_ll));
  if (dtt.overflow) {
    std::printf("3D DTT fixed-cap overflow: m2l_cap=%d max_m2l=%d  p2p_cap=%d max_p2p=%d\n",
                DTT_M2L_CAP, dtt.max_m2l_count, DTT_P2P_CAP, dtt.max_p2p_count);
    return false;
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

  return true;
}

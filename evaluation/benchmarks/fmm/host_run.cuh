#pragma once

#include "device_bind.cuh"
#include "dtt/device.cuh"
#include "dtt/host.cuh"
#if FMM3D_ENABLE_GTAP_DTT
#include "dtt/run_gtap.cuh"
#elif FMM3D_WORKLIST_DTT
#include "dtt/run_worklist.cuh"
#else
#include "dtt/run_host.cuh"
#endif
#include "gtap_thread.cuh"
#include "host_cuda.hpp"
#include "host_timing.hpp"
#include "host_types.hpp"
#include "numerics/host_l2l.hpp"
#include "numerics/host_upward.hpp"
#include "numerics/kernels.cuh"

#include <algorithm>
#include <cstdio>
#include <limits>

static int run_cpu_upward(FmmRun &run) {
#if !FMM3D_GPU_UPWARD
  const std::chrono::high_resolution_clock::time_point t_upward_start =
      std::chrono::high_resolution_clock::now();
  h_upward_pass(run.host.htree, run.host.hb);
  const std::chrono::high_resolution_clock::time_point t_upward_end =
      std::chrono::high_resolution_clock::now();
  run.timings.ms_upward = h_elapsed_ms(t_upward_start, t_upward_end);
#else
  (void)run;
#endif
  return 0;
}

static int run_gpu_upward(FmmRun &run) {
#if FMM3D_GPU_UPWARD
  const std::chrono::high_resolution_clock::time_point t_upward_start =
      std::chrono::high_resolution_clock::now();
  constexpr int UPWARD_BLOCK_SIZE = 128;
  const int nleaf = run.host.nleaf;
  if (nleaf > 0) {
    int blocks = std::max(1, (nleaf + UPWARD_BLOCK_SIZE - 1) / UPWARD_BLOCK_SIZE);
    fmm3d_p2m_leaf_kernel<<<blocks, UPWARD_BLOCK_SIZE>>>(run.device.d_leaf_idx, nleaf);
    CUDA_CHECK(cudaGetLastError());
  }
  for (int depth = run.host.max_cell_depth; depth >= 0; --depth) {
    const int begin = run.host.upward_level_offsets[depth];
    const int end = run.host.upward_level_offsets[depth + 1];
    const int count = end - begin;
    if (count == 0) continue;
    int blocks = (count + UPWARD_BLOCK_SIZE - 1) / UPWARD_BLOCK_SIZE;
    blocks = std::max(1, blocks);
    fmm3d_m2m_level_kernel<<<blocks, UPWARD_BLOCK_SIZE>>>(run.device.d_upward_cells + begin, count);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  const std::chrono::high_resolution_clock::time_point t_upward_end =
      std::chrono::high_resolution_clock::now();
  run.timings.ms_upward = h_elapsed_ms(t_upward_start, t_upward_end);
#else
  (void)run;
#endif
  return 0;
}

static int init_dtt(FmmRun &run) {
  const int ncells = run.host.ncells;
  const Real theta = run.options.theta;
  FmmDeviceBuffers &d = run.device;

#if FMM3D_WORKLIST_DTT
  d.pair_cap_size =
      static_cast<size_t>(std::max(1, ncells)) *
      static_cast<size_t>(std::max(1, WORKLIST_PAIR_CAP_FACTOR));
  if (d.pair_cap_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    std::printf("GPU worklist DTT pair cap too large for int counter: ncells=%d factor=%d cap=%zu\n",
                ncells, WORKLIST_PAIR_CAP_FACTOR, d.pair_cap_size);
    return 1;
  }
  d.worklist_pair_cap = static_cast<int>(d.pair_cap_size);
#endif

  const std::chrono::high_resolution_clock::time_point t_dtt_init_start =
      std::chrono::high_resolution_clock::now();
  CUDA_CHECK(cudaMalloc(&d.d_m2l_count, sizeof(int) * ncells));
  CUDA_CHECK(cudaMalloc(&d.d_p2p_count, sizeof(int) * ncells));
  CUDA_CHECK(cudaMalloc(&d.d_m2l_src, sizeof(int) * static_cast<size_t>(ncells) * DTT_M2L_CAP));
  CUDA_CHECK(cudaMalloc(&d.d_p2p_src, sizeof(int) * static_cast<size_t>(ncells) * DTT_P2P_CAP));
  CUDA_CHECK(cudaMalloc(&d.d_dtt_overflow, sizeof(int)));
#if FMM3D_PARTITIONED_FUSED_DTT
  {
    std::vector<int> h_child_lo;
    std::vector<int> h_child_hi;
    h_build_child_ranges(run.host.htree, h_child_lo, h_child_hi);
    CUDA_CHECK(cudaMalloc(&d.d_child_lo, sizeof(int) * std::max(1, ncells)));
    CUDA_CHECK(cudaMalloc(&d.d_child_hi, sizeof(int) * std::max(1, ncells)));
    if (ncells > 0) {
      CUDA_CHECK(cudaMemcpy(d.d_child_lo, h_child_lo.data(), sizeof(int) * ncells,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d.d_child_hi, h_child_hi.data(), sizeof(int) * ncells,
                            cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(bind_partitioned_dtt_ptrs(d.d_child_lo, d.d_child_hi, DTT_NSPAWN));
  }
#endif
  CUDA_CHECK(bind_dtt_ptrs(d.d_m2l_count, d.d_p2p_count, d.d_m2l_src, d.d_p2p_src,
                           d.d_dtt_overflow, theta));
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
    run.timings.ms_gtap_init = h_elapsed_ms(t_gtap_init_start, t_gtap_init_end);
  }
#elif FMM3D_WORKLIST_DTT
  {
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
  }
#else
  {
    const std::chrono::high_resolution_clock::time_point t_buffer_alloc_start =
        std::chrono::high_resolution_clock::now();
    h_prepare_fixed_lists(run.dtt.host_fixed_lists, ncells);
    const std::chrono::high_resolution_clock::time_point t_buffer_alloc_end =
        std::chrono::high_resolution_clock::now();
    run.dtt.host_dtt_breakdown.ms_buffer_alloc =
        h_elapsed_ms(t_buffer_alloc_start, t_buffer_alloc_end);
    run.timings.ms_host_list_buffer_alloc = run.dtt.host_dtt_breakdown.ms_buffer_alloc;
  }
#endif
  const std::chrono::high_resolution_clock::time_point t_dtt_init_end =
      std::chrono::high_resolution_clock::now();
  run.timings.ms_dtt_init = h_elapsed_ms(t_dtt_init_start, t_dtt_init_end);
  return 0;
}

static int run_dtt(FmmRun &run) {
#if FMM3D_ENABLE_GTAP_DTT
  if (run_dtt_gtap(run)) return 1;
#elif FMM3D_WORKLIST_DTT
  if (run_dtt_worklist(run)) return 1;
#else
  if (run_dtt_host(run)) return 1;
#endif

#if FMM3D_M2L_COMPACT_TARGETS
  if (run.dtt.m2l_target_count > 0) {
    CUDA_CHECK(cudaMemcpy(run.device.d_m2l_targets, run.dtt.h_m2l_targets.data(),
                          sizeof(int) * run.dtt.m2l_target_count, cudaMemcpyHostToDevice));
  }
#endif

#ifdef GTAP_ENABLE_PROFILING
  gtap_export_profile({
      .output_directory = "./profile",
      .overwrite = true,
  });
#endif
  return 0;
}

static int run_m2l_l2l_eval(FmmRun &run) {
  FmmDeviceBuffers &d = run.device;
  FmmTimings &t = run.timings;

#if !FMM3D_GTAP_FUSED
  CUDA_CHECK(cudaMemset(d.d_phi, 0, sizeof(Real) * run.options.n));
  CUDA_CHECK(cudaMemset(d.d_ax, 0, sizeof(Real) * run.options.n));
  CUDA_CHECK(cudaMemset(d.d_ay, 0, sizeof(Real) * run.options.n));
  CUDA_CHECK(cudaMemset(d.d_az, 0, sizeof(Real) * run.options.n));
#endif
  CUDA_CHECK(cudaEventCreate(&d.ev_gpu_start));
  CUDA_CHECK(cudaEventCreate(&d.ev_gpu_end));

#if !FMM3D_GTAP_FUSED
  CUDA_CHECK(cudaEventRecord(d.ev_gpu_start));
  fmm3d_m2l_kernel<<<std::max(1, run.dtt.m2l_target_count), M2L_BLOCK_SIZE,
                      sizeof(Real) * 2 * EXA_NTERM * M2L_BLOCK_SIZE>>>(
      FMM3D_M2L_COMPACT_TARGETS ? d.d_m2l_targets : nullptr, run.dtt.m2l_target_count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(d.ev_gpu_end));
  CUDA_CHECK(cudaEventSynchronize(d.ev_gpu_end));
  {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, d.ev_gpu_start, d.ev_gpu_end));
    t.ms_m2l = static_cast<double>(ms);
  }
#endif

  {
    const std::chrono::high_resolution_clock::time_point t_l2l_start =
        std::chrono::high_resolution_clock::now();
#if FMM3D_GPU_L2L
    for (int depth = 1; depth <= run.host.max_cell_depth; ++depth) {
      const int begin = run.host.l2l_level_offsets[depth];
      const int end = run.host.l2l_level_offsets[depth + 1];
      const int count = end - begin;
      if (count == 0) continue;
      constexpr int L2L_BLOCK_SIZE = 128;
      int blocks = std::max(1, (count + L2L_BLOCK_SIZE - 1) / L2L_BLOCK_SIZE);
      fmm3d_l2l_level_kernel<<<blocks, L2L_BLOCK_SIZE>>>(d.d_l2l_cells + begin, count);
      CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::chrono::high_resolution_clock::time_point t_l2l_end =
        std::chrono::high_resolution_clock::now();
    t.ms_l2l = h_elapsed_ms(t_l2l_start, t_l2l_end);
#else
    CUDA_CHECK(cudaMemcpy(run.host.htree.data(), d.d_tree, sizeof(TreeNode) * run.host.ncells, cudaMemcpyDeviceToHost));
    const std::chrono::high_resolution_clock::time_point t_l2l_d2h_end = std::chrono::high_resolution_clock::now();
    h_downward_pass_devtree(run.host.htree);
    const std::chrono::high_resolution_clock::time_point t_l2l_cpu_end = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d.d_tree, run.host.htree.data(), sizeof(TreeNode) * run.host.ncells, cudaMemcpyHostToDevice));
    const std::chrono::high_resolution_clock::time_point t_l2l_h2d_end = std::chrono::high_resolution_clock::now();
    t.ms_l2l_tree_d2h = h_elapsed_ms(t_l2l_start, t_l2l_d2h_end);
    t.ms_l2l = h_elapsed_ms(t_l2l_d2h_end, t_l2l_cpu_end);
    t.ms_l2l_tree_h2d = h_elapsed_ms(t_l2l_cpu_end, t_l2l_h2d_end);
#endif
  }

  CUDA_CHECK(cudaEventRecord(d.ev_gpu_start));
  fmm3d_eval_kernel<<<GTAP_BENCH_GRID_SIZE, GTAP_BENCH_BLOCK_SIZE>>>();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(d.ev_gpu_end));
  CUDA_CHECK(cudaEventSynchronize(d.ev_gpu_end));
  {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, d.ev_gpu_start, d.ev_gpu_end));
    t.ms_eval = static_cast<double>(ms);
  }
  return 0;
}

static int copy_results(FmmRun &run) {
  const int n = run.options.n;
  FmmValidation &v = run.validation;
  v.hphi.resize(n);
  v.hax.resize(n);
  v.hay.resize(n);
  v.haz.resize(n);
  const std::chrono::high_resolution_clock::time_point t_d2h_start = std::chrono::high_resolution_clock::now();
  CUDA_CHECK(cudaMemcpy(v.hphi.data(), run.device.d_phi, sizeof(Real) * n, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(v.hax.data(), run.device.d_ax, sizeof(Real) * n, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(v.hay.data(), run.device.d_ay, sizeof(Real) * n, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(v.haz.data(), run.device.d_az, sizeof(Real) * n, cudaMemcpyDeviceToHost));
  const std::chrono::high_resolution_clock::time_point t_d2h_end = std::chrono::high_resolution_clock::now();
  run.timings.ms_d2h = h_elapsed_ms(t_d2h_start, t_d2h_end);
  run.timings.ms_execution = h_elapsed_ms(run.timings.run_start, t_d2h_end);
  return 0;
}

static void release_run(FmmRun &run) {
  FmmDeviceBuffers &d = run.device;
  cudaFree(d.d_tree);
  cudaFree(d.d_bx);
  cudaFree(d.d_by);
  cudaFree(d.d_bz);
  cudaFree(d.d_bq);
  cudaFree(d.d_phi);
  cudaFree(d.d_ax);
  cudaFree(d.d_ay);
  cudaFree(d.d_az);
  cudaFree(d.d_leaf_idx);
  cudaFree(d.d_body_cell);
  cudaFree(d.d_l2l_cells);
  cudaFree(d.d_upward_cells);
  cudaFree(d.d_m2l_count);
  cudaFree(d.d_p2p_count);
  cudaFree(d.d_m2l_src);
  cudaFree(d.d_p2p_src);
  cudaFree(d.d_m2l_targets);
  cudaFree(d.d_dtt_overflow);
#if FMM3D_PARTITIONED_FUSED_DTT
  cudaFree(d.d_child_lo);
  cudaFree(d.d_child_hi);
#endif
#if FMM3D_WORKLIST_DTT
  cudaFree(d.d_pairs_a);
  cudaFree(d.d_pairs_b);
  cudaFree(d.d_worklist_out_count);
  cudaFree(d.d_worklist_overflow);
#endif
  if (d.ev_gpu_start) cudaEventDestroy(d.ev_gpu_start);
  if (d.ev_gpu_end) cudaEventDestroy(d.ev_gpu_end);
#if FMM3D_ENABLE_GTAP_DTT
  gtap_finalize();
#endif
}

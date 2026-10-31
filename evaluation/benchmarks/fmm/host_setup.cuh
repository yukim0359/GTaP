#pragma once

#include "device_bind.cuh"
#include "host_cuda.hpp"
#include "host_timing.hpp"
#include "host_types.hpp"
#include "tree/gpu_morton.cuh"
#include "tree/host_octree.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>

static int init_cuda() {
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
  return 0;
}

static int build_tree(FmmRun &run) {
  const int n = run.options.n;
  run.host.hb = HostBodies(n);
  run.host.body_seed = h_body_rng_seed(n);
  h_generate_bodies(run.host.hb, run.host.body_seed);
  try {
#if FMM3D_GPU_TREE_BUILD
    const bool need_host_sorted_bodies =
        (!FMM3D_GPU_UPWARD) || (DIRECT_VALIDATE_N > 0 && n <= DIRECT_VALIDATE_N);
    GpuTreeBuildResult build_result =
        h_build_tree_gpu_morton(run.host.hb, need_host_sorted_bodies);
    run.host.htree = std::move(build_result.tree);
    run.host.sorted_device_bodies = build_result.sorted_bodies;
#else
    run.host.htree = h_build_tree(run.host.hb);
#endif
  } catch (const std::exception &e) {
    std::printf("Tree build error: %s\n", e.what());
    return 1;
  }
  return 0;
}

static int build_metadata(FmmRun &run) {
  const int n = run.options.n;
  const Real theta = run.options.theta;
  std::vector<TreeNode> &htree = run.host.htree;
  const int ncells = static_cast<int>(htree.size());
  run.host.ncells = ncells;
  run.host.leaf_idx.reserve(std::max(1, ncells / 2));
  run.host.body_cell.assign(n, -1);
  run.host.cell_depth.assign(ncells, 0);
  run.host.max_cell_depth = 0;

  for (int ci = 0; ci < ncells; ++ci) {
    const int parent = htree[ci].parent;
    if (parent >= 0) {
      run.host.cell_depth[ci] = run.host.cell_depth[parent] + 1;
      run.host.max_cell_depth = std::max(run.host.max_cell_depth, run.host.cell_depth[ci]);
    }
    if (htree[ci].is_leaf) {
      run.host.leaf_idx.push_back(ci);
      for (int k = 0; k < htree[ci].nbody; ++k) {
        run.host.body_cell[htree[ci].ibody + k] = ci;
      }
    }
  }

  run.host.l2l_level_offsets.assign(run.host.max_cell_depth + 2, 0);
  for (int ci = 1; ci < ncells; ++ci) {
    run.host.l2l_level_offsets[run.host.cell_depth[ci] + 1]++;
  }
  for (int d = 1; d <= run.host.max_cell_depth + 1; ++d) {
    run.host.l2l_level_offsets[d] += run.host.l2l_level_offsets[d - 1];
  }
  run.host.l2l_cells.resize(std::max(0, ncells - 1));
  {
    std::vector<int> cursor = run.host.l2l_level_offsets;
    for (int ci = 1; ci < ncells; ++ci) {
      run.host.l2l_cells[cursor[run.host.cell_depth[ci]]++] = ci;
    }
  }

  run.host.upward_level_offsets.assign(run.host.max_cell_depth + 2, 0);
  for (int ci = 0; ci < ncells; ++ci) {
    run.host.upward_level_offsets[run.host.cell_depth[ci] + 1]++;
  }
  for (int d = 1; d <= run.host.max_cell_depth + 1; ++d) {
    run.host.upward_level_offsets[d] += run.host.upward_level_offsets[d - 1];
  }
  run.host.upward_cells.resize(ncells);
  {
    std::vector<int> cursor = run.host.upward_level_offsets;
    for (int ci = 0; ci < ncells; ++ci) {
      run.host.upward_cells[cursor[run.host.cell_depth[ci]]++] = ci;
    }
  }

  run.host.nleaf = static_cast<int>(run.host.leaf_idx.size());
  std::printf("Problem setup: N=%d  Nodes=%d  Leaves=%d  theta=%.2f  body_seed=%llu  per_n_seed=%d\n",
              n, run.host.ncells, run.host.nleaf, theta,
              static_cast<unsigned long long>(run.host.body_seed),
              FMM3D_BODY_SEED_PER_N);
  h_print_leaf_depth_stats(htree, run.host.cell_depth, run.host.leaf_idx, n);
  return 0;
}

static int alloc_and_bind(FmmRun &run) {
  const int n = run.options.n;
  const int ncells = run.host.ncells;
  const int nleaf = run.host.nleaf;
  FmmDeviceBuffers &d = run.device;

  d.d_bx = run.host.sorted_device_bodies.x;
  d.d_by = run.host.sorted_device_bodies.y;
  d.d_bz = run.host.sorted_device_bodies.z;
  d.d_bq = run.host.sorted_device_bodies.q;

  CUDA_CHECK(cudaMalloc(&d.d_tree, sizeof(TreeNode) * ncells));
  if (!d.d_bx) CUDA_CHECK(cudaMalloc(&d.d_bx, sizeof(Real) * n));
  if (!d.d_by) CUDA_CHECK(cudaMalloc(&d.d_by, sizeof(Real) * n));
  if (!d.d_bz) CUDA_CHECK(cudaMalloc(&d.d_bz, sizeof(Real) * n));
  if (!d.d_bq) CUDA_CHECK(cudaMalloc(&d.d_bq, sizeof(Real) * n));
  CUDA_CHECK(cudaMalloc(&d.d_phi, sizeof(Real) * n));
  CUDA_CHECK(cudaMalloc(&d.d_ax, sizeof(Real) * n));
  CUDA_CHECK(cudaMalloc(&d.d_ay, sizeof(Real) * n));
  CUDA_CHECK(cudaMalloc(&d.d_az, sizeof(Real) * n));
  CUDA_CHECK(cudaMalloc(&d.d_leaf_idx, sizeof(int) * std::max(1, nleaf)));
  CUDA_CHECK(cudaMalloc(&d.d_body_cell, sizeof(int) * n));
  CUDA_CHECK(cudaMalloc(&d.d_l2l_cells, sizeof(int) * std::max(1, ncells - 1)));
  CUDA_CHECK(cudaMalloc(&d.d_upward_cells, sizeof(int) * std::max(1, ncells)));
  CUDA_CHECK(cudaMalloc(&d.d_m2l_targets, sizeof(int) * std::max(1, ncells)));

  CUDA_CHECK(cudaMemcpy(d.d_tree, run.host.htree.data(), sizeof(TreeNode) * ncells, cudaMemcpyHostToDevice));
  if (!run.host.sorted_device_bodies.x) {
    CUDA_CHECK(cudaMemcpy(d.d_bx, run.host.hb.x.data(), sizeof(Real) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.d_by, run.host.hb.y.data(), sizeof(Real) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.d_bz, run.host.hb.z.data(), sizeof(Real) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.d_bq, run.host.hb.q.data(), sizeof(Real) * n, cudaMemcpyHostToDevice));
  }
  if (nleaf > 0) {
    CUDA_CHECK(cudaMemcpy(d.d_leaf_idx, run.host.leaf_idx.data(), sizeof(int) * nleaf, cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMemcpy(d.d_body_cell, run.host.body_cell.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
  if (ncells > 1) {
    CUDA_CHECK(cudaMemcpy(d.d_l2l_cells, run.host.l2l_cells.data(), sizeof(int) * (ncells - 1),
                          cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMemcpy(d.d_upward_cells, run.host.upward_cells.data(), sizeof(int) * ncells,
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(bind_params(n, ncells, nleaf));
  CUDA_CHECK(bind_ptrs(d.d_tree, d.d_bx, d.d_by, d.d_bz, d.d_bq, d.d_phi, d.d_ax, d.d_ay, d.d_az,
                       d.d_leaf_idx, d.d_body_cell));
  return 0;
}

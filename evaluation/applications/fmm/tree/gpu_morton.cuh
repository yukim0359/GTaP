#pragma once

#include "host_octree.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

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

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "gtap_thread.cuh"

__device__ int d_nodes_finished;
__device__ int d_root_observed;

#pragma gtap function
__device__ void visit(int depth, bool root) {
  if (depth > 0) {
#pragma gtap task
    visit(depth - 1, false);
#pragma gtap task
    visit(depth - 1, false);
#pragma gtap taskwait
  }
  atomicAdd(&d_nodes_finished, 1);
  if (root)
    d_root_observed = atomicAdd(&d_nodes_finished, 0);
}

__global__ void test_kernel(int depth) {
#pragma gtap entry
  visit(depth, true);
}

int main(int argc, char **argv) {
  int depth = argc > 1 ? std::atoi(argv[1]) : 6;
  int zero = 0;
  cudaError_t status = cudaMemcpyToSymbol(d_nodes_finished, &zero, sizeof(zero));
  if (status == cudaSuccess)
    status = cudaMemcpyToSymbol(d_root_observed, &zero, sizeof(zero));

  gtap_thread_config config;
  config.grid_size = 8;
  config.block_size = 32;
  config.max_tasks_per_warp = 1024;
  config.num_queues = 1;
  if (status == cudaSuccess)
    status = gtap_initialize(config);
  if (status == cudaSuccess)
    status = gtap_launch(test_kernel, depth);
  if (status == cudaSuccess)
    status = gtap_synchronize();

  int finished = 0;
  int observed = 0;
  if (status == cudaSuccess)
    status = cudaMemcpyFromSymbol(&finished, d_nodes_finished, sizeof(finished));
  if (status == cudaSuccess)
    status = cudaMemcpyFromSymbol(&observed, d_root_observed, sizeof(observed));
  cudaError_t finalize_status = gtap_finalize();
  if (status != cudaSuccess || finalize_status != cudaSuccess) {
    cudaError_t reported = status != cudaSuccess ? status : finalize_status;
    std::fprintf(stderr, "binary_tree_thread runtime failure: %s\n",
                 cudaGetErrorString(reported));
    return 1;
  }

  int expected = (1 << (depth + 1)) - 1;
  std::printf("binary_tree_thread(%d): finished=%d root_observed=%d expected=%d\n",
              depth, finished, observed, expected);
  return finished == expected && observed == expected ? 0 : 1;
}

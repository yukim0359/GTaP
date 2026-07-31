#include <cstdio>
#include <cuda_runtime.h>
#include "gtap_thread.cuh"

__device__ int d_result;

#pragma gtap function
__device__ int leaf(int value) { return value; }

#pragma gtap function
__device__ int two_waits() {
  int first;
  int second;
#pragma gtap task queue(1)
  first = leaf(1);
#pragma gtap taskwait queue(0)
#pragma gtap task queue(1)
  second = leaf(first + 1);
#pragma gtap taskwait queue(0)
  return first + second;
}

__global__ void test_kernel() {
#pragma gtap entry
  d_result = two_waits();
}

int main() {
  gtap_thread_config config;
  config.grid_size = 8;
  config.block_size = 32;
  config.max_tasks_per_warp = 128;
  config.num_queues = 2;

  cudaError_t status = gtap_initialize(config);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "gtap_initialize failed: %s\n",
                 cudaGetErrorString(status));
    return 1;
  }

  status = gtap_launch(test_kernel);
  if (status == cudaSuccess)
    status = gtap_synchronize();

  int result = 0;
  if (status == cudaSuccess)
    status = cudaMemcpyFromSymbol(&result, d_result, sizeof(result));

  cudaError_t finalize_status = gtap_finalize();
  if (status != cudaSuccess || finalize_status != cudaSuccess) {
    cudaError_t reported =
        status != cudaSuccess ? status : finalize_status;
    std::fprintf(stderr, "multiple_taskwait_queue runtime failure: %s\n",
                 cudaGetErrorString(reported));
    return 1;
  }

  constexpr int expected = 3;
  std::printf("multiple_taskwait_queue: result=%d expected=%d\n",
              result, expected);
  return result == expected ? 0 : 1;
}

#include <cstdio>
#include <cuda_runtime.h>
#include "gtap_thread.cuh"

__device__ int d_result;

#pragma gtap function
__device__ int leaf() { return 1; }

#pragma gtap function
__device__ int out_of_range_queue() {
  int value;
#pragma gtap task queue(1)
  value = leaf();
#pragma gtap taskwait
  return value;
}

__global__ void test_kernel() {
#pragma gtap entry
  d_result = out_of_range_queue();
}

int main() {
  gtap_thread_config config;
  config.grid_size = 8;
  config.block_size = 32;
  config.max_tasks_per_warp = 128;
  config.num_queues = 1;

  cudaError_t status = gtap_initialize(config);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "gtap_initialize failed: %s\n",
                 cudaGetErrorString(status));
    return 1;
  }

  cudaError_t launch_status = gtap_launch(test_kernel);
  cudaError_t sync_status = cudaSuccess;
  if (launch_status == cudaSuccess)
    sync_status = gtap_synchronize();

  cudaGetLastError();
  cudaError_t finalize_status = gtap_finalize();
  if (finalize_status != cudaSuccess)
    cudaGetLastError();

  if (launch_status != cudaSuccess) {
    std::fprintf(stderr, "invalid_queue_thread: gtap_launch failed: %s\n",
                 cudaGetErrorString(launch_status));
    return 1;
  }
  if (sync_status == cudaSuccess) {
    std::fprintf(stderr,
                 "invalid_queue_thread: gtap_synchronize returned success\n");
    return 1;
  }
  std::printf("invalid_queue_thread: gtap_synchronize failed as expected (%s)\n",
              cudaGetErrorString(sync_status));
  return 0;
}

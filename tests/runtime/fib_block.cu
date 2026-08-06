#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "gtap_block.cuh"

__device__ int d_result;
__device__ int d_non_spawner_received_result;
__device__ int d_entry_lane_results[GTAP_BLOCK_SIZE];
__device__ int d_non_master_entry_assignment;

#pragma gtap function
__device__ int fib(int n) {
  if (n < 2)
    return n;

  int a = 0;
  int b = 0;
  if (threadIdx.x == 0) {
#pragma gtap task
    a = fib(n - 1);
#pragma gtap task
    b = fib(n - 2);
  }
#pragma gtap taskwait

  if (threadIdx.x != 0 && (a != 0 || b != 0))
    atomicExch(&d_non_spawner_received_result, 1);
  return a + b;
}

__global__ void test_kernel(int n) {
  constexpr int untouched = -12345;
  int result = untouched;
#pragma gtap entry
  result = fib(n);

  // The entry assignment is performed by every thread in the master block.
  // Other blocks must retain their original local value.
  if (blockIdx.x == 0) {
    d_entry_lane_results[threadIdx.x] = result;
    if (threadIdx.x == 0)
      d_result = result;
  } else if (result != untouched) {
    atomicExch(&d_non_master_entry_assignment, 1);
  }
}

static int serial_fib(int n) {
  return n < 2 ? n : serial_fib(n - 1) + serial_fib(n - 2);
}

int main(int argc, char **argv) {
  int n = argc > 1 ? std::atoi(argv[1]) : 10;

  gtap_block_config config;
  config.grid_size = 8;
  config.max_tasks_per_block = 1024;

  cudaError_t status = gtap_initialize(config);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "gtap_initialize failed: %s\n",
                 cudaGetErrorString(status));
    return 1;
  }

  int zero = 0;
  status = cudaMemcpyToSymbol(d_non_spawner_received_result, &zero,
                              sizeof(zero));
  if (status == cudaSuccess)
    status = cudaMemcpyToSymbol(d_non_master_entry_assignment, &zero,
                                sizeof(zero));
  if (status == cudaSuccess)
    status = gtap_launch(test_kernel, n);
  if (status == cudaSuccess)
    status = gtap_synchronize();

  int result = 0;
  int non_spawner_received_result = 0;
  int non_master_entry_assignment = 0;
  int entry_lane_results[GTAP_BLOCK_SIZE] = {};
  if (status == cudaSuccess)
    status = cudaMemcpyFromSymbol(&result, d_result, sizeof(result));
  if (status == cudaSuccess)
    status = cudaMemcpyFromSymbol(&non_spawner_received_result,
                                  d_non_spawner_received_result,
                                  sizeof(non_spawner_received_result));
  if (status == cudaSuccess)
    status = cudaMemcpyFromSymbol(&non_master_entry_assignment,
                                  d_non_master_entry_assignment,
                                  sizeof(non_master_entry_assignment));
  if (status == cudaSuccess)
    status = cudaMemcpyFromSymbol(entry_lane_results, d_entry_lane_results,
                                  sizeof(entry_lane_results));

  cudaError_t finalize_status = gtap_finalize();
  if (status != cudaSuccess || finalize_status != cudaSuccess) {
    cudaError_t reported =
        status != cudaSuccess ? status : finalize_status;
    std::fprintf(stderr, "fib_block runtime failure: %s\n",
                 cudaGetErrorString(reported));
    return 1;
  }

  int expected = serial_fib(n);
  bool lane_results_ok = entry_lane_results[0] == expected;
  for (int lane = 1; lane < GTAP_BLOCK_SIZE; ++lane)
    lane_results_ok = lane_results_ok && entry_lane_results[lane] == 0;

  std::printf(
      "fib_block(%d): result=%d expected=%d non_spawner=%d "
      "non_master_entry=%d lane_results=%s\n",
      n, result, expected, non_spawner_received_result,
      non_master_entry_assignment, lane_results_ok ? "ok" : "bad");
  return result == expected && non_spawner_received_result == 0 &&
                 non_master_entry_assignment == 0 && lane_results_ok
             ? 0
             : 1;
}

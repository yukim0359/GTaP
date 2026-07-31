#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "gtap_thread.cuh"

__device__ int d_result;

#pragma gtap function
__device__ int fib(int n) {
  if (n < 2)
    return n;
  int a;
  int b;
#pragma gtap task
  a = fib(n - 1);
#pragma gtap task
  b = fib(n - 2);
#pragma gtap taskwait
  return a + b;
}

__global__ void test_kernel(int n) {
#pragma gtap entry
  d_result = fib(n);
}

static int serial_fib(int n) {
  return n < 2 ? n : serial_fib(n - 1) + serial_fib(n - 2);
}

int main(int argc, char **argv) {
  int n = argc > 1 ? std::atoi(argv[1]) : 10;

  gtap_thread_config config;
  config.grid_size = 8;
  config.block_size = 32;
  config.max_tasks_per_warp = 1024;
  config.num_queues = 1;

  cudaError_t status = gtap_initialize(config);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "gtap_initialize failed: %s\n",
                 cudaGetErrorString(status));
    return 1;
  }

  status = gtap_launch(test_kernel, n);
  if (status == cudaSuccess)
    status = gtap_synchronize();

  int result = 0;
  if (status == cudaSuccess)
    status = cudaMemcpyFromSymbol(&result, d_result, sizeof(result));

  cudaError_t finalize_status = gtap_finalize();
  if (status != cudaSuccess || finalize_status != cudaSuccess) {
    cudaError_t reported =
        status != cudaSuccess ? status : finalize_status;
    std::fprintf(stderr, "fib_thread runtime failure: %s\n",
                 cudaGetErrorString(reported));
    return 1;
  }

  int expected = serial_fib(n);
  std::printf("fib_thread(%d): result=%d expected=%d\n", n, result, expected);
  return result == expected ? 0 : 1;
}

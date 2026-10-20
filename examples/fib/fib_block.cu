#include <stdio.h>
#include <cuda_runtime.h>
#include "gtap_block.cuh"

__device__ int d_result[32];

#pragma gtap function
__device__ int fib_block(int n) {
    if (n < 2) {
        return n;
    }

    // In block mode every task is executed by a whole block. Only thread 0
    // spawns these children, so only thread 0 receives their return values.
    int a = 0;
    int b = 0;
    if (threadIdx.x == 0) {
        #pragma gtap task
        a = fib_block(n - 1);
        #pragma gtap task
        b = fib_block(n - 2);
    }

    // taskwait is collective in block mode.
    #pragma gtap taskwait
    return a + b;
}

__global__ void exec_block_kernel(int n) {
    #pragma gtap entry
    d_result[threadIdx.x] = fib_block(n);
}

int main(int argc, char** argv) {
    int n = 35;
    if (argc >= 2) n = atoi(argv[1]);

    gtap_block_config config;
    config.grid_size = 4000;
    config.block_size = 32;
    config.max_tasks_per_block = 10000;

    cudaError_t err = gtap_initialize(config);
    if (err != cudaSuccess) {
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    err = gtap_launch(exec_block_kernel, n);

    if (err == cudaSuccess) {
        cudaEventRecord(stop);
        err = gtap_synchronize();
    }
    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        gtap_finalize();
        return 1;
    }

    int h_result[32] = {};
    cudaMemcpyFromSymbol(h_result, d_result, sizeof(h_result));
    printf("Fibonacci of %d is %d\n", n, h_result[0]);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Execution time: %.3f ms\n", elapsed_time);

#ifdef GTAP_ENABLE_PROFILING
    gtap_export_profile({
        .output_directory = "./profile/fib_block",
        .overwrite = true,
    });
#endif

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    err = gtap_finalize();
    return err == cudaSuccess ? 0 : 1;
}

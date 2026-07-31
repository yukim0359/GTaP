#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "gtap_block.cuh"

__device__ int d_block_result;
__device__ int d_non_spawner_received_result;

#pragma gtap function
__device__ int fib_block(int n) {
    if (n < 2) {
        return n;
    }

    // In block mode every task is executed by a whole block.  Only thread 0
    // spawns these children, so only thread 0 must receive their return values.
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

    // A nonzero flag means that a thread which did not execute the spawn
    // incorrectly received a child result.
    if (threadIdx.x != 0 && (a != 0 || b != 0)) {
        atomicExch(&d_non_spawner_received_result, 1);
    }

    // Only thread 0 carries the recursive result.  Other threads return zero,
    // and their return values must not be delivered to the spawning thread.
    return a + b;
}

__global__ void exec_block_kernel(int n) {
    #pragma gtap entry
    d_block_result = fib_block(n);
}

static int fib_serial(int n) {
    if (n < 2) {
        return n;
    }
    return fib_serial(n - 1) + fib_serial(n - 2);
}

int main(int argc, char** argv) {
    int n = 20;
    if (argc >= 2) {
        n = atoi(argv[1]);
    }

    gtap_block_config config;
    config.grid_size = 4000;
    config.max_tasks_per_block = 10000;

    cudaError_t err = gtap_initialize(config);
    if (err != cudaSuccess) {
        fprintf(stderr, "gtap_initialize failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    int zero = 0;
    cudaMemcpyToSymbol(d_non_spawner_received_result, &zero, sizeof(zero));

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    err = gtap_launch(exec_block_kernel, n);
    if (err != cudaSuccess) {
        fprintf(stderr, "gtap_launch failed: %s\n", cudaGetErrorString(err));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        gtap_finalize();
        return 1;
    }
    cudaEventRecord(stop);

    err = gtap_synchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel failed: %s\n", cudaGetErrorString(err));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        gtap_finalize();
        return 1;
    }

    int result = 0;
    int non_spawner_received_result = 0;
    cudaMemcpyFromSymbol(&result, d_block_result, sizeof(result));
    cudaMemcpyFromSymbol(&non_spawner_received_result,
                         d_non_spawner_received_result,
                         sizeof(non_spawner_received_result));

    const int expected = fib_serial(n);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    printf("Block Fibonacci of %d is %d (expected %d)\n",
           n, result, expected);
    printf("Non-spawning thread received a result: %s\n",
           non_spawner_received_result ? "YES" : "no");
    printf("Execution time: %.3f ms\n", elapsed_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    gtap_finalize();

    return result == expected && !non_spawner_received_result ? 0 : 1;
}

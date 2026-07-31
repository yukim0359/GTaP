#include <stdio.h>
#include <cuda_runtime.h>
#include "gtap_thread.cuh"

__device__ int d_result;

#pragma gtap function
__device__ int fib(int n) {
    if (n < 2) {
        return n;
    }
    int a, b;
    #pragma gtap task
    a = fib(n - 1);
    #pragma gtap task
    b = fib(n - 2);
    #pragma gtap taskwait
    return a + b;
}

__global__ void exec_kernel(int n) {
    #pragma gtap entry
    d_result = fib(n);
}

int main(int argc, char** argv) {
    int n = 40;
    if (argc >= 2) n = atoi(argv[1]);

    gtap_thread_config config;
    config.grid_size = 4000;
    config.block_size = 32;
    config.max_tasks_per_warp = 150000;
    config.num_queues = 1;

    cudaError_t err = gtap_initialize(config);
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    err = gtap_launch(exec_kernel, n);
    if (err != cudaSuccess) {
        printf("gtap_launch failed: %s\n", cudaGetErrorString(err));
        gtap_finalize();
        return 1;
    }
    cudaEventRecord(stop);
    err = gtap_synchronize();
    if (err != cudaSuccess) {
        printf("gtap_synchronize failed: %s\n", cudaGetErrorString(err));
        gtap_finalize();
        return 1;
    }

    int h_result;
    cudaMemcpyFromSymbol(&h_result, d_result, sizeof(int));
    printf("Fibonacci of %d is %d\n", n, h_result);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Execution time: %.3f ms\n", elapsed_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    err = gtap_finalize();
    return err == cudaSuccess ? 0 : 1;
}

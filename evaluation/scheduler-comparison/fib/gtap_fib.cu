#include <stdio.h>
#include <cuda_runtime.h>
// #define GTAP_PROFILE

#ifdef GQ
#include "experimental/gtap_thread_gq.cuh"
#else
#ifdef CHASELEV
#include "experimental/gtap_thread_chaselev.cuh"
#else
#include "gtap_thread.cuh"
#endif
#endif

#ifndef GTAP_BENCH_GRID_SIZE
#define GTAP_BENCH_GRID_SIZE 4000
#endif
#ifndef GTAP_BENCH_BLOCK_SIZE
#define GTAP_BENCH_BLOCK_SIZE 32
#endif
#ifndef GTAP_BENCH_MAX_TASKS_PER_WARP
#define GTAP_BENCH_MAX_TASKS_PER_WARP 200000
#endif

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

    gtap_thread_config config{
        .grid_size = GTAP_BENCH_GRID_SIZE,
        .block_size = GTAP_BENCH_BLOCK_SIZE,
        .max_tasks_per_warp = GTAP_BENCH_MAX_TASKS_PER_WARP,
        .num_queues = 1,
    };
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
        printf("Launch error: %s\n", cudaGetErrorString(err));
        gtap_finalize();
        return 1;
    }
    gtap_synchronize();
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    
    int h_result;
    cudaMemcpyFromSymbol(&h_result, d_result, sizeof(int));
    printf("Fibonacci of %d is %d\n", n, h_result);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Execution time: %.3f ms\n", elapsed_time);

#ifdef GTAP_PROFILE
    visualize_working_time("fib");
#endif

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

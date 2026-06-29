#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
// #define PROFILE
// #define INIT_PROFILE
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

// static double elapsed_ms(timespec start, timespec end) {
//     return (end.tv_sec - start.tv_sec) * 1000.0 +
//            (end.tv_nsec - start.tv_nsec) / 1000000.0;
// }

int main(int argc, char** argv) {
    int n = 40;
    if (argc >= 2) n = atoi(argv[1]);

    cudaError_t err = cudaFree(0);
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // timespec init_start, init_end;
    // clock_gettime(CLOCK_MONOTONIC, &init_start);
    err = gtap_initialize();
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    // clock_gettime(CLOCK_MONOTONIC, &init_end);
    // printf("Initialization time: %.3f ms\n", elapsed_ms(init_start, init_end));

    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    exec_kernel<<<GTAP_GRID_SIZE, GTAP_BLOCK_SIZE>>>(n);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);

    int h_result;
    cudaMemcpyFromSymbol(&h_result, d_result, sizeof(int));
    printf("Fibonacci of %d is %d\n", n, h_result);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Execution time: %.3f ms\n", elapsed_time);

#ifdef PROFILE
    gtap_visualize_profile("fib");
#endif

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

#include <stdio.h>
#include <cuda_runtime.h>
// #define PROFILE
#include "gtap_thread.cuh"

__device__ int d_result;
__device__ int d_cutoff;

__device__ int fib_sequential(int n) {
    if (n < 2) return n;
    int result1 = fib_sequential(n - 1);
    int result2 = fib_sequential(n - 2);
    return result1 + result2;
}

#pragma gtap function
__device__ int fib(int n) {
    if (n < d_cutoff) return fib_sequential(n);
    int a, b;
    #pragma gtap task queue((n - 1) < d_cutoff ? 1 : 0)
    a = fib(n - 1);
    #pragma gtap task queue((n - 2) < d_cutoff ? 1 : 0)
    b = fib(n - 2);
    #pragma gtap taskwait queue(2)
    return a + b;
}

__global__ void exec_kernel(int n) {
    #pragma gtap entry
    d_result = fib(n);
}

int main(int argc, char** argv) {
    int n = 40;
    int cutoff = 2;
    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) cutoff = atoi(argv[2]);

    cudaError_t err = gtap_initialize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpyToSymbol(d_cutoff, &cutoff, sizeof(int));
    
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
    gtap_visualize_profile("fib_queue_3");
#endif

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

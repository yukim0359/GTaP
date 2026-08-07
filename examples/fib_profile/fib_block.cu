#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "gtap_block.cuh"

__device__ int d_result;

#pragma gtap function
__device__ int fib(int n) {
    if (n < 2) {
        return n;
    }

    int a = 0;
    int b = 0;
    if (threadIdx.x == 0) {
        #pragma gtap task
        a = fib(n - 1);
        #pragma gtap task
        b = fib(n - 2);
    }
    #pragma gtap taskwait
    return a + b;
}

__global__ void exec_kernel(int n) {
    int result;
    #pragma gtap entry
    result = fib(n);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        d_result = result;
    }
}

int main(int argc, char** argv) {
    int n = 25;
    if (argc >= 2) n = atoi(argv[1]);

    gtap_block_config config{
        .grid_size = 3000,
        .max_tasks_per_block = 10000,
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
    cudaEventRecord(stop);
    gtap_synchronize();
    cudaEventSynchronize(stop);

    int h_result;
    cudaMemcpyFromSymbol(&h_result, d_result, sizeof(int));
    printf("Fibonacci of %d is %d\n", n, h_result);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Execution time: %.3f ms\n", elapsed_time);

    gtap_export_profile({
        .output_directory = "./profile/fib_block",
        .overwrite = true,
    });

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

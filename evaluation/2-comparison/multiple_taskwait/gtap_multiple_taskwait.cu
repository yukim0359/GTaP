#include <stdio.h>
#include <cuda_runtime.h>
#define GTAP_MAX_TASK_DATA_SIZE 20
// #define PROFILE
#include "gtap_thread.cuh"

__device__ int d_result;

#pragma gtap function worker_size(thread)
__device__ void do_something() {
    d_result++;
}

#pragma gtap function worker_size(thread)
__device__ int multiple_taskwait(int n) {
    int a = 0;
    for (int i = 0; i < n; i++) {
        printf("blockIdx.x: %d\n", blockIdx.x);
        for (int j = 0; j < 34; j++) {
            #pragma gtap task
            do_something();
        }
        #pragma gtap taskwait
        a++;
    }
    return a;
}

__global__ void exec_kernel(int n) {
    #pragma gtap entry
    d_result = multiple_taskwait(n);
}

int main(int argc, char** argv) {
    int n = 10;
    if (argc >= 2) n = atoi(argv[1]);

    cudaError_t err = gtap_initialize();
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
    printf("Multiple taskwait of %d is %d\n", n, h_result);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Execution time: %.3f ms\n", elapsed_time);

#ifdef PROFILE
    visualize_working_time("multiple_taskwait");
#endif

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

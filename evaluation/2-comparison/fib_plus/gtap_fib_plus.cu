#include <stdio.h>
#include <cuda_runtime.h>
// #define PROFILE
#include "gtap_block.cuh"

#define DATA_LENGTH 2048

__device__ int d_result;

__device__ bool heavy_operation() {
    // Use FMA-heavy math on shared data to better exercise the GPU pipelines.
    __shared__ float sdata[DATA_LENGTH];
    float acc = static_cast<float>(threadIdx.x + 1);

    // Write phase: each thread strides through the array doing FMA work.
    for (int i = threadIdx.x; i < DATA_LENGTH; i += blockDim.x) {
        float base = static_cast<float>(i) * 1.001f;
        float val = __fmaf_rn(base, acc, 0.5f); // val = base * acc + 0.5
        sdata[i] = val;
        acc = __fmaf_rn(acc, 1.0001f, 0.123f);
    }

    __syncthreads();

    // Verification phase: recompute expected values and check with tolerance.
    acc = static_cast<float>(threadIdx.x + 1);
    for (int i = threadIdx.x; i < DATA_LENGTH; i += blockDim.x) {
        float base = static_cast<float>(i) * 1.001f;
        float expected = __fmaf_rn(base, acc, 0.5f);
        if (fabsf(sdata[i] - expected) > 1e-3f) {
            printf("FATAL ERROR: heavy_operation failed at index %d\n", i);
            __trap();
        }
        acc = __fmaf_rn(acc, 1.0001f, 0.123f);
    }

    return true;
}

#pragma gtap function worker_size(block) return_thread(0)
__device__ int fib_plus_heavy(int n) {
    if (n < 2) {
        heavy_operation();
        return n;
    }

    int a, b;
    if (threadIdx.x == 0) {
        #pragma gtap task
        a = fib_plus_heavy(n - 1);
        #pragma gtap task
        b = fib_plus_heavy(n - 2);
    }
    #pragma gtap taskwait

    heavy_operation();
    return a + b;
}

__global__ void my_kernel(int n) {
    #pragma gtap entry
    d_result = fib_plus_heavy(n);
}

int main(int argc, char** argv) {
    int n = 30;
    if (argc > 1) n = atoi(argv[1]);
    
    cudaError_t err = gtap_initialize();
    if (err != cudaSuccess) {
        printf("Error initializing task runtime: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    my_kernel<<<GTAP_GRID_SIZE, GTAP_BLOCK_SIZE>>>(n);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    
    int h_result;
    cudaMemcpyFromSymbol(&h_result, d_result, sizeof(int));
    printf("Fibonacci of %d is %d\n", n, h_result);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Execution time: %f ms\n", elapsed_time);

#ifdef PROFILE
    visualize_profile("fib_plus");
#endif

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

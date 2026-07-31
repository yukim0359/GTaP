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

#ifdef INIT_PROFILE
static double elapsed_ms(timespec start, timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 +
           (end.tv_nsec - start.tv_nsec) / 1000000.0;
}
#endif

int main(int argc, char** argv) {
    int n = 40;
    if (argc >= 2) n = atoi(argv[1]);

    cudaError_t err = cudaFree(0);
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

#ifdef INIT_PROFILE
    timespec init_start, init_end;
    clock_gettime(CLOCK_MONOTONIC, &init_start);
#endif
    size_t gtap_device_bytes = 0;
    gtap_thread_config config{
        .grid_size = GTAP_BENCH_GRID_SIZE,
        .block_size = GTAP_BENCH_BLOCK_SIZE,
        .max_tasks_per_warp = GTAP_BENCH_MAX_TASKS_PER_WARP,
        .num_queues = GTAP_BENCH_NUM_QUEUES,
    };
    err = gtap_initialize(config, &gtap_device_bytes);
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
#ifdef INIT_PROFILE
    clock_gettime(CLOCK_MONOTONIC, &init_end);
    printf("Initialization time: %.3f ms\n", elapsed_ms(init_start, init_end));
#endif

    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("GTaP runtime device pool: %.2f GiB (%zu bytes)\n",
           gtap_device_bytes / (1024.0 * 1024.0 * 1024.0), gtap_device_bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    err = gtap_launch(exec_kernel, n);
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaEventRecord(stop);
    err = gtap_synchronize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
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

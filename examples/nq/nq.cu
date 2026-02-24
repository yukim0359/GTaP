#include <stdio.h>
#include <cuda_runtime.h>
#include "gtap_thread.cuh"

__device__ int d_answer;
__device__ __constant__ int d_grid_size;
__device__ __constant__ int d_cutoff_depth;

__device__ void serial_search(int row, uint32_t column, uint32_t left, uint32_t down, uint32_t right) {
    int grid_size = d_grid_size;
    if (row == grid_size) {
        atomicAdd(&d_answer, 1);
        return;
    }
    uint32_t mask = (grid_size < 32 ? (1u << grid_size) - 1 : 0xFFFFFFFFu);
    uint32_t avail = mask & ~((column | left | right));
    while (avail) {
        uint32_t p = avail & -avail;
        avail -= p;
        serial_search(row + 1, column | p, (left | p) << 1, down | p, (right | p) >> 1);
    }
}

#pragma gtap function
__device__ void nq(int row, uint32_t column, uint32_t left, uint32_t down, uint32_t right) {
    int grid_size   = d_grid_size;
    int cutoff_depth = d_cutoff_depth;

    if (row > cutoff_depth) {
        serial_search(row, column, left, down, right);
        return;
    }

    uint32_t mask = (grid_size < 32 ? (1u << grid_size) - 1 : 0xFFFFFFFFu);
    uint32_t avail = mask & ~((column | left | right));

    while (avail) {
        uint32_t p = avail & -avail;
        avail -= p;
        int new_row = row + 1;
        uint32_t new_column = column | p;
        uint32_t new_left = (left | p) << 1;
        uint32_t new_down = down | p;
        uint32_t new_right = (right | p) >> 1;

        #pragma gtap task
        nq(new_row, new_column, new_left, new_down, new_right);
    }
}

__global__ void my_kernel() {
    #pragma gtap entry
    nq(0, 0, 0, 0, 0);
}

int main(int argc, char **argv) {
    int GRID_SIZE = (argc > 1 ? atoi(argv[1]) : 16);
    int CUTOFF_DEPTH = (argc > 2 ? atoi(argv[2]) : 7);

    int zero = 0;
    cudaMemcpyToSymbol(d_answer, &zero, sizeof(int));
    cudaMemcpyToSymbol(d_grid_size, &GRID_SIZE, sizeof(int));
    cudaMemcpyToSymbol(d_cutoff_depth, &CUTOFF_DEPTH, sizeof(int));

    cudaError_t err = gtap_initialize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    my_kernel<<<GTAP_GRID_SIZE, GTAP_BLOCK_SIZE>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    int result = 0;
    cudaMemcpyFromSymbol(&result, d_answer, sizeof(int), 0, cudaMemcpyDeviceToHost);

    printf("N-Queens(%d) = %d\n", GRID_SIZE, result);
    printf("Execution time: %.3f ms\n", milliseconds);
    gtap_finalize();
    return 0;
}

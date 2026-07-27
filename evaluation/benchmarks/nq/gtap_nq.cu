#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
// #define PROFILE
#include "gtap_thread.cuh"

__device__ unsigned long long d_answer;
__device__ __constant__ int d_n;
__device__ __constant__ int d_cutoff;

__device__ void serial_search(int row, uint32_t col, uint32_t ld, uint32_t rd) {
    int n = d_n;
    if (row == n) {
        atomicAdd(&d_answer, 1ULL);
        return;
    }
    uint32_t mask = (n < 32 ? (1u << n) - 1 : 0xFFFFFFFFu);
    uint32_t avail = mask & ~(col | ld | rd);
    while (avail) {
        uint32_t p = avail & -avail;
        avail -= p;
        serial_search(row + 1, col | p, (ld | p) << 1, (rd | p) >> 1);
    }
}

#pragma gtap function
__device__ void nq(int row, uint32_t col, uint32_t ld, uint32_t rd) {
    int n = d_n;
    int cutoff = d_cutoff;

    if (row > cutoff) {
        serial_search(row, col, ld, rd);
        return;
    }

    uint32_t mask = (n < 32 ? (1u << n) - 1 : 0xFFFFFFFFu);
    uint32_t avail = mask & ~(col | ld | rd);

    while (avail) {
        uint32_t p = avail & -avail;
        avail -= p;
        #pragma gtap task
        nq(row + 1, col | p, (ld | p) << 1, (rd | p) >> 1);
    }
}

__global__ void nq_kernel() {
    #pragma gtap entry
    nq(0, 0, 0, 0);
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s <n> [cutoff]\n", prog);
    fprintf(stderr, "  n: board size (default 16)\n");
    fprintf(stderr, "  cutoff: parallel depth limit (default 7)\n");
}

int main(int argc, char **argv) {
    int n = 16;
    int cutoff = 7;
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) cutoff = atoi(argv[2]);
    if (argc > 3 || n < 0 || cutoff < 0) {
        usage(argv[0]);
        return 1;
    }

    unsigned long long zero = 0ULL;
    cudaMemcpyToSymbol(d_answer, &zero, sizeof(unsigned long long));
    cudaMemcpyToSymbol(d_n, &n, sizeof(int));
    cudaMemcpyToSymbol(d_cutoff, &cutoff, sizeof(int));

    cudaError_t err = gtap_initialize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    nq_kernel<<<GTAP_GRID_SIZE, GTAP_BLOCK_SIZE>>>();
    gtap_synchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    unsigned long long result = 0ULL;
    cudaMemcpyFromSymbol(&result, d_answer, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);

    printf("N-Queens(%d) = %llu\n", n, result);
    printf("Execution time: %.3f ms\n", milliseconds);

#ifdef PROFILE
    gtap_visualize_profile("nq");
#endif

    return 0;
}

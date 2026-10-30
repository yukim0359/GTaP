#pragma once

#include <cuda_runtime.h>

#include "../config_defaults.cuh"

__device__ __constant__ const unsigned long long* g_nCr;

static inline cudaError_t bind_nCr(const unsigned long long* d_nCr) {
    return cudaMemcpyToSymbol(g_nCr, &d_nCr, sizeof(d_nCr));
}

__device__ __forceinline__ unsigned long long binomial_ull(int n, int k) {
    if (k < 0 || k > n) return 0ULL;
    if (k == 0 || k == n) return 1ULL;
    if (k > n - k) k = n - k;
    unsigned long long result = 1ULL;
    for (int i = 0; i < k; ++i) {
        result = result * (unsigned long long)(n - i) / (unsigned long long)(i + 1);
    }
    return result;
}

__device__ __forceinline__ unsigned long long lookup_nCr(int drop, int c) {
    if (drop >= 0 && drop < GTAP_NCR_ROWS && c >= 0 && c < GTAP_NCR_COLS) {
        return g_nCr[(size_t)drop * (size_t)GTAP_NCR_COLS + (size_t)c];
    }
    return binomial_ull(drop, c);
}

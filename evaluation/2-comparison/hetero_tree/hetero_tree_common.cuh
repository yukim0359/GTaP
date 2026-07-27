#pragma once
#include <stdint.h>

// Hetero tree: each leaf has kind = hash(node) % K and runs a distinct
// branch-free compute column. Per-iter work matches binary_tree's
// do_memory_and_compute (mem_ops=0): xorshift + FMA + mix^=bits.
// Kinds differ only in seed/stride constants (equal cost); mixing kinds
// in a warp hurts via SIMT divergence on separate noinline columns.

#ifndef HETERO_TREE_K
#define HETERO_TREE_K 4
#endif

// Same FMA as binary_tree/gtap_thread_binary_tree.cu
__device__ __forceinline__ double mix_fma(double x) {
    return fma(x, 1.0000001192092896, 0.9999999403953552);
}

__device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ uint32_t xorshift32(uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

__device__ __forceinline__ int node_kind(int node) {
    return (int)(hash32((uint32_t)node) % (uint32_t)HETERO_TREE_K);
}

// DAQ: queue 0 = expand (spawn only); queues 1..K = kind compute / leaf tasks.
__device__ __forceinline__ int kind_queue_idx(int node) {
    return 1 + node_kind(node);
}

// Matched to binary leaf compute (mem_ops=0). Kind-specific seed/stride only.
__device__ __noinline__ double compute_kind0(int node, int compute_iters) {
    uint32_t seed = xorshift32((uint32_t)node ^ 0x9e3779b9u);
    uint32_t mix = 0x9e3779b9u ^ (uint32_t)node;
    double y = 0.0;
    for (int it = 0; it < compute_iters; ++it) {
        double x = (double)xorshift32(seed + (uint32_t)it * 747796405u);
        y = mix_fma(x);
        mix ^= (uint32_t)__double_as_longlong(y);
    }
    return (double)mix;
}

__device__ __noinline__ double compute_kind1(int node, int compute_iters) {
    uint32_t seed = xorshift32((uint32_t)node ^ 0x3C6EF372u);
    uint32_t mix = 0x3C6EF372u ^ (uint32_t)node;
    double y = 0.0;
    for (int it = 0; it < compute_iters; ++it) {
        double x = (double)xorshift32(seed + (uint32_t)it * 1664525u);
        y = mix_fma(x);
        mix ^= (uint32_t)__double_as_longlong(y);
    }
    return (double)mix;
}

__device__ __noinline__ double compute_kind2(int node, int compute_iters) {
    uint32_t seed = xorshift32((uint32_t)node ^ 0xA5A5A5A5u);
    uint32_t mix = 0xA5A5A5A5u ^ (uint32_t)node;
    double y = 0.0;
    for (int it = 0; it < compute_iters; ++it) {
        double x = (double)xorshift32(seed + (uint32_t)it * 2654435761u);
        y = mix_fma(x);
        mix ^= (uint32_t)__double_as_longlong(y);
    }
    return (double)mix;
}

__device__ __noinline__ double compute_kind3(int node, int compute_iters) {
    uint32_t seed = xorshift32((uint32_t)node ^ 0xDEADBEEFu);
    uint32_t mix = 0xDEADBEEFu ^ (uint32_t)node;
    double y = 0.0;
    for (int it = 0; it < compute_iters; ++it) {
        double x = (double)xorshift32(seed + (uint32_t)it * 2246822519u);
        y = mix_fma(x);
        mix ^= (uint32_t)__double_as_longlong(y);
    }
    return (double)mix;
}

__device__ unsigned long long g_node_batches;
__device__ unsigned long long g_node_mixed_batches;
__device__ unsigned long long g_node_kind_sum;
__device__ unsigned long long g_node_lane_sum;

__device__ __forceinline__ void hetero_record_mix(int kind) {
    const unsigned mask = __activemask();
    const unsigned m0 = __ballot_sync(mask, kind == 0);
    const unsigned m1 = __ballot_sync(mask, kind == 1);
    const unsigned m2 = __ballot_sync(mask, kind == 2);
    const unsigned m3 = __ballot_sync(mask, kind == 3);
    const int n_kinds = (m0 != 0) + (m1 != 0) + (m2 != 0) + (m3 != 0);
    const int n_lanes = __popc(mask);
    const int leader = __ffs(mask) - 1;
    if ((int)(threadIdx.x & 31) == leader) {
        atomicAdd(&g_node_batches, 1ULL);
        if (n_kinds > 1) atomicAdd(&g_node_mixed_batches, 1ULL);
        atomicAdd(&g_node_kind_sum, (unsigned long long)n_kinds);
        atomicAdd(&g_node_lane_sum, (unsigned long long)n_lanes);
    }
}

__device__ __forceinline__ double do_hetero_compute(int node, int compute_iters) {
    const int kind = node_kind(node);
    hetero_record_mix(kind);
    if (kind == 0) return compute_kind0(node, compute_iters);
    if (kind == 1) return compute_kind1(node, compute_iters);
    if (kind == 2) return compute_kind2(node, compute_iters);
    return compute_kind3(node, compute_iters);
}

__device__ __forceinline__ double compute_kind_block_body(
    int node, int compute_iters, uint32_t seed_xor, uint32_t stride
) {
    const int nt = blockDim.x;
    const int tid = threadIdx.x;
    const int chunk = compute_iters / nt;
    const int rem = compute_iters % nt;
    const int start = tid * chunk + (tid < rem ? tid : rem);
    const int count = chunk + (tid < rem ? 1 : 0);
    uint32_t seed = xorshift32((uint32_t)node ^ seed_xor);
    uint32_t mix = seed_xor ^ (uint32_t)node;
    double y = 0.0;
    for (int i = 0; i < count; ++i) {
        const int it = start + i;
        double x = (double)xorshift32(seed + (uint32_t)it * stride);
        y = mix_fma(x);
        mix ^= (uint32_t)__double_as_longlong(y);
    }
    return (double)mix;
}

__device__ __noinline__ double compute_kind0_block(int node, int compute_iters) {
    return compute_kind_block_body(node, compute_iters, 0x9e3779b9u, 747796405u);
}

__device__ __noinline__ double compute_kind1_block(int node, int compute_iters) {
    return compute_kind_block_body(node, compute_iters, 0x3C6EF372u, 1664525u);
}

__device__ __noinline__ double compute_kind2_block(int node, int compute_iters) {
    return compute_kind_block_body(node, compute_iters, 0xA5A5A5A5u, 2654435761u);
}

__device__ __noinline__ double compute_kind3_block(int node, int compute_iters) {
    return compute_kind_block_body(node, compute_iters, 0xDEADBEEFu, 2246822519u);
}

__device__ __forceinline__ double do_hetero_compute_block(int node, int compute_iters) {
    const int kind = node_kind(node);
    hetero_record_mix(kind);
    if (kind == 0) return compute_kind0_block(node, compute_iters);
    if (kind == 1) return compute_kind1_block(node, compute_iters);
    if (kind == 2) return compute_kind2_block(node, compute_iters);
    return compute_kind3_block(node, compute_iters);
}

static inline __host__ void hetero_clear_mix_stats() {
    unsigned long long z = 0;
    cudaMemcpyToSymbol(g_node_batches, &z, sizeof(z));
    cudaMemcpyToSymbol(g_node_mixed_batches, &z, sizeof(z));
    cudaMemcpyToSymbol(g_node_kind_sum, &z, sizeof(z));
    cudaMemcpyToSymbol(g_node_lane_sum, &z, sizeof(z));
}

static inline __host__ void hetero_print_mix_stats(const char* tag) {
    unsigned long long batches = 0, mixed = 0, kind_sum = 0, lane_sum = 0;
    cudaMemcpyFromSymbol(&batches, g_node_batches, sizeof(batches));
    cudaMemcpyFromSymbol(&mixed, g_node_mixed_batches, sizeof(mixed));
    cudaMemcpyFromSymbol(&kind_sum, g_node_kind_sum, sizeof(kind_sum));
    cudaMemcpyFromSymbol(&lane_sum, g_node_lane_sum, sizeof(lane_sum));
    double mix_frac = (batches > 0) ? (100.0 * (double)mixed / (double)batches) : 0.0;
    double avg_kinds = (batches > 0) ? ((double)kind_sum / (double)batches) : 0.0;
    double avg_lanes = (batches > 0) ? ((double)lane_sum / (double)batches) : 0.0;
    printf("Node-mix [%s]: batches=%llu mixed=%llu (%.1f%%) avg_kinds=%.2f avg_active_lanes=%.1f\n",
           tag, (unsigned long long)batches, (unsigned long long)mixed, mix_frac, avg_kinds, avg_lanes);
}

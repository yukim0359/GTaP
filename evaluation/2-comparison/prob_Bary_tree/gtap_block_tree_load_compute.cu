#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>
#include <stdint.h>
#include <cmath>
#define MAX_PROFILE_DATA 40000
// #define PROFILE
#define GTAP_MAX_TASK_DATA_SIZE 24
#include "gtap_block.cuh"

// ------------------------------
// Device globals
// ------------------------------
__device__ const double* g_input;
__device__ int           g_input_n;

// fixed-size sink to avoid per-node allocation
__device__ double* g_sink;
__device__ int     g_sink_n;

// Bind helper
static inline cudaError_t bind_globals(const double* d_input, int input_n,
                                       double* d_sink, int sink_n) {
    cudaError_t st;
    st = cudaMemcpyToSymbol(g_input, &d_input, sizeof(d_input));
    if (st != cudaSuccess) return st;
    st = cudaMemcpyToSymbol(g_input_n, &input_n, sizeof(input_n));
    if (st != cudaSuccess) return st;

    st = cudaMemcpyToSymbol(g_sink, &d_sink, sizeof(d_sink));
    if (st != cudaSuccess) return st;
    st = cudaMemcpyToSymbol(g_sink_n, &sink_n, sizeof(sink_n));
    if (st != cudaSuccess) return st;

    return cudaSuccess;
}

__device__ __forceinline__ double mix_fma(double x) {
    return fma(x, 1.0000001192092896, 0.9999999403953552);
}

__device__ __forceinline__ uint32_t hash32(uint32_t x){
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

__device__ double do_memory_and_compute(uint32_t node_id, int mem_ops, int compute_iters) {
    double acc = 0.0;

    uint32_t seed = xorshift32(node_id ^ 0x9e3779b9u);
    uint32_t mask = (uint32_t)g_input_n - 1u; // input_n is power of two

    #pragma unroll 1
    for (int m = threadIdx.x; m < mem_ops; m += blockDim.x) {
        uint32_t r = xorshift32(seed + (uint32_t)m * 747796405u);
        int idx = (int)(r & mask);
        acc += g_input[idx];
    }

    double y = 0.0;
    uint32_t mix = (0x9e3779b9u ^ node_id) + (uint32_t)acc;

    #pragma unroll 1
    for (int it = threadIdx.x; it < compute_iters; it += blockDim.x) {
        double x = (double)xorshift32(seed + (uint32_t)it * 747796405u);
        y = mix_fma(x);
        mix ^= (uint32_t)__double_as_longlong(y);
    }

    // asm volatile("" :: "r"(mix), "f"(y));
    return (double)mix;
}

__device__ __forceinline__ void sink_store(uint32_t node_id, double x) {
    // fixed-size sink (power-of-two recommended)
    uint32_t m = (uint32_t)g_sink_n - 1u;
    uint32_t pos = hash32(node_id) & m;
    g_sink[pos] = x; // collisions are OK (last writer wins)
}

// ------------------------------
// Variable-branch tree task (your probability model)
//
// Depth d = D - rem_h (root: d=0, leaf: d=D)
// For a node at depth d, each of its B child candidates is generated
// independently with probability p(d) = rem_h / D = (D - d) / D.
//
// - d=0: rem_h=D => always generate B children
// - d=D: rem_h=0 => leaf
// ------------------------------
#pragma gtap function worker_size(block)
__device__ void tree_work(uint32_t node_id, int rem_h, int D, int B,
                          int mem_ops, int compute_iters) {
    // leaf
    if (rem_h == 0) {
        double v = do_memory_and_compute(node_id, mem_ops, compute_iters);
        if (threadIdx.x == 0) sink_store(node_id, v);
        return;
    }

    int d = D - rem_h; // depth from root (root: 0)

    // seed: node_id と d から一回だけ作る（hash32 1 回だけ）
    uint32_t s;
    if (threadIdx.x == 0) {
        s = hash32(node_id ^ (uint32_t)(d * 0x9e3779b9u));
    }

    // fork: consider B child candidates, each spawned independently with prob p
    __shared__ int spawn;
    if (threadIdx.x == 0) {
        spawn = 0;
        #pragma unroll 1
        for (int k = 0; k < B; ++k) {
            // 乱数更新（超軽量）
            s = xorshift32(s + (uint32_t)k);

            // r in [0, D) を作る（D は小さいので mod で十分）
            // D が 2^n なら r = s & (D-1) にできて更に軽い
            int r = (int)(s % (uint32_t)D);

            if (r < rem_h) {
                uint32_t child = node_id * (uint32_t)B + (uint32_t)(k + 1);
                #pragma gtap task
                tree_work(child, rem_h - 1, D, B, mem_ops, compute_iters);
                spawn++;
            }
        }
    }
    __syncthreads();

    if (spawn > 0) {
        #pragma gtap taskwait
    }

    // own work
    double own = do_memory_and_compute(node_id, mem_ops, compute_iters);
    if (threadIdx.x == 0) sink_store(node_id, own);
}

__global__ void exec_kernel(int height, int B, int mem_ops, int compute_iters) {
    #pragma gtap entry
    tree_work(0u, height, height, B, mem_ops, compute_iters);
}

int main(int argc, char** argv) {
    cudaSetDevice(0);

    int height = 15;
    int B = 3;
    int mem_ops = 64;
    int compute_iters = 512;

    int input_n = 1 << 20;    // power of two
    int sink_n  = 1 << 20;    // power of two (fixed-size)

    if (argc >= 2) height        = atoi(argv[1]);
    if (argc >= 3) compute_iters = atoi(argv[2]);
    if (argc >= 4) mem_ops       = atoi(argv[3]);
    if (argc >= 5) B             = atoi(argv[4]);

    printf("Tree workload (probabilistic B-ary): height=%d B=%d mem_ops=%d compute_iters=%d\n",
           height, B, mem_ops, compute_iters);
    printf("Child spawn prob at depth d: p(d)=1-d/D (d=0 => 1)\n");
    printf("input_n=%d sink_n=%d\n", input_n, sink_n);

    // Host init
    std::mt19937_64 rng(0xC0FFEE);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<double> h_input(input_n);
    for (int i = 0; i < input_n; ++i) h_input[i] = dist(rng);

    // Device alloc
    double* d_input = nullptr;
    double* d_sink  = nullptr;

    cudaMalloc(&d_input, sizeof(double) * (size_t)input_n);
    cudaMalloc(&d_sink,  sizeof(double) * (size_t)sink_n);

    cudaMemcpy(d_input, h_input.data(),
               sizeof(double) * (size_t)input_n, cudaMemcpyHostToDevice);
    cudaMemset(d_sink, 0, sizeof(double) * (size_t)sink_n);

    // Bind device globals
    auto st = bind_globals(d_input, input_n, d_sink, sink_n);
    if (st != cudaSuccess) {
        fprintf(stderr, "bind_globals failed: %s\n", cudaGetErrorString(st));
        return 1;
    }

    // Init task runtime
    st = gtap_initialize();
    if (st != cudaSuccess) {
        fprintf(stderr, "gtap_initialize failed: %s\n", cudaGetErrorString(st));
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    exec_kernel<<<GTAP_GRID_SIZE, GTAP_BLOCK_SIZE>>>(height, B, mem_ops, compute_iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaDeviceSynchronize();

    // Read back a few sink entries as a lightweight sanity check
    std::vector<double> h_sink(8);
    cudaMemcpy(h_sink.data(), d_sink, sizeof(double) * h_sink.size(),
               cudaMemcpyDeviceToHost);

    printf("Sink[0..7]: ");
    for (int i = 0; i < (int)h_sink.size(); ++i) printf("%.3e ", h_sink[i]);
    printf("\n");
    printf("Execution time: %.3f ms\n", ms);

#ifdef PROFILE
    visualize_profile("tree_block");
#endif

    cudaFree(d_input);
    cudaFree(d_sink);
    return 0;
}

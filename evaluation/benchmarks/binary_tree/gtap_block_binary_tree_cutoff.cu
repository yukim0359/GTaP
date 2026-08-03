#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>
// #define PROFILE
#include "gtap_block.cuh"

// Binary tree (block) with OpenMP-style subtree cutoff:
// - height > cutoff: spawn L/R tasks
// - height <= cutoff: ordinary device helper recurses sequentially
//   (same per-node block-tiled compute as gtap_block_binary_tree)
// Default CUTOFF=5 → 32 leaves / task.

#ifndef BINARY_BLOCK_CUTOFF
#define BINARY_BLOCK_CUTOFF 5
#endif

__device__ const double* g_input;
__device__ const int*    g_indices;
__device__ double*       g_out;
__device__ int           g_input_n;
__device__ int           g_indices_n;

static inline cudaError_t bind_globals(const double* d_input, int input_n,
                                       const int* d_indices, int indices_n,
                                       double* d_out) {
    cudaError_t st;
    st = cudaMemcpyToSymbol(g_input, &d_input, sizeof(d_input));
    if (st != cudaSuccess) return st;
    st = cudaMemcpyToSymbol(g_indices, &d_indices, sizeof(d_indices));
    if (st != cudaSuccess) return st;
    st = cudaMemcpyToSymbol(g_out, &d_out, sizeof(d_out));
    if (st != cudaSuccess) return st;
    st = cudaMemcpyToSymbol(g_input_n, &input_n, sizeof(input_n));
    if (st != cudaSuccess) return st;
    st = cudaMemcpyToSymbol(g_indices_n, &indices_n, sizeof(indices_n));
    if (st != cudaSuccess) return st;
    return cudaSuccess;
}

__device__ __forceinline__ double mix_fma(double x) {
    return fma(x, 1.0000001192092896, 0.9999999403953552);
}

__device__ __forceinline__ uint32_t xorshift32(uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

__device__ double do_memory_and_compute(int node, int mem_ops, int compute_iters) {
    double acc = 0.0;

    uint32_t seed = xorshift32((uint32_t)node ^ 0x9e3779b9u);
    uint32_t mask = (uint32_t)g_input_n - 1u;

    const int nt = blockDim.x;
    const int tid = threadIdx.x;

    {
        const int chunk = mem_ops / nt;
        const int rem = mem_ops % nt;
        const int start = tid * chunk + (tid < rem ? tid : rem);
        const int count = chunk + (tid < rem ? 1 : 0);
        for (int i = 0; i < count; ++i) {
            const int m = start + i;
            uint32_t r = xorshift32(seed + (uint32_t)m * 747796405u);
            int idx = (int)(r & mask);
            acc += g_input[idx];
        }
    }

    double y = 0.0;
    uint32_t mix = (0x9e3779b9u ^ (uint32_t)node) + (uint32_t)acc;

    {
        const int chunk = compute_iters / nt;
        const int rem = compute_iters % nt;
        const int start = tid * chunk + (tid < rem ? tid : rem);
        const int count = chunk + (tid < rem ? 1 : 0);
        for (int i = 0; i < count; ++i) {
            const int it = start + i;
            double x = (double)xorshift32(seed + (uint32_t)it * 747796405u);
            y = mix_fma(x);
            mix ^= (uint32_t)__double_as_longlong(y);
        }
    }

    return (double)mix;
}

// Ordinary device helper (NOT a gtap task function). Leaf-only compute.
// BINARY_CUTOFF_FLAT=1: iterate leaves (avoids recursive process_subtree_seq spill).
// default: recursive walk (same as hetero_tree).
#ifndef BINARY_CUTOFF_FLAT
#define BINARY_CUTOFF_FLAT 0
#endif

#if BINARY_CUTOFF_FLAT
__device__ void process_subtree_seq(int node, int height, int mem_ops, int compute_iters) {
    const int nleaves = 1 << height;
    const int first = (node + 1) * nleaves - 1;
    for (int i = 0; i < nleaves; ++i) {
        const int leaf = first + i;
        double v = do_memory_and_compute(leaf, mem_ops, compute_iters);
        if (threadIdx.x == 0) g_out[leaf] = v;
    }
}
#else
__device__ void process_subtree_seq(int node, int height, int mem_ops, int compute_iters) {
    if (height == 0) {
        double v = do_memory_and_compute(node, mem_ops, compute_iters);
        if (threadIdx.x == 0) g_out[node] = v;
        return;
    }
    process_subtree_seq(node * 2 + 1, height - 1, mem_ops, compute_iters);
    process_subtree_seq(node * 2 + 2, height - 1, mem_ops, compute_iters);
}
#endif

#pragma gtap function
__device__ void tree_work(const int node, const int height,
                          const int mem_ops, const int compute_iters,
                          const int cutoff) {
    if (height <= cutoff) {
        process_subtree_seq(node, height, mem_ops, compute_iters);
        return;
    }

    if (threadIdx.x == 0) {
        int l = node * 2 + 1;
        int r = node * 2 + 2;
        #pragma gtap task
        tree_work(l, height - 1, mem_ops, compute_iters, cutoff);
        #pragma gtap task
        tree_work(r, height - 1, mem_ops, compute_iters, cutoff);
    }
    #pragma gtap taskwait
}

__global__ void exec_kernel(int height, int mem_ops, int compute_iters, int cutoff) {
    #pragma gtap entry
    tree_work(0, height, mem_ops, compute_iters, cutoff);
}

int main(int argc, char** argv) {
    cudaSetDevice(0);

    int height = 20;
    int mem_ops = 0;
    int compute_iters = 512;
    int cutoff = BINARY_BLOCK_CUTOFF;
    int input_n = 1 << 20;
    int indices_n = 1 << 20;

    if (argc >= 2) height = atoi(argv[1]);
    if (argc >= 3) compute_iters = atoi(argv[2]);
    if (argc >= 4) mem_ops = atoi(argv[3]);
    if (argc >= 5) cutoff = atoi(argv[4]);

    if (cutoff < 0 || cutoff > height) {
        fprintf(stderr, "invalid cutoff=%d (need 0..height=%d)\n", cutoff, height);
        return 1;
    }

    const int total_nodes = (1 << (height + 1)) - 1;
    const int leaves_per_task = 1 << cutoff;
    printf("Tree workload (block, cutoff): height=%d cutoff=%d leaves/task=%d "
           "flat=%d nodes=%d mem_ops=%d compute_iters=%d\n",
           height, cutoff, leaves_per_task, BINARY_CUTOFF_FLAT, total_nodes, mem_ops,
           compute_iters);

    std::mt19937_64 rng(0xC0FFEE);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::uniform_int_distribution<int> idist(0, input_n - 1);

    std::vector<double> h_input(input_n);
    for (int i = 0; i < input_n; ++i) h_input[i] = dist(rng);

    std::vector<int> h_indices(indices_n);
    for (int i = 0; i < indices_n; ++i) h_indices[i] = idist(rng);

    double* d_input = nullptr;
    int* d_indices = nullptr;
    double* d_out = nullptr;

    cudaMalloc(&d_input,   sizeof(double) * (size_t)input_n);
    cudaMalloc(&d_indices, sizeof(int)    * (size_t)indices_n);
    cudaMalloc(&d_out,     sizeof(double) * (size_t)total_nodes);

    cudaMemcpy(d_input,   h_input.data(),   sizeof(double) * (size_t)input_n,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices.data(), sizeof(int)    * (size_t)indices_n, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(double) * (size_t)total_nodes);

    auto st = bind_globals(d_input, input_n, d_indices, indices_n, d_out);
    if (st != cudaSuccess) {
        fprintf(stderr, "bind_globals failed: %s\n", cudaGetErrorString(st));
        return 1;
    }

    gtap_block_config config{
        .grid_size = GTAP_BENCH_GRID_SIZE,
        .max_tasks_per_block = GTAP_BENCH_MAX_TASKS_PER_BLOCK,
    };
    st = gtap_initialize(config);
    if (st != cudaSuccess) {
        fprintf(stderr, "gtap_initialize failed: %s\n", cudaGetErrorString(st));
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    st = gtap_launch(
        exec_kernel, height, mem_ops, compute_iters, cutoff);
    if (st != cudaSuccess) {
        fprintf(stderr, "gtap_launch failed: %s\n", cudaGetErrorString(st));
        return 1;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaDeviceSynchronize();

    double root = 0.0;
    cudaMemcpy(&root, d_out, sizeof(double), cudaMemcpyDeviceToHost);

    printf("Root: %.6e\n", root);
    printf("Execution time: %.3f ms\n", ms);

#ifdef PROFILE
    gtap_visualize_profile("tree_block_cutoff");
#endif

    cudaFree(d_input);
    cudaFree(d_indices);
    cudaFree(d_out);
    return 0;
}

#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>
#include <stdint.h>

#define GTAP_MAX_TASK_DATA_SIZE 16
#include "gtap_block.cuh"

// ------------------------------
// Tunable synthetic workload (sequential mem access variant)
// ------------------------------
// Each task does:
//  - mem_ops global loads (SEQUENTIAL indices) from g_input
//  - compute_iters iterations of FMA-like arithmetic
// Output: one double per node (g_out[node])

// Device globals
__device__ const double* g_input;
__device__ const int*    g_indices;
__device__ double*       g_out;
__device__ int           g_input_n;
__device__ int           g_indices_n;

// Bind helper
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

// Sequential mem + compute (block-cooperative task)
__device__ double do_memory_and_compute(int node, int mem_ops, int compute_iters) {
    // 1) fixed number of irregular global loads from g_input
    double acc = 0.0;

    // If g_input_n is a power of two (default is 1<<20), masking is valid and fast.
    // Otherwise, replace "& mask" with "% g_input_n".
    uint32_t mask = (uint32_t)g_input_n - 1u;

    // Sequential access region per node
    uint32_t base = ((uint32_t)node * (uint32_t)mem_ops) & mask;

    for (int m = threadIdx.x; m < mem_ops; m += blockDim.x) {
        int idx = (int)((base + (uint32_t)m) & mask);
        acc += g_input[idx];
    }

    // 2) compute loop (distributed across threads)
    double x = acc + (double)(node & 0xFF) * 1e-9;
    for (int it = threadIdx.x; it < compute_iters; it += blockDim.x) {
        x = mix_fma(x);
    }

    asm volatile("" :: "f"(x));
    return x; // per-thread value (not reduced)
}

// ------------------------------
// Tree task: spawn two children and join
// Each node writes one scalar result
// ------------------------------
#pragma gtap function worker_size(block)
__device__ void tree_work(int node, int height, int mem_ops, int compute_iters) {
    if (height == 0) {
        // leaf
        double v = do_memory_and_compute(node, mem_ops, compute_iters); // all threads do work
        if (threadIdx.x == 0) g_out[node] = v; // store thread0's per-thread result (no reduction)
        __syncthreads();
        return;
    } else {
        if (threadIdx.x == 0) {
            int l = node * 2 + 1;
            int r = node * 2 + 2;
            #pragma gtap task
            tree_work(l, height - 1, mem_ops, compute_iters);
            #pragma gtap task
            tree_work(r, height - 1, mem_ops, compute_iters);
        }
        __syncthreads();
        #pragma gtap taskwait

        // own synthetic work only (no combine/reduction)
        double own = do_memory_and_compute(node, mem_ops, compute_iters); // all threads do work
        if (threadIdx.x == 0) g_out[node] = own;
        __syncthreads();
        return;
    }
}

__global__ void exec_kernel(int height, int mem_ops, int compute_iters) {
    #pragma gtap entry
    tree_work(0, height, mem_ops, compute_iters);
}

int main(int argc, char** argv) {
    cudaSetDevice(0);

    int height = 15;
    int mem_ops = 64;         // fixed memory "transactions"
    int compute_iters = 512;  // sweep this for compute intensity
    int input_n = 1 << 20;    // size of global input (tune to exceed caches)
    int indices_n = 1 << 20;  // index stream length

    if (argc >= 2) height = atoi(argv[1]);
    if (argc >= 3) compute_iters = atoi(argv[2]);
    if (argc >= 4) mem_ops = atoi(argv[3]);

    const int total_nodes = (1 << (height + 1)) - 1;
    printf("Tree workload (block task, SEQ mem): height=%d nodes=%d mem_ops=%d compute_iters=%d\n",
           height, total_nodes, mem_ops, compute_iters);

    // Host init
    std::mt19937_64 rng(0xC0FFEE);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::uniform_int_distribution<int> idist(0, input_n - 1);

    std::vector<double> h_input(input_n);
    for (int i = 0; i < input_n; ++i) h_input[i] = dist(rng);

    std::vector<int> h_indices(indices_n);
    for (int i = 0; i < indices_n; ++i) h_indices[i] = idist(rng);

    // Device alloc
    double* d_input = nullptr;
    int* d_indices = nullptr;
    double* d_out = nullptr;

    cudaMalloc(&d_input,   sizeof(double) * (size_t)input_n);
    cudaMalloc(&d_indices, sizeof(int)    * (size_t)indices_n);
    cudaMalloc(&d_out,     sizeof(double) * (size_t)total_nodes);

    cudaMemcpy(d_input,   h_input.data(),   sizeof(double) * (size_t)input_n,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices.data(), sizeof(int)    * (size_t)indices_n, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(double) * (size_t)total_nodes);

    // Bind device globals
    auto st = bind_globals(d_input, input_n, d_indices, indices_n, d_out);
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
    exec_kernel<<<GTAP_GRID_SIZE, GTAP_BLOCK_SIZE>>>(height, mem_ops, compute_iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaDeviceSynchronize();

    // Copy root output and checksum
    double root = 0.0;
    cudaMemcpy(&root, d_out, sizeof(double), cudaMemcpyDeviceToHost);

    printf("Root: %.6e\n", root);
    printf("Execution time: %.3f ms\n", ms);

    cudaFree(d_input);
    cudaFree(d_indices);
    cudaFree(d_out);
    return 0;
}

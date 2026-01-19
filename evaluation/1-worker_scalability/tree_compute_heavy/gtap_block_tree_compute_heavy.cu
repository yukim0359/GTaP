#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>
#define GTAP_MAX_TASK_DATA_SIZE 16

#ifdef GQ
#include "gtap_block_gq.cuh"
#else
#include "gtap_block.cuh"
#endif

// ------------------------------
// Tunable synthetic workload
// ------------------------------
// Each task does:
//  - mem_ops global loads (random indices) from g_input
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

// Prevent the compiler from optimizing away the loads too aggressively
__device__ __forceinline__ double mix_fma(double x) {
    // a small deterministic mix; uses fma to keep ALU busy
    x = fma(x, 1.0000001192092896, 0.9999999403953552);
    x = fma(x, 0.9999999403953552, 1.0000001192092896);
    return x;
}

__device__ __forceinline__ uint32_t hash32(uint32_t x){
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ double do_memory_and_compute(int node, int mem_ops, int compute_iters) {
    // 1) fixed number of random global loads
    double acc = 0.0;
    // make per-node starting point different
    uint32_t base = hash32((uint32_t)node) % (uint32_t)g_indices_n;

    for (int m = threadIdx.x; m < mem_ops; m += blockDim.x) {
        int idx = g_indices[(base + m) % g_indices_n];
        // idx in [0, g_input_n)
        acc += g_input[idx];
    }

    // reduction within the block (simple; deterministic)
    __shared__ double sh[GTAP_BLOCK_SIZE]; // THREADS_PER_BLK is 1024 in your config
    sh[threadIdx.x] = acc;
    __syncthreads();

    // tree-reduction
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) sh[threadIdx.x] += sh[threadIdx.x + offset];
        __syncthreads();
    }
    acc = sh[0];

    // 2) variable compute: FMA-heavy loop
    // all threads run to model compute, but only thread0 returns final
    double x = acc + (double)(node & 0xFF) * 1e-9;
    for (int it = threadIdx.x; it < compute_iters; it += blockDim.x) {
        x = mix_fma(x);
    }

    // fold x across threads similarly (so compute isn't trivially dead)
    sh[threadIdx.x] = x;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) sh[threadIdx.x] += sh[threadIdx.x + offset];
        __syncthreads();
    }
    return sh[0];
}

// ------------------------------
// Tree task: spawn two children and join
// Each node writes one scalar result
// ------------------------------
#pragma gtap function worker_size(block)
__device__ void tree_work(int node, int height, int mem_ops, int compute_iters) {
    // if (threadIdx.x == 0) printf("tree_work: node=%d height=%d mem_ops=%d compute_iters=%d\n", node, height, mem_ops, compute_iters);
    if (height == 0) {
        // leaf
        double v = do_memory_and_compute(node, mem_ops, compute_iters);
        if (threadIdx.x == 0) g_out[node] = v;
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

        // combine child results + own synthetic work
        double own = do_memory_and_compute(node, mem_ops, compute_iters);
        // if (threadIdx.x == 0) g_out[node] = own;
        if (threadIdx.x == 0) {
            double lv = g_out[node * 2 + 1];
            double rv = g_out[node * 2 + 2];
            // cheap combine; you can choose any associative-ish op
            g_out[node] = (lv + rv) * 0.5 + own * 1e-6;
        }
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
    printf("Tree workload: height=%d nodes=%d mem_ops=%d compute_iters=%d\n",
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

    // Optional: checksum over all nodes (costly if huge)
    // Here we just print root; for stronger validation, sum a subset.
    printf("Root: %.6e\n", root);
    printf("Execution time: %.3f ms\n", ms);

    cudaFree(d_input);
    cudaFree(d_indices);
    cudaFree(d_out);
    return 0;
}

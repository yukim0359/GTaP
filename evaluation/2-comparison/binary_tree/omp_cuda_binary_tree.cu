#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <omp.h>
#include <cuda_runtime.h>

#ifndef GPU_BLOCK_SIZE
#define GPU_BLOCK_SIZE 256
#endif

// ------------------------------
// Device helpers: hash + fma-mix
// ------------------------------
__host__ __device__ __forceinline__ uint32_t hash32(uint32_t x){
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ double mix_fma(double x) {
    return fma(x, 1.0000001192092896, 0.9999999403953552);
}

// ------------------------------
// One-node work kernel (leaf)
//   out[node] = work(node)
// ------------------------------
__device__ double do_memory_and_compute(int node, int mem_ops, int compute_iters,
                                         const double* __restrict__ input, int input_n) {
    // 1) fixed number of irregular global loads from input
    double acc = 0.0;

    // Per-node seed (same logical random stream as thread/OMP versions).
    // Threads within a block cooperate by iterating m = threadIdx.x, threadIdx.x + blockDim.x, ...
    // so the overall sequence of mem_ops indices per node matches the CPU / 1-thread-per-task GPU version.
    uint32_t seed = hash32((uint32_t)node ^ 0x9e3779b9u);

    // If input_n is a power of two (your default is 1<<20), masking is valid and fast.
    // Otherwise, replace "& mask" with "% input_n".
    uint32_t mask = (uint32_t)input_n - 1u;

    for (int m = threadIdx.x; m < mem_ops; m += blockDim.x) {
        uint32_t r = hash32(seed + (uint32_t)m);
        int idx = (int)(r & mask);              // power-of-two case
        // int idx = (int)(r % (uint32_t)input_n); // general case
        acc += input[idx];
    }

    // 2) compute loop (distributed across threads as before)
    double x = acc + (double)(node & 0xFF) * 1e-9;
    for (int it = threadIdx.x; it < compute_iters; it += blockDim.x) {
        x = mix_fma(x);
    }

    asm volatile("" :: "f"(x));
    return x; // per-thread value (not reduced)
}

__global__ void leaf_kernel(int node, int mem_ops, int compute_iters,
                            const double* __restrict__ input, int input_n,
                            double* __restrict__ out) {
    // all threads do work
    double v = do_memory_and_compute(node, mem_ops, compute_iters, input, input_n);
    if (threadIdx.x == 0) {
        out[node] = v; // store thread0's per-thread result (no reduction)
    }
}

// ------------------------------
// Internal-node kernel
//   out[node] = combine(out[l], out[r], work(node))
//   (caller ensures l/r are completed via stream waits)
// ------------------------------
__global__ void internal_kernel(int node, int mem_ops, int compute_iters,
                                const double* __restrict__ input, int input_n,
                                double* __restrict__ out)
{
    // own synthetic work only (no combine/reduction)
    // all threads do work
    double own = do_memory_and_compute(node, mem_ops, compute_iters, input, input_n);
    if (threadIdx.x == 0) {
        out[node] = own;
    }
}

// ------------------------------
// Host-side OpenMP task recursion
// Each node has an event that fires when out[node] is ready.
// We use one CUDA stream per OpenMP thread.
// ------------------------------
struct Ctx {
    const double* d_input;
    const int*    d_indices;
    double*       d_out;
    int input_n;
    int indices_n;
    int mem_ops;
    int compute_iters;

    std::vector<cudaStream_t> streams; // [omp_threads]
};

static inline void cuda_check(cudaError_t st, const char* msg) {
    if (st != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(st));
        std::exit(1);
    }
}

static void tree_omp_cuda(int node, int height, Ctx* ctx) {
    if (height == 0) {
        int tid = omp_get_thread_num();
        cudaStream_t s = ctx->streams[tid];
        leaf_kernel<<<1, GPU_BLOCK_SIZE, 0, s>>>(
            node, ctx->mem_ops, ctx->compute_iters,
            ctx->d_input, ctx->input_n,
            ctx->d_out
        );
        cuda_check(cudaStreamSynchronize(s), "StreamSync");
        return;
    }

    int l = node * 2 + 1;
    int r = node * 2 + 2;

    // spawn children
    #pragma omp task default(none) firstprivate(l, height) shared(ctx)
    tree_omp_cuda(l, height - 1, ctx);
    #pragma omp task default(none) firstprivate(r, height) shared(ctx)
    tree_omp_cuda(r, height - 1, ctx);

    // wait until child OpenMP tasks have at least enqueued & recorded their events
    #pragma omp taskwait

    int tid = omp_get_thread_num();
    cudaStream_t s = ctx->streams[tid];
    internal_kernel<<<1, GPU_BLOCK_SIZE, 0, s>>>(
        node, ctx->mem_ops, ctx->compute_iters,
        ctx->d_input, ctx->input_n,
        ctx->d_out
    );
    cuda_check(cudaStreamSynchronize(s), "StreamSync");
}

// ------------------------------
// main
// ------------------------------
int main(int argc, char** argv) {
    int height = 15;
    int compute_iters = 512;
    int mem_ops = 64;
    int input_n = 1 << 20;
    int indices_n = 1 << 20;

    if (argc >= 2) height = std::atoi(argv[1]);
    if (argc >= 3) compute_iters = std::atoi(argv[2]);
    if (argc >= 4) mem_ops = std::atoi(argv[3]);

    const int total_nodes = (1 << (height + 1)) - 1;

    std::printf("OMP+CUDA Tree workload: height=%d nodes=%d mem_ops=%d compute_iters=%d\n",
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
    int*    d_indices = nullptr;
    double* d_out = nullptr;

    cuda_check(cudaMalloc(&d_input,   sizeof(double) * (size_t)input_n),   "cudaMalloc(d_input)");
    cuda_check(cudaMalloc(&d_indices, sizeof(int)    * (size_t)indices_n), "cudaMalloc(d_indices)");
    cuda_check(cudaMalloc(&d_out,     sizeof(double) * (size_t)total_nodes),"cudaMalloc(d_out)");

    cuda_check(cudaMemcpy(d_input, h_input.data(),
                          sizeof(double) * (size_t)input_n, cudaMemcpyHostToDevice),
               "Memcpy input");
    cuda_check(cudaMemcpy(d_indices, h_indices.data(),
                          sizeof(int) * (size_t)indices_n, cudaMemcpyHostToDevice),
               "Memcpy indices");
    cuda_check(cudaMemset(d_out, 0, sizeof(double) * (size_t)total_nodes), "Memset out");

    // Setup context
    Ctx ctx;
    ctx.d_input = d_input;
    ctx.d_indices = d_indices;
    ctx.d_out = d_out;
    ctx.input_n = input_n;
    ctx.indices_n = indices_n;
    ctx.mem_ops = mem_ops;
    ctx.compute_iters = compute_iters;

    // One stream per OpenMP thread
    int omp_threads = omp_get_max_threads();
    ctx.streams.resize(omp_threads);
    for (int t = 0; t < omp_threads; ++t) {
        cuda_check(cudaStreamCreateWithFlags(&ctx.streams[t], cudaStreamNonBlocking),
                   "StreamCreate");
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            /* no op */
        }
    }

    // Timing
    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            tree_omp_cuda(0, height, &ctx);
        }
    }
    // Wait for root completion, then stop timer
    cuda_check(cudaDeviceSynchronize(), "DeviceSync");
    double t1 = omp_get_wtime();
    double ms = (t1 - t0) * 1000.0;

    double root = 0.0;
    cuda_check(cudaMemcpy(&root, d_out, sizeof(double), cudaMemcpyDeviceToHost), "Memcpy root");

    std::printf("Root: %.6e\n", root);
    std::printf("Execution time: %.3f ms\n", ms);

    // Cleanup
    for (int t = 0; t < omp_threads; ++t) cudaStreamDestroy(ctx.streams[t]);

    cudaFree(d_input);
    cudaFree(d_indices);
    cudaFree(d_out);
    return 0;
}

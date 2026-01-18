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
    x = fma(x, 1.0000001192092896, 0.9999999403953552);
    x = fma(x, 0.9999999403953552, 1.0000001192092896);
    return x;
}

// ------------------------------
// One-node work kernel (leaf)
//   out[node] = work(node)
// ------------------------------
__global__ void leaf_kernel(int node, int mem_ops, int compute_iters,
                            const double* __restrict__ input, int input_n,
                            const int* __restrict__ indices, int indices_n,
                            double* __restrict__ out) {
    // 1) fixed number of random global loads
    uint32_t base = hash32((uint32_t)node) % (uint32_t)indices_n;

    double acc = 0.0;
    for (int m = threadIdx.x; m < mem_ops; m += blockDim.x) {
        int idx = indices[(base + (uint32_t)m) % (uint32_t)indices_n];
        acc += input[idx];
    }

    // reduction in block
    __shared__ double sh[GPU_BLOCK_SIZE];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sh[threadIdx.x] += sh[threadIdx.x + off];
        __syncthreads();
    }
    acc = sh[0];

    // 2) compute
    double x = acc + (double)(node & 0xFF) * 1e-9;
    for (int it = threadIdx.x; it < compute_iters; it += blockDim.x) {
        x = mix_fma(x);
    }

    // fold again (avoid dead-code)
    sh[threadIdx.x] = x;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sh[threadIdx.x] += sh[threadIdx.x + off];
        __syncthreads();
    }

    if (threadIdx.x == 0) out[node] = sh[0];
}

// ------------------------------
// Internal-node kernel
//   out[node] = combine(out[l], out[r], work(node))
//   (caller ensures l/r are completed via stream waits)
// ------------------------------
__global__ void internal_kernel(int node, int l, int r, int mem_ops, int compute_iters,
                                const double* __restrict__ input, int input_n,
                                const int* __restrict__ indices, int indices_n,
                                double* __restrict__ out)
{
    // compute own work (same as leaf, but we'll combine)
    uint32_t base = hash32((uint32_t)node) % (uint32_t)indices_n;

    double acc = 0.0;
    for (int m = threadIdx.x; m < mem_ops; m += blockDim.x) {
        int idx = indices[(base + (uint32_t)m) % (uint32_t)indices_n];
        acc += input[idx];
    }

    __shared__ double sh[GPU_BLOCK_SIZE];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sh[threadIdx.x] += sh[threadIdx.x + off];
        __syncthreads();
    }
    acc = sh[0];

    double x = acc + (double)(node & 0xFF) * 1e-9;
    for (int it = threadIdx.x; it < compute_iters; it += blockDim.x) {
        x = mix_fma(x);
    }

    sh[threadIdx.x] = x;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sh[threadIdx.x] += sh[threadIdx.x + off];
        __syncthreads();
    }
    double own = sh[0];

    if (threadIdx.x == 0) {
        double lv = out[l];
        double rv = out[r];
        out[node] = (lv + rv) * 0.5 + own * 1e-6;
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
    int total_nodes;

    std::vector<cudaStream_t> streams; // [omp_threads]
    std::vector<cudaEvent_t>  done;    // [total_nodes]
};

static inline void cuda_check(cudaError_t st, const char* msg) {
    if (st != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(st));
        std::exit(1);
    }
}

static void tree_omp_cuda(int node, int height, Ctx* ctx) {
    if (node >= ctx->total_nodes) return;

    if (height == 0) {
        #pragma omp task default(none) firstprivate(node) shared(ctx)
        {
            int tid = omp_get_thread_num();
            cudaStream_t s = ctx->streams[tid];

            leaf_kernel<<<1, GPU_BLOCK_SIZE, 0, s>>>(
                node, ctx->mem_ops, ctx->compute_iters,
                ctx->d_input, ctx->input_n,
                ctx->d_indices, ctx->indices_n,
                ctx->d_out
            );
            cuda_check(cudaEventRecord(ctx->done[node], s), "EventRecord(leaf)");
        }
        return;
    }

    int l = node * 2 + 1;
    int r = node * 2 + 2;

    // spawn children
    tree_omp_cuda(l, height - 1, ctx);
    tree_omp_cuda(r, height - 1, ctx);

    // wait until child OpenMP tasks have at least enqueued & recorded their events
    #pragma omp taskwait

    // internal node task: BLOCK on GPU completion of children, then launch parent
    #pragma omp task default(none) firstprivate(node, l, r) shared(ctx)
    {
        // ここで「子のGPU完了」まで待つ
        cuda_check(cudaEventSynchronize(ctx->done[l]), "EventSync(l)");
        cuda_check(cudaEventSynchronize(ctx->done[r]), "EventSync(r)");

        int tid = omp_get_thread_num();
        cudaStream_t s = ctx->streams[tid];

        internal_kernel<<<1, GPU_BLOCK_SIZE, 0, s>>>(
            node, l, r, ctx->mem_ops, ctx->compute_iters,
            ctx->d_input, ctx->input_n,
            ctx->d_indices, ctx->indices_n,
            ctx->d_out
        );
        cuda_check(cudaEventRecord(ctx->done[node], s), "EventRecord(internal)");
    };
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
    ctx.total_nodes = total_nodes;

    // One stream per OpenMP thread
    int omp_threads = omp_get_max_threads();
    ctx.streams.resize(omp_threads);
    for (int t = 0; t < omp_threads; ++t) {
        cuda_check(cudaStreamCreateWithFlags(&ctx.streams[t], cudaStreamNonBlocking),
                   "StreamCreate");
    }

    // One event per node (disable timing to reduce overhead)
    ctx.done.resize((size_t)total_nodes);
    for (int i = 0; i < total_nodes; ++i) {
        cuda_check(cudaEventCreateWithFlags(&ctx.done[i], cudaEventDisableTiming),
                   "EventCreate");
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
    cuda_check(cudaEventSynchronize(ctx.done[0]), "EventSync(root)");
    cuda_check(cudaDeviceSynchronize(), "DeviceSync");
    double t1 = omp_get_wtime();
    double ms = (t1 - t0) * 1000.0;

    double root = 0.0;
    cuda_check(cudaMemcpy(&root, d_out, sizeof(double), cudaMemcpyDeviceToHost), "Memcpy root");

    std::printf("Root: %.6e\n", root);
    std::printf("Execution time: %.3f ms\n", ms);

    // Cleanup
    for (int i = 0; i < total_nodes; ++i) cudaEventDestroy(ctx.done[i]);
    for (int t = 0; t < omp_threads; ++t) cudaStreamDestroy(ctx.streams[t]);

    cudaFree(d_input);
    cudaFree(d_indices);
    cudaFree(d_out);
    return 0;
}

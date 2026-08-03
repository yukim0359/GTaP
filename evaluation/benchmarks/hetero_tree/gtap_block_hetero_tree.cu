#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
// #define PROFILE
#include "gtap_block.cuh"
#include "hetero_tree_common.cuh"

// Hetero tree (block): kind-specific compute on leaves only.

__device__ double* g_out;

static inline cudaError_t bind_globals(double* d_out) {
    return cudaMemcpyToSymbol(g_out, &d_out, sizeof(d_out));
}

#pragma gtap function
__device__ void tree_work(const int node, const int height,
                          const int compute_iters) {
    if (height == 0) {
        double v = do_hetero_compute_block(node, compute_iters);
        if (threadIdx.x == 0) g_out[node] = v;
        return;
    }

    if (threadIdx.x == 0) {
        int l = node * 2 + 1;
        int r = node * 2 + 2;
        #pragma gtap task
        tree_work(l, height - 1, compute_iters);
        #pragma gtap task
        tree_work(r, height - 1, compute_iters);
    }
    #pragma gtap taskwait
}

__global__ void exec_kernel(int height, int compute_iters) {
    #pragma gtap entry
    tree_work(0, height, compute_iters);
}

int main(int argc, char** argv) {
    cudaSetDevice(0);

    int height = 20;
    int compute_iters = 1024;
    if (argc >= 2) height = atoi(argv[1]);
    if (argc >= 3) compute_iters = atoi(argv[2]);

    if (height >= 30) {
        fprintf(stderr,
                "height=%d requires int64 node IDs (node*2 overflows int); aborting.\n",
                height);
        return 1;
    }
    if (height < 0 || height > 29) {
        fprintf(stderr, "invalid height=%d\n", height);
        return 1;
    }

    const int64_t total_nodes = (int64_t(1) << (height + 1)) - 1;
    const size_t out_bytes = sizeof(double) * (size_t)total_nodes;
    printf("Hetero tree (block): height=%d nodes=%lld K=%d compute_iters=%d (out=%.2f GiB)\n",
           height, (long long)total_nodes, HETERO_TREE_K, compute_iters,
           out_bytes / (1024.0 * 1024.0 * 1024.0));

    double* d_out = nullptr;
    cudaError_t st = cudaMalloc(&d_out, out_bytes);
    if (st != cudaSuccess) {
        fprintf(stderr, "cudaMalloc(g_out, %zu bytes) failed: %s\n", out_bytes, cudaGetErrorString(st));
        return 1;
    }
    st = bind_globals(d_out);
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

    hetero_clear_mix_stats();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    st = gtap_launch(exec_kernel, height, compute_iters);
    if (st != cudaSuccess) {
        fprintf(stderr, "gtap_launch failed: %s\n", cudaGetErrorString(st));
        return 1;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaDeviceSynchronize();

    double sample = 0.0;
    int64_t sample_idx = (height == 0) ? 0 : ((int64_t(1) << height) - 1);
    cudaMemcpy(&sample, d_out + sample_idx, sizeof(double), cudaMemcpyDeviceToHost);

    printf("Sample leaf[%lld]: %.6e\n", (long long)sample_idx, sample);
    printf("Execution time: %.3f ms\n", ms);
    hetero_print_mix_stats("block");

#ifdef PROFILE
    gtap_visualize_profile("hetero_tree_block");
#endif

    cudaFree(d_out);
    return 0;
}

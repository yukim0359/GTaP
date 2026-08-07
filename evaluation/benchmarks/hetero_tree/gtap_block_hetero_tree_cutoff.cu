#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
// #define GTAP_PROFILE
#include "gtap_block.cuh"
#include "hetero_tree_common.cuh"

// Hetero tree (block) with subtree cutoff (OpenMP-style):
// - height > cutoff: spawn L/R tasks (gtap function only via task/entry)
// - height <= cutoff: ordinary device helper recurses sequentially;
//   each leaf is block-tiled via do_hetero_compute_block
// Default CUTOFF=5 → 32 leaves / task (fewer tasks, same leaf compute model).

#ifndef HETERO_BLOCK_CUTOFF
#define HETERO_BLOCK_CUTOFF 5
#endif

__device__ double* g_out;

static inline cudaError_t bind_globals(double* d_out) {
    return cudaMemcpyToSymbol(g_out, &d_out, sizeof(d_out));
}

// Ordinary device recursion (NOT a gtap task function).
__device__ void process_subtree_seq(int node, int height, int compute_iters) {
    if (height == 0) {
        double v = do_hetero_compute_block(node, compute_iters);
        if (threadIdx.x == 0) g_out[node] = v;
        return;
    }
    process_subtree_seq(node * 2 + 1, height - 1, compute_iters);
    process_subtree_seq(node * 2 + 2, height - 1, compute_iters);
}

#pragma gtap function
__device__ void tree_work(const int node, const int height,
                          const int compute_iters, const int cutoff) {
    if (height <= cutoff) {
        process_subtree_seq(node, height, compute_iters);
        return;
    }

    if (threadIdx.x == 0) {
        int l = node * 2 + 1;
        int r = node * 2 + 2;
        #pragma gtap task
        tree_work(l, height - 1, compute_iters, cutoff);
        #pragma gtap task
        tree_work(r, height - 1, compute_iters, cutoff);
    }
    #pragma gtap taskwait
}

__global__ void exec_kernel(int height, int compute_iters, int cutoff) {
    #pragma gtap entry
    tree_work(0, height, compute_iters, cutoff);
}

int main(int argc, char** argv) {
    cudaSetDevice(0);

    int height = 20;
    int compute_iters = 1024;
    int cutoff = HETERO_BLOCK_CUTOFF;
    if (argc >= 2) height = atoi(argv[1]);
    if (argc >= 3) compute_iters = atoi(argv[2]);
    if (argc >= 4) cutoff = atoi(argv[3]);

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
    if (cutoff < 0 || cutoff > height) {
        fprintf(stderr, "invalid cutoff=%d (need 0..height=%d)\n", cutoff, height);
        return 1;
    }

    const int64_t total_nodes = (int64_t(1) << (height + 1)) - 1;
    const int leaves_per_task = 1 << cutoff;
    const size_t out_bytes = sizeof(double) * (size_t)total_nodes;
    printf("Hetero tree (block, cutoff): height=%d cutoff=%d leaves/task=%d "
           "(sequential recurse + block-tiled leaf) K=%d compute_iters=%d "
           "nodes=%lld (out=%.2f GiB)\n",
           height, cutoff, leaves_per_task, HETERO_TREE_K, compute_iters,
           (long long)total_nodes, out_bytes / (1024.0 * 1024.0 * 1024.0));

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
    st = gtap_launch(exec_kernel, height, compute_iters, cutoff);
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
    hetero_print_mix_stats("block-cutoff");

#ifdef GTAP_PROFILE
    gtap_export_profile();
#endif

    cudaFree(d_out);
    return 0;
}

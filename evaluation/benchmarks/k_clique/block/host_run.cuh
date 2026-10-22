#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>

#include "gtap_block.cuh"
#include "config_defaults.cuh"
#include "../common/host_graph.hpp"
#include "../common/host_options.hpp"
#include "../common/timing.hpp"
#include "block_device_state.cuh"

__global__ void exec_kernel_k();

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err__));        \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

struct GtapKCliqueDeviceBuffers {
    int* d_row_ptr = nullptr;
    int* d_col_idx = nullptr;
    int* d_edge_src = nullptr;
    int* d_cand_scratch = nullptr;
    unsigned long long* d_bit_scratch = nullptr;
    unsigned long long* d_encode_scratch = nullptr;
    int* d_encode_slot_stack = nullptr;
    unsigned long long* d_recurse_tree_stats = nullptr;

    size_t cand_scratch_ints = 0;
    size_t bit_scratch_words = 0;
    size_t encode_scratch_words = 0;
    size_t num_oriented_edges = 0;
    int num_workers = 0;
};

struct GtapKCliqueRunResult {
    // Narrower than the thread host: no count_e2e / transfer / orient split.
    // preprocess_ms here is one blob (orient + transfer + workspace).
    double preprocess_ms = 0.0;
    double count_phase_ms = 0.0;
    double kernel_ms = 0.0;
    unsigned long long gpu_count = 0ULL;
    int scratch_overflow = 0;
#ifdef GTAP_STATS
    int candidate_highwater = 0;
    unsigned long long recurse_encoded_calls = 0ULL;
    unsigned long long recurse_encoded_roots = 0ULL;
    unsigned long long recurse_max_encoded_tree_calls = 0ULL;
    unsigned long long task_max_cycles[K_PROFILE_KINDS] = {0ULL};
    int task_max_info[K_PROFILE_KINDS * K_PROFILE_FIELDS] = {0};
#endif
};

struct GtapKCliqueHostHooks {
    GtapOrientMode default_orient = GTAP_ORIENT_DEGREE;
    bool set_cuda_stack_limit = false;
    const char* profile_name = nullptr;
    bool (*on_count_phase_begin)(GtapKCliqueDeviceBuffers& buffers) = nullptr;
    cudaError_t (*bind_extras)() = nullptr;
    void (*free_extras)() = nullptr;
    void (*print_variant_tuning)() = nullptr;
};

static void gtap_k_clique_free_device_buffers(GtapKCliqueDeviceBuffers& buffers) {
    cudaFree(buffers.d_row_ptr);
    cudaFree(buffers.d_col_idx);
    cudaFree(buffers.d_edge_src);
    cudaFree(buffers.d_cand_scratch);
    cudaFree(buffers.d_bit_scratch);
    cudaFree(buffers.d_encode_scratch);
    cudaFree(buffers.d_encode_slot_stack);
    cudaFree(buffers.d_recurse_tree_stats);
    buffers = GtapKCliqueDeviceBuffers{};
}

static bool gtap_k_clique_run(
    const HostOptions& options,
    const HostGraph& graph,
    const GtapKCliqueHostHooks& hooks,
    GtapKCliqueDeviceBuffers& buffers,
    GtapKCliqueRunResult& result) {
    const int num_exec_edges = (int)buffers.num_oriented_edges;
    timespec count_phase_start, count_phase_stop;
    clock_gettime(CLOCK_MONOTONIC, &count_phase_start);

    if (hooks.on_count_phase_begin != nullptr && !hooks.on_count_phase_begin(buffers)) {
        if (hooks.free_extras != nullptr) hooks.free_extras();
        gtap_k_clique_free_device_buffers(buffers);
        return false;
    }

    {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
            fprintf(stderr,
                    "GPU mem before gtap_initialize: free %.1f / total %.1f GiB\n",
                    free_bytes / (1024.0 * 1024.0 * 1024.0),
                    total_bytes / (1024.0 * 1024.0 * 1024.0));
        }
        fprintf(stderr, "gtap_initialize: allocating + zeroing (may take 1-2 min)...\n");
        fflush(stderr);
    }

    timespec init_start, init_stop;
    clock_gettime(CLOCK_MONOTONIC, &init_start);
    gtap_block_config config{
        .grid_size = GTAP_PIVOT_BLOCK_GRID_SIZE,
        .block_size = GTAP_PIVOT_BLOCK_BLOCK_SIZE,
        .max_tasks_per_block = GTAP_PIVOT_BLOCK_MAX_TASKS_PER_BLOCK,
    };
    cudaError_t err = gtap_initialize(config);
    clock_gettime(CLOCK_MONOTONIC, &init_stop);
    fprintf(stderr, "gtap_initialize done in %.1f s\n",
            elapsed_ms(init_start, init_stop) / 1000.0);
    fflush(stderr);
    if (err != cudaSuccess) {
        fprintf(stderr, "gtap_initialize failed: %s\n", cudaGetErrorString(err));
        if (hooks.free_extras != nullptr) hooks.free_extras();
        gtap_k_clique_free_device_buffers(buffers);
        return false;
    }

    const int encode_slot_top_init = GTAP_ENCODE_SLOTS;
    unsigned long long zero = 0ULL;
    int scratch_overflow = 0;
    int encode_slot_lock_init = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(g_answer, &zero, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_scratch_overflow, &scratch_overflow, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_encode_slot_lock, &encode_slot_lock_init, sizeof(int)));
#ifdef GTAP_STATS
    int candidate_highwater = 0;
    unsigned long long profile_cycles_zero[K_PROFILE_KINDS] = {0ULL};
    int profile_info_zero[K_PROFILE_KINDS * K_PROFILE_FIELDS] = {0};
    unsigned long long recurse_zero = 0ULL;
    CUDA_CHECK(cudaMemcpyToSymbol(g_candidate_highwater, &candidate_highwater, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(
        g_task_max_cycles,
        profile_cycles_zero,
        sizeof(profile_cycles_zero)));
    CUDA_CHECK(cudaMemcpyToSymbol(
        g_task_max_info,
        profile_info_zero,
        sizeof(profile_info_zero)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_recurse_encoded_calls, &recurse_zero, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_recurse_encoded_roots, &recurse_zero, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyToSymbol(
        g_recurse_max_encoded_tree_calls, &recurse_zero, sizeof(unsigned long long)));
#endif
    CUDA_CHECK(cudaMemcpyToSymbol(g_encode_slot_top, &encode_slot_top_init, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_num_vertices, &graph.n, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_num_edges, &num_exec_edges, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_clique_k, &options.k, sizeof(int)));
    err = bind_graph(
        buffers.d_row_ptr,
        buffers.d_col_idx,
        buffers.d_edge_src,
        buffers.d_cand_scratch,
        buffers.d_bit_scratch,
        buffers.d_encode_scratch,
        buffers.d_encode_slot_stack);
    if (err != cudaSuccess) {
        fprintf(stderr, "bind_graph failed: %s\n", cudaGetErrorString(err));
        if (hooks.free_extras != nullptr) hooks.free_extras();
        gtap_k_clique_free_device_buffers(buffers);
        gtap_finalize();
        return false;
    }
    if (hooks.bind_extras != nullptr) {
        err = hooks.bind_extras();
        if (err != cudaSuccess) {
            fprintf(stderr, "bind_extras failed: %s\n", cudaGetErrorString(err));
            if (hooks.free_extras != nullptr) hooks.free_extras();
            gtap_k_clique_free_device_buffers(buffers);
            gtap_finalize();
            return false;
        }
    }
#ifdef GTAP_STATS
    err = bind_recurse_stats(buffers.d_recurse_tree_stats);
    if (err != cudaSuccess) {
        fprintf(stderr, "bind_recurse_stats failed: %s\n", cudaGetErrorString(err));
        if (hooks.free_extras != nullptr) hooks.free_extras();
        gtap_k_clique_free_device_buffers(buffers);
        gtap_finalize();
        return false;
    }
#endif

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    fprintf(stderr, "launching exec_kernel_k<<<GRID=%d, BLOCK=%d>>> ...\n",
            GTAP_PIVOT_BLOCK_GRID_SIZE, GTAP_PIVOT_BLOCK_BLOCK_SIZE);
    fflush(stderr);
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(gtap_launch(exec_kernel_k));
    CUDA_CHECK(gtap_synchronize());
    CUDA_CHECK(cudaGetLastError());
    fprintf(stderr, "exec_kernel_k finished\n");
    fflush(stderr);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    clock_gettime(CLOCK_MONOTONIC, &count_phase_stop);

    result.count_phase_ms = elapsed_ms(count_phase_start, count_phase_stop);
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    result.kernel_ms = kernel_ms;
    CUDA_CHECK(cudaMemcpyFromSymbol(&result.gpu_count, g_answer, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyFromSymbol(
        &result.scratch_overflow, g_scratch_overflow, sizeof(int)));
#ifdef GTAP_STATS
    CUDA_CHECK(cudaMemcpyFromSymbol(
        &result.candidate_highwater, g_candidate_highwater, sizeof(int)));
    CUDA_CHECK(cudaMemcpyFromSymbol(
        result.task_max_cycles,
        g_task_max_cycles,
        sizeof(result.task_max_cycles)));
    CUDA_CHECK(cudaMemcpyFromSymbol(
        result.task_max_info,
        g_task_max_info,
        sizeof(result.task_max_info)));
    CUDA_CHECK(cudaMemcpyFromSymbol(
        &result.recurse_encoded_calls, g_recurse_encoded_calls, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyFromSymbol(
        &result.recurse_encoded_roots, g_recurse_encoded_roots, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyFromSymbol(
        &result.recurse_max_encoded_tree_calls,
        g_recurse_max_encoded_tree_calls,
        sizeof(unsigned long long)));
    std::vector<unsigned long long> h_recurse_tree_encoded((size_t)buffers.num_workers, 0ULL);
    CUDA_CHECK(cudaMemcpy(
        h_recurse_tree_encoded.data(),
        buffers.d_recurse_tree_stats,
        (size_t)buffers.num_workers * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost));
    for (int worker = 0; worker < buffers.num_workers; ++worker) {
        if (h_recurse_tree_encoded[(size_t)worker] > result.recurse_max_encoded_tree_calls) {
            result.recurse_max_encoded_tree_calls = h_recurse_tree_encoded[(size_t)worker];
        }
    }
#endif

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return true;
}

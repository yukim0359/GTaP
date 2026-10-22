#pragma once

#include <cstdio>
#include <cuda_runtime.h>

#include "gtap_block.cuh"
#include "config_defaults.cuh"
#include "../common/host_options.hpp"
#include "block_device_state.cuh"

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
    unsigned long long task_max_cycles[K_PROFILE_KINDS] = {0ULL};
    int task_max_info[K_PROFILE_KINDS * K_PROFILE_FIELDS] = {0};
#endif
};

static void free_device_buffers(GtapKCliqueDeviceBuffers& buffers) {
    cudaFree(buffers.d_row_ptr);
    cudaFree(buffers.d_col_idx);
    cudaFree(buffers.d_edge_src);
    cudaFree(buffers.d_cand_scratch);
    cudaFree(buffers.d_bit_scratch);
    cudaFree(buffers.d_encode_scratch);
    cudaFree(buffers.d_encode_slot_stack);
    buffers = GtapKCliqueDeviceBuffers{};
}

static void release_run(
    GtapKCliqueDeviceBuffers& buffers,
    bool initialized,
    unsigned long long* d_nCr = nullptr) {
    cudaFree(d_nCr);
    free_device_buffers(buffers);
    if (initialized) gtap_finalize();
}

static cudaError_t init_device_state(
    const HostOptions& options,
    const GtapKCliqueDeviceBuffers& buffers) {
    const int num_exec_edges = (int)buffers.num_oriented_edges;
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
    CUDA_CHECK(cudaMemcpyToSymbol(g_candidate_highwater, &candidate_highwater, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(
        g_task_max_cycles,
        profile_cycles_zero,
        sizeof(profile_cycles_zero)));
    CUDA_CHECK(cudaMemcpyToSymbol(
        g_task_max_info,
        profile_info_zero,
        sizeof(profile_info_zero)));
#endif
    CUDA_CHECK(cudaMemcpyToSymbol(g_encode_slot_top, &encode_slot_top_init, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_num_edges, &num_exec_edges, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_clique_k, &options.k, sizeof(int)));
    cudaError_t err = bind_graph(
        buffers.d_row_ptr,
        buffers.d_col_idx,
        buffers.d_edge_src,
        buffers.d_cand_scratch,
        buffers.d_bit_scratch,
        buffers.d_encode_scratch,
        buffers.d_encode_slot_stack);
    if (err != cudaSuccess) {
        fprintf(stderr, "bind_graph failed: %s\n", cudaGetErrorString(err));
        return err;
    }
    return cudaSuccess;
}

static void launch_and_measure(void (*kernel)(), GtapKCliqueRunResult& result) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(gtap_launch(kernel));
    CUDA_CHECK(gtap_synchronize());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    result.kernel_ms = kernel_ms;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

static void collect_results(
    const GtapKCliqueDeviceBuffers& buffers,
    GtapKCliqueRunResult& result) {
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
#endif
    (void)buffers;
}

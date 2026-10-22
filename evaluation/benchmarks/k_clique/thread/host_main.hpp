#pragma once

#include <cstdio>
#include <time.h>
#include <vector>

#include "config_defaults.cuh"
#include "../common/gpu_orient.cuh"
#include "../common/host_graph.hpp"
#include "../common/timing.hpp"
#include "host_run.cuh"

static void orient_graph(
    const HostGraph& graph,
    GtapOrientMode orient_mode,
    GtapKCliqueDeviceBuffers& buffers,
    GtapKCliqueRunResult& result) {
    CUDA_CHECK(gtap_gpu_orient_csr(
        graph.row_ptr.data(),
        graph.col_idx.data(),
        graph.n,
        orient_mode,
        &buffers.d_row_ptr,
        &buffers.d_col_idx,
        &buffers.d_edge_src,
        &buffers.num_oriented_edges,
        &result.transfer_ms,
        &result.orient_ms));
}

static void alloc_scratch(
    const gtap_thread_config& config,
    GtapKCliqueDeviceBuffers& buffers,
    GtapKCliqueRunResult& result) {
    timespec workspace_start, workspace_stop;
    clock_gettime(CLOCK_MONOTONIC, &workspace_start);

    buffers.num_workers = config.grid_size * config.block_size;
    const int words_per_worker = (GTAP_MAX_CANDIDATES + 63) >> 6;
    buffers.cand_scratch_ints =
        (size_t)buffers.num_workers * (size_t)GTAP_MAX_CANDIDATES;
    buffers.bit_scratch_words =
        (size_t)buffers.num_workers * (size_t)GTAP_BIT_BUFFER_COUNT * (size_t)words_per_worker;
    buffers.encode_scratch_words =
        ((size_t)GTAP_ENCODE_SLOTS + (size_t)buffers.num_workers) *
        (size_t)GTAP_MAX_CANDIDATES *
        (size_t)words_per_worker;

    std::vector<int> encode_slot_stack((size_t)GTAP_ENCODE_SLOTS);
    for (int i = 0; i < GTAP_ENCODE_SLOTS; ++i) {
        encode_slot_stack[(size_t)i] = i;
    }

    CUDA_CHECK(cudaMalloc(&buffers.d_cand_scratch, buffers.cand_scratch_ints * sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        &buffers.d_bit_scratch, buffers.bit_scratch_words * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(
        &buffers.d_encode_scratch, buffers.encode_scratch_words * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(
        &buffers.d_encode_slot_stack, encode_slot_stack.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(
        buffers.d_encode_slot_stack,
        encode_slot_stack.data(),
        encode_slot_stack.size() * sizeof(int),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, GTAP_CUDA_STACK_SIZE));
    size_t actual_stack_size = 0;
    CUDA_CHECK(cudaDeviceGetLimit(&actual_stack_size, cudaLimitStackSize));
    printf(
        "CUDA stack limit: requested=%d actual=%zu bytes\n",
        GTAP_CUDA_STACK_SIZE, actual_stack_size);

    clock_gettime(CLOCK_MONOTONIC, &workspace_stop);
    result.workspace_ms = elapsed_ms(workspace_start, workspace_stop);
    result.preprocess_ms = result.orient_ms + result.workspace_ms;
}
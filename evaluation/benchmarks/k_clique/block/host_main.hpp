#pragma once

#include <vector>

#include "gtap_block.cuh"
#include "config_defaults.cuh"
#include "../common/gpu_orient.cuh"
#include "../common/host_graph.hpp"
#include "host_run.cuh"

static void orient_graph(
    const HostGraph& graph,
    GtapOrientMode orient_mode,
    GtapKCliqueDeviceBuffers& buffers) {
    CUDA_CHECK(gtap_gpu_orient_csr(
        graph.row_ptr.data(),
        graph.col_idx.data(),
        graph.n,
        orient_mode,
        &buffers.d_row_ptr,
        &buffers.d_col_idx,
        &buffers.d_edge_src,
        &buffers.num_oriented_edges));
}

static void alloc_scratch(
    const gtap_block_config& config,
    GtapKCliqueDeviceBuffers& buffers) {
    // Block runtime: one scratch set per CUDA block (blockIdx.x).
    buffers.num_workers = config.grid_size;
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
}
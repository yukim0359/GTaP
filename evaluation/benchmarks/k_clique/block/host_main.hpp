#pragma once

#include <vector>

#include "gtap_block.cuh"
#include "config_defaults.cuh"
#include "../common/gpu_orient.cuh"
#include "../common/host_graph.hpp"
#include "../common/measure_max_degree.cuh"
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
    GtapKCliqueDeviceBuffers& buffers,
    int clique_k, BitScratchKind bit_kind) {
    // Block runtime: one scratch set per CUDA block (blockIdx.x).
    int measured = 0;
    CUDA_CHECK(measure_max_endpoint_degree(
            buffers.d_row_ptr, buffers.d_col_idx, buffers.d_edge_src,
            (int)buffers.num_oriented_edges, &measured));
    buffers.max_candidates = measured < 1 ? 1 : measured;
    if (bit_kind == BitScratchKind::ByMaxCandidates) {
        buffers.bit_buffer_count = buffers.max_candidates * 2;
    } else {
        buffers.bit_buffer_count = clique_k < 2 ? 2 : clique_k;
    }

    buffers.num_workers = config.grid_size;
    const int max_cand = buffers.max_candidates;
    const int words_per_worker = (max_cand + 63) >> 6;
    buffers.cand_scratch_ints = (size_t)buffers.num_workers * (size_t)max_cand;
    buffers.bit_scratch_words =
        (size_t)buffers.num_workers * (size_t)buffers.bit_buffer_count * (size_t)words_per_worker;
    buffers.candidate_adjacency_scratch_words =
        ((size_t)GTAP_ADJACENCY_SLOTS + (size_t)buffers.num_workers) *
        (size_t)max_cand *
        (size_t)words_per_worker;
    int pivot_levels = buffers.bit_buffer_count >> 1;
    if (pivot_levels < 1) pivot_levels = 1;
    buffers.pivot_stack_ints =
        (size_t)buffers.num_workers * (size_t)kPivotStackFields * (size_t)pivot_levels;

    std::vector<int> candidate_adjacency_slot_stack((size_t)GTAP_ADJACENCY_SLOTS);
    for (int i = 0; i < GTAP_ADJACENCY_SLOTS; ++i) {
        candidate_adjacency_slot_stack[(size_t)i] = i;
    }

    CUDA_CHECK(cudaMalloc(&buffers.d_cand_scratch, buffers.cand_scratch_ints * sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        &buffers.d_bit_scratch, buffers.bit_scratch_words * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(
        &buffers.d_candidate_adjacency_scratch, buffers.candidate_adjacency_scratch_words * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(
        &buffers.d_candidate_adjacency_slot_stack, candidate_adjacency_slot_stack.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffers.d_pivot_stack, buffers.pivot_stack_ints * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(
        buffers.d_candidate_adjacency_slot_stack,
        candidate_adjacency_slot_stack.data(),
        candidate_adjacency_slot_stack.size() * sizeof(int),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, GTAP_CUDA_STACK_SIZE));
}

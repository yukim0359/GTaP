#pragma once

#include <cstdio>

#include "gtap_block.cuh"
#include "config_defaults.cuh"
#include "../common/gpu_orient.cuh"
#include "../common/host_graph.hpp"
#include "../common/host_options.hpp"
#include "host_run.cuh"

static void print_run_settings(
    const HostOptions& options,
    const HostGraph& graph,
    GtapOrientMode orient_mode,
    const gtap_block_config& config,
    const GtapKCliqueDeviceBuffers& buffers) {
    printf("k-clique graph file: %s n=%d k=%d edges=%zu mode=%s-oriented (GTAP block)\n",
           options.graph_path, graph.n, options.k, graph.undirected_edges,
           gtap_orient_mode_name(orient_mode));
    printf("oriented_edges: %zu\n", buffers.num_oriented_edges);
    printf("GTAP_PIVOT_BLOCK_BLOCK_SIZE: %d\n", config.block_size);
    printf("GTAP_PIVOT_BLOCK_GRID_SIZE: %d\n", config.grid_size);
    printf("GTAP_RANGE_CUTOFF: %d\n", GTAP_RANGE_CUTOFF);
    printf("max candidates: %d (measured)\n", buffers.max_candidates);
    printf("bit buffer count: %d\n", buffers.bit_buffer_count);
    printf("GTAP_CUDA_STACK_SIZE: %d\n", GTAP_CUDA_STACK_SIZE);
}

static void print_run_result(
    const GtapKCliqueDeviceBuffers& buffers,
    const GtapKCliqueRunResult& result) {
    printf("scratch blocks: %d\n", buffers.num_workers);
    printf("candidate scratch: %zu ints\n", buffers.cand_scratch_ints);
    printf("bit scratch: %zu uint64 words\n", buffers.bit_scratch_words);
    printf("adjacency slots: %d\n", GTAP_ADJACENCY_SLOTS);
    printf("adjacency scratch: %zu uint64 words (slots + per-block)\n", buffers.candidate_adjacency_scratch_words);
    printf("pivot stack: %zu ints\n", buffers.pivot_stack_ints);
#ifdef GTAP_STATS
    printf("GTaP max candidate length: %d / %d\n",
           result.candidate_highwater, buffers.max_candidates);
#endif
    printf("GTaP count: %llu\n", result.gpu_count);
    if (result.scratch_overflow) {
        printf("scratch overflow\n");
    }
    printf("GTaP count phase time: %.3f ms\n", result.count_phase_ms);
    printf("GTaP kernel time: %.3f ms\n", result.kernel_ms);
    printf("GTaP preprocess+transfer time: %.3f ms\n", result.preprocess_ms);
}

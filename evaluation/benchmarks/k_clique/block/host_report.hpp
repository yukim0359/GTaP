#pragma once

#include <cstdio>

#include "config_defaults.cuh"
#include "../common/gpu_orient.cuh"
#include "../common/host_graph.hpp"
#include "../common/host_options.hpp"
#include "host_run.cuh"

static void gtap_k_clique_report(
    const HostOptions& options,
    const HostGraph& graph,
    const GtapKCliqueHostHooks& hooks,
    GtapOrientMode orient_mode,
    const GtapKCliqueDeviceBuffers& buffers,
    const GtapKCliqueRunResult& result) {
    printf("k-clique graph file: %s n=%d k=%d edges=%zu mode=%s-oriented (GTAP block)\n",
           options.graph_path, graph.n, options.k, graph.undirected_edges,
           gtap_orient_mode_name(orient_mode));
    printf("oriented_edges: %zu\n", buffers.num_oriented_edges);
    printf("GTAP_PIVOT_BLOCK_BLOCK_SIZE: %d\n", GTAP_PIVOT_BLOCK_BLOCK_SIZE);
    printf("GTAP_PIVOT_BLOCK_GRID_SIZE: %d\n", GTAP_PIVOT_BLOCK_GRID_SIZE);
    printf("GTAP_RANGE_CUTOFF: %d\n", GTAP_RANGE_CUTOFF);
    printf("GTAP_MAX_CANDIDATES: %d\n", GTAP_MAX_CANDIDATES);
    printf("GTAP_BIT_BUFFER_COUNT: %d\n", GTAP_BIT_BUFFER_COUNT);
    printf("GTAP_CUDA_STACK_SIZE: %d\n", GTAP_CUDA_STACK_SIZE);
    if (hooks.print_variant_tuning != nullptr) {
        hooks.print_variant_tuning();
    }
    printf("scratch blocks: %d\n", buffers.num_workers);
    printf("candidate scratch: %zu ints\n", buffers.cand_scratch_ints);
    printf("bit scratch: %zu uint64 words\n", buffers.bit_scratch_words);
    printf("encode slots: %d\n", GTAP_ENCODE_SLOTS);
    printf("encode scratch: %zu uint64 words (slots + per-block)\n", buffers.encode_scratch_words);
#ifdef GTAP_STATS
    printf("GTaP max candidate length: %d / %d\n",
           result.candidate_highwater, GTAP_MAX_CANDIDATES);
#endif
    printf("GTaP count: %llu\n", result.gpu_count);
    if (result.scratch_overflow) {
        printf("scratch overflow\n");
    }
    printf("GTaP count phase time: %.3f ms\n", result.count_phase_ms);
    printf("GTaP kernel time: %.3f ms\n", result.kernel_ms);
    printf("GTaP preprocess+transfer time: %.3f ms\n", result.preprocess_ms);
}

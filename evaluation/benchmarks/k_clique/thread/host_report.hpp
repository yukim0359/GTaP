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
    printf("k-clique graph file: %s n=%d k=%d edges=%zu mode=%s-oriented\n",
           options.graph_path, graph.n, options.k, graph.undirected_edges,
           gtap_orient_mode_name(orient_mode));
    printf("oriented_edges: %zu\n", buffers.num_oriented_edges);
    printf("GTAP_GRID_SIZE: %d\n", buffers.grid_size);
    printf("GTAP_BLOCK_SIZE: %d\n", buffers.block_size);
    printf("GTAP_MAX_TASKS_PER_WARP: %d\n", buffers.max_tasks_per_warp);
    printf("GTAP_RANGE_CUTOFF: %d\n", GTAP_RANGE_CUTOFF);
    printf("GTAP_MAX_CANDIDATES: %d\n", GTAP_MAX_CANDIDATES);
    printf("GTAP_BIT_BUFFER_COUNT: %d\n", GTAP_BIT_BUFFER_COUNT);
    printf("GTAP_CUDA_STACK_SIZE: %d\n", GTAP_CUDA_STACK_SIZE);
    if (hooks.print_variant_tuning != nullptr) {
        hooks.print_variant_tuning();
    }
    printf("scratch workers: %d\n", buffers.num_workers);
    printf("candidate scratch: %zu ints\n", buffers.cand_scratch_ints);
    printf("bit scratch: %zu uint64 words\n", buffers.bit_scratch_words);
    printf("encode slots: %d\n", GTAP_ENCODE_SLOTS);
    printf("encode scratch: %zu uint64 words (slots + per-worker)\n", buffers.encode_scratch_words);
#ifdef GTAP_STATS
    printf("GTaP max candidate length: %d / %d\n",
           result.candidate_highwater, GTAP_MAX_CANDIDATES);
    {
        printf("GTaP recurse calls (encoded): %llu (roots: %llu",
               result.recurse_encoded_calls, result.recurse_encoded_roots);
        if (result.recurse_encoded_roots != 0ULL) {
            printf(", avg/root: %.2f",
                   (double)result.recurse_encoded_calls /
                   (double)result.recurse_encoded_roots);
        }
        printf(", max/root: %llu)\n", result.recurse_max_encoded_tree_calls);
    }
#endif
    printf("GTaP count: %llu\n", result.gpu_count);
    if (result.scratch_overflow) {
        printf("scratch overflow\n");
    }
    printf("GTaP count e2e time: %.3f ms\n", result.count_e2e_ms);
    printf("GTaP count phase time: %.3f ms\n", result.count_phase_ms);
    printf("GTaP kernel time: %.3f ms\n", result.kernel_ms);
    printf("GTaP preprocess time: %.3f ms\n", result.preprocess_ms);
    printf("GTaP transfer time: %.3f ms\n", result.transfer_ms);
    printf("GTaP orient time: %.3f ms\n", result.orient_ms);
    printf("GTaP workspace alloc time: %.3f ms\n", result.workspace_ms);
    printf("GTaP count setup time: %.3f ms\n", result.count_setup_ms);
    printf("GTaP gtap_initialize time: %.3f ms\n", result.gtap_initialize_ms);

#ifdef GTAP_STATS
    printf("Max task profile:\n");
    for (int kind = 0; kind < K_PROFILE_KINDS; ++kind) {
        int base = kind * K_PROFILE_FIELDS;
        printf("  %-22s cycles=%llu u=%d v=%d m=%d a=%d b=%d c=%d d=%d worker=%d\n",
               task_kind_name(kind),
               result.task_max_cycles[kind],
               result.task_max_info[base + 0],
               result.task_max_info[base + 1],
               result.task_max_info[base + 2],
               result.task_max_info[base + 3],
               result.task_max_info[base + 4],
               result.task_max_info[base + 5],
               result.task_max_info[base + 6],
               result.task_max_info[base + 7]);
    }
#endif
}

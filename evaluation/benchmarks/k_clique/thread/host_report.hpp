#pragma once

#include <cstdio>

#include "gtap_thread.cuh"
#include "config_defaults.cuh"
#include "../common/gpu_orient.cuh"
#include "../common/host_graph.hpp"
#include "../common/host_options.hpp"
#include "host_run.cuh"

static void print_run_settings(
        const HostOptions& options, const HostGraph& graph, GtapOrientMode orient_mode,
        const gtap_thread_config& config, const GtapKCliqueDeviceBuffers& buffers) {
    printf("k-clique graph file: %s n=%d k=%d edges=%zu mode=%s-oriented\n",
           options.graph_path, graph.n, options.k, graph.undirected_edges,
           gtap_orient_mode_name(orient_mode));
    printf("oriented_edges: %zu\n", buffers.num_oriented_edges);
    printf("GTAP_GRID_SIZE: %d\n", config.grid_size);
    printf("GTAP_BLOCK_SIZE: %d\n", config.block_size);
    printf("GTAP_MAX_TASKS_PER_WARP: %d\n", config.max_tasks_per_warp);
    printf("GTAP_RANGE_CUTOFF: %d\n", GTAP_RANGE_CUTOFF);
    printf("max candidates: %d (measured)\n", buffers.max_candidates);
    printf("bit buffer count: %d\n", buffers.bit_buffer_count);
    printf("GTAP_CUDA_STACK_SIZE: %d\n", GTAP_CUDA_STACK_SIZE);
}

static void print_run_result(
        const GtapKCliqueDeviceBuffers& buffers, const GtapKCliqueRunResult& result) {
    printf("scratch workers: %d\n", buffers.num_workers);
    printf("candidate scratch: %zu ints\n", buffers.cand_scratch_ints);
    printf("bit scratch: %zu uint64 words\n", buffers.bit_scratch_words);
    printf("adjacency slots: %d\n", GTAP_ADJACENCY_SLOTS);
    printf("adjacency scratch: %zu uint64 words (slots + per-worker)\n", buffers.candidate_adjacency_scratch_words);
    printf("pivot stack: %zu ints\n", buffers.pivot_stack_ints);
#ifdef GTAP_STATS
    printf("GTaP max candidate length: %d / %d\n",
           result.candidate_highwater, buffers.max_candidates);
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
    printf("GTaP max-degree measure time: %.3f ms\n", result.measure_ms);
    printf("GTaP workspace alloc time: %.3f ms\n", result.workspace_ms);
    printf("GTaP count setup time: %.3f ms\n", result.count_setup_ms);
    printf("GTaP gtap_initialize time: %.3f ms\n", result.gtap_initialize_ms);

#ifdef GTAP_STATS
    printf("Max task profile:\n");
    for (int kind = 0; kind < K_PROFILE_KINDS; ++kind) {
        int base = kind * K_PROFILE_FIELDS;
        printf("  %-22s cycles=%llu u=%d v=%d cand_count=%d a=%d b=%d c=%d d=%d worker=%d\n",
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

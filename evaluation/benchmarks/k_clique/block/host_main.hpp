#pragma once

#include <cstdio>
#include <time.h>
#include <vector>

#include "config_defaults.cuh"
#include "../common/gpu_orient.cuh"
#include "../common/host_graph.hpp"
#include "../common/host_options.hpp"
#include "../common/timing.hpp"
#include "host_run.cuh"
#include "host_report.hpp"

static bool gtap_k_clique_preprocess(
    const HostGraph& graph,
    const GtapKCliqueHostHooks& hooks,
    GtapOrientMode& orient_mode,
    GtapKCliqueDeviceBuffers& buffers,
    double& preprocess_ms) {
    timespec preprocess_start, preprocess_stop;
    clock_gettime(CLOCK_MONOTONIC, &preprocess_start);

    orient_mode = parse_gtap_orient_from_env(hooks.default_orient);

    CUDA_CHECK(gtap_gpu_orient_csr(
        graph.row_ptr.data(),
        graph.col_idx.data(),
        graph.n,
        orient_mode,
        &buffers.d_row_ptr,
        &buffers.d_col_idx,
        &buffers.d_edge_src,
        &buffers.num_oriented_edges));

    // Block runtime: one scratch set per CUDA block (blockIdx.x).
    buffers.num_workers = GTAP_PIVOT_BLOCK_GRID_SIZE;
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
    if (hooks.set_cuda_stack_limit) {
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, GTAP_CUDA_STACK_SIZE));
    }
#ifdef GTAP_STATS
    CUDA_CHECK(cudaMalloc(
        &buffers.d_recurse_tree_stats,
        (size_t)buffers.num_workers * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(
        buffers.d_recurse_tree_stats,
        0,
        (size_t)buffers.num_workers * sizeof(unsigned long long)));
#else
    (void)buffers.d_recurse_tree_stats;
#endif

    clock_gettime(CLOCK_MONOTONIC, &preprocess_stop);
    preprocess_ms = elapsed_ms(preprocess_start, preprocess_stop);
    return true;
}

#ifdef GTAP_ENABLE_PROFILING
static void gtap_k_clique_export_profile(const GtapKCliqueHostHooks& hooks) {
    if (hooks.profile_name == nullptr) return;
    char profile_directory[512];
    snprintf(profile_directory, sizeof(profile_directory), "./profile/%s",
             hooks.profile_name);
    gtap_export_profile({
        .output_directory = profile_directory,
        .overwrite = true,
    });
}
#endif

static int gtap_k_clique_host_main(int argc, char** argv, const GtapKCliqueHostHooks& hooks) {
    HostOptions options;
    if (!parse_options(argc, argv, options)) {
        return 1;
    }
    HostGraph graph;
    if (!load_graph_file(options.graph_path, graph)) {
        return 1;
    }
    if (!validate_problem(options, graph)) {
        return 1;
    }

    GtapOrientMode orient_mode = hooks.default_orient;
    GtapKCliqueDeviceBuffers buffers;
    GtapKCliqueRunResult result;

    if (!gtap_k_clique_preprocess(graph, hooks, orient_mode, buffers, result.preprocess_ms)) {
        return 1;
    }
    if (!gtap_k_clique_run(options, graph, hooks, buffers, result)) {
        return 1;
    }
    gtap_k_clique_report(options, graph, hooks, orient_mode, buffers, result);
#ifdef GTAP_ENABLE_PROFILING
    gtap_k_clique_export_profile(hooks);
#endif

    if (hooks.free_extras != nullptr) hooks.free_extras();
    gtap_k_clique_free_device_buffers(buffers);
    gtap_finalize();
    return result.scratch_overflow ? 2 : 0;
}

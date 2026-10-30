#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <time.h>

// #define GTAP_ENABLE_PROFILING
// #define GTAP_STATS
#include "gtap_thread.cuh"
#include "../config_defaults.cuh"

#define K_PROFILE_FIELDS 8
enum KProfileKind {
    K_KIND_EDGE_COUNT = 0,
    K_KIND_PIVOT_SHARED_WAIT,
    K_KIND_PIVOT_SHARED_PREFIX,
    K_KIND_PIVOT_PREFIX_BODY,
    K_KIND_PIVOT_PREFIX_WAIT,
    K_PROFILE_KINDS
};

#ifdef GTAP_STATS
static const char* task_kind_name(int kind) {
    switch (kind) {
        case K_KIND_EDGE_COUNT: return "edge_count";
        case K_KIND_PIVOT_SHARED_WAIT: return "pivot_shared_wait";
        case K_KIND_PIVOT_SHARED_PREFIX: return "pivot_shared_prefix";
        case K_KIND_PIVOT_PREFIX_BODY: return "pivot_prefix_body";
        case K_KIND_PIVOT_PREFIX_WAIT: return "pivot_prefix_wait";
        default: return "unknown";
    }
}
#endif

#include "../device_state.cuh"
#include "counting.cuh"
#include "tasks.cuh"
#include "../../common/host_graph.hpp"
#include "../../common/host_options.hpp"
#include "../../common/ncr_host.hpp"
#include "../../common/timing.hpp"
#include "../host_run.cuh"
#include "../host_setup.cuh"
#include "../host_report.hpp"

static void print_pivot_tuning() {
    printf("GTAP_PIVOT_PREFIX_CHUNK: %d\n", GTAP_PIVOT_PREFIX_CHUNK);
    printf("GTAP_PIVOT_SHARED_SPLIT_CANDIDATES: %d\n", GTAP_PIVOT_SHARED_SPLIT_CANDIDATES);
    printf("GTAP_PIVOT_PREFIX_HEAVY_CANDIDATES: %d\n", GTAP_PIVOT_PREFIX_HEAVY_CANDIDATES);
    printf("GTAP_PIVOT_PREFIX_WORK_THRESHOLD: %d\n", GTAP_PIVOT_PREFIX_WORK_THRESHOLD);
    printf("GTAP_PIVOT_PREFIX_FANOUT: %d\n", GTAP_PIVOT_PREFIX_FANOUT);
    printf("GTAP_PIVOT_PREFIX_MAX_WORDS: %d\n", GTAP_PIVOT_PREFIX_MAX_WORDS);
}

int main(int argc, char** argv) {
    HostOptions options;
    if (!parse_options(argc, argv, options)) return 1;
    HostGraph graph;
    if (!load_graph_file(options.graph_path, graph)) return 1;
    if (!validate_problem(options, graph)) return 1;

    gtap_thread_config config{
        .grid_size = 1584,
        .block_size = 32,
        .max_tasks_per_warp = 200000,
        .num_queues = 1,
    };
    if (!apply_launch_env(config.grid_size, config.block_size, config.max_tasks_per_warp)) {
        return 1;
    }

    const GtapOrientMode orient_mode = parse_gtap_orient_from_env(GTAP_ORIENT_DEGEN);
    GtapKCliqueDeviceBuffers buffers;
    GtapKCliqueRunResult result;
    unsigned long long* d_nCr = nullptr;
    bool initialized = false;

    orient_graph(graph, orient_mode, buffers, result);
    alloc_scratch(config, buffers, result, options.k, BitScratchKind::ByMaxCandidates);
    {
        const int prefix_cap = GTAP_PIVOT_PREFIX_MAX_WORDS * 64;
        if (buffers.max_candidates > prefix_cap) {
            fprintf(stderr,
                    "candidate capacity %d exceeds pivot task CL/PL (%d bits, "
                    "GTAP_PIVOT_PREFIX_MAX_WORDS=%d); rebuild with a larger "
                    "GTAP_PIVOT_PREFIX_MAX_WORDS\n",
                    buffers.max_candidates, prefix_cap,
                    GTAP_PIVOT_PREFIX_MAX_WORDS);
            release_run(buffers, false);
            return 1;
        }
    }

    timespec count_phase_start, count_phase_stop;
    timespec count_setup_start, count_setup_stop;
    timespec gtap_init_start, gtap_init_stop;
    clock_gettime(CLOCK_MONOTONIC, &count_phase_start);
    clock_gettime(CLOCK_MONOTONIC, &count_setup_start);

    if (!load_ncr_to_device(&d_nCr)) {
        release_run(buffers, initialized, d_nCr);
        return 1;
    }

    clock_gettime(CLOCK_MONOTONIC, &gtap_init_start);
    cudaError_t err = gtap_initialize(config);
    clock_gettime(CLOCK_MONOTONIC, &gtap_init_stop);
    result.gtap_initialize_ms = elapsed_ms(gtap_init_start, gtap_init_stop);
    if (err != cudaSuccess) {
        fprintf(stderr, "gtap_initialize failed: %s\n", cudaGetErrorString(err));
        release_run(buffers, initialized, d_nCr);
        return 1;
    }
    initialized = true;

    err = init_device_state(options, buffers);
    if (err != cudaSuccess) {
        release_run(buffers, initialized, d_nCr);
        return 1;
    }
    err = bind_nCr(d_nCr);
    if (err != cudaSuccess) {
        fprintf(stderr, "bind_nCr failed: %s\n", cudaGetErrorString(err));
        release_run(buffers, initialized, d_nCr);
        return 1;
    }

    clock_gettime(CLOCK_MONOTONIC, &count_setup_stop);
    result.count_setup_ms =
        elapsed_ms(count_setup_start, count_setup_stop) - result.gtap_initialize_ms;
    if (result.count_setup_ms < 0.0) result.count_setup_ms = 0.0;

    launch_and_measure(count_cliques_kernel, result);
    clock_gettime(CLOCK_MONOTONIC, &count_phase_stop);
    result.count_phase_ms = elapsed_ms(count_phase_start, count_phase_stop);
    result.count_e2e_ms = result.orient_ms + result.workspace_ms + result.count_phase_ms;
    collect_results(buffers, result);

    print_run_settings(options, graph, orient_mode, config, buffers);
    print_pivot_tuning();
    print_run_result(buffers, result);
#ifdef GTAP_ENABLE_PROFILING
    char graph_name[256];
    char profile_directory[512];
    graph_file_stem(options.graph_path, graph_name, sizeof(graph_name));
    snprintf(profile_directory, sizeof(profile_directory),
             "./profile/pivot/%s-%d", graph_name, options.k);
    gtap_export_profile({
        .output_directory = profile_directory,
        .overwrite = true,
    });
#endif

    release_run(buffers, initialized, d_nCr);
    return result.scratch_overflow ? 2 : 0;
}

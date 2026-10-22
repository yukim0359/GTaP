#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <time.h>

#include "config_defaults.cuh"

// #define GTAP_STATS
#include "gtap_block.cuh"

#define K_PROFILE_FIELDS 8
enum KProfileKind {
    K_KIND_EDGE_COUNT = 0,
    K_KIND_PIVOT_SHARED_WAIT,
    K_KIND_PIVOT_SHARED_PREFIX,
    K_KIND_PIVOT_PREFIX_BODY,
    K_KIND_PIVOT_PREFIX_WAIT,
    K_PROFILE_KINDS
};

#include "block_device_state.cuh"
#include "pivot_block_counting.cuh"
#include "pivot_block_tasks.cuh"
#include "../common/host_graph.hpp"
#include "../common/host_options.hpp"
#include "../common/ncr_host.hpp"
#include "../common/timing.hpp"
#include "host_run.cuh"
#include "host_main.hpp"
#include "host_report.hpp"

static void print_pivot_tuning() {
    printf("GTAP_PIVOT_PREFIX_CHUNK: %d\n", GTAP_PIVOT_PREFIX_CHUNK);
    printf("GTAP_PIVOT_SHARED_SPLIT_CANDIDATES: %d\n", GTAP_PIVOT_SHARED_SPLIT_CANDIDATES);
    printf("GTAP_PIVOT_PREFIX_HEAVY_CANDIDATES: %d\n", GTAP_PIVOT_PREFIX_HEAVY_CANDIDATES);
    printf("GTAP_PIVOT_PREFIX_WORK_THRESHOLD: %d\n", GTAP_PIVOT_PREFIX_WORK_THRESHOLD);
    printf("GTAP_PIVOT_PREFIX_FANOUT: %d\n", GTAP_PIVOT_PREFIX_FANOUT);
    printf("GTAP_RANGE_MAX_FIXED: %d\n", GTAP_RANGE_MAX_FIXED);
    printf("GTAP_PIVOT_STATE_LEVELS: %d\n", GTAP_PIVOT_STATE_LEVELS);
}

int main(int argc, char** argv) {
    HostOptions options;
    if (!parse_options(argc, argv, options)) return 1;
    HostGraph graph;
    if (!load_graph_file(options.graph_path, graph)) return 1;
    if (!validate_problem(options, graph)) return 1;

    gtap_block_config config{
        .grid_size = GTAP_PIVOT_BLOCK_GRID_SIZE,
        .block_size = GTAP_PIVOT_BLOCK_BLOCK_SIZE,
        .max_tasks_per_block = GTAP_PIVOT_BLOCK_MAX_TASKS_PER_BLOCK,
    };

    const GtapOrientMode orient_mode = parse_gtap_orient_from_env(GTAP_ORIENT_DEGEN);
    GtapKCliqueDeviceBuffers buffers;
    GtapKCliqueRunResult result;
    unsigned long long* d_nCr = nullptr;
    bool initialized = false;

    timespec preprocess_start, preprocess_stop;
    clock_gettime(CLOCK_MONOTONIC, &preprocess_start);
    orient_graph(graph, orient_mode, buffers);
    alloc_scratch(config, buffers);
    clock_gettime(CLOCK_MONOTONIC, &preprocess_stop);
    result.preprocess_ms = elapsed_ms(preprocess_start, preprocess_stop);

    timespec count_phase_start, count_phase_stop;
    clock_gettime(CLOCK_MONOTONIC, &count_phase_start);

    if (!load_ncr_to_device(&d_nCr)) {
        release_run(buffers, initialized, d_nCr);
        return 1;
    }

    {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
            fprintf(stderr,
                    "GPU mem before gtap_initialize: free %.1f / total %.1f GiB\n",
                    free_bytes / (1024.0 * 1024.0 * 1024.0),
                    total_bytes / (1024.0 * 1024.0 * 1024.0));
        }
        fprintf(stderr, "gtap_initialize: allocating + zeroing (may take 1-2 min)...\n");
        fflush(stderr);
    }

    timespec init_start, init_stop;
    clock_gettime(CLOCK_MONOTONIC, &init_start);
    cudaError_t err = gtap_initialize(config);
    clock_gettime(CLOCK_MONOTONIC, &init_stop);
    fprintf(stderr, "gtap_initialize done in %.1f s\n",
            elapsed_ms(init_start, init_stop) / 1000.0);
    fflush(stderr);
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

    fprintf(stderr, "launching exec_kernel<<<GRID=%d, BLOCK=%d>>> ...\n",
            config.grid_size, config.block_size);
    fflush(stderr);
    launch_and_measure(exec_kernel, result);
    fprintf(stderr, "exec_kernel finished\n");
    fflush(stderr);
    clock_gettime(CLOCK_MONOTONIC, &count_phase_stop);
    result.count_phase_ms = elapsed_ms(count_phase_start, count_phase_stop);
    collect_results(buffers, result);

    print_run_settings(options, graph, orient_mode, config, buffers);
    print_pivot_tuning();
    print_run_result(buffers, result);
#ifdef GTAP_ENABLE_PROFILING
    char graph_name[256];
    char profile_directory[512];
    graph_file_stem(options.graph_path, graph_name, sizeof(graph_name));
    snprintf(profile_directory, sizeof(profile_directory),
             "./profile/pivot_block/%s-%d", graph_name, options.k);
    gtap_export_profile({
        .output_directory = profile_directory,
        .overwrite = true,
    });
#endif

    release_run(buffers, initialized, d_nCr);
    return result.scratch_overflow ? 2 : 0;
}

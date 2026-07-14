#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// #define PROFILE
// #define GTAP_K_STATS
#include "gtap_thread.cuh"
#include "k_clique_config_defaults.cuh"

#define K_PROFILE_FIELDS 8
enum KProfileKind {
    K_KIND_EDGE_COUNT = 0,
    K_KIND_PIVOT_SHARED_WAIT,
    K_KIND_PIVOT_SHARED_PREFIX,
    K_KIND_PIVOT_PREFIX_BODY,
    K_KIND_PIVOT_PREFIX_WAIT,
    K_PROFILE_KINDS
};

#ifdef GTAP_K_STATS
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

#include "k_clique_device_state.cuh"
#include "k_clique_pivot_counting.cuh"
#include "k_clique_pivot_gtap_tasks.cuh"
#include "k_clique_host_graph.hpp"
#include "k_clique_ncr_host.hpp"
#include "gtap_k_clique_host_main.hpp"

#ifndef GTAP_DEFAULT_ORIENT
#define GTAP_DEFAULT_ORIENT GTAP_ORIENT_DEGEN
#endif

static unsigned long long* g_d_nCr = nullptr;

static bool pivot_count_phase_begin(GtapKCliqueDeviceBuffers& buffers) {
    (void)buffers;
    std::vector<unsigned long long> ncr_host;
    const std::string default_ncr_path = gtap_k_default_ncr_path(__FILE__);
    const char* ncr_file = getenv("GTAP_K_NCR_FILE");
    if (ncr_file == nullptr) ncr_file = default_ncr_path.c_str();
    if (!gtap_k_load_ncr_table_from_file(ncr_file, ncr_host)) {
        fprintf(stderr, "Failed to load nCr table from %s\n", ncr_file);
        return false;
    }
    CUDA_CHECK(cudaMalloc(
        &g_d_nCr, gtap_k_ncr_table_elems() * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(
        g_d_nCr,
        ncr_host.data(),
        gtap_k_ncr_table_elems() * sizeof(unsigned long long),
        cudaMemcpyHostToDevice));
    return true;
}

static cudaError_t pivot_bind_extras() {
    return bind_nCr(g_d_nCr);
}

static void pivot_free_extras() {
    cudaFree(g_d_nCr);
    g_d_nCr = nullptr;
}

static void pivot_print_tuning() {
    printf("GTAP_K_PIVOT_PREFIX_CHUNK: %d\n", GTAP_K_PIVOT_PREFIX_CHUNK);
    printf("GTAP_K_PIVOT_SHARED_SPLIT_CANDIDATES: %d\n", GTAP_K_PIVOT_SHARED_SPLIT_CANDIDATES);
    printf("GTAP_K_PIVOT_PREFIX_HEAVY_CANDIDATES: %d\n", GTAP_K_PIVOT_PREFIX_HEAVY_CANDIDATES);
    printf("GTAP_K_PIVOT_PREFIX_WORK_THRESHOLD: %d\n", GTAP_K_PIVOT_PREFIX_WORK_THRESHOLD);
    printf("GTAP_K_PIVOT_PREFIX_FANOUT: %d\n", GTAP_K_PIVOT_PREFIX_FANOUT);
    printf("GTAP_K_RANGE_MAX_FIXED: %d\n", GTAP_K_RANGE_MAX_FIXED);
    printf("GTAP_K_PIVOT_STATE_LEVELS: %d\n", GTAP_K_PIVOT_STATE_LEVELS);
}

int main(int argc, char** argv) {
    GtapKCliqueHostHooks hooks = {};
    hooks.set_cuda_stack_limit = true;
    hooks.profile_name = "k_clique_pivot";
    hooks.on_count_phase_begin = pivot_count_phase_begin;
    hooks.bind_extras = pivot_bind_extras;
    hooks.free_extras = pivot_free_extras;
    hooks.print_variant_tuning = pivot_print_tuning;
    return gtap_k_clique_host_main(argc, argv, hooks);
}

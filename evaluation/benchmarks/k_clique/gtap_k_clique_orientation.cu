#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

// #define PROFILE
// #define GTAP_K_STATS
#include "gtap_thread.cuh"
#include "k_clique_config_defaults.cuh"

#define K_PROFILE_FIELDS 8
enum KProfileKind {
    K_KIND_EDGE_COUNT = 0,
    K_KIND_SHARED_FIRST,
    K_KIND_SHARED_SECOND,
    K_KIND_SHARED_THIRD,
    K_KIND_EDGE_WAIT,
    K_KIND_SHARED_FIRST_WAIT,
    K_KIND_SHARED_SECOND_WAIT,
    K_PROFILE_KINDS
};

#ifdef GTAP_K_STATS
static const char* task_kind_name(int kind) {
    switch (kind) {
        case K_KIND_EDGE_COUNT: return "edge_count";
        case K_KIND_SHARED_FIRST: return "shared_encoded_first";
        case K_KIND_SHARED_SECOND: return "shared_encoded_second";
        case K_KIND_SHARED_THIRD: return "shared_encoded_third";
        case K_KIND_EDGE_WAIT: return "edge_wait";
        case K_KIND_SHARED_FIRST_WAIT: return "shared_encoded_first_wait";
        case K_KIND_SHARED_SECOND_WAIT: return "shared_encoded_second_wait";
        default: return "unknown";
    }
}
#endif

#include "k_clique_device_state.cuh"
#include "k_clique_orientation_counting.cuh"
#include "k_clique_orientation_gtap_tasks.cuh"
#include "k_clique_host_graph.hpp"
#include "gtap_k_clique_host_main.hpp"

#ifndef GTAP_DEFAULT_ORIENT
#define GTAP_DEFAULT_ORIENT GTAP_ORIENT_DEGREE
#endif

static void orientation_print_tuning() {
    printf("GTAP_K_HEAVY_CANDIDATES: %d\n", GTAP_K_HEAVY_CANDIDATES);
    printf("GTAP_K_SECOND_HEAVY_CANDIDATES: %d\n", GTAP_K_SECOND_HEAVY_CANDIDATES);
    printf("GTAP_K_THIRD_HEAVY_CANDIDATES: %d\n", GTAP_K_THIRD_HEAVY_CANDIDATES);
    printf("GTAP_K_WAIT_FANOUT: %d\n", GTAP_K_WAIT_FANOUT);
}

int main(int argc, char** argv) {
    GtapKCliqueHostHooks hooks = {};
    hooks.set_cuda_stack_limit = true;
    hooks.profile_name = "k_clique_orientation";
    hooks.print_variant_tuning = orientation_print_tuning;
    return gtap_k_clique_host_main(argc, argv, hooks);
}

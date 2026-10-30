#pragma once

#include "counting.cuh"

// Parent CL/PL plus scalars, stolen by value. Starting state for a pivot task,
// not a search-path prefix. W=4 covers 256 candidates (Orkut max core 253).
struct PivotTaskState {
    unsigned long long cl[GTAP_PIVOT_PREFIX_MAX_WORDS];
    unsigned long long pl[GTAP_PIVOT_PREFIX_MAX_WORDS];
    int rsize;
    int drop;
    int pivot;
    int pl_count;
};

__device__ __forceinline__ void restore_pivot_task_bitsets(
        int level, int words, const PivotTaskState& src) {
    unsigned long long* cl = pivot_cl_buffer(level);
    unsigned long long* pl = pivot_pl_buffer(level);
    for (int w = 0; w < words; ++w) {
        cl[w] = src.cl[w];
        pl[w] = src.pl[w];
    }
}

__device__ __forceinline__ void capture_pivot_task_state(
        int level, int words, const PivotChildState& child, PivotTaskState* out) {
    const unsigned long long* cl = pivot_cl_buffer(level);
    const unsigned long long* pl = pivot_pl_buffer(level);
    for (int w = 0; w < GTAP_PIVOT_PREFIX_MAX_WORDS; ++w) {
        out->cl[w] = (w < words) ? cl[w] : 0ULL;
        out->pl[w] = (w < words) ? pl[w] : 0ULL;
    }
    out->rsize = child.rsize;
    out->drop = child.drop;
    out->pivot = child.pivot;
    out->pl_count = child.pl_count;
}

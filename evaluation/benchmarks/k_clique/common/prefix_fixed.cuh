#pragma once

#include "config_defaults.cuh"

// Candidate index chosen at each search depth.
// Only the first `depth` entries are valid.
struct SearchPrefix {
    int candidate_indices[GTAP_MAX_PREFIX_DEPTH];
};

__device__ __forceinline__ SearchPrefix extend_search_prefix(
        const SearchPrefix& prefix,
        int depth,
        int idx) {
    SearchPrefix next = prefix;
    if (depth >= 0 && depth < GTAP_MAX_PREFIX_DEPTH) {
        next.candidate_indices[depth] = idx;
    }
    return next;
}

#pragma once

#include "k_clique_block_config_defaults.cuh"

struct KRangeFixed {
    int idx[GTAP_K_RANGE_MAX_FIXED];
};

__device__ __forceinline__ KRangeFixed append_range_fixed(
    const KRangeFixed& fixed,
    int depth,
    int idx) {
    KRangeFixed next = fixed;
    if (depth >= 0 && depth < GTAP_K_RANGE_MAX_FIXED) {
        next.idx[depth] = idx;
    }
    return next;
}

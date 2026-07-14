#include "k_clique_config_defaults.cuh"

// Prefix path for stateless GTAP tasks: idx[level] is the pl branch taken at each depth.
struct KRangeFixed {
    int idx[GTAP_K_RANGE_MAX_FIXED];
};

static constexpr int K_PIVOT_PREFIX_WORDS = (GTAP_K_MAX_CANDIDATES + 63) >> 6;

struct KPivotPrefixState {
    unsigned long long pl[K_PIVOT_PREFIX_WORDS];
    unsigned long long cl[K_PIVOT_PREFIX_WORDS];
    int pl_count;
    int rsize;
    int drop;
    int pivot;
    int level;
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

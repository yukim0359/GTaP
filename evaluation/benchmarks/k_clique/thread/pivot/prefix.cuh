#pragma once

#include "counting.cuh"
#include "../../common/search_prefix.cuh"

#define K_PREFIX_REBUILD_INVALID 0
#define K_PREFIX_REBUILD_READY 1
#define K_PREFIX_REBUILD_TERMINAL 2

__device__ int rebuild_pivot_prefix(
        const unsigned long long* adjacency, int cand_count, int need, int root_pivot, int depth,
        const SearchPrefix& fixed, int* pl_count_out, int* rsize_out,
        int* drop_out, int* pivot_out, unsigned long long* terminal_count) {
    *terminal_count = 0ULL;
    const int max_state_levels = pivot_state_levels();
    if (depth < 0 || depth >= max_state_levels || depth > GTAP_MAX_PREFIX_DEPTH) {
        return K_PREFIX_REBUILD_INVALID;
    }
    int target = need + 1;

    int cur_pivot = -1;
    int cur_pl_count = init_pivot_root(adjacency, cand_count, root_pivot, &cur_pivot, pivot_pl_buffer(0));
    if (cur_pl_count <= 0) return K_PREFIX_REBUILD_INVALID;
    int cur_rsize = 1;
    int cur_drop = 0;

    for (int level = 0; level < depth; ++level) {
        int new_idx = fixed.candidate_indices[level];
        if (new_idx < 0 || new_idx >= cand_count) return K_PREFIX_REBUILD_INVALID;
        unsigned long long* parent_pl = pivot_pl_buffer(level);
        if (!is_bit_set(parent_pl, new_idx)) return K_PREFIX_REBUILD_INVALID;

        int child_level = level + 1;
        if (child_level >= max_state_levels) {
            g_scratch_overflow = 1;
            return K_PREFIX_REBUILD_INVALID;
        }

        PivotChildState child;
        PivotChildKind kind = make_pivot_child_state(
            adjacency, cand_count, target, cur_rsize, cur_drop, cur_pivot,
            new_idx, new_idx >> 6, child_level, &child);
        if (kind == PivotChildKind::Terminal) {
            *terminal_count = child.terminal_count;
            return K_PREFIX_REBUILD_TERMINAL;
        }
        if (kind != PivotChildKind::Ready) return K_PREFIX_REBUILD_INVALID;
        cur_rsize = child.rsize;
        cur_drop = child.drop;
        cur_pivot = child.pivot;
        cur_pl_count = child.pl_count;
    }

    *pl_count_out = cur_pl_count;
    *rsize_out = cur_rsize;
    *drop_out = cur_drop;
    *pivot_out = cur_pivot;
    return K_PREFIX_REBUILD_READY;
}

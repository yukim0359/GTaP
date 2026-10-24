#pragma once

#include "gtap_thread.cuh"
#include "../device_state.cuh"
#include "../bitset.cuh"
#include "../candidate_graph.cuh"
#include "ncr.cuh"
#include "pivot_stack.cuh"

__device__ __forceinline__ unsigned long long* pivot_pl_buffer(int level) {
    return candidate_bitset_buffer(level << 1);
}

__device__ __forceinline__ unsigned long long* pivot_cl_buffer(int level) {
    return candidate_bitset_buffer((level << 1) + 1);
}

__device__ __forceinline__ int choose_kcgpu_pivot_from_cl(
        const unsigned long long* adjacency, int cand_count, const unsigned long long* cl,
        int words, int* max_intersection) {
    int stride = max_words();
    int best_idx = -1;
    int best_count = -1;

    for (int word = 0; word < words; ++word) {
        unsigned long long bits = cl[word];
        while (bits) {
            unsigned long long bit = bits & (~bits + 1ULL);
            int idx = (word << 6) + (__ffsll((long long)bit) - 1);
            bits ^= bit;
            if (idx >= cand_count) continue;

            const unsigned long long* row = adjacency + (size_t)idx * (size_t)stride;
            int count = 0;
            for (int w = 0; w < words; ++w) {
                count += __popcll(cl[w] & row[w]);
            }
            if (count > best_count || (count == best_count && idx < best_idx)) {
                best_count = count;
                best_idx = idx;
            }
        }
    }

    *max_intersection = best_count;
    return best_idx;
}

__device__ __forceinline__ int make_kcgpu_pl(
        const unsigned long long* adjacency, int cand_count, const unsigned long long* cl,
        int pivot, unsigned long long* pl) {
    int words = words_for_len(cand_count);
    int stride = max_words();
    const unsigned long long* row = adjacency + (size_t)pivot * (size_t)stride;
    int count = 0;
    for (int w = 0; w < words; ++w) {
        unsigned long long bits = cl[w] & ~row[w] & valid_bits_mask_for_word(w, words, cand_count);
        pl[w] = bits;
        count += __popcll(bits);
    }
    return count;
}

__device__ __forceinline__ int make_kcgpu_child_cl(
        const unsigned long long* adjacency, int cand_count,
        const unsigned long long* parent_cl, const unsigned long long* parent_pl,
        int new_idx, int new_word, unsigned long long* child_cl) {
    int words = words_for_len(cand_count);
    int stride = max_words();
    const unsigned long long* row = adjacency + (size_t)new_idx * (size_t)stride;
    int bit = new_idx & 63;
    unsigned long long same_word_mask = (~((1ULL << bit) - 1ULL)) | ~parent_pl[new_word];
    int count = 0;

    for (int w = 0; w < words; ++w) {
        unsigned long long order_mask =
            (new_word < w) ? ~parent_pl[w] :
            ((new_word > w) ? ~0ULL : same_word_mask);
        unsigned long long bits =
            parent_cl[w] & row[w] & order_mask & valid_bits_mask_for_word(w, words, cand_count);
        child_cl[w] = bits;
        count += __popcll(bits);
    }
    return count;
}

enum class PivotChildKind {
    Dead = 0,
    Terminal,
    Ready
};

struct PivotChildState {
    int rsize;
    int drop;
    int pivot;
    int pl_count;
    int cl_count;
    int max_intersection;
    unsigned long long terminal_count;
};

// Level-0 CL = all candidates; PL = vertices outside the chosen pivot's neighborhood.
__device__ __forceinline__ int init_pivot_root(
        const unsigned long long* adjacency, int cand_count, int known_pivot,
        int* pivot_out, unsigned long long* root_pl) {
    int words = words_for_len(cand_count);
    unsigned long long* root_cl = pivot_cl_buffer(0);
    init_full_bitset(root_cl, cand_count);

    int max_intersection = 0;
    int pivot = known_pivot;
    if (pivot >= 0) {
        const unsigned long long* row = adjacency + (size_t)pivot * (size_t)max_words();
        for (int w = 0; w < words; ++w) {
            max_intersection += __popcll(root_cl[w] & row[w]);
        }
    } else {
        pivot = choose_kcgpu_pivot_from_cl(adjacency, cand_count, root_cl, words, &max_intersection);
    }
    *pivot_out = pivot;
    if (pivot < 0 || max_intersection <= 0) return 0;
    return make_kcgpu_pl(adjacency, cand_count, root_cl, pivot, root_pl);
}

// After selecting `new_idx` from the parent PL, write child CL/PL at `child_level`.
__device__ __forceinline__ PivotChildKind make_pivot_child_state(
        const unsigned long long* adjacency, int cand_count, int target,
        int parent_rsize, int parent_drop, int parent_pivot,
        int new_idx, int new_word, int child_level, PivotChildState* child) {
    child->terminal_count = 0ULL;
    child->pl_count = 0;
    child->rsize = parent_rsize + 1;
    child->drop = parent_drop + (new_idx == parent_pivot ? 1 : 0);
    if (child->rsize - child->drop > target) return PivotChildKind::Dead;

    int words = words_for_len(cand_count);
    unsigned long long* parent_pl = pivot_pl_buffer(child_level - 1);
    unsigned long long* parent_cl = pivot_cl_buffer(child_level - 1);
    unsigned long long* child_cl = pivot_cl_buffer(child_level);
    child->cl_count = make_kcgpu_child_cl(
        adjacency, cand_count, parent_cl, parent_pl, new_idx, new_word, child_cl);

    child->max_intersection = 0;
    child->pivot = choose_kcgpu_pivot_from_cl(
        adjacency, cand_count, child_cl, words, &child->max_intersection);
    if (child->pivot < 0) {
        if (child->rsize >= target) {
            child->terminal_count = lookup_nCr(child->drop, child->rsize - target);
            return PivotChildKind::Terminal;
        }
        return PivotChildKind::Dead;
    }

    child->pl_count = make_kcgpu_pl(
        adjacency, cand_count, child_cl, child->pivot, pivot_pl_buffer(child_level));
    if (child->pl_count == 0) return PivotChildKind::Dead;
    return PivotChildKind::Ready;
}

// KCGPU-style pivot DFS over an encoded candidate bitset, starting from a ready state.
__device__ unsigned long long count_pivot_encoded_from_state(
        const unsigned long long* adjacency, int cand_count, int target, int start_level,
        int start_rsize, int start_drop, int start_pivot) {
    int words = words_for_len(cand_count);
    const int max_state_levels = pivot_state_levels();
    if (start_level < 0 || start_level >= max_state_levels) {
        g_scratch_overflow = 1;
        return 0ULL;
    }

    PivotStackCol next_candidate = pivot_stack(PivotStackField::Next);
    PivotStackCol drop = pivot_stack(PivotStackField::Drop);
    PivotStackCol pivot = pivot_stack(PivotStackField::Pivot);
    drop[start_level] = start_drop;
    pivot[start_level] = start_pivot;
    next_candidate[start_level] = 0;

    unsigned long long local = 0ULL;
    int level = start_level;
    while (level >= start_level) {
        unsigned long long* parent_pl = pivot_pl_buffer(level);
        int new_word = 0;
        int new_idx = next_set_bit_from(
            parent_pl, words, next_candidate[level], &new_word);
        if (new_idx < 0 || new_idx >= cand_count) {
            --level;
            continue;
        }
        next_candidate[level] = new_idx + 1;

        int child_level = level + 1;
        if (child_level >= max_state_levels) {
            g_scratch_overflow = 1;
            continue;
        }

        int parent_rsize = start_rsize + (level - start_level);
        PivotChildState child;
        PivotChildKind kind = make_pivot_child_state(
            adjacency, cand_count, target,
            parent_rsize, drop[level], pivot[level],
            new_idx, new_word, child_level, &child);
        if (kind == PivotChildKind::Dead) continue;
        if (kind == PivotChildKind::Terminal) {
            local += child.terminal_count;
            continue;
        }

        drop[child_level] = child.drop;
        pivot[child_level] = child.pivot;
        next_candidate[child_level] = 0;
        level = child_level;
    }

    return local;
}

__device__ unsigned long long count_pivot_encoded(
        const unsigned long long* adjacency, int cand_count, int need) {
    if (need == 0) return 1ULL;
    if (need == 1) return (unsigned long long)cand_count;
    if (need == 2) {
        unsigned long long* full = pivot_cl_buffer(0);
        init_full_bitset(full, cand_count);
        return count_edges_in_bitset(adjacency, cand_count, full);
    }

    int target = need + 1;  // KCGPU edge pivot uses kcount = k - 1.
    int pivot = -1;
    int pl_count = init_pivot_root(adjacency, cand_count, -1, &pivot, pivot_pl_buffer(0));
    if (pl_count <= 0) return 0ULL;
    return count_pivot_encoded_from_state(
        adjacency, cand_count, target, 0, 1, 0, pivot);
}

#pragma once

#include "gtap_thread.cuh"
#include "../device_state.cuh"
#include "../bitset.cuh"
#include "../candidate_graph.cuh"

// Parent set stays at bitset[buffer_index]; the child is written at +1.
__device__ unsigned long long count_cliques_in_bitset(
        const unsigned long long* adjacency, int cand_count,
        const unsigned long long* active_candidates, int active_count,
        int need, int buffer_index) {
    if (need == 0) return 1ULL;
    if (active_count < need) return 0ULL;
    if (need == 1) return (unsigned long long)active_count;
    if (need == 2) {
        return count_edges_in_bitset(
            adjacency, cand_count, active_candidates);
    }

    if (buffer_index + 1 >= g_bit_buffer_count) {
        mark_overflow();
        return 0ULL;
    }

    unsigned long long* child_candidates =
        candidate_bitset_buffer(buffer_index + 1);
    unsigned long long clique_count = 0ULL;
    int child_need = need - 1;
    int words = words_for_len(cand_count);

    // Bitset is uint64 words, so the outer loop is the word and the inner
    // loop peels one set bit at a time. Each set bit is one candidate to extend.
    for (int word = 0; word < words; ++word) {
        unsigned long long bits = active_candidates[word];
        while (bits) {
            unsigned long long bit = bits & (~bits + 1ULL);
            int candidate_index = (word << 6) + (__ffsll((long long)bit) - 1);
            bits ^= bit;
            if (candidate_index >= cand_count) continue;

            int child_count = build_child_candidates(
                adjacency, cand_count, active_candidates, candidate_index,
                child_candidates);
            if (child_count >= child_need) {
                clique_count += count_cliques_in_bitset(
                    adjacency, cand_count, child_candidates, child_count,
                    child_need, buffer_index + 1);
            }
        }
    }

    return clique_count;
}

// Count need-cliques in `cand` on this worker. No child tasks.
__device__ __forceinline__ unsigned long long count_cliques_in_candidates(
        const int* cand, int cand_count, int need) {
    if (need == 0) return 1ULL;
    if (cand_count < need) return 0ULL;
    if (cand_count > g_max_candidates) {
        mark_overflow();
        return 0ULL;
    }
    if (need == 1) return (unsigned long long)cand_count;
    if (need == 2) return count_edges_in_candidates(cand, cand_count);

    unsigned long long* adjacency = candidate_adjacency_private_buffer();
    build_candidate_adjacency(cand, cand_count, adjacency);

    // Depth 0: every candidate is still selectable.
    unsigned long long* active_candidates = candidate_bitset_buffer(0);
    init_full_bitset(active_candidates, cand_count);
    return count_cliques_in_bitset(
        adjacency, cand_count, active_candidates, cand_count, need, 0);
}

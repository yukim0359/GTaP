#pragma once

#include "device_state.cuh"
#include "bitset.cuh"

__device__ __forceinline__ int vertices_needed_after_edge() {
    return g_clique_k - 2;
}

// Writes common neighbors to candidates.
// Returns their count, or -1 if the buffer capacity is exceeded.
__device__ __forceinline__ int collect_common_neighbors(int u, int v, int* cand) {
    int iu = g_row_ptr[u];
    int eu = g_row_ptr[u + 1];
    int iv = g_row_ptr[v];
    int ev = g_row_ptr[v + 1];
    int len = 0;
    while (iu < eu && iv < ev) {
        int a = g_col_idx[iu];
        int b = g_col_idx[iv];
        if (a == b) {
            if (len >= g_max_candidates) {
                mark_overflow();
                return -1;
            }
            cand[len++] = a;
            ++iu;
            ++iv;
        } else if (a < b) {
            ++iu;
        } else {
            ++iv;
        }
    }
#ifdef GTAP_STATS
    atomicMax(&g_candidate_highwater, len);
#endif
    return len;
}

__device__ __forceinline__ int count_out_neighbors_in_candidates(
        const int* cand, int cand_count, int x) {
    int i = 0;
    int j = g_row_ptr[x];
    int j_end = g_row_ptr[x + 1];
    int count = 0;
    while (i < cand_count && j < j_end) {
        int a = cand[i];
        int b = g_col_idx[j];
        if (a == b) {
            ++count;
            ++i;
            ++j;
        } else if (a < b) {
            ++i;
        } else {
            ++j;
        }
    }
    return count;
}

__device__ __forceinline__ unsigned long long count_edges_in_candidates(
        const int* cand, int cand_count) {
    unsigned long long edge_count = 0ULL;
    for (int i = 0; i < cand_count; ++i) {
        edge_count += (unsigned long long)count_out_neighbors_in_candidates(
            cand, cand_count, cand[i]);
    }
    return edge_count;
}

// Directed adjacency: row i stores N+(cand[i]) ∩ cand from the oriented CSR.
// Bits are candidate indices, not vertex IDs; `cand` order does not filter
// which neighbors are kept.
__device__ __forceinline__ void build_candidate_adjacency(
        const int* cand, int cand_count, unsigned long long* adjacency) {
    int words = words_for_len(cand_count);
    int stride = max_words();

    for (int row = 0; row < cand_count; ++row) {
        unsigned long long* dst = adjacency + (size_t)row * (size_t)stride;
        clear_bits(dst, words);

        int i = 0;
        int x = cand[row];
        int j = g_row_ptr[x];
        int j_end = g_row_ptr[x + 1];

        while (i < cand_count && j < j_end) {
            int a = cand[i];
            int b = g_col_idx[j];

            if (a == b) {
                dst[i >> 6] |= 1ULL << (i & 63);
                ++i;
                ++j;
            } else if (a < b) {
                ++i;
            } else {
                ++j;
            }
        }
    }
}

// Symmetric adjacency: each undirected candidate edge is stored in both rows.
__device__ __forceinline__ void build_candidate_adjacency_symmetric(
        const int* cand, int cand_count, unsigned long long* adjacency) {
    int words = words_for_len(cand_count);
    int stride = max_words();

    for (int row = 0; row < cand_count; ++row) {
        unsigned long long* dst = adjacency + (size_t)row * (size_t)stride;
        clear_bits(dst, words);
    }

    for (int row = 0; row < cand_count; ++row) {
        int i = 0;
        int x = cand[row];
        int j = g_row_ptr[x];
        int j_end = g_row_ptr[x + 1];

        while (i < cand_count && j < j_end) {
            int a = cand[i];
            int b = g_col_idx[j];

            if (a == b) {
                adjacency[(size_t)row * (size_t)stride + (i >> 6)] |= 1ULL << (i & 63);
                adjacency[(size_t)i * (size_t)stride + (row >> 6)] |= 1ULL << (row & 63);
                ++i;
                ++j;
            } else if (a < b) {
                ++i;
            } else {
                ++j;
            }
        }
    }
}

__device__ __forceinline__ int build_child_candidates(
        const unsigned long long* adjacency, int cand_count,
        const unsigned long long* active_candidates, int candidate_index,
        unsigned long long* child_candidates) {
    int words = words_for_len(cand_count);
    int stride = max_words();
    const unsigned long long* row =
        adjacency + (size_t)candidate_index * (size_t)stride;

    int child_count = 0;
    for (int w = 0; w < words; ++w) {
        unsigned long long bits = active_candidates[w] & row[w];
        child_candidates[w] = bits;
        child_count += __popcll(bits);
    }
    return child_count;
}

__device__ __forceinline__ unsigned long long count_edges_in_bitset(
        const unsigned long long* adjacency, int cand_count,
        const unsigned long long* active_candidates) {
    int words = words_for_len(cand_count);
    int stride = max_words();
    unsigned long long clique_count = 0ULL;

    for (int word = 0; word < words; ++word) {
        unsigned long long bits = active_candidates[word];
        while (bits) {
            unsigned long long bit = bits & (~bits + 1ULL);
            int candidate_index = (word << 6) + (__ffsll((long long)bit) - 1);
            bits ^= bit;
            if (candidate_index >= cand_count) continue;

            const unsigned long long* row =
                adjacency + (size_t)candidate_index * (size_t)stride;

            for (int w = 0; w < words; ++w) {
                unsigned long long p = active_candidates[w];
                if (p != 0ULL) {
                    clique_count += (unsigned long long)__popcll(p & row[w]);
                }
            }
        }
    }

    return clique_count;
}

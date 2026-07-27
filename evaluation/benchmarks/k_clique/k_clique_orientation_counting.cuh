__device__ __forceinline__ bool is_bit_set(const unsigned long long* bits, int idx) {
    return (bits[idx >> 6] >> (idx & 63)) & 1ULL;
}

__device__ __forceinline__ void clear_bits(unsigned long long* bits, int words) {
    for (int w = 0; w < words; ++w) bits[w] = 0ULL;
}

__device__ __forceinline__ void init_full_bits(unsigned long long* bits, int len) {
    int words = words_for_len(len);
    for (int w = 0; w < words; ++w) bits[w] = ~0ULL;
    int rem = len & 63;
    if (rem != 0) bits[words - 1] = (1ULL << rem) - 1ULL;
}

__device__ __forceinline__ int popcount_bits(const unsigned long long* bits, int words) {
    int total = 0;
    for (int w = 0; w < words; ++w) total += __popcll(bits[w]);
    return total;
}

__device__ __forceinline__ int clique_tail_need() {
    return g_clique_k - 2;
}

__device__ __forceinline__ int intersect_rows_to_candidate(int u, int v, int* cand) {
    int iu = g_row_ptr[u];
    int eu = g_row_ptr[u + 1];
    int iv = g_row_ptr[v];
    int ev = g_row_ptr[v + 1];
    int len = 0;
    while (iu < eu && iv < ev) {
        int a = g_col_idx[iu];
        int b = g_col_idx[iv];
        if (a == b) {
            if (len >= GTAP_K_MAX_CANDIDATES) {
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
#ifdef GTAP_K_STATS
    atomicMax(&g_candidate_highwater, len);
#endif
    return len;
}

__device__ __forceinline__ int count_child_in_full_candidate(
    const int* cand,
    int m,
    int x) {
    int i = 0;
    int j = g_row_ptr[x];
    int j_end = g_row_ptr[x + 1];
    int count = 0;
    while (i < m && j < j_end) {
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

__device__ __forceinline__ unsigned long long count_full_candidate_edges(
    const int* cand,
    int m) {
    unsigned long long local = 0ULL;
    for (int i = 0; i < m; ++i) {
        local += (unsigned long long)count_child_in_full_candidate(cand, m, cand[i]);
    }
    return local;
}

__device__ __forceinline__ void build_encoded_candidate_graph(
    const int* cand,
    int m,
    unsigned long long* enc) {
    int words = words_for_len(m);
    int stride = max_words();

    for (int row = 0; row < m; ++row) {
        unsigned long long* dst = enc + (size_t)row * (size_t)stride;
        clear_bits(dst, words);

        int i = 0;
        int x = cand[row];
        int j = g_row_ptr[x];
        int j_end = g_row_ptr[x + 1];

        while (i < m && j < j_end) {
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

__device__ __forceinline__ int make_child_bits_encoded(
    const unsigned long long* enc,
    int m,
    const unsigned long long* P,
    int local_idx,
    unsigned long long* out) {
    int words = words_for_len(m);
    int stride = max_words();
    const unsigned long long* row = enc + (size_t)local_idx * (size_t)stride;

    int out_count = 0;
    for (int w = 0; w < words; ++w) {
        unsigned long long bits = P[w] & row[w];
        out[w] = bits;
        out_count += __popcll(bits);
    }
    return out_count;
}

__device__ __forceinline__ unsigned long long count_encoded_edges_in_bitset(
    const unsigned long long* enc,
    int m,
    const unsigned long long* P) {
    int words = words_for_len(m);
    int stride = max_words();
    unsigned long long local = 0ULL;

    for (int word = 0; word < words; ++word) {
        unsigned long long bits = P[word];
        while (bits) {
            unsigned long long bit = bits & (~bits + 1ULL);
            int local_idx = (word << 6) + (__ffsll((long long)bit) - 1);
            bits ^= bit;
            if (local_idx >= m) continue;

            const unsigned long long* row =
                enc + (size_t)local_idx * (size_t)stride;

            for (int w = 0; w < words; ++w) {
                unsigned long long p = P[w];
                if (p != 0ULL) {
                    local += (unsigned long long)__popcll(p & row[w]);
                }
            }
        }
    }

    return local;
}

__device__ unsigned long long count_encoded_recursive(
    const unsigned long long* enc,
    int m,
    const unsigned long long* P,
    int need,
    int buf_level) {
#ifdef GTAP_K_STATS
    stats_on_encoded_recurse(buf_level);
#endif
    if (need == 0) return 1ULL;

    int words = words_for_len(m);
    int pc = popcount_bits(P, words);
    if (pc < need) return 0ULL;
    if (need == 1) return (unsigned long long)pc;

    if (need == 2) {
        return count_encoded_edges_in_bitset(enc, m, P);
    }

    if (buf_level + 1 >= GTAP_K_BIT_BUFFER_COUNT) {
        mark_overflow();
        return 0ULL;
    }

    unsigned long long* child = bit_buffer(buf_level + 1);
    unsigned long long local = 0ULL;
    int child_need = need - 1;

    for (int word = 0; word < words; ++word) {
        unsigned long long bits = P[word];
        while (bits) {
            unsigned long long bit = bits & (~bits + 1ULL);
            int local_idx = (word << 6) + (__ffsll((long long)bit) - 1);
            bits ^= bit;
            if (local_idx >= m) continue;

            int child_count =
                make_child_bits_encoded(enc, m, P, local_idx, child);
            if (child_count >= child_need) {
                local += count_encoded_recursive(
                    enc, m, child, child_need, buf_level + 1);
            }
        }
    }

    return local;
}

__device__ __forceinline__ unsigned long long count_all_from_candidate_encoded(
    const int* cand,
    int m,
    int need) {
    if (need == 0) return 1ULL;
    if (m < need) return 0ULL;
    if (m > GTAP_K_MAX_CANDIDATES) return 0ULL;
    if (need == 2) return count_full_candidate_edges(cand, m);

    unsigned long long* enc = encode_buffer();
    build_encoded_candidate_graph(cand, m, enc);

    unsigned long long* P0 = bit_buffer(0);
    init_full_bits(P0, m);
    return count_encoded_recursive(enc, m, P0, need, 0);
}

#pragma once

#include "k_clique_block_prefix_fixed.cuh"

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

__device__ __forceinline__ unsigned long long binomial_ull(int n, int k) {
    if (k < 0 || k > n) return 0ULL;
    if (k == 0 || k == n) return 1ULL;
    if (k > n - k) k = n - k;
    unsigned long long result = 1ULL;
    for (int i = 0; i < k; ++i) {
        result = result * (unsigned long long)(n - i) / (unsigned long long)(i + 1);
    }
    return result;
}

__device__ __forceinline__ unsigned long long lookup_nCr(int drop, int c) {
    if (drop >= 0 && drop < GTAP_K_NCR_ROWS && c >= 0 && c < GTAP_K_NCR_COLS) {
        return g_nCr[(size_t)drop * (size_t)GTAP_K_NCR_COLS + (size_t)c];
    }
    return binomial_ull(drop, c);
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

__device__ __forceinline__ unsigned long long count_full_candidate_edges(
    const int* cand,
    int m) {
    unsigned long long local = 0ULL;
    for (int i = 0; i < m; ++i) {
        int j = 0;
        int x = cand[i];
        int p = g_row_ptr[x];
        int p_end = g_row_ptr[x + 1];
        int count = 0;
        while (j < m && p < p_end) {
            int a = cand[j];
            int b = g_col_idx[p];
            if (a == b) {
                ++count;
                ++j;
                ++p;
            } else if (a < b) {
                ++j;
            } else {
                ++p;
            }
        }
        local += (unsigned long long)count;
    }
    return local;
}

__device__ __forceinline__ unsigned long long* pivot_pl_buffer(int level) {
    return bit_buffer(level << 1);
}

__device__ __forceinline__ unsigned long long* pivot_cl_buffer(int level) {
    return bit_buffer((level << 1) + 1);
}

#define K_PIVOT_STATE_LEVELS \
    ((GTAP_K_PIVOT_STATE_LEVELS < (GTAP_K_BIT_BUFFER_COUNT >> 1)) \
         ? GTAP_K_PIVOT_STATE_LEVELS \
         : (GTAP_K_BIT_BUFFER_COUNT >> 1))

#if K_PIVOT_STATE_LEVELS <= 1
#error "K_PIVOT_STATE_LEVELS must be at least 2"
#endif

__device__ __forceinline__ unsigned long long valid_bits_mask_for_word(int word, int words, int m) {
    if (word + 1 != words) return ~0ULL;
    int rem = m & 63;
    return rem == 0 ? ~0ULL : ((1ULL << rem) - 1ULL);
}

__device__ __forceinline__ int next_set_bit_from(
    const unsigned long long* bits,
    int words,
    int start,
    int* out_word) {
    int word = start >> 6;
    int bit = start & 63;
    if (word >= words) return -1;

    unsigned long long mask = bits[word] & (~0ULL << bit);
    while (mask == 0ULL) {
        ++word;
        if (word >= words) return -1;
        mask = bits[word];
    }

    *out_word = word;
    return (word << 6) + (__ffsll((long long)mask) - 1);
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

// ---- Block-cooperative bit operations (KCGPU-style within a CUDA block) ----

__device__ __forceinline__ void block_reduce_pivot_choice(
    int local_count,
    int local_idx,
    int* best_count,
    int* best_idx) {
    __shared__ int sh_count[GTAP_BLOCK_SIZE];
    __shared__ int sh_idx[GTAP_BLOCK_SIZE];
    int tid = threadIdx.x;
    sh_count[tid] = local_count;
    sh_idx[tid] = local_idx;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int o = tid + stride;
            if (o < (int)blockDim.x) {
                if (sh_count[o] > sh_count[tid] ||
                    (sh_count[o] == sh_count[tid] &&
                     sh_idx[o] >= 0 &&
                     (sh_idx[tid] < 0 || sh_idx[o] < sh_idx[tid]))) {
                    sh_count[tid] = sh_count[o];
                    sh_idx[tid] = sh_idx[o];
                }
            }
        }
        __syncthreads();
    }

    __syncthreads();
    *best_count = sh_count[0];
    *best_idx = sh_idx[0];
}

__device__ __forceinline__ bool row_contains_vertex_block(
    int begin,
    int end,
    int target) {
    int lo = begin;
    int hi = end;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        int value = g_col_idx[mid];
        if (value < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo < end && g_col_idx[lo] == target;
}

__device__ __forceinline__ void build_encoded_candidate_graph_symmetric_block(
    const int* cand,
    int m,
    unsigned long long* enc) {
    int words = words_for_len(m);
    int stride = max_words();

#if GTAP_K_BLOCK_ENCODE_KCGPU_STYLE
    int total_words = m * words;
    for (int offset = threadIdx.x; offset < total_words; offset += blockDim.x) {
        int row = offset / words;
        int word = offset - row * words;
        enc[(size_t)row * (size_t)stride + word] = 0ULL;
    }
    __syncthreads();

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = (blockDim.x + 31) >> 5;
    for (int row = warp; row < m; row += warps_per_block) {
        int x = cand[row];
        int begin = g_row_ptr[x];
        int end = g_row_ptr[x + 1];
        for (int i = lane; i < m; i += 32) {
            if (i == row) continue;
            if (row_contains_vertex_block(begin, end, cand[i])) {
                atomicOr(
                    enc + (size_t)row * (size_t)stride + (i >> 6),
                    1ULL << (i & 63));
                atomicOr(
                    enc + (size_t)i * (size_t)stride + (row >> 6),
                    1ULL << (row & 63));
            }
        }
    }
    __syncthreads();
#else
    for (int row = threadIdx.x; row < m; row += blockDim.x) {
        unsigned long long* dst = enc + (size_t)row * (size_t)stride;
        clear_bits(dst, words);
    }
    __syncthreads();

    for (int row = threadIdx.x; row < m; row += blockDim.x) {
        int i = 0;
        int x = cand[row];
        int j = g_row_ptr[x];
        int j_end = g_row_ptr[x + 1];

        while (i < m && j < j_end) {
            int a = cand[i];
            int b = g_col_idx[j];

            if (a == b) {
                atomicOr(
                    enc + (size_t)row * (size_t)stride + (i >> 6),
                    1ULL << (i & 63));
                atomicOr(
                    enc + (size_t)i * (size_t)stride + (row >> 6),
                    1ULL << (row & 63));
                ++i;
                ++j;
            } else if (a < b) {
                ++i;
            } else {
                ++j;
            }
        }
    }
    __syncthreads();
#endif
}

__device__ __forceinline__ int make_kcgpu_child_cl_block(
    const unsigned long long* enc,
    int m,
    const unsigned long long* parent_cl,
    const unsigned long long* parent_pl,
    int new_idx,
    int new_word,
    unsigned long long* child_cl) {
    int words = words_for_len(m);
    int stride = max_words();
    const unsigned long long* row = enc + (size_t)new_idx * (size_t)stride;
    int bit = new_idx & 63;
    unsigned long long same_word_mask = (~((1ULL << bit) - 1ULL)) | ~parent_pl[new_word];

    for (int w = threadIdx.x; w < words; w += blockDim.x) {
        unsigned long long order_mask =
            (new_word < w) ? ~parent_pl[w] :
            ((new_word > w) ? ~0ULL : same_word_mask);
        child_cl[w] =
            parent_cl[w] & row[w] & order_mask & valid_bits_mask_for_word(w, words, m);
    }
    __syncthreads();

    __shared__ int s_count;
    if (threadIdx.x == 0) {
        s_count = popcount_bits(child_cl, words);
    }
    __syncthreads();
    return s_count;
}

__device__ __forceinline__ int choose_kcgpu_pivot_from_cl_block(
    const unsigned long long* enc,
    int m,
    const unsigned long long* cl,
    int words,
    int* max_intersection) {
    int stride = max_words();
    int local_best_count = -1;
    int local_best_idx = -1;

    for (int idx = threadIdx.x; idx < m; idx += blockDim.x) {
        if (!is_bit_set(cl, idx)) continue;
        const unsigned long long* row = enc + (size_t)idx * (size_t)stride;
        int count = 0;
        for (int w = 0; w < words; ++w) {
            count += __popcll(cl[w] & row[w]);
        }
        if (count > local_best_count ||
            (count == local_best_count && idx < local_best_idx)) {
            local_best_count = count;
            local_best_idx = idx;
        }
    }

    int best_count = -1;
    int best_idx = -1;
    block_reduce_pivot_choice(local_best_count, local_best_idx, &best_count, &best_idx);
    *max_intersection = best_count;
    return best_idx;
}

__device__ __forceinline__ int make_kcgpu_pl_block(
    const unsigned long long* enc,
    int m,
    const unsigned long long* cl,
    int pivot,
    unsigned long long* pl) {
    int words = words_for_len(m);
    int stride = max_words();
    const unsigned long long* row = enc + (size_t)pivot * (size_t)stride;

    __shared__ int s_count;
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();

    for (int w = threadIdx.x; w < words; w += blockDim.x) {
        unsigned long long bits = cl[w] & ~row[w] & valid_bits_mask_for_word(w, words, m);
        pl[w] = bits;
        atomicAdd(&s_count, __popcll(bits));
    }
    __syncthreads();
    return s_count;
}

__device__ unsigned long long count_pivot_encoded_loop_block(
    const unsigned long long* enc,
    int m,
    int target,
    int start_level,
    int start_pl_count,
    int start_rsize,
    int start_drop,
    int start_pivot) {
    int words = words_for_len(m);
    constexpr int max_state_levels = K_PIVOT_STATE_LEVELS;
    int max_buffer_levels = GTAP_K_BIT_BUFFER_COUNT >> 1;
    if (start_level < 0 || start_level >= max_state_levels) {
        if (threadIdx.x == 0) g_scratch_overflow = 1;
        return 0ULL;
    }

    int level_count[max_state_levels];
    int level_index[max_state_levels];
    int level_prev[max_state_levels];
    int rsize[max_state_levels];
    int drop[max_state_levels];
    int level_pivot[max_state_levels];
    level_count[start_level] = start_pl_count;
    rsize[start_level] = start_rsize;
    drop[start_level] = start_drop;
    level_pivot[start_level] = start_pivot;
    level_index[start_level] = 0;
    level_prev[start_level] = 0;

    unsigned long long local = 0ULL;
    int level = start_level;
    while (level >= start_level) {
        if (level_count[level] <= level_index[level]) {
            --level;
            continue;
        }

        unsigned long long* parent_pl = pivot_pl_buffer(level);
        unsigned long long* parent_cl = pivot_cl_buffer(level);
        int new_word = 0;
        int new_idx = next_set_bit_from(parent_pl, words, level_prev[level], &new_word);
        if (new_idx < 0 || new_idx >= m) {
            level_index[level] = level_count[level];
            continue;
        }

        level_prev[level] = new_idx + 1;
        ++level_index[level];

        int child_level = level + 1;
        if (child_level >= max_state_levels || child_level >= max_buffer_levels) {
            if (threadIdx.x == 0) g_scratch_overflow = 1;
            continue;
        }
        rsize[child_level] = rsize[level] + 1;
        drop[child_level] = drop[level] + (new_idx == level_pivot[level] ? 1 : 0);

        if (rsize[child_level] - drop[child_level] > target) {
            continue;
        }

        unsigned long long* child_cl = pivot_cl_buffer(child_level);
        make_kcgpu_child_cl_block(
            enc, m, parent_cl, parent_pl, new_idx, new_word, child_cl);

        int child_max_intersection = 0;
        int child_pivot = choose_kcgpu_pivot_from_cl_block(
            enc, m, child_cl, words, &child_max_intersection);

        if (child_pivot < 0) {
            if (rsize[child_level] >= target) {
                local += lookup_nCr(drop[child_level], rsize[child_level] - target);
            }
            continue;
        }

        level_pivot[child_level] = child_pivot;
        level_count[child_level] = make_kcgpu_pl_block(
            enc, m, child_cl, child_pivot, pivot_pl_buffer(child_level));
        level_index[child_level] = 0;
        level_prev[child_level] = 0;

        if (level_count[child_level] == 0) {
            continue;
        }

        level = child_level;
    }

    __shared__ unsigned long long s_local;
    if (threadIdx.x == 0) s_local = local;
    __syncthreads();
    return s_local;
}

__device__ unsigned long long count_pivot_encoded_block(
    const unsigned long long* enc,
    int m,
    int need) {
    if (need == 0) return 1ULL;
    if (need == 1) return (unsigned long long)m;
    if (need == 2) {
        __shared__ unsigned long long s_edge_count;
        if (threadIdx.x == 0) {
            unsigned long long* full = pivot_cl_buffer(0);
            init_full_bits(full, m);
            s_edge_count = count_encoded_edges_in_bitset(enc, m, full);
        }
        __syncthreads();
        return s_edge_count;
    }

    int target = need + 1;
    int words = words_for_len(m);

    if (threadIdx.x == 0) {
        unsigned long long* cl0 = pivot_cl_buffer(0);
        init_full_bits(cl0, m);
    }
    __syncthreads();

    int max_intersection = 0;
    int pivot = choose_kcgpu_pivot_from_cl_block(
        enc, m, pivot_cl_buffer(0), words, &max_intersection);
    if (pivot < 0 || max_intersection <= 0) return 0ULL;

    int pl_count = make_kcgpu_pl_block(enc, m, pivot_cl_buffer(0), pivot, pivot_pl_buffer(0));
    return count_pivot_encoded_loop_block(
        enc, m, target, 0, pl_count, 1, 0, pivot);
}

__device__ unsigned long long count_pivot_encoded_from_state_block(
    const unsigned long long* enc,
    int m,
    int target,
    int start_level,
    int start_pl_count,
    int start_rsize,
    int start_drop,
    int start_pivot) {
    return count_pivot_encoded_loop_block(
        enc, m, target, start_level, start_pl_count, start_rsize, start_drop, start_pivot);
}

__device__ __forceinline__ int prepare_pivot_root_pl_block(
    const unsigned long long* enc,
    int m,
    int* pivot_out,
    unsigned long long* root_pl) {
    int words = words_for_len(m);
    if (threadIdx.x == 0) {
        init_full_bits(pivot_cl_buffer(0), m);
        clear_bits(root_pl, words);
    }
    __syncthreads();

    int max_intersection = 0;
    int pivot = choose_kcgpu_pivot_from_cl_block(
        enc, m, pivot_cl_buffer(0), words, &max_intersection);
    *pivot_out = pivot;
    if (pivot < 0 || max_intersection <= 0) return 0;
    return make_kcgpu_pl_block(enc, m, pivot_cl_buffer(0), pivot, root_pl);
}

#define K_PREFIX_REBUILD_INVALID 0
#define K_PREFIX_REBUILD_READY 1
#define K_PREFIX_REBUILD_TERMINAL 2

__device__ int rebuild_pivot_prefix_block(
    const unsigned long long* enc,
    int m,
    int need,
    int root_pivot,
    int depth,
    const KRangeFixed& fixed,
    int* pl_count_out,
    int* rsize_out,
    int* drop_out,
    int* pivot_out,
    unsigned long long* terminal_count) {
    *terminal_count = 0ULL;
    constexpr int max_state_levels = K_PIVOT_STATE_LEVELS;
    if (depth < 0 || depth >= max_state_levels || depth > GTAP_K_RANGE_MAX_FIXED) {
        return K_PREFIX_REBUILD_INVALID;
    }
    int words = words_for_len(m);
    int target = need + 1;

    if (threadIdx.x == 0) {
        init_full_bits(pivot_cl_buffer(0), m);
    }
    __syncthreads();

    __shared__ int s_max_intersection;
    __shared__ int s_pivot;
    int max_intersection = 0;
    int pivot = root_pivot;
    if (pivot >= 0) {
        if (threadIdx.x == 0) {
            const unsigned long long* row =
                enc + (size_t)pivot * (size_t)max_words();
            int mi = 0;
            for (int w = 0; w < words; ++w) {
                mi += __popcll(pivot_cl_buffer(0)[w] & row[w]);
            }
            s_max_intersection = mi;
            s_pivot = pivot;
        }
        __syncthreads();
        max_intersection = s_max_intersection;
        pivot = s_pivot;
    } else {
        pivot = choose_kcgpu_pivot_from_cl_block(
            enc, m, pivot_cl_buffer(0), words, &max_intersection);
    }
    if (pivot < 0 || max_intersection <= 0) return K_PREFIX_REBUILD_INVALID;

    int cur_rsize = 1;
    int cur_drop = 0;
    int cur_pivot = pivot;
    int cur_pl_count = make_kcgpu_pl_block(
        enc, m, pivot_cl_buffer(0), pivot, pivot_pl_buffer(0));

    for (int level = 0; level < depth; ++level) {
        int new_idx = fixed.idx[level];
        if (new_idx < 0 || new_idx >= m) return K_PREFIX_REBUILD_INVALID;
        unsigned long long* parent_pl = pivot_pl_buffer(level);
        unsigned long long* parent_cl = pivot_cl_buffer(level);
        if (!is_bit_set(parent_pl, new_idx)) return K_PREFIX_REBUILD_INVALID;

        int child_level = level + 1;
        if (child_level >= max_state_levels || child_level >= (GTAP_K_BIT_BUFFER_COUNT >> 1)) {
            if (threadIdx.x == 0) g_scratch_overflow = 1;
            return K_PREFIX_REBUILD_INVALID;
        }

        int new_word = new_idx >> 6;
        cur_rsize = cur_rsize + 1;
        cur_drop = cur_drop + (new_idx == cur_pivot ? 1 : 0);
        if (cur_rsize - cur_drop > target) return K_PREFIX_REBUILD_INVALID;

        unsigned long long* child_cl = pivot_cl_buffer(child_level);
        make_kcgpu_child_cl_block(
            enc, m, parent_cl, parent_pl, new_idx, new_word, child_cl);

        int child_max_intersection = 0;
        int child_pivot = choose_kcgpu_pivot_from_cl_block(
            enc, m, child_cl, words, &child_max_intersection);
        if (child_pivot < 0) {
            if (cur_rsize >= target) {
                *terminal_count = lookup_nCr(cur_drop, cur_rsize - target);
                return K_PREFIX_REBUILD_TERMINAL;
            }
            return K_PREFIX_REBUILD_INVALID;
        }

        cur_pivot = child_pivot;
        cur_pl_count = make_kcgpu_pl_block(
            enc, m, child_cl, child_pivot, pivot_pl_buffer(child_level));
        if (cur_pl_count == 0) {
            return K_PREFIX_REBUILD_INVALID;
        }
    }

    *pl_count_out = cur_pl_count;
    *rsize_out = cur_rsize;
    *drop_out = cur_drop;
    *pivot_out = cur_pivot;
    __syncthreads();
    return K_PREFIX_REBUILD_READY;
}

__device__ __forceinline__ unsigned long long pivot_prefix_work_estimate(
    int child_pl_count,
    int child_cl_count,
    int child_max_intersection,
    int remaining) {
    const unsigned long long cap = (unsigned long long)GTAP_K_PIVOT_PREFIX_WORK_THRESHOLD * 1024ULL;
    unsigned long long width = (unsigned long long)(child_pl_count > 0 ? child_pl_count : 1);
    if (remaining <= 0) {
        int tail_n = child_max_intersection;
        if (tail_n < child_cl_count) tail_n = child_cl_count;
        if (tail_n < 1) tail_n = 1;
        if (width > cap / (unsigned long long)tail_n) return cap;
        unsigned long long estimate = width * (unsigned long long)tail_n;
        return estimate > cap ? cap : estimate;
    }
    if (remaining == 1) return width;

    int choose_n = child_max_intersection;
    if (choose_n < child_cl_count) choose_n = child_cl_count;
    unsigned long long tail = lookup_nCr(choose_n, remaining - 1);
    if (tail != 0ULL && width > cap / tail) return cap;
    unsigned long long estimate = width * tail;
    return estimate > cap ? cap : estimate;
}

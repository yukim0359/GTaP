#include "k_clique_prefix_fixed.cuh"

__device__ __forceinline__ int heavy_threshold_for_depth(int depth) {
    if (depth == 0) return GTAP_K_SECOND_HEAVY_CANDIDATES;
    if (depth == 1) return GTAP_K_THIRD_HEAVY_CANDIDATES;
    return 1000000000;
}

__device__ __forceinline__ int range_fanout_for_depth(int m, int depth, bool shared_slot_path) {
    int fanout = shared_slot_path ? GTAP_K_WAIT_FANOUT : GTAP_K_TASK_FANOUT;
    (void)depth;
    return (m + fanout - 1) / fanout;
}

#ifdef GTAP_K_STATS
__device__ __forceinline__ int shared_range_profile_kind(int depth) {
    if (depth <= 0) return K_KIND_SHARED_FIRST;
    if (depth == 1) return K_KIND_SHARED_SECOND;
    return K_KIND_SHARED_THIRD;
}

__device__ __forceinline__ int shared_range_wait_profile_kind(int depth) {
    if (depth <= 0) return K_KIND_SHARED_FIRST_WAIT;
    return K_KIND_SHARED_SECOND_WAIT;
}
#endif

__device__ bool rebuild_encoded_prefix(
    const unsigned long long* enc,
    int m,
    int depth,
    const KRangeFixed& fixed) {
    if (depth <= 0) return true;
    if (depth + 1 >= GTAP_K_BIT_BUFFER_COUNT) return false;
    if (depth > GTAP_K_RANGE_MAX_FIXED) return false;

    init_full_bits(bit_buffer(0), m);
    for (int i = 0; i < depth; ++i) {
        int fi = fixed.idx[i];
        if (fi < 0 || fi >= m) return false;
        int child_count = make_child_bits_encoded(
            enc, m, bit_buffer(i), fi, bit_buffer(i + 1));
        if (child_count == 0) return false;
    }
    return true;
}

#pragma gtap function
__device__ void edgek_shared_encoded_range_task(
    int slot,
    int u,
    int v,
    int m,
    int depth,
    KRangeFixed fixed,
    int range_begin,
    int range_end) {
#ifdef GTAP_K_STATS
    unsigned long long t0 = global_time();
#endif
    int need = clique_tail_need();
    if (m < need || m > GTAP_K_MAX_CANDIDATES) return;
    if (depth < 0 || depth + 1 >= GTAP_K_BIT_BUFFER_COUNT) return;

    if (range_begin < 0) range_begin = 0;
    if (range_end > m) range_end = m;
    if (range_end <= range_begin) return;

    const unsigned long long* enc = encode_slot_buffer(slot);

    if (depth > 0) {
        if (!rebuild_encoded_prefix(enc, m, depth, fixed)) {
            return;
        }
        int words = words_for_len(m);
        int pc = popcount_bits(bit_buffer(depth), words);
        if (pc < need - depth) return;
    } else {
        init_full_bits(bit_buffer(0), m);
    }

    unsigned long long local = 0ULL;
    int child_need = need - depth - 1;
    bool spawned = false;
    for (int idx = range_begin; idx < range_end; ++idx) {
        unsigned long long* P_cur = bit_buffer(depth);
        unsigned long long* P_child = bit_buffer(depth + 1);
        if (depth > 0 && !is_bit_set(P_cur, idx)) continue;

        if (child_need == 0) {
            ++local;
            continue;
        }

        int child_count = make_child_bits_encoded(enc, m, P_cur, idx, P_child);
        int heavy = heavy_threshold_for_depth(depth);
        if (child_count >= heavy && child_need > 2) {
            int chunk = range_fanout_for_depth(m, depth, true);
            KRangeFixed child_fixed = append_range_fixed(fixed, depth, idx);
            for (int s = 0; s < m; s += chunk) {
                int e = s + chunk;
                if (e > m) e = m;
                spawned = true;
                #pragma gtap task
                edgek_shared_encoded_range_task(
                    slot, u, v, m, depth + 1,
                    child_fixed,
                    s, e);
            }
        } else if (child_count >= child_need) {
            local += count_encoded_recursive(enc, m, P_child, child_need, depth + 1);
        }
    }

    if (local != 0ULL) atomicAdd(&g_answer, local);
    if (spawned) {
#ifdef GTAP_K_STATS
        unsigned long long wait_start = global_time();
#endif
        #pragma gtap taskwait
#ifdef GTAP_K_STATS
        record_task_profile(
            shared_range_wait_profile_kind(depth),
            global_time() - wait_start,
            u,
            v,
            m,
            range_begin,
            range_end,
            spawned ? 1 : 0,
            (int)local);
#endif
    }
#ifdef GTAP_K_STATS
    record_task_profile(
        shared_range_profile_kind(depth),
        global_time() - t0,
        u,
        v,
        m,
        range_begin,
        range_end,
        spawned ? 1 : 0,
        (int)local);
#endif
}

#pragma gtap function
__device__ void edgek_encoded_range_task(
    int u,
    int v,
    int depth,
    KRangeFixed fixed,
    int range_begin,
    int range_end) {

    int* cand = cand_buffer();
    int m = intersect_rows_to_candidate(u, v, cand);
    int need = clique_tail_need();
    if (m < need || m > GTAP_K_MAX_CANDIDATES) return;
    if (depth < 0 || depth + 1 >= GTAP_K_BIT_BUFFER_COUNT) return;

    if (range_begin < 0) range_begin = 0;
    if (range_end > m) range_end = m;
    if (range_end <= range_begin) return;

    unsigned long long* enc = encode_buffer();
    build_encoded_candidate_graph(cand, m, enc);

    if (depth > 0) {
        if (!rebuild_encoded_prefix(enc, m, depth, fixed)) {
            return;
        }
        int words = words_for_len(m);
        int pc = popcount_bits(bit_buffer(depth), words);
        if (pc < need - depth) return;
    } else {
        init_full_bits(bit_buffer(0), m);
    }

    unsigned long long local = 0ULL;
    int child_need = need - depth - 1;
    for (int idx = range_begin; idx < range_end; ++idx) {
        unsigned long long* P_cur = bit_buffer(depth);
        unsigned long long* P_child = bit_buffer(depth + 1);
        if (depth > 0 && !is_bit_set(P_cur, idx)) continue;

        if (child_need == 0) {
            ++local;
            continue;
        }

        int child_count = make_child_bits_encoded(enc, m, P_cur, idx, P_child);
        int heavy = heavy_threshold_for_depth(depth);
        if (child_count >= heavy && child_need > 2) {
            int chunk = range_fanout_for_depth(m, depth, false);
            KRangeFixed child_fixed = append_range_fixed(fixed, depth, idx);
            for (int s = 0; s < m; s += chunk) {
                int e = s + chunk;
                if (e > m) e = m;
                #pragma gtap task
                edgek_encoded_range_task(
                    u, v, depth + 1,
                    child_fixed,
                    s, e);
            }
        } else if (child_count >= child_need) {
            local += count_encoded_recursive(enc, m, P_child, child_need, depth + 1);
        }
    }

    if (local != 0ULL) atomicAdd(&g_answer, local);
}

#pragma gtap function
__device__ void edgek_edge(int ei) {
    {
        int u = g_edge_src[ei];
        int v = g_col_idx[ei];
#ifdef GTAP_K_STATS
        unsigned long long t0 = global_time();
#endif
        int need = clique_tail_need();
        if (g_row_ptr[u + 1] - g_row_ptr[u] < need ||
            g_row_ptr[v + 1] - g_row_ptr[v] < need) {
            return;
        }

        int* cand = cand_buffer();
        int m = intersect_rows_to_candidate(u, v, cand);
        if (m < need) return;

        if (need == 2) {
            unsigned long long local = count_full_candidate_edges(cand, m);
            if (local != 0ULL) atomicAdd(&g_answer, local);
#ifdef GTAP_K_STATS
            record_task_profile(
                K_KIND_EDGE_COUNT,
                global_time() - t0,
                u,
                v,
                m,
                -1,
                5,
                0,
                (int)local);
#endif
            return;
        }

        if (m >= GTAP_K_HEAVY_CANDIDATES) {
            int slot = acquire_encode_slot();
            if (slot >= 0) {
                unsigned long long* enc = encode_slot_buffer(slot);
                build_encoded_candidate_graph(cand, m, enc);

                KRangeFixed fixed;
                int chunk = (m + GTAP_K_WAIT_FANOUT - 1) / GTAP_K_WAIT_FANOUT;
                for (int s = 0; s < m; s += chunk) {
                    int e = s + chunk;
                    if (e > m) e = m;
                    #pragma gtap task
                    edgek_shared_encoded_range_task(
                        slot, u, v, m, 0,
                        fixed,
                        s, e);
                }

#ifdef GTAP_K_STATS
                unsigned long long wait_start = global_time();
#endif
                #pragma gtap taskwait
#ifdef GTAP_K_STATS
                record_task_profile(
                    K_KIND_EDGE_WAIT,
                    global_time() - wait_start,
                    u,
                    v,
                    m,
                    slot,
                    1,
                    0,
                    0);
#endif

                release_encode_slot(slot);
#ifdef GTAP_K_STATS
                record_task_profile(
                    K_KIND_EDGE_COUNT,
                    global_time() - t0,
                    u,
                    v,
                    m,
                    slot,
                    1,
                    0,
                    0);
#endif
                return;
            }

            KRangeFixed fixed;
            int chunk = (m + GTAP_K_TASK_FANOUT - 1) / GTAP_K_TASK_FANOUT;
            for (int s = 0; s < m; s += chunk) {
                int e = s + chunk;
                if (e > m) e = m;
                #pragma gtap task
                edgek_encoded_range_task(
                    u, v, 0,
                    fixed,
                    s, e);
            }
#ifdef GTAP_K_STATS
            record_task_profile(
                K_KIND_EDGE_COUNT,
                global_time() - t0,
                u,
                v,
                m,
                -1,
                3,
                0,
                0);
#endif
            return;
        }

        unsigned long long local = count_all_from_candidate_encoded(cand, m, need);
        if (local != 0ULL) atomicAdd(&g_answer, local);
#ifdef GTAP_K_STATS
        record_task_profile(
            K_KIND_EDGE_COUNT,
            global_time() - t0,
            u,
            v,
            m,
            -1,
            2,
            0,
            (int)local);
#endif
    }
}

#pragma gtap function
__device__ void edgek_range(int begin_edge, int end_edge) {
    if (end_edge <= begin_edge) return;

    if (end_edge - begin_edge > GTAP_K_RANGE_CUTOFF) {
        int chunk = (end_edge - begin_edge + GTAP_K_TASK_FANOUT - 1) / GTAP_K_TASK_FANOUT;
        for (int s = begin_edge; s < end_edge; s += chunk) {
            int e = s + chunk;
            if (e > end_edge) e = end_edge;
            #pragma gtap task
            edgek_range(s, e);
        }
        return;
    }

    for (int ei = begin_edge; ei < end_edge; ++ei) {
        #pragma gtap task
        edgek_edge(ei);
    }
}

__global__ void exec_kernel_k() {
    #pragma gtap entry
    edgek_range(0, g_num_edges);
}

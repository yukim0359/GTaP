#pragma gtap function
__device__ void edgek_pivot_shared_prefix_task(
    int slot,
    int u,
    int v,
    int m,
    int root_pivot,
    int depth,
    KRangeFixed fixed,
    int range_begin,
    int range_end) {
#ifdef GTAP_K_STATS
    unsigned long long t0 = global_time();
#endif
    int need = clique_tail_need();
    if (m < need || m > GTAP_K_MAX_CANDIDATES) return;
    if (depth < 0 || depth >= GTAP_K_RANGE_MAX_FIXED) return;
    if (range_begin < 0) range_begin = 0;

    const unsigned long long* enc = encode_slot_buffer(slot);
    constexpr int max_state_levels = K_PIVOT_STATE_LEVELS;
    int parent_pl_count = 0;
    int parent_rsize = 0;
    int parent_drop = 0;
    int parent_pivot = 0;
    unsigned long long terminal_count = 0ULL;
    int rebuild_status = rebuild_pivot_prefix(
            enc,
            m,
            need,
            root_pivot,
            depth,
            fixed,
            &parent_pl_count,
            &parent_rsize,
            &parent_drop,
            &parent_pivot,
            &terminal_count);
    if (rebuild_status == K_PREFIX_REBUILD_TERMINAL) {
        if (terminal_count != 0ULL) atomicAdd(&g_answer, terminal_count);
        return;
    }
    if (rebuild_status != K_PREFIX_REBUILD_READY) {
        return;
    }

    int target = need + 1;
    int words = words_for_len(m);
    if (range_end > parent_pl_count) range_end = parent_pl_count;
    if (range_end <= range_begin) return;
    unsigned long long local = 0ULL;
    bool spawned = false;
#ifdef GTAP_K_STATS
    int spawned_count = 0;
    unsigned long long max_work_estimate = 0ULL;
#endif
    unsigned long long* parent_pl = pivot_pl_buffer(depth);
    unsigned long long* parent_cl = pivot_cl_buffer(depth);

    int scan_from = 0;
    int new_word = 0;
    for (int skip = 0; skip < range_begin; ++skip) {
        int idx = next_set_bit_from(parent_pl, words, scan_from, &new_word);
        if (idx < 0 || idx >= m) return;
        scan_from = idx + 1;
    }

    for (int ordinal = range_begin; ordinal < range_end; ++ordinal) {
        int new_idx = next_set_bit_from(parent_pl, words, scan_from, &new_word);
        if (new_idx < 0 || new_idx >= m) break;
        scan_from = new_idx + 1;
        int child_level = depth + 1;
        if (child_level >= max_state_levels || child_level >= (GTAP_K_BIT_BUFFER_COUNT >> 1)) {
            g_scratch_overflow = 1;
            continue;
        }

        int child_rsize = parent_rsize + 1;
        int child_drop = parent_drop + (new_idx == parent_pivot ? 1 : 0);
        if (child_rsize - child_drop > target) continue;

        unsigned long long* child_cl = pivot_cl_buffer(child_level);
        int child_cl_count = make_kcgpu_child_cl(
            enc, m, parent_cl, parent_pl, new_idx, new_word, child_cl);

        int child_max_intersection = 0;
        int child_pivot = choose_kcgpu_pivot_from_cl(
            enc, m, child_cl, words, &child_max_intersection);
        if (child_pivot < 0) {
            if (child_rsize >= target) {
                local += lookup_nCr(child_drop, child_rsize - target);
            }
            continue;
        }

        int child_pl_count = make_kcgpu_pl(
            enc, m, child_cl, child_pivot, pivot_pl_buffer(child_level));

        if (child_pl_count == 0) {
            continue;
        }

        int effective_rsize = child_rsize - child_drop;
        int remaining = target - effective_rsize;
        unsigned long long work_estimate = pivot_prefix_work_estimate(
            child_pl_count,
            child_cl_count,
            child_max_intersection,
            remaining);
#ifdef GTAP_K_STATS
        if (work_estimate > max_work_estimate) max_work_estimate = work_estimate;
#endif
        bool heavy_prefix =
            child_cl_count >= GTAP_K_PIVOT_PREFIX_HEAVY_CANDIDATES ||
            child_max_intersection >= GTAP_K_PIVOT_PREFIX_HEAVY_CANDIDATES ||
            child_pl_count >= GTAP_K_PIVOT_PREFIX_HEAVY_CANDIDATES;
        bool should_spawn =
            child_level < GTAP_K_RANGE_MAX_FIXED &&
            effective_rsize < target &&
            heavy_prefix &&
            work_estimate >= (unsigned long long)GTAP_K_PIVOT_PREFIX_WORK_THRESHOLD;
        if (should_spawn) {
            KRangeFixed child_fixed = append_range_fixed(fixed, depth, new_idx);
            int chunk = (child_pl_count + GTAP_K_PIVOT_PREFIX_FANOUT - 1) / GTAP_K_PIVOT_PREFIX_FANOUT;
            if (chunk < 1) chunk = 1;
            for (int s = 0; s < child_pl_count; s += chunk) {
                int e = s + chunk;
                if (e > child_pl_count) e = child_pl_count;
                spawned = true;
#ifdef GTAP_K_STATS
                ++spawned_count;
#endif
                #pragma gtap task
                edgek_pivot_shared_prefix_task(
                    slot, u, v, m, root_pivot, child_level, child_fixed, s, e);
            }
        } else {
            local += count_pivot_encoded_from_state(
                enc,
                m,
                target,
                child_level,
                child_pl_count,
                child_rsize,
                child_drop,
                child_pivot);
        }
    }

    if (local != 0ULL) atomicAdd(&g_answer, local);
#ifdef GTAP_K_STATS
    record_task_profile(
        K_KIND_PIVOT_PREFIX_BODY,
        global_time() - t0,
        u,
        v,
        m,
        depth,
        range_begin,
        spawned_count,
        (int)(max_work_estimate > 2147483647ULL ? 2147483647ULL : max_work_estimate));
#endif
    if (spawned) {
#ifdef GTAP_K_STATS
        unsigned long long wait_start = global_time();
#endif
        #pragma gtap taskwait
#ifdef GTAP_K_STATS
        record_task_profile(
            K_KIND_PIVOT_PREFIX_WAIT,
            global_time() - wait_start,
            u,
            v,
            m,
            depth,
            range_begin,
            spawned_count,
            0);
#endif
    }
#ifdef GTAP_K_STATS
    record_task_profile(
        K_KIND_PIVOT_SHARED_PREFIX,
        global_time() - t0,
        u,
        v,
        m,
        depth,
        range_begin,
        spawned_count,
        (int)local);
#endif
}

#pragma gtap function
__device__ void edgek_edge(int ei) {
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

    if (m >= GTAP_K_PIVOT_SHARED_SPLIT_CANDIDATES) {
        int slot = acquire_encode_slot();
        if (slot >= 0) {
            unsigned long long* enc = encode_slot_buffer(slot);
            build_encoded_candidate_graph_symmetric(cand, m, enc);
            int root_pivot = -1;
            unsigned long long* root_pl = pivot_pl_buffer(0);
            int root_count = prepare_pivot_root_pl(enc, m, &root_pivot, root_pl);
            if (root_count <= 0) {
                release_encode_slot(slot);
#ifdef GTAP_K_STATS
                record_task_profile(
                    K_KIND_EDGE_COUNT,
                    global_time() - t0,
                    u,
                    v,
                    m,
                    root_pivot,
                    10,
                    root_count,
                    0);
#endif
                return;
            }
            bool spawned = false;
            int words = words_for_len(m);
            for (int word = 0; word < words; ++word) {
                unsigned long long bits = root_pl[word];
                while (bits) {
                    unsigned long long bit = bits & (~bits + 1ULL);
                    int root_idx = (word << 6) + (__ffsll((long long)bit) - 1);
                    bits ^= bit;
                    if (root_idx >= m) continue;
                    KRangeFixed root_fixed;
                    root_fixed.idx[0] = root_idx;
                    for (int s = 0; s < m; s += GTAP_K_PIVOT_PREFIX_CHUNK) {
                        int e = s + GTAP_K_PIVOT_PREFIX_CHUNK;
                        if (e > m) e = m;
                        spawned = true;
                        #pragma gtap task
                        edgek_pivot_shared_prefix_task(
                            slot, u, v, m, root_pivot, 1, root_fixed, s, e);
                    }
                }
            }
            if (spawned) {
#ifdef GTAP_K_STATS
                unsigned long long wait_start = global_time();
#endif
                #pragma gtap taskwait
#ifdef GTAP_K_STATS
                record_task_profile(
                    K_KIND_PIVOT_SHARED_WAIT,
                    global_time() - wait_start,
                    u,
                    v,
                    m,
                    root_pivot,
                    root_count,
                    GTAP_K_PIVOT_PREFIX_CHUNK,
                    0);
#endif
            }
            release_encode_slot(slot);
#ifdef GTAP_K_STATS
            record_task_profile(
                K_KIND_EDGE_COUNT,
                global_time() - t0,
                u,
                v,
                m,
                root_pivot,
                10,
                root_count,
                0);
#endif
            return;
        }
    }

    unsigned long long* enc = encode_buffer();
    build_encoded_candidate_graph_symmetric(cand, m, enc);
    unsigned long long local = count_pivot_encoded(enc, m, need);
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

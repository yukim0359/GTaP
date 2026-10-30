// Pivot counting: where work becomes a GTaP task (thread mode).
//
//   count_cliques_kernel
//     count_edge_range[0, E)
//       count_from_edge(ei)
//         light / no slot        inline count_cliques_in_pivot_bitset
//         heavy + encode slot    encode; spawn count_pivot_prefix_from_slot
//         heavy prefix child     another count_pivot_prefix_from_slot at depth+1
//
// slot  = shared encode-buffer index.
// parent = PivotTaskState: parent CL/PL + rsize/drop/pivot/pl_count (stolen by value).
// cl    = current candidate bitset (KCGPU CL).
// pl    = pivot-restricted remaining set (KCGPU PL).
// See README.md "Task split".

#pragma once

#include "gtap_thread.cuh"
#include "counting.cuh"
#include "prefix.cuh"

__device__ __forceinline__ unsigned long long pivot_prefix_work_estimate(
        int child_pl_count, int child_cl_count, int remaining) {
    const unsigned long long cap = (unsigned long long)GTAP_PIVOT_PREFIX_WORK_THRESHOLD * 1024ULL;
    unsigned long long width = (unsigned long long)(child_pl_count > 0 ? child_pl_count : 1);
    if (remaining <= 0) {
        int tail_n = child_cl_count;
        if (tail_n < 1) tail_n = 1;
        if (width > cap / (unsigned long long)tail_n) return cap;
        unsigned long long estimate = width * (unsigned long long)tail_n;
        return estimate > cap ? cap : estimate;
    }
    if (remaining == 1) return width;

    unsigned long long tail = lookup_nCr(child_cl_count, remaining - 1);
    if (tail != 0ULL && width > cap / tail) return cap;
    unsigned long long estimate = width * tail;
    return estimate > cap ? cap : estimate;
}

// Copy stolen parent CL/PL into this worker, then scan PL ordinals [range_begin, range_end).
#pragma gtap function
__device__ void count_pivot_prefix_from_slot(
        int slot, int cand_count, int depth,
        PivotTaskState parent, int range_begin, int range_end) {
#ifdef GTAP_STATS
    unsigned long long start_time = global_time();
#endif
    int need = vertices_needed_after_edge();
    int words = words_for_len(cand_count);
    if (cand_count < need || cand_count > g_max_candidates) return;
    if (words > GTAP_PIVOT_PREFIX_MAX_WORDS) {
        mark_overflow();
        return;
    }
    if (depth < 0 || depth >= GTAP_MAX_PREFIX_DEPTH) return;
    if (range_begin < 0) range_begin = 0;

    const unsigned long long* adjacency = candidate_adjacency_shared_buffer(slot);
    const int max_state_levels = pivot_state_levels();
    if (depth >= max_state_levels) return;
    restore_pivot_task_bitsets(depth, words, parent);

    int target = need + 1;
    if (range_end > parent.pl_count) range_end = parent.pl_count;
    if (range_end <= range_begin) return;
    unsigned long long clique_count = 0ULL;
    bool spawned = false;
#ifdef GTAP_STATS
    int spawned_count = 0;
    unsigned long long max_work_estimate = 0ULL;
#endif
    unsigned long long* parent_pl = pivot_pl_buffer(depth);

    int scan_from = 0;
    int new_word = 0;
    for (int skip = 0; skip < range_begin; ++skip) {
        int idx = next_set_bit_from(parent_pl, words, scan_from, &new_word);
        if (idx < 0 || idx >= cand_count) return;
        scan_from = idx + 1;
    }

    for (int ordinal = range_begin; ordinal < range_end; ++ordinal) {
        int new_idx = next_set_bit_from(parent_pl, words, scan_from, &new_word);
        if (new_idx < 0 || new_idx >= cand_count) break;
        scan_from = new_idx + 1;
        int child_level = depth + 1;
        if (child_level >= max_state_levels) {
            mark_overflow();
            continue;
        }

        PivotChildState child;
        PivotChildKind kind = make_pivot_child_state(
            adjacency, cand_count, target, parent.rsize, parent.drop, parent.pivot,
            new_idx, new_word, child_level, &child);
        if (kind == PivotChildKind::Dead) continue;
        if (kind == PivotChildKind::Terminal) {
            clique_count += child.terminal_count;
            continue;
        }

        int effective_rsize = child.rsize - child.drop;
        int remaining = target - effective_rsize;
        unsigned long long work_estimate = pivot_prefix_work_estimate(
            child.pl_count, child.cl_count, remaining);
#ifdef GTAP_STATS
        if (work_estimate > max_work_estimate) max_work_estimate = work_estimate;
#endif
        bool should_spawn =
            child_level < GTAP_MAX_PREFIX_DEPTH &&
            effective_rsize < target &&
            child.cl_count >= GTAP_PIVOT_PREFIX_HEAVY_CANDIDATES &&
            work_estimate >= (unsigned long long)GTAP_PIVOT_PREFIX_WORK_THRESHOLD;
        if (should_spawn) {
            PivotTaskState child_parent;
            capture_pivot_task_state(child_level, words, child, &child_parent);
            int chunk = (child.pl_count + GTAP_PIVOT_PREFIX_FANOUT - 1) / GTAP_PIVOT_PREFIX_FANOUT;
            if (chunk < 1) chunk = 1;
            for (int s = 0; s < child.pl_count; s += chunk) {
                int e = s + chunk;
                if (e > child.pl_count) e = child.pl_count;
                spawned = true;
#ifdef GTAP_STATS
                ++spawned_count;
#endif
                #pragma gtap task
                count_pivot_prefix_from_slot(
                    slot, cand_count, child_level, child_parent, s, e);
            }
        } else {
            clique_count += count_cliques_from_pivot_state(
                adjacency, cand_count, target, child_level,
                child.rsize, child.drop, child.pivot);
        }
    }

    if (clique_count != 0ULL) atomicAdd(&g_answer, clique_count);
#ifdef GTAP_STATS
    record_task_profile(K_KIND_PIVOT_PREFIX_BODY, global_time() - start_time, -1, -1, cand_count, depth, range_begin,
                        spawned_count, (int)(max_work_estimate > 2147483647ULL ? 2147483647ULL : max_work_estimate));
#endif
    if (spawned) {
#ifdef GTAP_STATS
        unsigned long long wait_start = global_time();
#endif
        #pragma gtap taskwait
#ifdef GTAP_STATS
        record_task_profile(K_KIND_PIVOT_PREFIX_WAIT, global_time() - wait_start,
                            -1, -1, cand_count, depth, range_begin, spawned_count, 0);
#endif
    }
#ifdef GTAP_STATS
    record_task_profile(K_KIND_PIVOT_SHARED_PREFIX, global_time() - start_time,
                        -1, -1, cand_count, depth, range_begin, spawned_count, (int)clique_count);
#endif
}

// One oriented edge: intersect, then inline pivot DFS or spawn prefix tasks.
#pragma gtap function
__device__ void count_from_edge(int ei) {
    int u = g_edge_src[ei];
    int v = g_col_idx[ei];
#ifdef GTAP_STATS
    unsigned long long start_time = global_time();
#endif
    int need = vertices_needed_after_edge();
    if (g_row_ptr[u + 1] - g_row_ptr[u] < need ||
        g_row_ptr[v + 1] - g_row_ptr[v] < need) {
        return;
    }

    int* cand = cand_buffer();
    int cand_count = collect_common_neighbors(u, v, cand);
    if (cand_count < need) return;

    if (need == 2) {
        unsigned long long clique_count = count_edges_in_candidates(cand, cand_count);
        if (clique_count != 0ULL) atomicAdd(&g_answer, clique_count);
#ifdef GTAP_STATS
        record_task_profile(K_KIND_EDGE_COUNT, global_time() - start_time, u, v, cand_count, -1, 5, 0, (int)clique_count);
#endif
        return;
    }

    // 1) Heavy + slot: encode once, spawn count_pivot_prefix_from_slot.
    int words = words_for_len(cand_count);
    if (cand_count >= GTAP_PIVOT_SHARED_SPLIT_CANDIDATES) {
        int slot = acquire_candidate_adjacency_slot();
        if (slot >= 0) {
            unsigned long long* adjacency = candidate_adjacency_shared_buffer(slot);
            build_candidate_adjacency_symmetric(cand, cand_count, adjacency);

            int root_pivot = -1;
            unsigned long long* root_pl = pivot_pl_buffer(0);
            int root_count = init_pivot_root(adjacency, cand_count, -1, &root_pivot, root_pl);
            if (root_count <= 0) {
                release_candidate_adjacency_slot(slot);
#ifdef GTAP_STATS
                record_task_profile(K_KIND_EDGE_COUNT, global_time() - start_time, u, v, cand_count, root_pivot, 10, root_count, 0);
#endif
                return;
            }

            bool spawned = false;
            unsigned long long clique_count = 0ULL;
            int target = need + 1;
            const int max_state_levels = pivot_state_levels();
            for (int word = 0; word < words; ++word) {
                unsigned long long bits = root_pl[word];
                while (bits) {
                    unsigned long long bit = bits & (~bits + 1ULL);
                    int root_idx = (word << 6) + (__ffsll((long long)bit) - 1);
                    bits ^= bit;
                    if (root_idx >= cand_count) continue;

                    int child_level = 1;
                    if (child_level >= max_state_levels) {
                        mark_overflow();
                        continue;
                    }
                    PivotChildState child;
                    PivotChildKind kind = make_pivot_child_state(
                        adjacency, cand_count, target, 1, 0, root_pivot,
                        root_idx, word, child_level, &child);
                    if (kind == PivotChildKind::Dead) continue;
                    if (kind == PivotChildKind::Terminal) {
                        clique_count += child.terminal_count;
                        continue;
                    }

                    PivotTaskState parent;
                    capture_pivot_task_state(child_level, words, child, &parent);
                    for (int s = 0; s < child.pl_count; s += GTAP_PIVOT_PREFIX_CHUNK) {
                        int e = s + GTAP_PIVOT_PREFIX_CHUNK;
                        if (e > child.pl_count) e = child.pl_count;
                        spawned = true;
                        #pragma gtap task
                        count_pivot_prefix_from_slot(
                            slot, cand_count, child_level, parent, s, e);
                    }
                }
            }

            if (clique_count != 0ULL) atomicAdd(&g_answer, clique_count);
            if (spawned) {
#ifdef GTAP_STATS
                unsigned long long wait_start = global_time();
#endif
                #pragma gtap taskwait
#ifdef GTAP_STATS
                record_task_profile(K_KIND_PIVOT_SHARED_WAIT, global_time() - wait_start,
                                    u, v, cand_count, root_pivot, root_count, GTAP_PIVOT_PREFIX_CHUNK, 0);
#endif
            }
            release_candidate_adjacency_slot(slot);
#ifdef GTAP_STATS
            record_task_profile(K_KIND_EDGE_COUNT, global_time() - start_time, u, v, cand_count, root_pivot, 10, root_count, 0);
#endif
            return;
        }
    }

    // 2) Light / no slot: count inline.
    unsigned long long* adjacency = candidate_adjacency_private_buffer();
    build_candidate_adjacency_symmetric(cand, cand_count, adjacency);
    unsigned long long clique_count = count_cliques_in_pivot_bitset(adjacency, cand_count, need);
    if (clique_count != 0ULL) atomicAdd(&g_answer, clique_count);
#ifdef GTAP_STATS
    record_task_profile(K_KIND_EDGE_COUNT, global_time() - start_time, u, v, cand_count, -1, 2, 0, (int)clique_count);
#endif
}

// Split [begin_edge, end_edge) until each piece is <= RANGE_CUTOFF edges.
#pragma gtap function
__device__ void count_edge_range(int begin_edge, int end_edge) {
    if (end_edge <= begin_edge) return;

    if (end_edge - begin_edge > GTAP_RANGE_CUTOFF) {
        int chunk = (end_edge - begin_edge + GTAP_TASK_FANOUT - 1) / GTAP_TASK_FANOUT;
        for (int s = begin_edge; s < end_edge; s += chunk) {
            int e = s + chunk;
            if (e > end_edge) e = end_edge;
            #pragma gtap task
            count_edge_range(s, e);
        }
        return;
    }

    for (int ei = begin_edge; ei < end_edge; ++ei) {
        #pragma gtap task
        count_from_edge(ei);
    }
}

__global__ void count_cliques_kernel() {
    #pragma gtap entry
    count_edge_range(0, g_num_edges);
}

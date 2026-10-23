// Orientation counting: where work becomes a GTaP task (thread mode).
//
//   count_cliques_kernel
//     count_edge_range[0, E)
//       count_from_edge(ei)
//         light cand_count                count inline
//         heavy + encode slot    encode once; spawn count_candidate_range_from_slot
//         heavy, no slot         spawn count_candidate_range_from_edge
//         heavy child at depth   another count_candidate_range_from_slot
//
// slot   = shared adjacency-buffer index.
// prefix = SearchPrefix: candidate index chosen at each search depth.
// See README.md "Task split".

#pragma once

#include "gtap_thread.cuh"
#include "counting.cuh"
#include "../../common/prefix_fixed.cuh"

__device__ __forceinline__ int split_threshold_for_depth(int depth) {
    if (depth == 0) return GTAP_SECOND_HEAVY_CANDIDATES;
    if (depth == 1) return GTAP_THIRD_HEAVY_CANDIDATES;
    return 1000000000;
}

__device__ __forceinline__ int chunk_size(int count, int max_tasks) {
    return (count + max_tasks - 1) / max_tasks;
}

#ifdef GTAP_STATS
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

__device__ bool restore_candidates_from_prefix(
        const unsigned long long* adjacency, int candidate_count, int depth,
        const SearchPrefix& prefix) {
    if (depth <= 0) return true;
    if (depth + 1 >= g_bit_buffer_count) return false;
    if (depth > GTAP_MAX_PREFIX_DEPTH) return false;

    init_full_bitset(candidate_bitset_buffer(0), candidate_count);
    for (int i = 0; i < depth; ++i) {
        int candidate_index = prefix.candidate_indices[i];
        if (candidate_index < 0 || candidate_index >= candidate_count) {
            return false;
        }
        int child_count = build_child_candidates(
            adjacency, candidate_count, candidate_bitset_buffer(i),
            candidate_index, candidate_bitset_buffer(i + 1));
        if (child_count == 0) return false;
    }
    return true;
}

// Walk candidate indices [range_begin, range_end) using the adjacency in `slot`.
#pragma gtap function
__device__ void count_candidate_range_from_slot(
        int slot, int u, int v, int candidate_count, int depth,
        SearchPrefix prefix, int range_begin, int range_end) {
#ifdef GTAP_STATS
    unsigned long long start_time = global_time();
#endif
    const int target_size = vertices_needed_after_edge();
    const int child_vertices_needed = target_size - depth - 1;
    if (candidate_count < target_size || candidate_count > g_max_candidates) {
        return;
    }
    if (depth < 0 || depth + 1 >= g_bit_buffer_count) return;

    if (range_begin < 0) range_begin = 0;
    if (range_end > candidate_count) range_end = candidate_count;
    if (range_end <= range_begin) return;

    const unsigned long long* adjacency = candidate_adjacency_slot_buffer(slot);

    if (depth > 0) {
        if (!restore_candidates_from_prefix(
                adjacency, candidate_count, depth, prefix)) {
            return;
        }
        int words = words_for_len(candidate_count);
        int active_count = popcount_bits(
            candidate_bitset_buffer(depth), words);
        if (active_count < target_size - depth) return;
    } else {
        init_full_bitset(candidate_bitset_buffer(0), candidate_count);
    }

    unsigned long long* active_candidates = candidate_bitset_buffer(depth);
    unsigned long long* child_candidates = candidate_bitset_buffer(depth + 1);
    int split_threshold = split_threshold_for_depth(depth);
    int chunk = chunk_size(candidate_count, GTAP_WAIT_FANOUT);
    unsigned long long clique_count = 0ULL;
    bool spawned = false;
    for (int candidate_index = range_begin; candidate_index < range_end; ++candidate_index) {
        if (depth > 0 && !is_bit_set(active_candidates, candidate_index)) {
            continue;
        }

        if (child_vertices_needed == 0) {
            ++clique_count;
            continue;
        }

        int child_count = build_child_candidates(
            adjacency, candidate_count, active_candidates, candidate_index,
            child_candidates);
        if (child_count >= split_threshold && child_vertices_needed > 2) {
            SearchPrefix child_prefix = extend_search_prefix(
                prefix, depth, candidate_index);
            for (int s = 0; s < candidate_count; s += chunk) {
                int e = s + chunk;
                if (e > candidate_count) e = candidate_count;
                spawned = true;
                #pragma gtap task
                count_candidate_range_from_slot(
                    slot, u, v, candidate_count, depth + 1,
                    child_prefix,
                    s, e);
            }
        } else if (child_count >= child_vertices_needed) {
            clique_count += count_cliques_in_bitset(
                adjacency, candidate_count, child_candidates,
                child_vertices_needed, depth + 1);
        }
    }

    if (clique_count != 0ULL) atomicAdd(&g_answer, clique_count);
    if (spawned) {
#ifdef GTAP_STATS
        unsigned long long wait_start = global_time();
#endif
        #pragma gtap taskwait
#ifdef GTAP_STATS
        record_task_profile(shared_range_wait_profile_kind(depth), global_time() - wait_start,
                            u, v, candidate_count, range_begin, range_end, spawned ? 1 : 0,
                            (int)clique_count);
#endif
    }
#ifdef GTAP_STATS
    record_task_profile(shared_range_profile_kind(depth), global_time() - start_time,
                        u, v, candidate_count, range_begin, range_end, spawned ? 1 : 0,
                        (int)clique_count);
#endif
}

// Same candidate-range walk: rebuild the adjacency from (u, v) on this worker.
#pragma gtap function
__device__ void count_candidate_range_from_edge(
        int u, int v, int depth, SearchPrefix prefix, int range_begin,
        int range_end) {
    int* cand = cand_buffer();
    int candidate_count = collect_common_neighbors(u, v, cand);
    const int target_size = vertices_needed_after_edge();
    const int child_vertices_needed = target_size - depth - 1;
    if (candidate_count < target_size || candidate_count > g_max_candidates) {
        return;
    }
    if (depth < 0 || depth + 1 >= g_bit_buffer_count) return;

    if (range_begin < 0) range_begin = 0;
    if (range_end > candidate_count) range_end = candidate_count;
    if (range_end <= range_begin) return;

    unsigned long long* adjacency = candidate_adjacency_buffer();
    build_candidate_adjacency(cand, candidate_count, adjacency);

    if (depth > 0) {
        if (!restore_candidates_from_prefix(
                adjacency, candidate_count, depth, prefix)) {
            return;
        }
        int words = words_for_len(candidate_count);
        int active_count = popcount_bits(
            candidate_bitset_buffer(depth), words);
        if (active_count < target_size - depth) return;
    } else {
        init_full_bitset(candidate_bitset_buffer(0), candidate_count);
    }

    unsigned long long* active_candidates = candidate_bitset_buffer(depth);
    unsigned long long* child_candidates = candidate_bitset_buffer(depth + 1);
    int split_threshold = split_threshold_for_depth(depth);
    int chunk = chunk_size(candidate_count, GTAP_TASK_FANOUT);
    unsigned long long clique_count = 0ULL;
    for (int candidate_index = range_begin; candidate_index < range_end;
         ++candidate_index) {
        if (depth > 0 && !is_bit_set(active_candidates, candidate_index)) {
            continue;
        }

        if (child_vertices_needed == 0) {
            ++clique_count;
            continue;
        }

        int child_count = build_child_candidates(
            adjacency, candidate_count, active_candidates, candidate_index,
            child_candidates);
        if (child_count >= split_threshold && child_vertices_needed > 2) {
            SearchPrefix child_prefix = extend_search_prefix(
                prefix, depth, candidate_index);
            for (int s = 0; s < candidate_count; s += chunk) {
                int e = s + chunk;
                if (e > candidate_count) e = candidate_count;
                #pragma gtap task
                count_candidate_range_from_edge(
                    u, v, depth + 1,
                    child_prefix,
                    s, e);
            }
        } else if (child_count >= child_vertices_needed) {
            clique_count += count_cliques_in_bitset(
                adjacency, candidate_count, child_candidates,
                child_vertices_needed, depth + 1);
        }
    }

    if (clique_count != 0ULL) atomicAdd(&g_answer, clique_count);
}

// One oriented edge: intersect, then count inline or spawn candidate-range tasks.
#pragma gtap function
__device__ void count_from_edge(int ei) {
    int u = g_edge_src[ei];
    int v = g_col_idx[ei];
#ifdef GTAP_STATS
    unsigned long long start_time = global_time();
#endif
    const int target_size = vertices_needed_after_edge();
    if (g_row_ptr[u + 1] - g_row_ptr[u] < target_size ||
        g_row_ptr[v + 1] - g_row_ptr[v] < target_size) {
        return;
    }

    int* cand = cand_buffer();
    int candidate_count = collect_common_neighbors(u, v, cand);
    if (candidate_count < target_size) return;

    if (target_size == 2) {
        unsigned long long clique_count = count_edges_in_candidates(
            cand, candidate_count);
        if (clique_count != 0ULL) atomicAdd(&g_answer, clique_count);
#ifdef GTAP_STATS
        record_task_profile(K_KIND_EDGE_COUNT, global_time() - start_time,
                            u, v, candidate_count, -1, 5, 0, (int)clique_count);
#endif
        return;
    }

    // 1) Heavy + slot: encode once, spawn count_candidate_range_from_slot.
    if (candidate_count >= GTAP_HEAVY_CANDIDATES) {
        int slot = acquire_candidate_adjacency_slot();
        if (slot >= 0) {
            unsigned long long* adjacency = candidate_adjacency_slot_buffer(slot);
            build_candidate_adjacency(cand, candidate_count, adjacency);

            SearchPrefix prefix;
            int chunk = chunk_size(candidate_count, GTAP_WAIT_FANOUT);
            for (int s = 0; s < candidate_count; s += chunk) {
                int e = s + chunk;
                if (e > candidate_count) e = candidate_count;
                #pragma gtap task
                count_candidate_range_from_slot(
                    slot, u, v, candidate_count, 0,
                    prefix,
                    s, e);
            }

#ifdef GTAP_STATS
            unsigned long long wait_start = global_time();
#endif
            #pragma gtap taskwait
#ifdef GTAP_STATS
            record_task_profile(K_KIND_EDGE_WAIT, global_time() - wait_start,
                                u, v, candidate_count, slot, 1, 0, 0);
#endif

            release_candidate_adjacency_slot(slot);
#ifdef GTAP_STATS
            record_task_profile(K_KIND_EDGE_COUNT, global_time() - start_time,
                                u, v, candidate_count, slot, 1, 0, 0);
#endif
            return;
        }

        // 2) Heavy, no slot: spawn count_candidate_range_from_edge.
        SearchPrefix prefix;
        int chunk = chunk_size(candidate_count, GTAP_TASK_FANOUT);
        for (int s = 0; s < candidate_count; s += chunk) {
            int e = s + chunk;
            if (e > candidate_count) e = candidate_count;
            #pragma gtap task
            count_candidate_range_from_edge(
                u, v, 0,
                prefix,
                s, e);
        }
#ifdef GTAP_STATS
        record_task_profile(K_KIND_EDGE_COUNT, global_time() - start_time,
                            u, v, candidate_count, -1, 3, 0, 0);
#endif
        return;
    }

    // 3) Light: count inline.
    unsigned long long clique_count = count_cliques_in_candidates(
        cand, candidate_count, target_size);
    if (clique_count != 0ULL) atomicAdd(&g_answer, clique_count);
#ifdef GTAP_STATS
    record_task_profile(K_KIND_EDGE_COUNT, global_time() - start_time,
                        u, v, candidate_count, -1, 2, 0, (int)clique_count);
#endif
}

// Split [begin_edge, end_edge) until each piece is <= RANGE_CUTOFF edges.
#pragma gtap function
__device__ void count_edge_range(int begin_edge, int end_edge) {
    if (end_edge <= begin_edge) return;

    if (end_edge - begin_edge > GTAP_RANGE_CUTOFF) {
        int chunk = chunk_size(end_edge - begin_edge, GTAP_TASK_FANOUT);
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

#pragma once

#include "scheduler.cuh"
#include "task_pool.cuh"
#include "termination.cuh"

namespace gtap::detail::thread {

using namespace gtap::detail;

// Get the current state of a task (reads from TaskHeader)
__device__ __forceinline__ int get_task_state(int tid) {
#ifdef GTAP_ASSUME_NO_TASKWAIT
    (void)tid;
    return 0;
#else
    return load_L2(&d_task_headers[tid].state);
#endif
}

// TODO: Rename set_state_for_join and __gtap_set_state_for_join to a
// prepare_for_join style name. The function stores the resume queue and the
// child count, then reports whether the task suspends.
__device__ __forceinline__ bool set_state_for_join(int tid, int child_count, int next_state, int queue_idx_after_join) {
    if (queue_idx_after_join < 0 ||
        queue_idx_after_join >= d_launch_config.num_queues) {
        GTAP_DETAIL_RECORD_INVALID_QUEUE_IDX_AFTER_JOIN(
            tid, queue_idx_after_join, d_launch_config.num_queues);
    }
#ifndef GTAP_ASSUME_NO_TASKWAIT
    TaskHeader* hdr = &d_task_headers[tid];
    hdr->queue_idx = queue_idx_after_join;
    hdr->state = next_state;
    hdr->waiting_child_count = child_count;
#else
    d_task_headers[tid].queue_idx = queue_idx_after_join;
    (void)next_state;
#endif
    return child_count != 0;
}

#ifndef GTAP_ASSUME_NO_TASKWAIT
__device__ __forceinline__ int notify_parent(int parentId, TaskContext* ctx) {
    TaskHeader* parent_hdr = &d_task_headers[parentId];
    __threadfence();
    int rem = atomicSub(&parent_hdr->waiting_child_count, 1);
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
    int lane = get_lane_id();
    printf("notify_parent: %d (remaining child count: %d) in lane %d of warp %d of block %d\n", parentId, rem, lane, get_warp_id_in_block(), blockIdx.x);
#endif
    if (rem == 1) {
        int parent_queue_idx = load_L2(&parent_hdr->queue_idx);
        reserve_unpublished_task_id(ctx, parent_queue_idx, parentId);
    }
    return rem;
}
#endif

__device__ __forceinline__ void finish_task(int tid, TaskContext* ctx) {
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
    printf("finish_task: %d in lane %d of warp %d of block %d\n", tid, get_lane_id(), get_warp_id_in_block(), blockIdx.x);
#endif

#ifdef GTAP_ASSUME_NO_TASKWAIT
    (void)ctx;
    release_task_id_to_warp_pool(tid);
#else
    int lane = get_lane_id();
    int parent_tid = ctx->task_parent_tids[lane];
    uint32_t cached_generations = ctx->task_generations[lane];
    uint16_t generation = static_cast<uint16_t>(cached_generations);
    uint16_t parent_generation =
        static_cast<uint16_t>(cached_generations >> 16);
    d_task_headers[tid].generation = generation + 1;

    if (tid != 0 &&
        load_L2(&d_task_headers[parent_tid].generation) ==
            parent_generation) {
        notify_parent(parent_tid, ctx);
    }
    release_task_id_to_warp_pool(tid);
#endif

    if (tid == 0) {
        store_L2(&d_root_task_finished, 1);
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
        int lane = get_lane_id();
        printf("root task finished in lane %d of warp %d of block %d\n", lane, get_warp_id_in_block(), blockIdx.x);
#endif
    }
}

// Allocates task ID, sets up TaskHeader, returns task data pointer
// Caller stores task data fields after this call
__device__ __forceinline__ void* spawn_task(
    TaskContext* ctx,
    int self_tid,
    int* child_count,
    void (*func)(void*, int, TaskContext*),
    int child_queue_idx
) {
    if (child_queue_idx < 0 ||
        child_queue_idx >= d_launch_config.num_queues) {
        GTAP_DETAIL_RECORD_INVALID_QUEUE_IDX(
            self_tid, child_queue_idx, d_launch_config.num_queues);
    }
    int warp_id_global = get_warp_id_global();
    int new_tid = get_task_id_from_warp_pool(
        &d_task_id_list_free_positions[warp_id_global],
        &ctx->id_list_alloc_pos,
        &ctx->id_list_free_pos_stale);
    TaskHeader* new_hdr = &d_task_headers[new_tid];
    new_hdr->func = func;
    new_hdr->queue_idx = child_queue_idx;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    int lane = get_lane_id();
    new_hdr->parent_tid = self_tid;
    new_hdr->parent_generation =
        static_cast<uint16_t>(ctx->task_generations[lane]);
    new_hdr->state = 0;
    new_hdr->waiting_child_count = 0;
#endif

    reserve_unpublished_task_id(ctx, child_queue_idx, new_tid);

#ifndef GTAP_ASSUME_NO_TASKWAIT
    (*child_count)++;
#else
    (void)child_count;
#endif
    return get_task_data(new_tid);
}

}  // namespace gtap::detail::thread

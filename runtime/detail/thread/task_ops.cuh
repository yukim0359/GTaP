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

__device__ __forceinline__ bool prepare_for_join(int tid, int child_count, int next_state, int queue_idx_after_join) {
    if (queue_idx_after_join < 0 ||
        queue_idx_after_join >= d_launch_config.num_queues) {
        GTAP_DETAIL_RECORD_INVALID_QUEUE_IDX_AFTER_JOIN(
            tid, queue_idx_after_join, d_launch_config.num_queues);
    }
#ifndef GTAP_ASSUME_NO_TASKWAIT
    TaskHeader* const hdr = &d_task_headers[tid];
    hdr->queue_idx = queue_idx_after_join;
    hdr->state = next_state;
    hdr->waiting_child_count = child_count;
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
    printf("prepare_for_join: %d (child count: %d) in lane %d of warp %d of block %d\n",
           tid, child_count, get_lane_id(), get_warp_id_in_block(), blockIdx.x);
#endif
#else
    (void)next_state;
#endif
    return child_count != 0;
}

#ifndef GTAP_ASSUME_NO_TASKWAIT
__device__ __forceinline__ int notify_parent(int parent_tid, TaskContext* ctx) {
    TaskHeader* const parent_hdr = &d_task_headers[parent_tid];
    // Fence before the decrement. Another worker can bring waiting_child_count
    // to zero and resume the parent before this worker's global stores are
    // visible.
    __threadfence();
    const int rem = atomicSub(&parent_hdr->waiting_child_count, 1);
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
    const int lane = get_lane_id();
    printf("notify_parent: %d (remaining child count: %d) in lane %d of warp %d of block %d\n", parent_tid, rem, lane, get_warp_id_in_block(), blockIdx.x);
#endif
    if (rem == 1) {
        const int parent_queue_idx = load_L2(&parent_hdr->queue_idx);
        stage_task_id(ctx, parent_queue_idx, parent_tid);
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
#else
    const int lane = get_lane_id();
    const int parent_tid = ctx->parent_tids[lane];
    const uint32_t cached_generations = ctx->generations[lane];
    const uint16_t generation = static_cast<uint16_t>(cached_generations);
    const uint16_t parent_generation =
        static_cast<uint16_t>(cached_generations >> 16);
    const uint16_t next_generation = static_cast<uint16_t>(generation + 1);
    d_task_headers[tid].generation = next_generation;
    // prepare_for_join leaves the low half unchanged, so the execute loop can
    // tell a finished task from one that suspended at taskwait.
    ctx->generations[lane] =
        (cached_generations & 0xffff0000u) | next_generation;

    if (tid != 0 &&
        load_L2(&d_task_headers[parent_tid].generation) ==
            parent_generation) {
        notify_parent(parent_tid, ctx);
    }
#endif

    if (tid == 0) {
        store_L2(&d_root_task_finished, 1);
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
        printf("root task finished in lane %d of warp %d of block %d\n",
               get_lane_id(), get_warp_id_in_block(), blockIdx.x);
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
    const int new_tid = get_task_id_from_warp_pool(
        &ctx->id_list_free_pos,
        &ctx->id_list_alloc_pos);
    TaskHeader* const new_hdr = &d_task_headers[new_tid];
    new_hdr->func = func;
    new_hdr->queue_idx = child_queue_idx;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    const int lane = get_lane_id();
    new_hdr->parent_tid = self_tid;
    new_hdr->parent_generation =
        static_cast<uint16_t>(ctx->generations[lane]);
    new_hdr->state = 0;
    new_hdr->waiting_child_count = 0;
    (*child_count)++;
#else
    (void)child_count;
#endif

    stage_task_id(ctx, child_queue_idx, new_tid);
    return get_task_data(new_tid);
}

}  // namespace gtap::detail::thread

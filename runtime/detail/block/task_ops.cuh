#pragma once

// Shared block task operations, matching the standard scheduler.
// Include after the backend defines reserve_unpublished_task_id.
// The initial push stays in each scheduler.

#include "task_pool.cuh"
#include "termination.cuh"

namespace gtap::detail::block {

using namespace gtap::detail;

__device__ __forceinline__ int get_task_state(int tid) {
#ifdef GTAP_ASSUME_NO_TASKWAIT
    return 0;
#else
    return load_L2(&d_task_headers[tid].state);
#endif
}

__device__ __forceinline__ bool set_state_for_join_block(
    int tid,
    TaskContext* ctx,
    int next_state,
    int unused_value
) {
    (void)unused_value;
    __syncthreads();
    int child_count = ctx->generated_task_count;
    if (threadIdx.x == 0) {
        TaskHeader* hdr = &d_task_headers[tid];
#ifndef GTAP_ASSUME_NO_TASKWAIT
        hdr->state = next_state;
        hdr->waiting_child_count = child_count;
#endif
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
        printf("set_state_for_join_block: tid=%d child_count=%d\n", tid, child_count);
#endif
    }
    __syncthreads();
    return child_count != 0;
}

#ifndef GTAP_ASSUME_NO_TASKWAIT
__device__ __forceinline__ int notify_parent(int parentId, TaskContext* ctx) {
    TaskHeader* parent_hdr = &d_task_headers[parentId];
    __threadfence();
    int rem = atomicSub(&parent_hdr->waiting_child_count, 1);
    if (rem == 1) {
        ctx->have_task_id_resumable = true;
        ctx->task_id_resumable = parentId;
    }
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
    printf("notify_parent: %d, rem: %d\n", parentId, rem);
#endif
    return rem;
}
#endif

__device__ void finish_task(int tid, TaskContext* ctx) {
    __syncthreads();
    if (threadIdx.x == 0) {
#ifdef GTAP_ASSUME_NO_TASKWAIT
        release_task_id_to_block_pool(tid);
        if (tid == 0) store_L2(&d_first_task_finished, 1);
#else
        TaskHeader* cached_hdr = &ctx->cached_task_header;
        int parent_tid = cached_hdr->parent_tid;
        d_task_headers[tid].generation = cached_hdr->generation + 1;

        if (tid != 0 && load_L2(&d_task_headers[parent_tid].generation) == cached_hdr->parent_generation) {
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
            printf("finish_task: %d, parent_tid: %d\n", tid, parent_tid);
#endif
            notify_parent(parent_tid, ctx);
            release_task_id_to_block_pool(tid);
        } else {
            release_task_id_to_block_pool(tid);
        }
        if (tid == 0) store_L2(&d_first_task_finished, 1);
#endif
    }
}

__device__ __forceinline__ void* spawn_task(
    TaskContext* ctx,
    int self_tid,
    int* child_count,
    void (*func)(void*, int, TaskContext*),
    int unused_value
) {
    (void)unused_value;
    int new_tid = get_task_id_from_block_pool(
        &d_task_id_list_free_positions[blockIdx.x],
        &ctx->id_list_alloc_pos,
        &ctx->id_list_free_pos_stale);
    TaskHeader* new_hdr = &d_task_headers[new_tid];
    new_hdr->func = func;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    TaskHeader* cached_hdr = &ctx->cached_task_header;
    new_hdr->parent_tid = self_tid;
    new_hdr->parent_generation = cached_hdr->generation;
    new_hdr->state = 0;
    new_hdr->waiting_child_count = 0;
#endif

    reserve_unpublished_task_id(ctx, new_tid);
    (void)child_count;
    return get_task_data(new_tid);
}

}  // namespace gtap::detail::block

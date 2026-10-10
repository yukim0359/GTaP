#pragma once

#include "scheduler.cuh"
#include "task_pool.cuh"
#include "termination.cuh"

namespace gtap::detail::block {

using namespace gtap::detail;

// TODO: These functions share the compiler entry points with thread mode. A block-mode lowering could give them their own signatures.
// TODO: Clarify which of the block task ops and push_initial_task are thread 0
// only and which run on every thread in the block.

__device__ __forceinline__ int get_task_state(int tid) {
#ifdef GTAP_ASSUME_NO_TASKWAIT
    return 0;
#else
    return load_L2(&d_task_headers[tid].state);
#endif
}

// TODO: Rename set_state_for_join_block and __gtap_set_state_for_join_block
// to a prepare_for_join style name. The function takes the generated child
// count across the block, then reports whether the task suspends.
__device__ __forceinline__ bool set_state_for_join_block(
    int tid,
    TaskContext* ctx,
    int next_state,
    int unused_value
) {
    (void)unused_value;
    __syncthreads();
    const int child_count = ctx->generated_task_count;
    if (threadIdx.x == 0) {
        TaskHeader* const hdr = &d_task_headers[tid];
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
    TaskHeader* const parent_hdr = &d_task_headers[parentId];
    __threadfence();
    const int rem = atomicSub(&parent_hdr->waiting_child_count, 1);
    if (rem == 1) {
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
#else
        const int parent_tid = ctx->parent_tid;
        d_task_headers[tid].generation = ctx->generation + 1;

        if (tid != 0 && load_L2(&d_task_headers[parent_tid].generation) == ctx->parent_generation) {
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
            printf("finish_task: %d, parent_tid: %d\n", tid, parent_tid);
#endif
            notify_parent(parent_tid, ctx);
        }
        release_task_id_to_block_pool(tid);
#endif
        if (tid == 0) store_L2(&d_root_task_finished, 1);
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
    const int new_tid = get_task_id_from_block_pool(
        &d_task_id_list_free_positions[blockIdx.x],
        &ctx->id_list_alloc_pos,
        &ctx->id_list_free_pos_stale);
    TaskHeader* const new_hdr = &d_task_headers[new_tid];
    new_hdr->func = func;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    new_hdr->parent_tid = self_tid;
    new_hdr->parent_generation = ctx->generation;
    new_hdr->state = 0;
    new_hdr->waiting_child_count = 0;
#endif

    stage_task_id(ctx, new_tid);
    (void)child_count;
    return get_task_data(new_tid);
}

}  // namespace gtap::detail::block

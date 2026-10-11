#pragma once

#include "scheduler.cuh"
#include "task_pool.cuh"
#include "termination.cuh"

namespace gtap::detail::block {

using namespace gtap::detail;

// May be called independently by any thread. No block synchronization.
__device__ __forceinline__ int get_task_state(int tid) {
#ifdef GTAP_ASSUME_NO_TASKWAIT
    (void)tid;
    return 0;
#else
    return load_L2(&d_task_headers[tid].state);
#endif
}

// Every thread in the block must call this.
__device__ __forceinline__ bool prepare_for_join(
    int tid,
    TaskContext* ctx,
    int next_state
) {
    __syncthreads();
    const int child_count = ctx->generated_task_count;
    if (threadIdx.x == 0) {
#ifndef GTAP_ASSUME_NO_TASKWAIT
        TaskHeader* const hdr = &d_task_headers[tid];
        hdr->state = next_state;
        hdr->waiting_child_count = child_count;
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
        printf("prepare_for_join: %d (child count: %d) in block %d\n",
               tid, child_count, blockIdx.x);
#endif
#else
        (void)tid;
        (void)next_state;
#endif
    }
    __syncthreads();
    return child_count != 0;
}

#ifndef GTAP_ASSUME_NO_TASKWAIT
// Called only from finish_task's thread 0 path.
__device__ __forceinline__ int notify_parent(int parent_tid, TaskContext* ctx) {
    TaskHeader* const parent_hdr = &d_task_headers[parent_tid];
    // Fence before the decrement. Another worker can bring waiting_child_count
    // to zero and resume the parent before this worker's global stores are
    // visible.
    __threadfence();
    const int rem = atomicSub(&parent_hdr->waiting_child_count, 1);
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
    printf("notify_parent: %d (remaining child count: %d) in block %d\n",
           parent_tid, rem, blockIdx.x);
#endif
    if (rem == 1) {
        ctx->task_id_resumable = parent_tid;
    }
    return rem;
}
#endif

// Every thread in the block must call this.
__device__ __forceinline__ void finish_task(int tid, TaskContext* ctx) {
    __syncthreads();
    if (threadIdx.x == 0) {
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
        printf("finish_task: %d in block %d\n", tid, blockIdx.x);
#endif
#ifdef GTAP_ASSUME_NO_TASKWAIT
        (void)ctx;
#else
        const int parent_tid = ctx->parent_tid;
        const uint16_t next_generation =
            static_cast<uint16_t>(ctx->generation + 1);
        d_task_headers[tid].generation = next_generation;
        // prepare_for_join leaves this unchanged, so the execute loop can tell
        // a finished task from one that suspended at taskwait.
        ctx->generation = next_generation;

        if (tid != 0 && load_L2(&d_task_headers[parent_tid].generation) == ctx->parent_generation) {
            notify_parent(parent_tid, ctx);
        }
#endif
        if (tid == 0) {
            store_L2(&d_root_task_finished, 1);
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
            printf("root task finished in block %d\n", blockIdx.x);
#endif
        }
    }
}

// Any thread may call this; each call creates one child task.
__device__ __forceinline__ void* spawn_task(
    TaskContext* ctx,
    int self_tid,
    void (*func)(void*, int, TaskContext*)
) {
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
#else
    (void)self_tid;
#endif

    stage_task_id(ctx, new_tid);
    return get_task_data(new_tid);
}

}  // namespace gtap::detail::block

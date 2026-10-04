#pragma once

#include <climits>
#include <cuda_runtime.h>

#include "../common/cuda_primitives.cuh"
#include "../common/profile_buffer.cuh"
#include "../common/runtime_error.cuh"
#include "../common/termination.cuh"
#include "../common/victim_select.cuh"

#include "core.cuh"

#define GTAP_PROFILE_HAS_DROPPED_COUNTER 1

extern const size_t __gtap_auto_entry_result_size;

// Exposed device globals
// Note: gtap::detail::block::d_task_data_bytes is now char* (byte array) to support type-erased task data (static allocation)
namespace gtap::detail::block {
using namespace gtap::detail;

struct BlockTaskQueueMetadata {
    int top;
    int bottom;
};

__constant__ BlockTaskQueueMetadata* d_block_task_queue_metadata;
__constant__ int* d_block_task_queue_storage;

__device__ __forceinline__ int* block_queue_slot(
    int block_idx, int slot
) {
    return &d_block_task_queue_storage[
        static_cast<size_t>(block_idx) *
            d_launch_config.queue_capacity + slot];
}

__device__ __forceinline__ void reserve_unpublished_task_id(TaskContext* ctx, int task_id) {
    BlockTaskQueueMetadata* q = &d_block_task_queue_metadata[blockIdx.x];
    int old_tail = atomicAdd(&ctx->queue_tail, 1);
    int top = load_L2(&q->top);
    const int queue_capacity = d_launch_config.queue_capacity;
    if (old_tail + 1 - top > queue_capacity - GTAP_DETAIL_QUEUE_MARGIN) {
        GTAP_DETAIL_RECORD_QUEUE_OVERFLOW(
            task_id, -1, old_tail + 1 - top, queue_capacity - GTAP_DETAIL_QUEUE_MARGIN);
    }
    store_L2(
        block_queue_slot(blockIdx.x, old_tail % queue_capacity), task_id);
    atomicAdd(&ctx->task_id_generated_count, 1);
}

// Chase-Lev pop: owner pops from bottom
__device__ __forceinline__ int pop(int* taskId) {
    BlockTaskQueueMetadata* myQueue = &d_block_task_queue_metadata[blockIdx.x];

    int b = myQueue->bottom - 1;
    store_L2(&myQueue->bottom, b);
    __threadfence();

    int t = load_L2(&myQueue->top);
    int size = b - t;

    if (size < 0) {
        store_L2(&myQueue->bottom, t);
        *taskId = -1;
        return false;
    }

    int task_id = load_L2(block_queue_slot(
        blockIdx.x, b % d_launch_config.queue_capacity));

    if (size > 0) {
        *taskId = task_id;
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
        printf("pop: %d (block: %d)\n", task_id, blockIdx.x);
#endif
        return true;
    }

    if (atomicCAS(&myQueue->top, t, t + 1) != t) {
        *taskId = -1;
        store_L2(&myQueue->bottom, t + 1);
        return false;
    }

    *taskId = task_id;
    store_L2(&myQueue->bottom, t + 1);
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
    printf("pop: %d (block: %d)\n", task_id, blockIdx.x);
#endif
    return true;
}

template<TerminationMode M>
__device__ __forceinline__ int steal(int* taskId, bool prev_get_task) {
    int targetBlock = get_random_block_id(blockIdx.x);
    BlockTaskQueueMetadata* targetBq = &d_block_task_queue_metadata[targetBlock];

    int t = load_L2(&targetBq->top);
    __threadfence();
    int b = load_L2(&targetBq->bottom);

    int size = b - t;
    if (size <= 0) {
        *taskId = -1;
        return false;
    }

    int task_id = load_L2(block_queue_slot(
        targetBlock, t % d_launch_config.queue_capacity));

    if (atomicCAS(&targetBq->top, t, t + 1) != t) {
        *taskId = -1;
        return false;
    }

    if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
        if (!prev_get_task) atomicAdd(&d_active_block_count, 1);
    }

    *taskId = task_id;
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
    printf("steal: %d (block: %d -> %d)\n", task_id, targetBlock, blockIdx.x);
#endif
    return true;
}

// Chase-Lev push: owner pushes to bottom
__device__ __forceinline__ void push(
    TaskContext* ctx,
    int push_total,
    int* execute_task_id
) {
    BlockTaskQueueMetadata* myQueue = &d_block_task_queue_metadata[blockIdx.x];
    (void)push_total;

#ifdef GTAP_ASSUME_NO_TASKWAIT
    int publish_bottom = ctx->queue_tail;
    if (ctx->task_id_generated_count > 0) {
        publish_bottom = ctx->queue_tail - 1;
        if (threadIdx.x == 0) {
            *execute_task_id = load_L2(block_queue_slot(
                blockIdx.x,
                publish_bottom % d_launch_config.queue_capacity));
        }
    }
#else
    int publish_bottom = ctx->queue_tail;
    if (ctx->have_task_id_resumable) {
        if (threadIdx.x == 0) {
            *execute_task_id = ctx->task_id_resumable;
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
            printf("resume: %d (block: %d)\n", *execute_task_id, blockIdx.x);
#endif
        }
    } else if (ctx->task_id_generated_count > 0) {
        publish_bottom = ctx->queue_tail - 1;
        if (threadIdx.x == 0) {
            *execute_task_id = load_L2(block_queue_slot(
                blockIdx.x,
                publish_bottom % d_launch_config.queue_capacity));
        }
    }
#endif
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        store_L2(&myQueue->bottom, publish_bottom);
    }
}

__device__ __forceinline__ void set_state_for_join(
    int tid,
    int child_count,
    int next_state,
    int unused_value
) {
    (void)unused_value;
    __syncthreads();
    if (threadIdx.x == 0) {
        TaskHeader* hdr = &d_task_headers[tid];
#ifndef GTAP_ASSUME_NO_TASKWAIT
        hdr->state = next_state;
        hdr->waiting_child_count = child_count;
#endif
    }
}

__device__ __forceinline__ bool set_state_for_join_block(
    int tid,
    TaskContext* ctx,
    int next_state,
    int unused_value
) {
    (void)unused_value;
    __syncthreads();
    int child_count = ctx->task_id_generated_count;
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

__device__ __forceinline__ int get_task_state(int tid) {
#ifdef GTAP_ASSUME_NO_TASKWAIT
    return 0;
#else
    return load_L2(&d_task_headers[tid].state);
#endif
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

__device__ __forceinline__ void push_initial_task(
    void (*func)(void*, int, TaskContext*),
    int unused_value
) {
    (void)unused_value;
    TaskHeader* initial_hdr = &d_task_headers[0];
    initial_hdr->func = func;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    initial_hdr->state = 0;
    initial_hdr->parent_tid = 0;
    initial_hdr->parent_generation = 0;
    initial_hdr->waiting_child_count = 0;
#endif

    // Task data is copied from the compiler-generated code (out of this function)

    BlockTaskQueueMetadata* bq = &d_block_task_queue_metadata[blockIdx.x];
    store_L2(block_queue_slot(blockIdx.x, 0), 0);
    __threadfence();
    store_L2(&bq->bottom, 1);
}

template<TerminationMode M>
__device__ __forceinline__ void execute_task_loop() {
    __shared__ int execute_task_id;
    __shared__ bool have_execute_task;
    __shared__ bool prev_get_task;
    __shared__ bool should_continue;
    __shared__ TaskContext block_ctx;
#ifdef GTAP_ENABLE_PROFILING
    __shared__ int working_time_idx;
#endif

    if (threadIdx.x == 0) {
        should_continue = true;
        have_execute_task = false;
#ifndef GTAP_ASSUME_NO_TASKWAIT
        block_ctx.have_task_id_resumable = false;
#endif
        block_ctx.task_id_generated_count = 0;
        block_ctx.id_list_free_pos_stale = d_launch_config.tasks_per_worker;
#ifdef GTAP_ENABLE_PROFILING
        working_time_idx = 0;
#endif
        if (blockIdx.x == 0) {
            block_ctx.id_list_alloc_pos = 1;
            prev_get_task = true;
        } else {
            block_ctx.id_list_alloc_pos = 0;
            prev_get_task = false;
        }
    }
    __syncthreads();

    while (should_continue) {
        if (threadIdx.x == 0) {
            if (!have_execute_task) {
                if (prev_get_task) {
                    have_execute_task = pop(&execute_task_id);
                }
            }
            if (!have_execute_task) {
                have_execute_task = steal<M>(&execute_task_id, prev_get_task);
            }
        }
        __syncthreads();

        if (!have_execute_task) {
            if (threadIdx.x == 0) {
                if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
                    if (prev_get_task) {
                        int active_block_count = atomicSub(&d_active_block_count, 1) - 1;
                        if (active_block_count == 0) {
                            bool all_tasks_finished = 1;
                            BlockTaskQueueMetadata* q = &d_block_task_queue_metadata[blockIdx.x];
                            int t = load_L2(&q->top);
                            int b = load_L2(&q->bottom);
                            if (t < b) {
                                all_tasks_finished = 0;
                            }
                            atomicExch(&d_all_tasks_finished, all_tasks_finished);
                        }
                    }
                }
                prev_get_task = false;
                if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
                    should_continue = (load_L2(&d_all_tasks_finished) == 0);
                    // if (active_block_count == 0) consecutive_idle_count++;
                    // else consecutive_idle_count = 0;
                    // should_continue = (consecutive_idle_count != NUMBER_OF_CONSECUTIVE_IDLE_COUNTS_TO_TERMINATE);
                } else {
                    should_continue = (load_L2(&d_first_task_finished) == 0);
                }
            }
            __syncthreads();
            continue;
        } else {
            if (threadIdx.x == 0) {
                prev_get_task = true;
                block_ctx.task_id_generated_count = 0;
                block_ctx.queue_tail = load_L2(&d_block_task_queue_metadata[blockIdx.x].bottom);
#ifndef GTAP_ASSUME_NO_TASKWAIT
                block_ctx.have_task_id_resumable = false;
#endif
            }
            __syncthreads();
        }

        if (have_execute_task) {
#ifndef GTAP_ASSUME_NO_TASKWAIT
            // Copy task header to TaskContext for reuse in task function (using L2 load)
            if (threadIdx.x == 0) {
                TaskHeader* src_hdr = &d_task_headers[execute_task_id];
                TaskHeader* dst_hdr = &block_ctx.cached_task_header;
                dst_hdr->generation = load_L2(&src_hdr->generation);
                dst_hdr->parent_tid = load_L2(&src_hdr->parent_tid);
                dst_hdr->parent_generation = load_L2(&src_hdr->parent_generation);
            }
            __syncthreads();
#endif

#ifdef GTAP_ENABLE_PROFILING
            if (threadIdx.x == 0) {
                if (working_time_idx + 1 < profile_timestamp_capacity()) {
                    working_time[
                        blockIdx.x * profile_timestamp_capacity() +
                        working_time_idx] = get_global_time();
                    working_time_idx++;
                } else {
                    atomicAdd(&profile_dropped_events[blockIdx.x], 1ULL);
                }
            }
#endif
            void* task_data = get_task_data(execute_task_id);
            // Read function pointer atomically (64-bit) via L2 cache
            void* func_ptr = load_L2(reinterpret_cast<void**>(&d_task_headers[execute_task_id].func));
            void (*task_func)(void*, int, TaskContext*) = reinterpret_cast<void (*)(void*, int, TaskContext*)>(func_ptr);
            task_func(task_data, execute_task_id, &block_ctx);
            // if(threadIdx.x == 0) printf("finish_execute_task: %d\n", tid);
        }
        __syncthreads();
        __threadfence();
#ifdef GTAP_ENABLE_PROFILING
        if (threadIdx.x == 0) {
            if (working_time_idx < profile_timestamp_capacity()) {
                working_time[
                    blockIdx.x * profile_timestamp_capacity() +
                    working_time_idx] = get_global_time();
                working_time_idx++;
            }
        }
#endif

        int total_count =
#ifdef GTAP_ASSUME_NO_TASKWAIT
            block_ctx.task_id_generated_count;
#else
            (block_ctx.have_task_id_resumable ? 1 : 0) + block_ctx.task_id_generated_count;
#endif
        int push_total = max(total_count - 1, 0);
        push(&block_ctx, push_total, &execute_task_id);
        if (threadIdx.x == 0) {
            // printf("total_count: %d\n", total_count);
            have_execute_task = (total_count > 0);
        }
    }
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
    if (threadIdx.x == 0) printf("execute_task_loop: end (block_id = %d)\n", blockIdx.x);
#endif
}

}  // namespace gtap::detail::block

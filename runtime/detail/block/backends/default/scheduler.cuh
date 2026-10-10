#pragma once

#include <cuda_runtime.h>

#include "../../../common/cuda_primitives.cuh"
#include "../../../common/runtime_error.cuh"
#include "../../../common/termination.cuh"
#include "../../../common/victim_select.cuh"

#include "../../profile_buffer.cuh"
#include "../../scheduler_helpers.cuh"
#include "../../task_pool.cuh"
#include "../../task_types.cuh"
#include "../../termination.cuh"
#include "queue_storage.cuh"

namespace gtap::detail::block {
using namespace gtap::detail;

__device__ __forceinline__ void stage_task_id(TaskContext* ctx, int task_id) {
    BlockTaskQueueMetadata* q = &d_block_task_queue_metadata[blockIdx.x];
    int old_bottom = atomicAdd(&ctx->queue_bottom, 1);
    int top = load_L2(&q->top);
    const int queue_capacity = d_launch_config.queue_capacity;
    if (old_bottom + 1 - top > queue_capacity - GTAP_DETAIL_QUEUE_MARGIN) {
        GTAP_DETAIL_RECORD_QUEUE_OVERFLOW(
            task_id, -1, old_bottom + 1 - top, queue_capacity - GTAP_DETAIL_QUEUE_MARGIN);
    }
    store_L2(
        block_queue_slot(blockIdx.x, old_bottom % queue_capacity), task_id);
    atomicAdd(&ctx->generated_task_count, 1);
}

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

// Pop from this block's queue, then steal if it is still empty.
template<TerminationMode M>
__device__ __forceinline__ void fill_execution_batch(
    int* execute_task_id,
    bool* have_execute_task,
    bool prev_get_task
) {
    if (threadIdx.x == 0) {
        if (!*have_execute_task) {
            if (prev_get_task) {
                *have_execute_task = pop(execute_task_id);
            }
        }
        if (!*have_execute_task) {
            *have_execute_task = steal<M>(execute_task_id, prev_get_task);
        }
    }
}

__device__ __forceinline__ void push(
    TaskContext* ctx,
    int* execute_task_id,
    bool* have_execute_task
) {
    BlockTaskQueueMetadata* myQueue = &d_block_task_queue_metadata[blockIdx.x];

#ifdef GTAP_ASSUME_NO_TASKWAIT
    int publish_bottom = ctx->queue_bottom;
    if (ctx->generated_task_count > 0) {
        publish_bottom = ctx->queue_bottom - 1;
        if (threadIdx.x == 0) {
            *execute_task_id = load_L2(block_queue_slot(
                blockIdx.x,
                publish_bottom % d_launch_config.queue_capacity));
        }
    }
#else
    int publish_bottom = ctx->queue_bottom;
    if (ctx->task_id_resumable != -1) {
        if (threadIdx.x == 0) {
            *execute_task_id = ctx->task_id_resumable;
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
            printf("resume: %d (block: %d)\n", *execute_task_id, blockIdx.x);
#endif
        }
    } else if (ctx->generated_task_count > 0) {
        publish_bottom = ctx->queue_bottom - 1;
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
#ifdef GTAP_ASSUME_NO_TASKWAIT
        *have_execute_task = ctx->generated_task_count > 0;
#else
        *have_execute_task =
            ctx->task_id_resumable != -1 || ctx->generated_task_count > 0;
        ctx->task_id_resumable = -1;
#endif
        store_L2(&myQueue->bottom, publish_bottom);
        ctx->generated_task_count = 0;
    }
}

// TODO: Clarify which of the block task ops and push_initial_task are thread 0
// only and which run on every thread in the block.
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

__device__ __forceinline__ void initialize_loop(
    bool* have_execute_task,
    bool* prev_get_task,
    TaskContext* task_context
#ifdef GTAP_ENABLE_PROFILING
    , int* working_time_idx
#endif
) {
    if (threadIdx.x == 0) {
        *have_execute_task = false;
#ifndef GTAP_ASSUME_NO_TASKWAIT
        task_context->task_id_resumable = -1;
#endif
        task_context->generated_task_count = 0;
        task_context->id_list_free_pos_stale = d_launch_config.tasks_per_scheduling_unit;
#ifdef GTAP_ENABLE_PROFILING
        *working_time_idx = 0;
#endif
        if (blockIdx.x == 0) {
            task_context->id_list_alloc_pos = 1;
            *prev_get_task = true;
        } else {
            task_context->id_list_alloc_pos = 0;
            *prev_get_task = false;
        }
    }
    __syncthreads();
}

template<TerminationMode M>
__device__ __forceinline__ bool mark_idle_and_check_termination(
    bool* prev_get_task
) {
    __shared__ bool terminate;
    if (threadIdx.x == 0) {
        if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
            if (*prev_get_task) {
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
        *prev_get_task = false;
        if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
            terminate = (load_L2(&d_all_tasks_finished) != 0);
            // if (active_block_count == 0) consecutive_idle_count++;
            // else consecutive_idle_count = 0;
            // should_continue = (consecutive_idle_count != NUMBER_OF_CONSECUTIVE_IDLE_COUNTS_TO_TERMINATE);
        } else {
            terminate = (load_L2(&d_root_task_finished) != 0);
        }
    }
    __syncthreads();
    return terminate;
}

template<TerminationMode M>
__device__ __forceinline__ void execute_task_loop() {
    __shared__ int execute_task_id;
    // TODO: have_execute_task can be removed. Store -1 in execute_task_id on the
    // no-task paths and test that instead.
    __shared__ bool have_execute_task;
    __shared__ bool prev_get_task;
    __shared__ TaskContext task_context;
#ifdef GTAP_ENABLE_PROFILING
    __shared__ int working_time_idx;
#endif

    initialize_loop(
        &have_execute_task, &prev_get_task, &task_context
#ifdef GTAP_ENABLE_PROFILING
        , &working_time_idx
#endif
    );

    while (true) {
        fill_execution_batch<M>(&execute_task_id, &have_execute_task, prev_get_task);
        __syncthreads();

        if (!have_execute_task) {
            if (mark_idle_and_check_termination<M>(&prev_get_task))
                break;
            continue;
        } else {
            if (threadIdx.x == 0) {
                prev_get_task = true;
                task_context.queue_bottom = load_L2(&d_block_task_queue_metadata[blockIdx.x].bottom);
            }
            __syncthreads();
        }

        if (have_execute_task) {
            void* task_data = get_task_data(execute_task_id);
            prefetch_global_L2(task_data);
#ifndef GTAP_ASSUME_NO_TASKWAIT
            copy_task_header(execute_task_id, &task_context);
#endif

#ifdef GTAP_ENABLE_PROFILING
            record_execution_start(&working_time_idx);
#endif
            void* func_ptr = load_L2(reinterpret_cast<void**>(&d_task_headers[execute_task_id].func));
            void (*task_func)(void*, int, TaskContext*) = reinterpret_cast<void (*)(void*, int, TaskContext*)>(func_ptr);
            task_func(task_data, execute_task_id, &task_context);
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
            if (threadIdx.x == 0)
                printf("executed_task_id: %d in block %d\n", execute_task_id, blockIdx.x);
#endif
        }
        __syncthreads();
        __threadfence();
#ifdef GTAP_ENABLE_PROFILING
        record_execution_end(&working_time_idx);
#endif

        push(&task_context, &execute_task_id, &have_execute_task);
    }
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
    if (threadIdx.x == 0) printf("execute_task_loop: end (block_id = %d)\n", blockIdx.x);
#endif
}

}  // namespace gtap::detail::block

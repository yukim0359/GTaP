#pragma once

#include <cuda_runtime.h>

#include "../../../../common/cuda_primitives.cuh"
#include "../../../../common/runtime_error.cuh"
#include "../../../../common/termination.cuh"

#include "../../../profile_buffer.cuh"
#include "../../../task_pool.cuh"
#include "../../../task_types.cuh"
#include "../../../termination.cuh"
#include "queue_storage.cuh"

namespace gtap::detail::block {
using namespace gtap::detail;

__device__ __forceinline__ void reserve_unpublished_task_id(
    TaskContext* ctx, int task_id
) {
    int gen_idx = atomicAdd(&ctx->generated_task_count, 1);
    set_task_id_generated(blockIdx.x, gen_idx, task_id);
}

// ============================================================================
// Global Queue Operations (no steal needed - all blocks pop from global queue)
// ============================================================================

// Pop from global queue - block pops a single task
template<TerminationMode M>
__device__ __forceinline__ bool pop_global_queue(int* execute_task_id, bool prev_get_task) {
    bool pop_success = false;
    unsigned int head;
    // Try to claim a slot from global queue
    while (true) {
        unsigned int old_head = load_L2(&d_queue_head);
        unsigned int tail = load_L2(&d_queue_tail);
        unsigned int available = tail - old_head;  // unsigned subtraction handles wrap-around

        if (available == 0) break;

        // CAS to claim slot
        unsigned int new_head = old_head + 1;
        if (atomicCAS(&d_queue_head, old_head, new_head) == old_head) {
            head = old_head;
            pop_success = true;
            // Increment the active block count if this block was previously idle
            if (M == TERMINATE_ON_ALL_TASKS_FINISH && !prev_get_task) {
                atomicAdd(&d_active_block_count, 1);
            }
            break;
        }
        // CAS failed, retry
    }

    if (pop_success) {
        int idx = head % (d_launch_config.total_scheduling_units * d_launch_config.tasks_per_scheduling_unit);
        *execute_task_id = load_L2(&d_global_task_queue[idx]);
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
        printf("pop_global: tid=%d in block %d\n", *execute_task_id, blockIdx.x);
#endif
    } else {
        *execute_task_id = -1;
    }
    return pop_success;
}

// Push to global queue
template<TerminationMode M>
__device__ __forceinline__ void push_global_queue(
    TaskContext* ctx,
    int* execute_task_id,
    bool* have_execute_task
) {
    __shared__ unsigned int base_pos;
    __shared__ int first_idx_to_push;
    __shared__ int push_cnt;

    int total_count =
#ifdef GTAP_ASSUME_NO_TASKWAIT
        ctx->generated_task_count;
#else
        (ctx->task_id_resumable != -1 ? 1 : 0) + ctx->generated_task_count;
#endif

    if (total_count == 0) {
        *have_execute_task = false;
        return;
    }

    // Determine task to execute immediately vs push to queue
    if (threadIdx.x == 0) {
        first_idx_to_push = 0;
#ifdef GTAP_ASSUME_NO_TASKWAIT
        if (ctx->generated_task_count > 0) {
            *execute_task_id = get_task_id_generated(blockIdx.x, 0);
            *have_execute_task = true;
            first_idx_to_push = 1;
        } else {
            *have_execute_task = false;
        }
#else
        if (ctx->task_id_resumable != -1) {
            *execute_task_id = ctx->task_id_resumable;
            *have_execute_task = true;
        } else if (ctx->generated_task_count > 0) {
            *execute_task_id = get_task_id_generated(blockIdx.x, 0);
            *have_execute_task = true;
            first_idx_to_push = 1;
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
            printf("execute_immediately: tid=%d in block %d\n", *execute_task_id, blockIdx.x);
#endif
        } else {
            *have_execute_task = false;
        }
#endif
        push_cnt = ctx->generated_task_count - first_idx_to_push;
    }
    __syncthreads();

    if (threadIdx.x == 0)
        ctx->generated_task_count = 0;

    // Push remaining tasks to global queue
    if (push_cnt <= 0) return;

    // Reserve slots in global queue (allocate exclusive range)
    if (threadIdx.x == 0) {
        base_pos = atomicAdd(&d_queue_alloc, (unsigned int)push_cnt);
        // Overflow check (unsigned subtraction handles wrap-around)
        unsigned int head_val = load_L2(&d_queue_head);
        if (base_pos + (unsigned int)push_cnt - head_val > (d_launch_config.total_scheduling_units * d_launch_config.tasks_per_scheduling_unit) - GTAP_DETAIL_QUEUE_MARGIN) {
            GTAP_DETAIL_RECORD_QUEUE_OVERFLOW(
                -1, 0,
                static_cast<int>(base_pos + (unsigned int)push_cnt - head_val),
                (d_launch_config.total_scheduling_units * d_launch_config.tasks_per_scheduling_unit) - GTAP_DETAIL_QUEUE_MARGIN);
        }
    }
    __syncthreads();

    // Write tasks to reserved slots (parallel using block threads)
    for (int j = threadIdx.x; j < push_cnt; j += blockDim.x) {
        int tid = get_task_id_generated(blockIdx.x, first_idx_to_push + j);
        unsigned int pos = (base_pos + (unsigned int)j) % (d_launch_config.total_scheduling_units * d_launch_config.tasks_per_scheduling_unit);
        store_L2(&d_global_task_queue[pos], tid);
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
        printf("push_global: tid=%d to pos %d in block %d\n", tid, pos, blockIdx.x);
#endif
        }
    __threadfence();
    __syncthreads();

    // Wait for prior commits and update tail (ensures in-order visibility)
    if (threadIdx.x == 0) {
        while (load_L2(&d_queue_tail) != base_pos) {
            // spin - wait for prior pushers to commit
    }
        atomicAdd(&d_queue_tail, (unsigned int)push_cnt);
    }
    __syncthreads();
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

    // Push to global queue (only block 0)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        store_L2(&d_global_task_queue[0], 0);
        __threadfence();
        store_L2(&d_queue_head, 0u);
        store_L2(&d_queue_alloc, 1u);
        store_L2(&d_queue_tail, 1u);
        __threadfence();
    }
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
__device__ __forceinline__ void fill_execution_batch(
    int* execute_task_id,
    bool* have_execute_task,
    bool prev_get_task
) {
    if (threadIdx.x == 0) {
        if (!*have_execute_task) {
            *have_execute_task = pop_global_queue<M>(execute_task_id, prev_get_task);
        }
    }
}

#ifdef GTAP_ENABLE_PROFILING
__device__ __forceinline__ void record_execution_start(int* working_time_idx) {
    if (threadIdx.x == 0) {
        if (*working_time_idx + 1 < profile_timestamp_capacity()) {
            working_time[
                blockIdx.x * profile_timestamp_capacity() +
                *working_time_idx] = get_global_time();
            (*working_time_idx)++;
        } else {
            atomicAdd(&profile_dropped_events[blockIdx.x], 1ULL);
        }
    }
}

__device__ __forceinline__ void record_execution_end(int* working_time_idx) {
    if (threadIdx.x == 0) {
        if (*working_time_idx < profile_timestamp_capacity()) {
            working_time[
                blockIdx.x * profile_timestamp_capacity() +
                *working_time_idx] = get_global_time();
            (*working_time_idx)++;
        }
    }
}
#endif

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
                    // Check if queue is empty (unsigned comparison handles wrap-around)
                    bool all_tasks_finished = 1;
                    unsigned int head = load_L2(&d_queue_head);
                    unsigned int tail = load_L2(&d_queue_tail);
                    if (tail - head > 0) {  // unsigned subtraction
                        all_tasks_finished = 0;
                    }
                    atomicExch(&d_all_tasks_finished, all_tasks_finished);
                }
            }
        }
        *prev_get_task = false;
        if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
            terminate = (load_L2(&d_all_tasks_finished) != 0);
        } else {
            terminate = (load_L2(&d_root_task_finished) != 0);
        }
    }
    __syncthreads();
    return terminate;
}

#ifndef GTAP_ASSUME_NO_TASKWAIT
// Copy task header to TaskContext for reuse in task function (using L2 load)
__device__ __forceinline__ void copy_task_header(
    int execute_task_id,
    TaskContext* task_context
) {
    if (threadIdx.x == 0) {
        TaskHeader* src_hdr = &d_task_headers[execute_task_id];
        task_context->parent_tid = load_L2(&src_hdr->parent_tid);
        *reinterpret_cast<unsigned int*>(&task_context->generation) =
            load_L2(reinterpret_cast<unsigned int*>(&src_hdr->generation));
    }
    __syncthreads();
}
#endif

template<TerminationMode M>
__device__ __forceinline__ void execute_task_loop() {
    __shared__ int execute_task_id;
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
#ifndef GTAP_ASSUME_NO_TASKWAIT
                task_context.task_id_resumable = -1;
#endif
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
        }
        __syncthreads();
        __threadfence();
#ifdef GTAP_ENABLE_PROFILING
        record_execution_end(&working_time_idx);
#endif

        push_global_queue<M>(&task_context, &execute_task_id, &have_execute_task);
    }
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
    if (threadIdx.x == 0) printf("execute_task_loop: end (block_id = %d)\n", blockIdx.x);
#endif
}

}  // namespace gtap::detail::block

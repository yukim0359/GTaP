#pragma once

#include <cuda_runtime.h>

#include "../../../../common/cuda_primitives.cuh"
#include "../../../../common/runtime_error.cuh"
#include "../../../../common/termination.cuh"
#include "../../../../common/victim_select.cuh"
#include "../../../../common/warp_index.cuh"

#include "../../../profile_buffer.cuh"
#include "../../../queue_select.cuh"
#include "../../../shared_layout.cuh"
#include "../../../task_pool.cuh"
#include "../../../task_types.cuh"
#include "../../../termination.cuh"
#include "queue_storage.cuh"

namespace gtap::detail::thread {
using namespace gtap::detail;

// Whether make_shared_layout reserves per-queue tails. gtap_initialize and the execute loop both pass this.
inline constexpr bool include_queue_tails = true;

extern __shared__ unsigned char dynamic_shared[];

// Chase-Lev style sequential pop/steal operations

// Chase-Lev popBottom (single item) - called only by lane 0
// Returns task_id on success, -1 on failure (Empty)
__device__ __forceinline__ int pop_single_chase_lev(
    WarpTaskQueueMetadata* q, int queue_idx, int warp_idx
) {
    int b = q->bottom - 1;
    store_L2(&q->bottom, b);
    __threadfence();
    int t = load_L2(&q->top);
    int size = b - t;

    if (size < 0) {
        q->bottom = t;
        return -1;
    }

    int task_id = load_L2(chaselev_queue_slot(
        queue_idx, warp_idx,
        b % d_launch_config.queue_capacity));

    if (size > 0) {
        return task_id;
    }

    if (atomicCAS(&q->top, t, t + 1) != t) {
        // Lost race to stealer
        task_id = -1;
    }
    store_L2(&q->bottom, t + 1);
    return task_id;
}

// Sequential pop using chase-lev (repeats single pops)
__device__ __forceinline__ int pop_chase_lev(int* execute_task_id, int max_count_to_pop, int queue_idx) {
    int lane = get_lane_id();
    WarpTaskQueueMetadata* myQueue = warp_queue_metadata(queue_idx, get_warp_id_global());
    int pop_count = 0;

    for (int i = 0; i < max_count_to_pop; i++) {
        int task_id = -1;
        if (lane == 0) {
            task_id = pop_single_chase_lev(
                myQueue, queue_idx, get_warp_id_global());
        }
        task_id = __shfl_sync(0xFFFFFFFFu, task_id, 0);

        if (task_id == -1) break;

        // Assign to lane (filling from high lanes: warp_size-max_count_to_pop, ...)
        int target_lane = warp_size - max_count_to_pop + i;
        if (lane == target_lane) {
            *execute_task_id = task_id;
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
            printf("pop_task_id: %d (kind %d) in lane %d of warp %d of block %d\n", task_id, queue_idx, lane, get_warp_id_in_block(), blockIdx.x);
#endif
        }
        pop_count++;
    }

    return pop_count;
}

// Chase-Lev steal (single item) - called only by lane 0
// Returns task_id on success, -1 on failure (Empty or Abort)
__device__ __forceinline__ int steal_single_chase_lev(
    WarpTaskQueueMetadata* q, int queue_idx, int warp_idx
) {
    int t = load_L2(&q->top);
    __threadfence();
    int b = load_L2(&q->bottom);

    int size = b - t;
    if (size <= 0) return -1;

    int task_id = load_L2(chaselev_queue_slot(
        queue_idx, warp_idx,
        t % d_launch_config.queue_capacity));

    if (atomicCAS(&q->top, t, t + 1) != t) {
        return -1;  // Abort - lost race
    }

    return task_id;
}

// Sequential steal using chase-lev (repeats single steals)
template<TerminationMode M>
__device__ __forceinline__ int steal_chase_lev(int* execute_task_id, int max_count_to_steal, int queue_idx, bool prev_get_task) {
    int warp_id_global = get_warp_id_global();
    int lane = get_lane_id();
    int target_warp_id_global = 0;
    WarpTaskQueueMetadata* targetWq = nullptr;
    int steal_count = 0;
    bool active_count_incremented = false;

    // Select a random victim (lane 0 only)
    if (lane == 0) {
        target_warp_id_global = get_random_warp_id_global(warp_id_global);
        targetWq = warp_queue_metadata(queue_idx, target_warp_id_global);
    }
    target_warp_id_global = __shfl_sync(0xFFFFFFFFu, target_warp_id_global, 0);
    targetWq = warp_queue_metadata(queue_idx, target_warp_id_global);

    // Sequential steals using chase-lev
    for (int i = 0; i < max_count_to_steal; i++) {
        int task_id = -1;
        if (lane == 0) {
            task_id = steal_single_chase_lev(
                targetWq, queue_idx, target_warp_id_global);
        }
        task_id = __shfl_sync(0xFFFFFFFFu, task_id, 0);

        if (task_id == -1) break;

        // Increment the active warp count on the first successful steal
        if (M == TERMINATE_ON_ALL_TASKS_FINISH && !active_count_incremented && !prev_get_task) {
            if (lane == 0) atomicAdd(&d_active_warp_count, 1);
            active_count_incremented = true;
        }

        // Assign to lane (filling from high lanes: warp_size-max_count_to_steal, ...)
        int target_lane = warp_size - max_count_to_steal + i;
        if (lane == target_lane) {
            *execute_task_id = task_id;
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
            printf("steal_task_id: %d (kind %d) in lane %d of warp %d of block %d\n", task_id, queue_idx, lane, get_warp_id_in_block(), blockIdx.x);
#endif
        }
        steal_count++;
    }

    return steal_count;
}

// Fill empty lanes from one queue, then steal if the batch is still short of a warp.
template<TerminationMode M>
__device__ __forceinline__ void fill_batch_from_queue(
    int* execute_task_id,
    int* execute_task_count,
    int queue_idx,
    bool prev_get_task
) {
    if (*execute_task_count >= warp_size) return;
    if (prev_get_task) {
        int remaining = warp_size - *execute_task_count;
        *execute_task_count += pop_chase_lev(
            execute_task_id, remaining, queue_idx);
    }
    if (*execute_task_count < warp_size) {
        int remaining = warp_size - *execute_task_count;
        *execute_task_count += steal_chase_lev<M>(
            execute_task_id, remaining, queue_idx, prev_get_task);
    }
}

// Prepare the execution batch, including which queue to take it from.
// A short batch is topped up from the queue it already uses.
template<TerminationMode M>
__device__ __forceinline__ void fill_execution_batch(
    int warp_id_in_block,
    int warp_id_global,
    int lane,
    int* execute_task_id,
    int* execute_task_count,
    bool prev_get_task,
    const shared_layout& layout,
    TaskContext* task_context
) {
    if (d_launch_config.num_queues == 1) {
        fill_batch_from_queue<M>(
            execute_task_id, execute_task_count, 0, prev_get_task);
    } else if (*execute_task_count < warp_size) {
        if (*execute_task_count == 0) {
            int* queue_lengths = reinterpret_cast<int*>(
                dynamic_shared + layout.queue_lengths) +
                warp_id_in_block * d_launch_config.num_queues;
            if (lane == 0) {
                for (int k = 0; k < d_launch_config.num_queues; ++k) {
                    WarpTaskQueueMetadata* q = warp_queue_metadata(k, warp_id_global);
                    queue_lengths[k] =
                        load_L2(&q->bottom) - load_L2(&q->top);
                }
            }
            for (int attempt = 0; attempt < d_launch_config.num_queues; ++attempt) {
                int queue_idx;
                if (lane == 0) {
                    queue_idx = select_next_fullest_queue_idx(
                        queue_lengths,
                        d_launch_config.num_queues);
                    task_context->queue_idx = queue_idx;
                }
                queue_idx = __shfl_sync(0xFFFFFFFFu, task_context->queue_idx, 0);
                fill_batch_from_queue<M>(
                    execute_task_id, execute_task_count, queue_idx,
                    prev_get_task);
                if (*execute_task_count != 0) break;
            }
        } else {
            int queue_idx = __shfl_sync(
                0xFFFFFFFFu, task_context->queue_idx, 0);
            fill_batch_from_queue<M>(
                execute_task_id, execute_task_count, queue_idx,
                prev_get_task);
        }
    }
}

// Chase-Lev pushBottom (multiple items)
// NOTE: the template parameter is not used
__device__ __forceinline__ void reserve_unpublished_task_id(
    TaskContext* ctx, int queue_idx, int task_id
) {
    int gen_idx = atomicAdd(
        &ctx->generated_task_counts[queue_idx], 1);
    if (gen_idx < warp_size) {
        ctx->staged_task_ids[queue_idx * warp_size + gen_idx] = task_id;
        return;
    }

    // Keep overflow tasks in the owner's deque, but do not publish the new
    // bottom until push_batch after all producing lanes have synchronized.
    int old_tail = atomicAdd(&ctx->queue_tails[queue_idx], 1);
    WarpTaskQueueMetadata* q =
        warp_queue_metadata(queue_idx, get_warp_id_global());
    int top = load_L2(&q->top);
    const int capacity = d_launch_config.queue_capacity;
    if (old_tail + 1 - top > capacity - GTAP_DETAIL_QUEUE_MARGIN) {
        GTAP_DETAIL_RECORD_QUEUE_OVERFLOW(
            task_id, queue_idx, old_tail + 1 - top,
            capacity - GTAP_DETAIL_QUEUE_MARGIN);
    }
    *chaselev_queue_slot(
        queue_idx, get_warp_id_global(), old_tail % capacity) = task_id;
}

template<TerminationMode M>
__device__ __forceinline__ void push_batch (
    TaskContext* ctx,
    int* execute_task_id,
    int* execute_task_count
) {
    int warp_id_global = get_warp_id_global();
    int lane = get_lane_id();
    int k_max = 0;
    int max_gen = -1;
    int all_generated_count = 0;
    if (lane == 0) {
        for (int k = 0; k < d_launch_config.num_queues; ++k) {
            int cnt = ctx->generated_task_counts[k];
            all_generated_count += cnt;
            if (cnt > max_gen) {
                max_gen = cnt;
                k_max = k;
            }
        }
        ctx->queue_idx = k_max;
    }
    all_generated_count = __shfl_sync(0xFFFFFFFFu, all_generated_count, 0);
    if (all_generated_count == 0) {
        *execute_task_count = 0;
        return;
    }
    k_max = __shfl_sync(0xFFFFFFFFu, k_max, 0);
    max_gen = __shfl_sync(0xFFFFFFFFu, max_gen, 0);

    *execute_task_count = max(0, min(warp_size, max_gen));
    if (lane < *execute_task_count) {
        *execute_task_id =
            ctx->staged_task_ids[k_max * warp_size + lane];
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
        printf("push_task_id: %d (kind %d) in lane %d of warp %d of block %d\n", *execute_task_id, k_max, lane, get_warp_id_in_block(), blockIdx.x);
#endif
    }

    for (int kind = 0; kind < d_launch_config.num_queues; ++kind) {
        int first_idx_to_push = (kind == k_max) ? *execute_task_count : 0;
        int push_cnt = ctx->generated_task_counts[kind] - first_idx_to_push;
        if (push_cnt <= 0) continue;

        WarpTaskQueueMetadata* q = warp_queue_metadata(kind, warp_id_global);
        int total = ctx->generated_task_counts[kind];
        int staged_n = min(total, warp_size);
        if (kind != k_max) {
            int base = ctx->queue_tails[kind];
            for (int j = lane; j < staged_n; j += warp_size) {
                int idx_to_push =
                    (base + j) % d_launch_config.queue_capacity;
                int val =
                    ctx->staged_task_ids[kind * warp_size + j];
                *chaselev_queue_slot(
                    kind, warp_id_global, idx_to_push) = val;
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
                printf("push_task_id: %d to %d (kind %d) in lane %d of warp %d of block %d\n", val, idx_to_push, kind, lane, get_warp_id_in_block(), blockIdx.x);
#endif
            }
            if (lane == 0)
                ctx->queue_tails[kind] += staged_n;
        }
        __syncwarp();
        __threadfence();
        if (lane == 0) {
            store_L2(&q->bottom, ctx->queue_tails[kind]);
        }
    }
    if (lane == 0) {
        for (int kind = 0; kind < d_launch_config.num_queues; ++kind) {
            ctx->generated_task_counts[kind] = 0;
        }
    }
}

// push_initial_task: Device function to push initial task
// This function is called from compiler-generated kernel code
__device__ __forceinline__ void push_initial_task(
    void (*func)(void*, int, TaskContext*),
    int initial_queue_idx
) {
    int warp_id_global = get_warp_id_global();
    int new_tid = 0;

    TaskHeader* initial_hdr = &d_task_headers[new_tid];
    initial_hdr->func = func;
    initial_hdr->queue_idx = initial_queue_idx;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    initial_hdr->state = 0;
    initial_hdr->parent_tid = 0;
    initial_hdr->parent_generation = 0;
    initial_hdr->waiting_child_count = 0;
#endif

    // Task data is copied from the compiler-generated code (out of this function)

    *chaselev_queue_slot(
        initial_queue_idx, warp_id_global, 0) = new_tid;
    __threadfence();
    // atomicExch(&d_active_warp_count, 1);
}

__device__ __forceinline__ void initialize_loop(
    int warp_id_in_block,
    int warp_id_global,
    int lane,
    const shared_layout& layout,
    TaskContext*& task_context
#ifdef GTAP_ENABLE_PROFILING
    , int*& working_time_idx
#endif
) {
    task_context =
        reinterpret_cast<TaskContext*>(dynamic_shared) + warp_id_in_block;

#ifdef GTAP_ENABLE_PROFILING
    working_time_idx = reinterpret_cast<int*>(
        dynamic_shared + layout.working_time_idx);
    if (lane == 0) {
        working_time_idx[warp_id_in_block] = 0;
    }
#endif

    if (lane == 0) {
        int* queue_tails = reinterpret_cast<int*>(
            dynamic_shared + layout.queue_tails) +
            warp_id_in_block * d_launch_config.num_queues;
        task_context->generated_task_counts =
            reinterpret_cast<int*>(dynamic_shared + layout.generated_task_counts) +
            warp_id_in_block * d_launch_config.num_queues;
        task_context->queue_tails = queue_tails;
        task_context->staged_task_ids =
            reinterpret_cast<int*>(dynamic_shared + layout.staged_task_ids) +
            warp_id_in_block * d_launch_config.num_queues * warp_size;
        task_context->queue_idx = 0;
        task_context->id_list_free_pos_stale = d_launch_config.tasks_per_scheduling_unit;
        for (int k = 0; k < d_launch_config.num_queues; ++k) {
            task_context->generated_task_counts[k] = 0;
            queue_tails[k] = 0;
        }
        if (warp_id_global == 0) {
            task_context->id_list_alloc_pos = 1;
            // Chase-Lev: set bottom = 1 (initial task at position 0)
            WarpTaskQueueMetadata* q = warp_queue_metadata(0, 0);
            q->bottom = 1;
            queue_tails[0] = 1;
        } else {
            task_context->id_list_alloc_pos = 0;
        }
    }
    __syncwarp();
}

#ifdef GTAP_ENABLE_PROFILING
__device__ __forceinline__ void record_execution_start(
    int warp_id_in_block,
    int warp_id_global,
    int lane,
    int execute_task_count,
    int* working_time_idx
) {
    if (lane == 0) {
        if (working_time_idx[warp_id_in_block] + 1 <
            profile_timestamp_capacity()) {
            const int profile_idx =
                warp_id_global * profile_timestamp_capacity() +
                working_time_idx[warp_id_in_block];
            working_time[profile_idx] = get_global_time();
            tasks_processed_count[profile_idx] = execute_task_count;
            working_time_idx[warp_id_in_block]++;
        } else {
            atomicAdd(&profile_dropped_events[warp_id_global], 1ULL);
        }
    }
}

__device__ __forceinline__ void record_execution_end(
    int warp_id_in_block,
    int warp_id_global,
    int lane,
    int execute_task_count,
    int* working_time_idx
) {
    if (lane == 0) {
        if (working_time_idx[warp_id_in_block] < profile_timestamp_capacity()) {
            const int profile_idx =
                warp_id_global * profile_timestamp_capacity() +
                working_time_idx[warp_id_in_block];
            working_time[profile_idx] = get_global_time();
            tasks_processed_count[profile_idx] = execute_task_count;
            working_time_idx[warp_id_in_block]++;
        }
    }
    __syncwarp();
}
#endif

template<TerminationMode M>
__device__ __forceinline__ bool mark_idle_and_check_termination(
    int warp_id_global,
    int lane,
    bool* prev_get_task
) {
    if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
        if (lane == 0) {
            if (*prev_get_task) {
                int active_warp_count = atomicSub(&d_active_warp_count, 1) - 1;
                if (active_warp_count == 0) {
                    bool all_tasks_finished = 1;
                    for (int k = 0; k < d_launch_config.num_queues; ++k) {
                        // Chase-Lev: check if queue is empty (top >= bottom)
                        WarpTaskQueueMetadata* q = warp_queue_metadata(k, warp_id_global);
                        if (q->top < q->bottom) {
                            all_tasks_finished = 0;
                            break;
                        }
                    }
                    atomicExch(&d_all_tasks_finished, all_tasks_finished);
                }
            }
        }
        __syncwarp();
    }
    *prev_get_task = false;
    bool terminate = false;
    if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
        if (lane == 0) terminate = (load_L2(&d_all_tasks_finished) != 0);
    } else {
        if (lane == 0) terminate = (load_L2(&d_root_task_finished) != 0);
    }
    return __shfl_sync(0xFFFFFFFFu, terminate, 0);
}

#ifndef GTAP_ASSUME_NO_TASKWAIT
// Copy task header to TaskContext for reuse in task function (using L2 load)
__device__ __forceinline__ void copy_task_header(
    int lane,
    int execute_task_id,
    TaskContext* task_context
) {
    TaskHeader* src_hdr = &d_task_headers[execute_task_id];
    task_context->task_parent_tids[lane] = load_L2(&src_hdr->parent_tid);
    task_context->task_generations[lane] =
        load_L2(reinterpret_cast<unsigned int*>(&src_hdr->generation));
}
#endif

template<TerminationMode M>
__device__ __forceinline__ void execute_task_loop() {
    const int warp_id_in_block = get_warp_id_in_block();
    const int warp_id_global = get_warp_id_global();
    const int lane = get_lane_id();

    int execute_task_id = 0;
    int execute_task_count = 0;
    bool prev_get_task = (warp_id_global == 0);

    const shared_layout layout = make_shared_layout(
        d_launch_config.warps_per_block, d_launch_config.num_queues,
        include_queue_tails);
    TaskContext* task_context;
#ifdef GTAP_ENABLE_PROFILING
    int* working_time_idx;
#endif
    initialize_loop(
        warp_id_in_block, warp_id_global, lane, layout,
        task_context
#ifdef GTAP_ENABLE_PROFILING
        , working_time_idx
#endif
    );

    while (true) {
        fill_execution_batch<M>(
            warp_id_in_block, warp_id_global, lane,
            &execute_task_id, &execute_task_count, prev_get_task,
            layout, task_context);

        if (execute_task_count == 0) {
            if (mark_idle_and_check_termination<M>(
                    warp_id_global, lane, &prev_get_task))
                break;
            continue;
        } else {
            prev_get_task = true;
            if (lane == 0) {
                for (int k = 0; k < d_launch_config.num_queues; ++k) {
                    task_context->queue_tails[k] =
                        load_L2(&warp_queue_metadata(k, warp_id_global)->bottom);
                }
            }
            __syncwarp();
        }

        if (lane < execute_task_count) {
            void* task_data = get_task_data(execute_task_id);
            prefetch_global_L2(task_data);
#ifndef GTAP_ASSUME_NO_TASKWAIT
            copy_task_header(lane, execute_task_id, task_context);
#endif
            // __syncwarp(active_mask);

#ifdef GTAP_ENABLE_PROFILING
            record_execution_start(
                warp_id_in_block, warp_id_global, lane,
                execute_task_count, working_time_idx);
#endif
            void* func_ptr = load_L2(reinterpret_cast<void**>(&d_task_headers[execute_task_id].func));
            void (*task_func)(void*, int, TaskContext*) = reinterpret_cast<void (*)(void*, int, TaskContext*)>(func_ptr);
            task_func(task_data, execute_task_id, task_context);
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
            printf("executed_task_id: %d in lane %d of warp %d of block %d\n", execute_task_id, lane, warp_id_in_block, blockIdx.x);
#endif
        }
        __syncwarp();
        __threadfence();
#ifdef GTAP_ENABLE_PROFILING
        record_execution_end(
            warp_id_in_block, warp_id_global, lane,
            execute_task_count, working_time_idx);
#endif

        push_batch<M>(
            task_context, &execute_task_id,
            &execute_task_count
        );
    }
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
    if (lane == 0) printf("execute_task_loop: end (warp_id_global = %d)\n", warp_id_global);
#endif
}

}  // namespace gtap::detail::thread

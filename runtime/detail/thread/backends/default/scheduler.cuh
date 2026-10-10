#pragma once

#include <cuda_runtime.h>

#include "../../../common/cuda_primitives.cuh"
#include "../../../common/runtime_error.cuh"
#include "../../../common/termination.cuh"
#include "../../../common/victim_select.cuh"
#include "../../../common/warp_index.cuh"

#include "../../profile_buffer.cuh"
#include "../../queue_select.cuh"
#include "../../shared_layout.cuh"
#include "../../task_pool.cuh"
#include "../../task_types.cuh"
#include "../../termination.cuh"
#include "queue_storage.cuh"

namespace gtap::detail::thread {
using namespace gtap::detail;

// Whether make_shared_layout reserves per-queue tails. gtap_initialize and the execute loop both pass this.
inline constexpr bool include_queue_tails = true;

extern __shared__ unsigned char dynamic_shared[];

__device__ __forceinline__ void reserve_unpublished_task_id(TaskContext* ctx, int queue_idx, int task_id) {
    int gen_idx = atomicAdd(&ctx->generated_task_counts[queue_idx], 1);
    if (gen_idx < warp_size) {
        ctx->staged_task_ids[queue_idx * warp_size + gen_idx] = task_id;
        return;
    }

    WarpTaskQueueMetadata* q = warp_queue_metadata_ptr(queue_idx, get_warp_id_global());
    int old_tail = atomicAdd(&ctx->queue_tails[queue_idx], 1);
    int head = load_L2(&q->head);
    const int queue_capacity = d_launch_config.queue_capacity;
    if (old_tail + 1 - head > queue_capacity - GTAP_DETAIL_QUEUE_MARGIN) {
        GTAP_DETAIL_RECORD_QUEUE_OVERFLOW(
            task_id, queue_idx, old_tail + 1 - head, queue_capacity - GTAP_DETAIL_QUEUE_MARGIN);
    }
    *warp_queue_slot_ptr(
        queue_idx, get_warp_id_global(), old_tail % queue_capacity) = task_id;
}

__device__ __forceinline__ int pop_batch(int* execute_task_id, int max_count_to_pop, int* tail, int queue_idx) {
    int lane = get_lane_id();
    WarpTaskQueueMetadata* myQueue = warp_queue_metadata_ptr(queue_idx, get_warp_id_global());
    int pop_count = 0;
    if (lane == 0) {
        while (true) {
            int old_queue_count = load_L2(&myQueue->count);
            if (old_queue_count <= 0) break;
            int claim = min(max_count_to_pop, old_queue_count);
            if (atomicCAS(&myQueue->count, old_queue_count, old_queue_count - claim) == old_queue_count) {
                pop_count = claim;
                *tail -= claim;
                break;
            }
        }
    }
    pop_count = __shfl_sync(0xFFFFFFFFu, pop_count, 0);
    if (lane >= warp_size - max_count_to_pop && lane < warp_size - max_count_to_pop + pop_count) {
        int pop_task_id = load_L2(warp_queue_slot_ptr(
            queue_idx,
            get_warp_id_global(),
            (*tail + (lane - warp_size + max_count_to_pop)) %
                d_launch_config.queue_capacity));
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
        printf("pop_task_id: %d (kind %d) in lane %d of warp %d of block %d\n", pop_task_id, queue_idx, lane, get_warp_id_in_block(), blockIdx.x);
#endif
        *execute_task_id = pop_task_id;
    }
    return pop_count;
}

template<TerminationMode M>
__device__ __forceinline__ int steal_batch(int* execute_task_id, int max_count_to_steal, int queue_idx, bool prev_get_task) {
    int warp_id_global = get_warp_id_global();
    int lane = get_lane_id();
    int target_warp_id_global = 0;
    int old_head = 0;
    int steal_count = 0;
    WarpTaskQueueMetadata* targetWq = nullptr;
    if (lane == 0) {
        unsigned lock_backoff_ns = 32;
        while (true) {
            target_warp_id_global = get_random_warp_id_global(warp_id_global);
            targetWq = warp_queue_metadata_ptr(queue_idx, target_warp_id_global);
            if (atomicCAS(&targetWq->lock, 0, 1) == 0) break;
            __nanosleep(lock_backoff_ns);
            if (lock_backoff_ns < (1u << 12)) {
                lock_backoff_ns <<= 1u;
            }
        }
        while (true) {
            int old_queue_count = load_L2(&targetWq->count);
            if (old_queue_count <= 0) break;
            int claim = min(max_count_to_steal, old_queue_count);
            if (atomicCAS(&targetWq->count, old_queue_count, old_queue_count - claim) == old_queue_count) {
                if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
                    if (!prev_get_task) atomicAdd(&d_active_warp_count, 1);
                }
                steal_count = claim;
                old_head = load_L2(&targetWq->head);
                break;
            }
        }
    }
    steal_count = __shfl_sync(0xFFFFFFFFu, steal_count, 0);
    if (steal_count == 0) {
        if (lane == 0) atomicExch(&targetWq->lock, 0);
        return 0;
    }
    target_warp_id_global = __shfl_sync(0xFFFFFFFFu, target_warp_id_global, 0);
    old_head = __shfl_sync(0xFFFFFFFFu, old_head, 0);
    if (lane >= warp_size - max_count_to_steal && lane < warp_size - max_count_to_steal + steal_count) {
        targetWq = warp_queue_metadata_ptr(queue_idx, target_warp_id_global);
        int steal_task_id = load_L2(warp_queue_slot_ptr(
            queue_idx,
            target_warp_id_global,
            (old_head + (lane - warp_size + max_count_to_steal)) %
                d_launch_config.queue_capacity));
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
        printf("steal_task_id: %d (kind %d) in lane %d of warp %d of block %d\n", steal_task_id, queue_idx, lane, get_warp_id_in_block(), blockIdx.x);
#endif
        *execute_task_id = steal_task_id;
    }
    __syncwarp();
    if (lane == 0) {
        targetWq->head = old_head + steal_count;
        __threadfence();
        atomicExch(&targetWq->lock, 0);
    }
    return steal_count;
}

// Fill empty lanes from one queue, then steal if the batch is still short of a warp.
template<TerminationMode M>
__device__ __forceinline__ void fill_batch_from_queue(
    int* execute_task_id,
    int* execute_task_count,
    int* queue_tail,
    int queue_idx,
    bool prev_get_task
) {
    if (*execute_task_count >= warp_size) return;
    if (prev_get_task) {
        int remaining = warp_size - *execute_task_count;
        *execute_task_count += pop_batch(
            execute_task_id, remaining, queue_tail, queue_idx);
    }
    if (*execute_task_count < warp_size) {
        int remaining = warp_size - *execute_task_count;
        *execute_task_count += steal_batch<M>(
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
    TaskContext* task_context,
    int* queue_tails
) {
    if (d_launch_config.num_queues == 1) {
        // Single-queue fast path: skip DAQ count collection and selection.
        fill_batch_from_queue<M>(
            execute_task_id, execute_task_count, &queue_tails[0], 0,
            prev_get_task);
    } else if (*execute_task_count < warp_size) {
        // Multi-queue DAQ path.
        if (*execute_task_count == 0) {
            int* queue_lengths = reinterpret_cast<int*>(
                dynamic_shared + layout.queue_lengths) +
                warp_id_in_block * d_launch_config.num_queues;
            if (lane == 0) {
                for (int k = 0; k < d_launch_config.num_queues; ++k) {
                    queue_lengths[k] = load_L2(
                        &warp_queue_metadata_ptr(k, warp_id_global)->count);
                }
            }
            for (int attempt = 0; attempt < d_launch_config.num_queues; ++attempt) {
                int queue_idx;
                if (lane == 0) {
                    queue_idx = select_next_fullest_queue_idx(
                        queue_lengths, d_launch_config.num_queues);
                    task_context->queue_idx = queue_idx;
                }
                queue_idx = __shfl_sync(
                    0xFFFFFFFFu,
                    task_context->queue_idx,
                    0);
                fill_batch_from_queue<M>(
                    execute_task_id, execute_task_count,
                    &queue_tails[queue_idx], queue_idx, prev_get_task);
                if (*execute_task_count != 0) break;
            }
        } else {
            int queue_idx = __shfl_sync(
                0xFFFFFFFFu,
                task_context->queue_idx,
                0);
            fill_batch_from_queue<M>(
                execute_task_id, execute_task_count,
                &queue_tails[queue_idx], queue_idx, prev_get_task);
        }
    }
}

__device__ __forceinline__ void push_batch_single_queue(
    TaskContext* ctx,
    int* execute_task_id,
    int* execute_task_count
) {
    int warp_id_global = get_warp_id_global();
    int lane = get_lane_id();

    int count = ctx->generated_task_counts[0];
    if (count == 0) {
        *execute_task_count = 0;
        return;
    }
    *execute_task_count = min(count, warp_size);
    if (lane < *execute_task_count) {
        *execute_task_id = ctx->staged_task_ids[lane];
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
        printf("push_task_id: %d (kind %d) in lane %d of warp %d of block %d\n", *execute_task_id, 0, lane, get_warp_id_in_block(), blockIdx.x);
#endif
    }
    __syncwarp();
    if (lane == 0) {
        if (count > warp_size) {
            atomicAdd(
                &warp_queue_metadata_ptr(0, warp_id_global)->count,
                count - warp_size);
        }
        ctx->generated_task_counts[0] = 0;
    }
}

__device__ __forceinline__ void push_batch_multi_queue(
    TaskContext* ctx,
    int* execute_task_id,
    int* execute_task_count,
    int* queue_tails
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
        *execute_task_id = ctx->staged_task_ids[k_max * warp_size + lane];
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
        printf("push_task_id: %d (kind %d) in lane %d of warp %d of block %d\n", *execute_task_id, k_max, lane, get_warp_id_in_block(), blockIdx.x);
#endif
    }
    __syncwarp();

    for (int kind = 0; kind < d_launch_config.num_queues; ++kind) {
        int push_cnt = ctx->generated_task_counts[kind];
        if (kind == k_max) {
            push_cnt -= *execute_task_count;
        }
        if (push_cnt <= 0) continue;

        WarpTaskQueueMetadata* q = warp_queue_metadata_ptr(kind, warp_id_global);
        int total = ctx->generated_task_counts[kind];
        int staged_n = min(total, warp_size);
        if (kind != k_max) {
            for (int j = lane; j < staged_n; j += warp_size) {
                *warp_queue_slot_ptr(
                    kind,
                    warp_id_global,
                    (queue_tails[kind] + j) %
                        d_launch_config.queue_capacity) =
                    ctx->staged_task_ids[kind * warp_size + j];
            }
            if (lane == 0) {
                queue_tails[kind] += staged_n;
            }
            __syncwarp();
        }
        if (lane == 0) {
            atomicAdd(&q->count, push_cnt);
        }
    }
    if (lane == 0) {
        for (int kind = 0; kind < d_launch_config.num_queues; ++kind) {
            ctx->generated_task_counts[kind] = 0;
        }
    }
}

__device__ __forceinline__ void push_batch(
    TaskContext* ctx,
    int* execute_task_id,
    int* execute_task_count,
    int* queue_tails
) {
    if (d_launch_config.num_queues == 1) {
        push_batch_single_queue(ctx, execute_task_id, execute_task_count);
        return;
    }
    push_batch_multi_queue(ctx, execute_task_id, execute_task_count, queue_tails);
}

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

    *warp_queue_slot_ptr(initial_queue_idx, warp_id_global, 0) = new_tid;
    __threadfence();
    // atomicExch(&d_active_warp_count, 1);
}

__device__ __forceinline__ void initialize_loop(
    int warp_id_in_block,
    int warp_id_global,
    int lane,
    const shared_layout& layout,
    TaskContext*& task_context,
    int*& queue_tails
#ifdef GTAP_ENABLE_PROFILING
    , int*& working_time_idx
#endif
) {
    task_context =
        reinterpret_cast<TaskContext*>(dynamic_shared) + warp_id_in_block;
    queue_tails = reinterpret_cast<int*>(
        dynamic_shared + layout.queue_tails) +
        warp_id_in_block * d_launch_config.num_queues;

#ifdef GTAP_ENABLE_PROFILING
    working_time_idx = reinterpret_cast<int*>(
        dynamic_shared + layout.working_time_idx);
    if (lane == 0) {
        working_time_idx[warp_id_in_block] = 0;
    }
#endif

    if (lane == 0) {
        task_context->queue_idx = 0;
        task_context->generated_task_counts =
            reinterpret_cast<int*>(dynamic_shared + layout.generated_task_counts) +
            warp_id_in_block * d_launch_config.num_queues;
        task_context->queue_tails = queue_tails;
        task_context->staged_task_ids =
            reinterpret_cast<int*>(dynamic_shared + layout.staged_task_ids) +
            warp_id_in_block * d_launch_config.num_queues * warp_size;
        task_context->id_list_free_pos_stale =
            d_launch_config.tasks_per_scheduling_unit;
        for (int k = 0; k < d_launch_config.num_queues; ++k) {
            task_context->generated_task_counts[k] = 0;
            queue_tails[k] = 0;
        }
        if (warp_id_global == 0) {
            task_context->id_list_alloc_pos = 1;
            WarpTaskQueueMetadata* q = warp_queue_metadata_ptr(0, 0);
            store_L2(&q->count, 1);
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
    bool* prev_get_task,
    int* queue_tails
) {
    if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
        if (lane == 0) {
            if (*prev_get_task) {
                int active_warp_count = atomicSub(&d_active_warp_count, 1) - 1;
                if (active_warp_count == 0) {
                    bool all_tasks_finished = 1;
                    for (int k = 0; k < d_launch_config.num_queues; ++k) {
                        if (warp_queue_metadata_ptr(k, warp_id_global)->head < queue_tails[k]) {
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
    int* queue_tails;
#ifdef GTAP_ENABLE_PROFILING
    int* working_time_idx;
#endif
    initialize_loop(
        warp_id_in_block, warp_id_global, lane, layout,
        task_context, queue_tails
#ifdef GTAP_ENABLE_PROFILING
        , working_time_idx
#endif
    );

    while (true) {
        fill_execution_batch<M>(
            warp_id_in_block, warp_id_global, lane,
            &execute_task_id, &execute_task_count, prev_get_task,
            layout, task_context, queue_tails);

        if (execute_task_count == 0) {
            if (mark_idle_and_check_termination<M>(
                    warp_id_global, lane, &prev_get_task, queue_tails))
                break;
            continue;
        } else {
            prev_get_task = true;
            __syncwarp();
        }

        if (lane < execute_task_count) {
            void* task_data = get_task_data(execute_task_id);
            prefetch_global_L2(task_data);
#ifndef GTAP_ASSUME_NO_TASKWAIT
            copy_task_header(lane, execute_task_id, task_context);
#endif

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
        push_batch(
            task_context, &execute_task_id,
            &execute_task_count, queue_tails
        );
    }
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
    if (lane == 0) printf("execute_task_loop: end (warp_id_global = %d)\n", warp_id_global);
#endif
}

}  // namespace gtap::detail::thread

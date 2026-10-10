#pragma once

#include <cuda_runtime.h>

#include "../../../../common/cuda_primitives.cuh"
#include "../../../../common/runtime_error.cuh"
#include "../../../../common/termination.cuh"
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
inline constexpr bool include_queue_tails = false;

extern __shared__ unsigned char dynamic_shared[];

// ============================================================================
// Global Queue Operations (no steal needed - all warps pop from global queue)
// ============================================================================

// Keep the common case in warp-local shared memory.  Overflow remains in the
// private global staging buffer and is not published to the global queue until
// push_global_queue(), after the spawning lanes have finished task-data setup.
__device__ __forceinline__ void reserve_unpublished_task_id(
    TaskContext* ctx, int queue_idx, int task_id
) {
    int idx = atomicAdd(
        &ctx->generated_task_counts[queue_idx], 1);
    if (idx < warp_size) {
        ctx->staged_task_ids[queue_idx * warp_size + idx] = task_id;
        return;
    }
    set_task_id_generated(
        get_warp_id_global(), queue_idx, idx - warp_size, task_id);
}

__device__ __forceinline__ int get_unpublished_task_id(
    TaskContext* ctx, int queue_idx, int idx
) {
    if (idx < warp_size)
        return ctx->staged_task_ids[queue_idx * warp_size + idx];
    return get_task_id_generated(
        get_warp_id_global(), queue_idx, idx - warp_size);
}

// Pop from global queue - returns number of tasks popped (up to max_count)
// Each lane gets a different task if available
template<TerminationMode M>
__device__ __forceinline__ int pop_global_queue(int* execute_task_id, int max_count, int queue_idx, bool prev_get_task) {
    int lane = get_lane_id();
    int count = 0;
    int base_head = 0;

    if (lane == 0) {
        // Try to claim slots from global queue
        while (true) {
            int old_head = load_L2(&d_queue_head[queue_idx]);
            int tail = load_L2(&d_queue_tail[queue_idx]);
            int available = max(0, tail - old_head);
            count = min(max_count, available);

            if (count == 0) break;

            // CAS to claim slots
            int new_head = old_head + count;
            if (atomicCAS(&d_queue_head[queue_idx], old_head, new_head) == old_head) {
                base_head = old_head;
                // Increment the active warp count if this warp was previously idle
                if (M == TERMINATE_ON_ALL_TASKS_FINISH && !prev_get_task) {
                    atomicAdd(&d_active_warp_count, 1);
                }
                break;
            }
            // CAS failed, retry
        }
    }

    // Broadcast results to all lanes
    count = __shfl_sync(0xFFFFFFFFu, count, 0);
    base_head = __shfl_sync(0xFFFFFFFFu, base_head, 0);

    // Write into the empty suffix. Existing ids stay in lanes below warp_size - max_count.
    if (lane >= warp_size - max_count &&
        lane < warp_size - max_count + count) {
        int idx = (base_head + (lane - warp_size + max_count)) %
            (d_launch_config.total_scheduling_units * d_launch_config.queue_capacity);
        int tid = load_L2(global_queue_slot(queue_idx, idx));
        *execute_task_id = tid;
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
        printf("pop_global: tid=%d (queue %d) in lane %d\n", tid, queue_idx, lane);
#endif
    }

    return count;
}

// Pop from one global queue into the lanes that still have no task.
template<TerminationMode M>
__device__ __forceinline__ void fill_execution_batch(
    int* execute_task_id,
    int* execute_task_count,
    int queue_idx,
    bool prev_get_task
) {
    if (*execute_task_count >= warp_size) return;
    int remaining = warp_size - *execute_task_count;
    *execute_task_count += pop_global_queue<M>(
        execute_task_id, remaining, queue_idx, prev_get_task);
}

// Push to global queue
template<TerminationMode M>
__device__ __forceinline__ void push_global_queue(
    TaskContext* ctx,
    int* execute_task_id,
    int* execute_task_count
) {
    int lane = get_lane_id();
    // Calculate total generated tasks
    int all_generated_count = 0;
    int k_max = 0;
    int max_gen = -1;

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

    // Determine tasks to execute immediately vs push to queue
    *execute_task_count = max(0, min(warp_size, max_gen));
    if (lane < *execute_task_count) {
        *execute_task_id = get_unpublished_task_id(ctx, k_max, lane);
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
        printf("execute_immediately: tid=%d (queue %d) in lane %d\n", *execute_task_id, k_max, lane);
#endif
    }

    // Push remaining tasks to global queue
    for (int kind = 0; kind < d_launch_config.num_queues; ++kind) {
        int first_idx_to_push = (kind == k_max) ? *execute_task_count : 0;
        int push_cnt = ctx->generated_task_counts[kind] - first_idx_to_push;
        if (push_cnt <= 0) continue;

        // Reserve slots in global queue (allocate exclusive range)
        int base_pos = 0;
        if (lane == 0) {
            base_pos = atomicAdd(&d_queue_alloc[kind], push_cnt);
            // Overflow check
            int head_val = load_L2(&d_queue_head[kind]);
            if (base_pos + push_cnt - head_val > (d_launch_config.total_scheduling_units * d_launch_config.queue_capacity) - GTAP_DETAIL_QUEUE_MARGIN) {
            GTAP_DETAIL_RECORD_QUEUE_OVERFLOW(
                -1, kind, base_pos + push_cnt - head_val,
                (d_launch_config.total_scheduling_units * d_launch_config.queue_capacity) - GTAP_DETAIL_QUEUE_MARGIN);
            }
        }
        base_pos = __shfl_sync(0xFFFFFFFFu, base_pos, 0);

        // Write tasks to reserved slots
        for (int j = lane; j < push_cnt; j += warp_size) {
            int tid = get_unpublished_task_id(
                ctx, kind, first_idx_to_push + j);
            int pos = (base_pos + j) % (d_launch_config.total_scheduling_units * d_launch_config.queue_capacity);
            store_L2(global_queue_slot(kind, pos), tid);
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
            printf("push_global: tid=%d to queue %d, pos %d in lane %d\n", tid, kind, pos, lane);
#endif
        }
        __threadfence();
        __syncwarp();

        // Wait for prior commits and update tail (ensures in-order visibility)
        if (lane == 0) {
            while (load_L2(&d_queue_tail[kind]) != base_pos) {
                // spin - wait for prior pushers to commit
            }
            atomicAdd(&d_queue_tail[kind], push_cnt);
        }
    }
    if (lane == 0) {
        for (int kind = 0; kind < d_launch_config.num_queues; ++kind) {
            ctx->generated_task_counts[kind] = 0;
        }
    }
    __syncwarp();
}

// Push initial task to global queue
__device__ __forceinline__ void push_initial_task(
    void (*func)(void*, int, TaskContext*),
    int initial_queue_idx
) {
    int warp_id_global = get_warp_id_global();
    int lane = get_lane_id();
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

    // Push to global queue (only warp 0, lane 0)
    if (warp_id_global == 0 && lane == 0) {
        store_L2(global_queue_slot(initial_queue_idx, 0), new_tid);
        __threadfence();
        store_L2(&d_queue_head[initial_queue_idx], 0);
        store_L2(&d_queue_alloc[initial_queue_idx], 1);
        store_L2(&d_queue_tail[initial_queue_idx], 1);
        __threadfence();
    }
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
        task_context->generated_task_counts =
            reinterpret_cast<int*>(dynamic_shared + layout.generated_task_counts) +
            warp_id_in_block * d_launch_config.num_queues;
        task_context->staged_task_ids =
            reinterpret_cast<int*>(dynamic_shared + layout.staged_task_ids) +
            warp_id_in_block * d_launch_config.num_queues * warp_size;
        task_context->queue_idx = 0;
        task_context->id_list_free_pos_stale = d_launch_config.tasks_per_scheduling_unit;
        for (int k = 0; k < d_launch_config.num_queues; ++k) {
            task_context->generated_task_counts[k] = 0;
        }
        if (warp_id_global == 0) {
            task_context->id_list_alloc_pos = 1;
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
    int lane,
    bool* prev_get_task
) {
    if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
        if (lane == 0) {
            if (*prev_get_task) {
                int active_warp_count = atomicSub(&d_active_warp_count, 1) - 1;
                if (active_warp_count == 0) {
                    // Check if all queues are empty
                    bool all_tasks_finished = 1;
                    for (int k = 0; k < d_launch_config.num_queues; ++k) {
                        int head = load_L2(&d_queue_head[k]);
                        int tail = load_L2(&d_queue_tail[k]);
                        if (head < tail) {
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
        // Touch the global queue only when execute_task_count == 0. Popping it
        // hits the device-wide head, so a non-empty batch keeps its local tasks
        // and leaves the queue alone.
        if (execute_task_count == 0) {
            if (d_launch_config.num_queues > 1) {
                int* queue_lengths = reinterpret_cast<int*>(
                    dynamic_shared + layout.queue_lengths) +
                    warp_id_in_block * d_launch_config.num_queues;
                if (lane == 0) {
                    for (int k = 0; k < d_launch_config.num_queues; ++k) {
                        int head = load_L2(&d_queue_head[k]);
                        int tail = load_L2(&d_queue_tail[k]);
                        queue_lengths[k] = max(0, tail - head);
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
                    fill_execution_batch<M>(
                        &execute_task_id, &execute_task_count, queue_idx,
                        prev_get_task);
                    if (execute_task_count != 0) break;
                }
            } else {
                fill_execution_batch<M>(
                    &execute_task_id, &execute_task_count, 0, prev_get_task);
            }
        }

        if (execute_task_count == 0) {
            if (mark_idle_and_check_termination<M>(lane, &prev_get_task))
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

        push_global_queue<M>(
            task_context, &execute_task_id,
            &execute_task_count
        );
    }
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
    if (lane == 0) printf("execute_task_loop: end (warp_id_global = %d)\n", warp_id_global);
#endif
}

}  // namespace gtap::detail::thread

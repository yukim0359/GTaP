#pragma once

#include <cuda_runtime.h>
#include <climits>
#include "../common/gtap_runtime_common.cuh"
#include "gtap_thread_core.cuh"

#define GTAP_PROFILE_HAS_DROPPED_COUNTER 1

namespace gtap::detail::thread {
using namespace gtap::detail;

struct WarpTaskQueueMetadata {
    int count;
    int lock;
    int head;
    // tail is placed in shared memory
};

__constant__ WarpTaskQueueMetadata** d_warp_task_queue_metadata;
__constant__ int* d_warp_task_queue_storage;
extern __shared__ unsigned char dynamic_shared[];

__device__ __forceinline__ int* warp_queue_slot(
    int queue_idx, int warp_idx, int slot
) {
    const size_t index =
        (static_cast<size_t>(queue_idx) * d_launch_config.total_workers + warp_idx) * d_launch_config.queue_capacity + slot;
    return &d_warp_task_queue_storage[index];
}

__device__ __forceinline__ void reserve_unpublished_task_id(TaskContext* ctx, int queue_idx, int task_id) {
    int gen_idx = atomicAdd(&ctx->generated_task_counts[queue_idx], 1);
    if (gen_idx < warp_size) {
        ctx->staged_task_ids[queue_idx * warp_size + gen_idx] = task_id;
        return;
    }

    WarpTaskQueueMetadata* q = &d_warp_task_queue_metadata[queue_idx][get_warp_id_global()];
    int old_tail = atomicAdd(&ctx->queue_tails[queue_idx], 1);
    int head = load_L2(&q->head);
    const int queue_capacity = d_launch_config.queue_capacity;
    if (old_tail + 1 - head > queue_capacity - GTAP_DETAIL_QUEUE_MARGIN) {
        GTAP_DETAIL_RECORD_QUEUE_OVERFLOW(
            task_id, queue_idx, old_tail + 1 - head, queue_capacity - GTAP_DETAIL_QUEUE_MARGIN);
    }
    *warp_queue_slot(
        queue_idx, get_warp_id_global(), old_tail % queue_capacity) = task_id;
}

#ifdef GTAP_ENABLE_PROFILING
cudaError_t get_warp_working_time_data(long long* host_working_time) {
    long long* ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, working_time, sizeof(ptr)));
    return cudaMemcpy(
        host_working_time, ptr,
        sizeof(long long) * stored_launch_config().total_workers * profile_capacity(),
        cudaMemcpyDeviceToHost);
}

cudaError_t get_warp_tasks_processed_count_data(int* host_counts) {
    int* ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, tasks_processed_count, sizeof(ptr)));
    return cudaMemcpy(
        host_counts, ptr,
        sizeof(int) * stored_launch_config().total_workers * profile_capacity(),
        cudaMemcpyDeviceToHost);
}

cudaError_t get_warp_profile_dropped_events_data(
    unsigned long long* host_counts
) {
    unsigned long long* ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &ptr, profile_dropped_events, sizeof(ptr)));
    return cudaMemcpy(
        host_counts, ptr,
        sizeof(unsigned long long) *
            stored_launch_config().total_workers,
        cudaMemcpyDeviceToHost);
}

__global__ void get_warp_working_time_counts(int* counts) {
    if (threadIdx.x == 0) {
        int wid = blockIdx.x;
        int count = 0;
        for (int i = 0; i < profile_capacity(); i++) {
            if (working_time[wid * profile_capacity() + i] > 0) count++;
        }
        counts[wid] = count;
    }
}
#endif

// define pop_batch, steal_batch, push_batch
__device__ __forceinline__ int pop_batch(int* execute_task_id, int max_count_to_pop, int* tail, int queue_idx) {
    int lane = get_lane_id();
    WarpTaskQueueMetadata* myQueue = &d_warp_task_queue_metadata[queue_idx][get_warp_id_global()];
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
        int pop_task_id = load_L2(warp_queue_slot(
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
            targetWq = &d_warp_task_queue_metadata[queue_idx][target_warp_id_global];
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
        targetWq = &d_warp_task_queue_metadata[queue_idx][target_warp_id_global];
        int steal_task_id = load_L2(warp_queue_slot(
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

__device__ __forceinline__ void push_batch (
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

        WarpTaskQueueMetadata* q = &d_warp_task_queue_metadata[kind][warp_id_global];
        int total = ctx->generated_task_counts[kind];
        int staged_n = min(total, warp_size);
        if (kind != k_max) {
            for (int j = lane; j < staged_n; j += warp_size) {
                *warp_queue_slot(
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

// Get the current state of a task (reads from TaskHeader)
__device__ __forceinline__ int get_task_state(int tid) {
#ifdef GTAP_ASSUME_NO_TASKWAIT
    (void)tid;
    return 0;
#else
    return load_L2(&d_task_headers[tid].state);
#endif
}

__device__ __forceinline__ bool set_state_for_join(int tid, int child_count, int next_state, int queue_idx_after_join) {
    if (queue_idx_after_join >= d_launch_config.num_queues) {
        GTAP_DETAIL_RECORD_INVALID_QUEUE_IDX_AFTER_JOIN(
            tid, queue_idx_after_join, d_launch_config.num_queues);
    }
#ifndef GTAP_ASSUME_NO_TASKWAIT
    TaskHeader* hdr = &d_task_headers[tid];
    hdr->queue_idx = queue_idx_after_join;
    hdr->state = next_state;
    hdr->waiting_child_count = child_count;
#else
    d_task_headers[tid].queue_idx = queue_idx_after_join;
    (void)next_state;
#endif
    return child_count != 0;
}

#ifndef GTAP_ASSUME_NO_TASKWAIT
__device__ __forceinline__ int notify_parent(int parentId, TaskContext* ctx) {
    TaskHeader* parent_hdr = &d_task_headers[parentId];
    __threadfence();
    int rem = atomicSub(&parent_hdr->waiting_child_count, 1);
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
    int lane = get_lane_id();
    printf("notify_parent: %d (remaining child count: %d) in lane %d of warp %d of block %d\n", parentId, rem, lane, get_warp_id_in_block(), blockIdx.x);
#endif
    if (rem == 1) {
        int parent_queue_idx = load_L2(&parent_hdr->queue_idx);
        reserve_unpublished_task_id(ctx, parent_queue_idx, parentId);
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
    release_task_id_to_warp_pool(tid);
#else
    int lane = get_lane_id();
    int parent_tid = ctx->task_parent_tids[lane];
    uint32_t cached_generations = ctx->task_generations[lane];
    uint16_t generation = static_cast<uint16_t>(cached_generations);
    uint16_t parent_generation =
        static_cast<uint16_t>(cached_generations >> 16);
    d_task_headers[tid].generation = generation + 1;

    if (tid != 0 &&
        load_L2(&d_task_headers[parent_tid].generation) ==
            parent_generation) {
        notify_parent(parent_tid, ctx);
    }
    release_task_id_to_warp_pool(tid);
#endif

    if (tid == 0) {
        store_L2(&d_first_task_finished, 1);
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
        int lane = get_lane_id();
        printf("first task finished in lane %d of warp %d of block %d\n", lane, get_warp_id_in_block(), blockIdx.x);
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
    if (child_queue_idx >= d_launch_config.num_queues) {
        GTAP_DETAIL_RECORD_INVALID_QUEUE_IDX(
            self_tid, child_queue_idx, d_launch_config.num_queues);
    }
    int warp_id_global = get_warp_id_global();
    int new_tid = get_task_id_from_warp_pool(
        &d_task_id_list_free_positions[warp_id_global],
        &ctx->id_list_alloc_pos,
        &ctx->id_list_free_pos_stale);
    TaskHeader* new_hdr = &d_task_headers[new_tid];
    new_hdr->func = func;
    new_hdr->queue_idx = child_queue_idx;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    int lane = get_lane_id();
    new_hdr->parent_tid = self_tid;
    new_hdr->parent_generation =
        static_cast<uint16_t>(ctx->task_generations[lane]);
    new_hdr->state = 0;
    new_hdr->waiting_child_count = 0;
#endif

    reserve_unpublished_task_id(ctx, child_queue_idx, new_tid);

#ifndef GTAP_ASSUME_NO_TASKWAIT
    (*child_count)++;
#else
    (void)child_count;
#endif
    return get_task_data(new_tid);
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

    *warp_queue_slot(initial_queue_idx, warp_id_global, 0) = new_tid;
    __threadfence();
    // atomicExch(&d_active_warp_count, 1);
}

template<TerminationMode M>
__device__ __forceinline__ void execute_task_loop() {
    const int warp_id_in_block = get_warp_id_in_block();
    const int warp_id_global = get_warp_id_global();
    const int lane = get_lane_id();

    int execute_task_id = 0;
    int execute_task_count = 0;
    bool prev_get_task = (warp_id_global == 0);
    bool should_continue = true;

    const shared_layout layout = shared_layout_for(
        d_launch_config.warps_per_block, d_launch_config.num_queues, true);
    TaskContext* task_context =
        reinterpret_cast<TaskContext*>(dynamic_shared) + warp_id_in_block;
    int* queue_tails = reinterpret_cast<int*>(
        dynamic_shared + layout.queue_tails) +
        warp_id_in_block * d_launch_config.num_queues;

#ifdef GTAP_ENABLE_PROFILING
    int* working_time_idx = reinterpret_cast<int*>(
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
            d_launch_config.tasks_per_worker;
        for (int k = 0; k < d_launch_config.num_queues; ++k) {
            task_context->generated_task_counts[k] = 0;
            queue_tails[k] = 0;
        }
        if (warp_id_global == 0) {
            task_context->id_list_alloc_pos = 1;
            WarpTaskQueueMetadata* q = &d_warp_task_queue_metadata[0][0];
            store_L2(&q->count, 1);
            queue_tails[0] = 1;
        } else {
            task_context->id_list_alloc_pos = 0;
        }
    }
    __syncwarp();

    while (should_continue) {
        if (d_launch_config.num_queues == 1) {
            // Single-queue fast path: skip DAQ count collection and selection.
            if (execute_task_count < warp_size) {
                if (prev_get_task) {
                    int remaining = warp_size - execute_task_count;
                    int pop_count = pop_batch(
                        &execute_task_id, remaining, &queue_tails[0], 0
                    );
                    execute_task_count += pop_count;
                }
                if (execute_task_count < warp_size) {
                    int remaining = warp_size - execute_task_count;
                    int steal_count = steal_batch<M>(
                        &execute_task_id, remaining, 0, prev_get_task
                    );
                    execute_task_count += steal_count;
                }
            }
        } else {
            // Multi-queue DAQ path.
            if (execute_task_count < warp_size) {
                if (execute_task_count == 0) {
                    int* queue_lengths = reinterpret_cast<int*>(
                        dynamic_shared + layout.queue_lengths) +
                        warp_id_in_block * d_launch_config.num_queues;
                    if (lane == 0) {
                        for (int k = 0; k < d_launch_config.num_queues; ++k) {
                            queue_lengths[k] = load_L2(
                                &d_warp_task_queue_metadata[k][warp_id_global].count);
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
                        if (prev_get_task &&
                            execute_task_count < warp_size) {
                            int remaining =
                                warp_size - execute_task_count;
                            int pop_count = pop_batch(
                                &execute_task_id, remaining,
                                &queue_tails[queue_idx], queue_idx
                            );
                            execute_task_count += pop_count;
                        }
                        if (execute_task_count < warp_size) {
                            int remaining =
                                warp_size - execute_task_count;
                            int steal_count = steal_batch<M>(
                                &execute_task_id, remaining, queue_idx,
                                prev_get_task
                            );
                            execute_task_count += steal_count;
                        }
                        if (execute_task_count != 0) break;
                    }
                } else {
                    int queue_idx = __shfl_sync(
                        0xFFFFFFFFu,
                        task_context->queue_idx,
                        0
                    );
                    if (prev_get_task) {
                        int remaining = warp_size - execute_task_count;
                        int pop_count = pop_batch(
                            &execute_task_id, remaining,
                            &queue_tails[queue_idx], queue_idx
                        );
                        execute_task_count += pop_count;
                    }
                    if (execute_task_count < warp_size) {
                        int remaining = warp_size - execute_task_count;
                        int steal_count = steal_batch<M>(
                            &execute_task_id, remaining, queue_idx, prev_get_task
                        );
                        execute_task_count += steal_count;
                    }
                }
            }
        }
        if (execute_task_count == 0) {
            if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
                if (lane == 0) {
                    if (prev_get_task) {
                        int active_warp_count = atomicSub(&d_active_warp_count, 1) - 1;
                        if (active_warp_count == 0) {
                            bool all_tasks_finished = 1;
                            for (int k = 0; k < d_launch_config.num_queues; ++k) {
                                if (d_warp_task_queue_metadata[k][warp_id_global].head < queue_tails[k]) {
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
            prev_get_task = false;
            if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
                if (lane == 0) should_continue = (load_L2(&d_all_tasks_finished) == 0);
                should_continue = __shfl_sync(0xFFFFFFFFu, should_continue, 0);
            } else {
                if (lane == 0) should_continue = (load_L2(&d_first_task_finished) == 0);
                should_continue = __shfl_sync(0xFFFFFFFFu, should_continue, 0);
            }
            continue;
        } else {
            prev_get_task = true;
            if (lane == 0) {
                for (int k = 0; k < d_launch_config.num_queues; ++k) {
                    task_context->generated_task_counts[k] = 0;
                }
            }
            __syncwarp();
        }

        if (lane < execute_task_count) {
            prefetch_global_L2(get_task_data(execute_task_id));
            // Copy task header to TaskContext for reuse in task function (using L2 load)
#ifndef GTAP_ASSUME_NO_TASKWAIT
            {
                TaskHeader* src_hdr = &d_task_headers[execute_task_id];
                uint16_t generation = load_L2(&src_hdr->generation);
                uint16_t parent_generation =
                    load_L2(&src_hdr->parent_generation);
                task_context->task_parent_tids[lane] =
                    load_L2(&src_hdr->parent_tid);
                task_context->task_generations[lane] =
                    static_cast<uint32_t>(generation) |
                    (static_cast<uint32_t>(parent_generation) << 16);
            }
#endif

#ifdef GTAP_ENABLE_PROFILING
            if (lane == 0) {
                if (working_time_idx[warp_id_in_block] + 1 <
                    profile_capacity()) {
                    const int profile_idx =
                        warp_id_global * profile_capacity() +
                        working_time_idx[warp_id_in_block];
                    working_time[profile_idx] = get_global_time();
                    tasks_processed_count[profile_idx] = execute_task_count;
                    working_time_idx[warp_id_in_block]++;
                } else {
                    atomicAdd(&profile_dropped_events[warp_id_global], 1ULL);
                }
            }
#endif
            void* task_data = get_task_data(execute_task_id);
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
        if (lane == 0) {
            if (working_time_idx[warp_id_in_block] < profile_capacity()) {
                const int profile_idx =
                    warp_id_global * profile_capacity() +
                    working_time_idx[warp_id_in_block];
                working_time[profile_idx] = get_global_time();
                tasks_processed_count[profile_idx] = execute_task_count;
                working_time_idx[warp_id_in_block]++;
            }
        }
        __syncwarp();
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

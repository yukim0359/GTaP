#pragma once

#include <cuda_runtime.h>
#include <climits>
#include "../../common/runtime.cuh"
#include "../../block/core.cuh"

#define GTAP_PROFILE_HAS_DROPPED_COUNTER 1

// Depth of the per-block unpublished child-task buffer. Override with -D.
#ifndef GTAP_MAX_CHILD_TASKS
#define GTAP_MAX_CHILD_TASKS 32
#endif
static_assert(GTAP_MAX_CHILD_TASKS >= 0, "GTAP_MAX_CHILD_TASKS must be non-negative");

extern const size_t __gtap_auto_entry_result_size;

__constant__ int* d_global_task_queue;
namespace gtap::detail::block {
using namespace gtap::detail;

__device__ unsigned int d_queue_head;     // Global queue head (consumer reads from here)
__device__ unsigned int d_queue_tail;     // Global queue tail (consumer-visible, committed)
__device__ unsigned int d_queue_alloc;    // Write allocation position (producers reserve here)
__constant__ int* d_task_id_generated;

__device__ __forceinline__ int get_task_id_generated(int block_id, int idx) {
    int offset = block_id * GTAP_MAX_CHILD_TASKS + idx;
    return d_task_id_generated[offset];
}

__device__ __forceinline__ void set_task_id_generated(int block_id, int idx, int task_id) {
    if (idx >= GTAP_MAX_CHILD_TASKS) {
        GTAP_DETAIL_RECORD_GENERATED_TASK_ID_BUFFER_OVERFLOW(
            task_id, -1, idx, GTAP_MAX_CHILD_TASKS);
    }
    int offset = block_id * GTAP_MAX_CHILD_TASKS + idx;
    d_task_id_generated[offset] = task_id;
}

#define GTAP_RUNTIME_GRID_SIZE (stored_launch_config().grid_size)
#define GTAP_RUNTIME_TOTAL_TASKS \
    (stored_launch_config().total_workers * \
     stored_launch_config().tasks_per_worker)
#define GTAP_RUNTIME_TASKS_PER_BLOCK \
    (stored_launch_config().tasks_per_worker)

#ifdef GTAP_ENABLE_PROFILING
cudaError_t get_working_time_data(long long* host_working_time) {
    long long* ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, working_time, sizeof(ptr)));
    return cudaMemcpy(host_working_time, ptr, sizeof(long long) *
        stored_launch_config().total_workers * profile_capacity(),
        cudaMemcpyDeviceToHost);
}

cudaError_t get_block_profile_dropped_events_data(
    unsigned long long* host_counts
) {
    unsigned long long* ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &ptr, profile_dropped_events, sizeof(ptr)));
    return cudaMemcpy(
        host_counts, ptr,
        sizeof(unsigned long long) * stored_launch_config().grid_size,
        cudaMemcpyDeviceToHost);
}

cudaError_t get_block_working_time_data(int block_id, long long* host_working_time, int max_samples) {
    long long* ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, working_time, sizeof(ptr)));
    const int count = max_samples < profile_capacity() ? max_samples : profile_capacity();
    return cudaMemcpy(host_working_time,
        ptr + static_cast<size_t>(block_id) * profile_capacity(),
        sizeof(long long) * count, cudaMemcpyDeviceToHost);
}

__global__ void get_block_working_time_counts(int* counts) {
    if (threadIdx.x == 0) {
        // Count actual recorded samples for this block
        int count = 0;
        for (int i = 0; i < profile_capacity(); i++) {
            if (working_time[blockIdx.x * profile_capacity() + i] > 0) {
                count++;
            }
        }
        counts[blockIdx.x] = count;
    }
}
#endif

// ============================================================================
// Global Queue Operations (no steal needed - all workers pop from global queue)
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
            // Increment active worker count if this worker was previously idle
            if (M == TERMINATE_ON_ALL_TASKS_FINISH && !prev_get_task) {
                atomicAdd(&d_active_block_count, 1);
            }
            break;
        }
        // CAS failed, retry
    }

    if (pop_success) {
        int idx = head % (d_launch_config.total_workers * d_launch_config.tasks_per_worker);
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

    int total_count = (ctx->have_task_id_resumable ? 1 : 0) + ctx->task_id_generated_count;

    if (total_count == 0) {
        *have_execute_task = false;
        return;
    }

    // Determine task to execute immediately vs push to queue
    if (threadIdx.x == 0) {
        first_idx_to_push = 0;
        if (ctx->have_task_id_resumable) {
            *execute_task_id = ctx->task_id_resumable;
            *have_execute_task = true;
        } else if (ctx->task_id_generated_count > 0) {
            *execute_task_id = get_task_id_generated(blockIdx.x, 0);
            *have_execute_task = true;
            first_idx_to_push = 1;
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
            printf("execute_immediately: tid=%d in block %d\n", *execute_task_id, blockIdx.x);
#endif
        } else {
            *have_execute_task = false;
        }
        push_cnt = ctx->task_id_generated_count - first_idx_to_push;
    }
    __syncthreads();

    // Push remaining tasks to global queue
    if (push_cnt <= 0) return;

    // Reserve slots in global queue (allocate exclusive range)
    if (threadIdx.x == 0) {
        base_pos = atomicAdd(&d_queue_alloc, (unsigned int)push_cnt);
        // Overflow check (unsigned subtraction handles wrap-around)
        unsigned int head_val = load_L2(&d_queue_head);
        if (base_pos + (unsigned int)push_cnt - head_val > (d_launch_config.total_workers * d_launch_config.tasks_per_worker) - GTAP_DETAIL_QUEUE_MARGIN) {
            GTAP_DETAIL_RECORD_QUEUE_OVERFLOW(
                -1, 0,
                static_cast<int>(base_pos + (unsigned int)push_cnt - head_val),
                (d_launch_config.total_workers * d_launch_config.tasks_per_worker) - GTAP_DETAIL_QUEUE_MARGIN);
        }
    }
    __syncthreads();

    // Write tasks to reserved slots (parallel using block threads)
    for (int j = threadIdx.x; j < push_cnt; j += blockDim.x) {
        int tid = get_task_id_generated(blockIdx.x, first_idx_to_push + j);
        unsigned int pos = (base_pos + (unsigned int)j) % (d_launch_config.total_workers * d_launch_config.tasks_per_worker);
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

__device__ __forceinline__ void set_state_for_join(
    int tid,
    int child_count,
    int next_state,
    int unused_value
) {
    (void)unused_value;
    if (threadIdx.x == 0) {
        TaskHeader* hdr = &d_task_headers[tid];
        hdr->state = next_state;
#ifndef GTAP_ASSUME_NO_TASKWAIT
        hdr->waiting_child_count = child_count;
#endif
    }
}

__device__ __forceinline__ int get_task_state(int tid) {
    return load_L2(&d_task_headers[tid].state);
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
        hdr->state = next_state;
#ifndef GTAP_ASSUME_NO_TASKWAIT
        hdr->waiting_child_count = child_count;
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
    if (threadIdx.x == 0) {
        TaskHeader* cached_hdr = &ctx->cached_task_header;
        int parent_tid = cached_hdr->parent_tid;
        d_task_headers[tid].generation = cached_hdr->generation + 1;

        if (tid != 0 && load_L2(&d_task_headers[parent_tid].generation) == cached_hdr->parent_generation) {
#ifndef GTAP_ASSUME_NO_TASKWAIT
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
            printf("finish_task: %d, parent_tid: %d\n", tid, parent_tid);
#endif
            notify_parent(parent_tid, ctx);
            release_task_id_to_block_pool(tid);
#else
            // NO_TASKWAIT: no need to notify parent or release child IDs
            release_task_id_to_block_pool(tid);
#endif
        } else {
            release_task_id_to_block_pool(tid);
        }
        if (tid == 0) store_L2(&d_first_task_finished, 1);
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
    TaskHeader* cached_hdr = &ctx->cached_task_header;
    new_hdr->func = func;
    new_hdr->state = 0;
    new_hdr->parent_tid = self_tid;
    new_hdr->parent_generation = cached_hdr->generation;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    new_hdr->waiting_child_count = 0;
#endif

    int gen_idx = atomicAdd(&ctx->task_id_generated_count, 1);
    set_task_id_generated(blockIdx.x, gen_idx, new_tid);
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
    initial_hdr->state = 0;
    initial_hdr->parent_tid = 0;
    initial_hdr->parent_generation = 0;
#ifndef GTAP_ASSUME_NO_TASKWAIT
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
        block_ctx.have_task_id_resumable = false;
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
                // Try to pop from global queue
                have_execute_task = pop_global_queue<M>(&execute_task_id, prev_get_task);
            }
        }
        __syncthreads();

        if (!have_execute_task) {
            if (threadIdx.x == 0) {
                if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
                    if (prev_get_task) {
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
                prev_get_task = false;
                if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
                    should_continue = (load_L2(&d_all_tasks_finished) == 0);
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
                block_ctx.have_task_id_resumable = false;
            }
            __syncthreads();
        }

        if (have_execute_task) {
            // Copy task header to TaskContext for reuse in task function (using L2 load)
            if (threadIdx.x == 0) {
                TaskHeader* src_hdr = &d_task_headers[execute_task_id];
                TaskHeader* dst_hdr = &block_ctx.cached_task_header;
                dst_hdr->generation = load_L2(&src_hdr->generation);
                dst_hdr->parent_tid = load_L2(&src_hdr->parent_tid);
                dst_hdr->parent_generation = load_L2(&src_hdr->parent_generation);
            }
            __syncthreads();

#ifdef GTAP_ENABLE_PROFILING
            if (threadIdx.x == 0) {
                if (working_time_idx + 1 < profile_capacity()) {
                    working_time[
                        blockIdx.x * profile_capacity() +
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
            __threadfence();
        }
        __syncthreads();
#ifdef GTAP_ENABLE_PROFILING
        if (threadIdx.x == 0) {
            if (working_time_idx < profile_capacity()) {
                working_time[blockIdx.x * profile_capacity() + working_time_idx] = get_global_time();
                working_time_idx++;
            }
        }
#endif

        push_global_queue<M>(&block_ctx, &execute_task_id, &have_execute_task);
    }
#ifdef GTAP_DETAIL_INTERNAL_DEBUG
    if (threadIdx.x == 0) printf("execute_task_loop: end (block_id = %d)\n", blockIdx.x);
#endif
}

}  // namespace gtap::detail::block

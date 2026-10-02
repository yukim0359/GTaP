#pragma once

#include "../common/gtap_runtime_common.cuh"

#ifndef __GTAP_IS_BLOCK_MODE
#define __GTAP_IS_BLOCK_MODE
#endif

extern const size_t __gtap_auto_block_task_data_sizes[
    GTAP_MAX_THREADS_PER_BLOCK / GTAP_WARP_SIZE + 1];

namespace gtap::detail::block {

using namespace gtap::detail;

struct TaskContext;

inline size_t host_task_data_stride() {
    return align_up(
        __gtap_auto_block_task_data_sizes[
            stored_launch_config().block_size / GTAP_WARP_SIZE],
        16);
}

inline cudaError_t init_device_task_data_stride() {
    size_t stride = host_task_data_stride();
    return cudaMemcpyToSymbol(d_task_data_stride, &stride, sizeof(size_t));
}

struct TaskHeader {
    void (*func)(void* task, int tid, TaskContext* ctx);
#ifndef GTAP_ASSUME_NO_TASKWAIT
    // Info of current task
    uint16_t  generation;
    uint16_t  state;
    // Info of parent task
    int       parent_tid;
    uint16_t  parent_generation;
    // Info of child tasks
    int       waiting_child_count;
#endif
};

struct TaskContext {
    int task_id_generated_count;
    int queue_tail;
    int id_list_alloc_pos;
    int id_list_free_pos_stale;
    bool have_task_id_resumable;
    int task_id_resumable;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    TaskHeader cached_task_header;
#endif
};

__constant__ TaskHeader* d_task_headers;
__constant__ char* d_task_data_bytes;
__constant__ char* d_entry_result_bytes;
__constant__ int* d_task_id_list_free_positions;
__constant__ int* d_task_id_storage;
__device__ int d_first_task_finished;
__device__ int d_all_tasks_finished;
__device__ int d_active_block_count;

#ifdef GTAP_ENABLE_PROFILING
#ifdef GTAP_EXPERIMENTAL_PROFILE_LEGACY
__constant__ long long* having_task_time;
#endif
__constant__ long long* working_time;
__constant__ unsigned long long* profile_dropped_events;
#endif

__global__ void init_block_id_pools_metadata() {
    if (threadIdx.x == 0) {
        d_task_id_list_free_positions[blockIdx.x] =
            d_launch_config.tasks_per_worker;
    }
    __threadfence();
}

__device__ __forceinline__ int get_task_id_from_block_pool(
    int* id_list_free_pos,
    int* id_list_alloc_pos,
    int* id_list_free_pos_stale
) {
    int old_alloc = atomicAdd(id_list_alloc_pos, 1);
    const int tasks_per_block = d_launch_config.tasks_per_worker;
    int idx = old_alloc % tasks_per_block;
    int block_id = static_cast<int>(
        id_list_free_pos - d_task_id_list_free_positions);
    int id;
    bool first_use = (old_alloc < tasks_per_block);
    if (first_use) {
        id = block_id * tasks_per_block + idx;
    } else {
        id = load_L2(&d_task_id_storage[block_id * tasks_per_block + idx]);
    }
    int free_count = *id_list_free_pos_stale - old_alloc;
    if (free_count < GTAP_DETAIL_TASK_ID_POOL_MIN_FREE) {
        int new_free_pos = load_L2(id_list_free_pos);
        *id_list_free_pos_stale = new_free_pos;
        free_count = new_free_pos - old_alloc;
        if (free_count < GTAP_DETAIL_TASK_ID_POOL_MIN_FREE) {
            GTAP_DETAIL_RECORD_TASK_ID_POOL_LOW_HEADROOM(
                id, free_count, GTAP_DETAIL_TASK_ID_POOL_MIN_FREE);
        }
    }
    return id;
}

__device__ __forceinline__ void release_task_id_to_block_pool(int id) {
    const int tasks_per_block = d_launch_config.tasks_per_worker;
    int block_id = id / tasks_per_block;
    int* id_list_free_pos = &d_task_id_list_free_positions[block_id];
    int old_free = atomicAdd(id_list_free_pos, 1);
    store_L2(
        &d_task_id_storage[
            block_id * tasks_per_block + old_free % tasks_per_block],
        id);
}

__device__ __forceinline__ void* get_task_data(int tid) {
    return d_task_data_bytes + (size_t)tid * d_task_data_stride;
}

__device__ __forceinline__ void* get_entry_result_data() {
    return d_entry_result_bytes;
}

}  // namespace gtap::detail::block

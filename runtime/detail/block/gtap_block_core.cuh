#pragma once

#include "../common/gtap_runtime_common.cuh"

#ifndef __GTAP_IS_BLOCK_MODE
#define __GTAP_IS_BLOCK_MODE
#endif

extern const size_t __gtap_auto_block_task_data_sizes[
    GTAP_MAX_THREADS_PER_BLOCK / GTAP_WARP_SIZE + 1];

inline size_t gtap_host_task_data_stride() {
    return gtap_align_up(
        __gtap_auto_block_task_data_sizes[
            gtap_stored_launch_config().block_size / GTAP_WARP_SIZE],
        16);
}

inline cudaError_t gtap_init_device_task_data_stride() {
    size_t stride = gtap_host_task_data_stride();
    return cudaMemcpyToSymbol(d_gtap_task_data_stride, &stride, sizeof(size_t));
}

inline constexpr size_t __gtap_max_task_size = gtap_compile_time_task_data_size_limit();

struct TaskContext;

struct TaskHeader {
    void (*func)(void* task, int tid, TaskContext* ctx);
#ifndef GTAP_ASSUME_NO_TASKWAIT
    // Info of current task
    uint16_t generation;
    uint16_t state;
    // Info of parent task
    int parent_tid;
    uint16_t parent_generation;
    // Info of child tasks
    int waiting_child_count;
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

struct TaskIdList {
    int id_list_free_pos;
};

__constant__ TaskIdList* d_task_id_lists;
__constant__ int* d_task_id_storage;
__constant__ TaskHeader* d_task_headers;
__constant__ char* d_task_data_bytes;
__constant__ char* d_gtap_entry_result_bytes;
__constant__ int* d_task_id_generated;
__device__ int d_first_task_finished;
__device__ int d_all_tasks_finished_flag;
__device__ int d_active_worker_count;

#ifdef GTAP_ENABLE_PROFILING
#ifdef GTAP_EXPERIMENTAL_PROFILE_LEGACY
__constant__ long long* having_task_time;
#endif
__constant__ long long* working_time;
__constant__ unsigned long long* profile_dropped_events;
#endif

__device__ __forceinline__ int get_task_id_generated(int block_id, int idx) {
    int offset = block_id * GTAP_MAX_CHILD_TASKS + idx;
    return d_task_id_generated[offset];
}

__device__ __forceinline__ void set_task_id_generated(int block_id, int idx, int task_id) {
    if (idx >= GTAP_MAX_CHILD_TASKS) {
        GTAP_RECORD_GENERATED_TASK_ID_BUFFER_OVERFLOW(
            task_id, -1, idx, GTAP_MAX_CHILD_TASKS);
    }
    int offset = block_id * GTAP_MAX_CHILD_TASKS + idx;
    d_task_id_generated[offset] = task_id;
}

__global__ void init_block_id_pools_metadata() {
    if (threadIdx.x == 0) {
        TaskIdList* tid_list = &d_task_id_lists[blockIdx.x];
        tid_list->id_list_free_pos = d_gtap_launch_config.tasks_per_worker;
    }
    __threadfence();
}

__device__ __forceinline__ int get_task_id_from_block_pool(
    TaskIdList* tid_list,
    int* id_list_alloc_pos,
    int* id_list_free_pos_stale
) {
    int old_alloc = atomicAdd(id_list_alloc_pos, 1);
    const int tasks_per_block = d_gtap_launch_config.tasks_per_worker;
    int idx = old_alloc % tasks_per_block;
    int block_id = static_cast<int>(tid_list - d_task_id_lists);
    int id;
    bool first_use = (old_alloc < tasks_per_block);
    if (first_use) {
        id = block_id * tasks_per_block + idx;
    } else {
        id = load_L2(
            &d_task_id_storage[block_id * tasks_per_block + idx]);
    }
    int free_count = *id_list_free_pos_stale - old_alloc;
    if (free_count < GTAP_TASK_ID_POOL_MIN_FREE) {
        int new_free_pos = load_L2(&tid_list->id_list_free_pos);
        *id_list_free_pos_stale = new_free_pos;
        free_count = new_free_pos - old_alloc;
        if (free_count < GTAP_TASK_ID_POOL_MIN_FREE) {
            GTAP_RECORD_TASK_ID_POOL_LOW_HEADROOM(
                id, free_count, GTAP_TASK_ID_POOL_MIN_FREE);
        }
    }
    return id;
}

__device__ __forceinline__ void release_task_id_to_block_pool(int id) {
    const int tasks_per_block = d_gtap_launch_config.tasks_per_worker;
    int block_id = id / tasks_per_block;
    TaskIdList* tid_list = &d_task_id_lists[block_id];
    int old_free = atomicAdd(&tid_list->id_list_free_pos, 1);
    store_L2(
        &d_task_id_storage[
            block_id * tasks_per_block + old_free % tasks_per_block],
        id);
}

__device__ __forceinline__ void* __gtap_get_task_data(int tid) {
    return d_task_data_bytes + (size_t)tid * gtap_device_task_data_stride();
}

__device__ __forceinline__ void* __gtap_get_entry_result_data() {
    return d_gtap_entry_result_bytes;
}

template <typename TaskType>
__device__ __forceinline__ TaskType* __gtap_get_task_data(int tid) {
    return reinterpret_cast<TaskType*>(__gtap_get_task_data(tid));
}

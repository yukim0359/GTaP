#pragma once

#include "../common/gtap_runtime_common.cuh"

#ifndef __GTAP_WORKER_IS_BLOCK
#define __GTAP_WORKER_IS_BLOCK
#endif

inline constexpr size_t __gtap_max_task_size = gtap_compile_time_task_data_size_limit();

#ifndef GTAP_RESULT_HANDLE_CAPACITY
#define GTAP_RESULT_HANDLE_CAPACITY 10000
#endif

struct TaskContext;

struct TaskHeader {
    void (*func)(void* task, int tid, TaskContext* ctx);
#ifndef GTAP_ASSUME_NO_TASKWAIT
    // Info of current task
    uint16_t generation;
    uint16_t state;
    uint16_t retain_parent_result;
    // Info of parent task
    int parent_tid;
    uint16_t parent_generation;
    // Info of child tasks
    int waiting_child_count;
    int result_handle_begin;
    int result_handle_last;
    int result_handle_count;
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
    int id_list[GTAP_MAX_TASKS_PER_BLOCK];
    int id_list_free_pos;
};

struct GTaPResultHandle {
    int child_tid;
    int kind;
    int next;
    void* lhs_addr;
};

__constant__ TaskIdList* d_task_id_lists;
__constant__ TaskHeader* d_task_headers;
__constant__ char* d_task_data_bytes;
__constant__ int* d_task_id_generated;
__constant__ GTaPResultHandle* d_result_handles;
__device__ int d_result_handle_top;
__device__ int d_first_task_finished;
__device__ int d_all_tasks_finished_flag;
__device__ int d_active_worker_count;

#ifdef PROFILE
__device__ long long having_task_time[GTAP_GRID_SIZE][MAX_PROFILE_DATA];
__device__ long long working_time[GTAP_GRID_SIZE][MAX_PROFILE_DATA];
#endif

__device__ __forceinline__ int get_task_id_generated(int block_id, int idx) {
    int offset = block_id * GTAP_MAX_CHILD_TASKS + idx;
    return d_task_id_generated[offset];
}

__device__ __forceinline__ void set_task_id_generated(int block_id, int idx, int task_id) {
    int offset = block_id * GTAP_MAX_CHILD_TASKS + idx;
    d_task_id_generated[offset] = task_id;
}

__global__ void init_block_id_pools_metadata() {
    if (threadIdx.x == 0) {
        TaskIdList* tid_list = &d_task_id_lists[blockIdx.x];
        tid_list->id_list_free_pos = GTAP_MAX_TASKS_PER_BLOCK;
    }
    __threadfence();
}

struct TaskIdFromPool {
    int tid;
    bool first_use;
};

__device__ __forceinline__ TaskIdFromPool get_task_id_from_block_pool(
    TaskIdList* tid_list,
    int* id_list_alloc_pos,
    int* id_list_free_pos_stale
) {
    int old_alloc = atomicAdd(id_list_alloc_pos, 1);
    int idx = old_alloc % GTAP_MAX_TASKS_PER_BLOCK;
    int block_id = static_cast<int>(tid_list - d_task_id_lists);
    int id;
    bool first_use = (old_alloc < GTAP_MAX_TASKS_PER_BLOCK);
    if (first_use) {
        id = block_id * GTAP_MAX_TASKS_PER_BLOCK + idx;
    } else {
        id = load_L2(&tid_list->id_list[idx]);
    }
    int free_count = *id_list_free_pos_stale - old_alloc;
    if (free_count < GTAP_TASK_ID_POOL_MIN_FREE) {
        int new_free_pos = load_L2(&tid_list->id_list_free_pos);
        *id_list_free_pos_stale = new_free_pos;
        free_count = new_free_pos - old_alloc;
        if (free_count < GTAP_TASK_ID_POOL_MIN_FREE) {
            gtap_record_runtime_error_and_trap(
                GTAP_ERROR_TASK_ID_POOL_EXHAUSTED, block_id, id, -1,
                free_count, GTAP_TASK_ID_POOL_MIN_FREE, __LINE__);
        }
    }
    return TaskIdFromPool{id, first_use};
}

__device__ __forceinline__ void release_task_id_to_block_pool(int id) {
    int block_id = id / GTAP_MAX_TASKS_PER_BLOCK;
    TaskIdList* tid_list = &d_task_id_lists[block_id];
    int old_free = atomicAdd(&tid_list->id_list_free_pos, 1);
    store_L2(&tid_list->id_list[old_free % GTAP_MAX_TASKS_PER_BLOCK], id);
}

extern "C" __device__ __forceinline__ void __gtap_release_task_id(int tid) {
    release_task_id_to_block_pool(tid);
}

__device__ __forceinline__ void* __gtap_get_task_data(int tid) {
    return d_task_data_bytes + (size_t)tid * gtap_device_task_data_stride();
}

template <typename TaskType>
__device__ __forceinline__ TaskType* __gtap_get_task_data(int tid) {
    return reinterpret_cast<TaskType*>(__gtap_get_task_data(tid));
}

extern "C" __device__ __forceinline__ void __gtap_append_result_handle(
    int parent_tid,
    int kind,
    int child_tid,
    void* lhs_addr
) {
#ifdef GTAP_ASSUME_NO_TASKWAIT
    (void)parent_tid;
    (void)kind;
    (void)child_tid;
    (void)lhs_addr;
#else
    int slot = atomicAdd(&d_result_handle_top, 1);
    if (slot >= GTAP_RESULT_HANDLE_CAPACITY) {
        gtap_record_runtime_error_and_trap(
            GTAP_ERROR_QUEUE_OVERFLOW, blockIdx.x, child_tid, -1,
            slot, GTAP_RESULT_HANDLE_CAPACITY, __LINE__);
    }
    d_result_handles[slot].child_tid = child_tid;
    d_result_handles[slot].kind = kind;
    d_result_handles[slot].next = -1;
    d_result_handles[slot].lhs_addr = lhs_addr;

    TaskHeader* parent_hdr = &d_task_headers[parent_tid];
    int prev_last = parent_hdr->result_handle_last;
    if (prev_last >= 0) {
        d_result_handles[prev_last].next = slot;
    } else {
        parent_hdr->result_handle_begin = slot;
    }
    parent_hdr->result_handle_last = slot;
    atomicAdd(&parent_hdr->result_handle_count, 1);
#endif
}

extern "C" __device__ __forceinline__ int __gtap_get_result_handle_begin(int tid) {
#ifdef GTAP_ASSUME_NO_TASKWAIT
    (void)tid;
    return 0;
#else
    return load_L2(&d_task_headers[tid].result_handle_begin);
#endif
}

extern "C" __device__ __forceinline__ int __gtap_get_result_handle_count(int tid) {
#ifdef GTAP_ASSUME_NO_TASKWAIT
    (void)tid;
    return 0;
#else
    return load_L2(&d_task_headers[tid].result_handle_count);
#endif
}

extern "C" __device__ __forceinline__ int __gtap_get_result_handle_child_tid(int handle_index) {
    return load_L2(&d_result_handles[handle_index].child_tid);
}

extern "C" __device__ __forceinline__ int __gtap_get_result_handle_kind(int handle_index) {
    return load_L2(&d_result_handles[handle_index].kind);
}

extern "C" __device__ __forceinline__ int __gtap_get_result_handle_next(int handle_index) {
    return load_L2(&d_result_handles[handle_index].next);
}

extern "C" __device__ __forceinline__ void* __gtap_get_result_handle_lhs_addr(int handle_index) {
    return load_L2_ptr(&d_result_handles[handle_index].lhs_addr);
}

extern "C" __device__ __forceinline__ void __gtap_clear_result_handles(int tid) {
#ifdef GTAP_ASSUME_NO_TASKWAIT
    (void)tid;
#else
    d_task_headers[tid].result_handle_begin = -1;
    d_task_headers[tid].result_handle_last = -1;
    d_task_headers[tid].result_handle_count = 0;
#endif
}

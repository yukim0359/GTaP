#pragma once

#include "../common/gtap_runtime_common.cuh"

#ifndef __GTAP_WORKER_IS_THREAD
#define __GTAP_WORKER_IS_THREAD
#endif

inline constexpr size_t __gtap_max_task_size = gtap_compile_time_task_data_size_limit();

#ifndef GTAP_RESULT_HANDLE_CAPACITY
#define GTAP_RESULT_HANDLE_CAPACITY 1000
#endif

GTAP_VALIDATE_RESULT_HANDLE_CONFIG();

// #define DEBUG
// #define INIT_PROFILE

struct TaskContext;

struct TaskHeader {
    void (*func)(void* task, int tid, TaskContext* __ctx);
#ifdef GTAP_ASSUME_NO_TASKWAIT
#if (GTAP_NUM_QUEUES > 1)
    uint16_t   queue_idx;
#endif
#else
    // Info of current task
    uint16_t   generation;
    uint16_t   state;
    uint16_t   queue_idx;
    uint16_t   retain_parent_result;
    // Info of parent task
    int        parent_tid;
    uint16_t   parent_generation;
    // Info of child tasks
    int        waiting_child_count;
    int        result_handle_begin;
    int        result_handle_last;
    int        result_handle_count;
#endif
};

struct TaskContext {
    int queue_idx;
    int task_id_generated_count_by_queue_idx[GTAP_NUM_QUEUES];
    int* tail_by_queue_idx;
    int id_list_alloc_pos;
    int id_list_free_pos_stale;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    TaskHeader task_headers[GTAP_WARP_SIZE];
#endif
};

struct TaskIdList {
    int id_list[GTAP_TOTAL_TASK_IDS_PER_WARP];
    int valid[GTAP_TOTAL_TASK_IDS_PER_WARP];
    int id_list_free_pos;
};

struct GTaPResultHandle {
    int child_tid;
    int kind;
    int next;
    void* lhs_addr;
};

__constant__ TaskHeader* d_task_headers;
__constant__ char* d_task_data_bytes;
__constant__ TaskIdList* d_task_id_lists;
__constant__ GTaPResultHandle* d_result_handles;
#ifdef GTAP_THREAD_HAS_GENERATED_TASK_IDS
__constant__ int* d_task_id_generated_by_queue_idx;
#endif
__device__ int d_result_handle_top;
__device__ int d_first_task_finished;
__device__ int d_all_tasks_finished_flag;
__device__ int d_active_worker_count;

#ifdef PROFILE
__device__ long long having_task_time[GTAP_GRID_SIZE * GTAP_NUM_WARPS][MAX_PROFILE_DATA];
__device__ long long working_time[GTAP_GRID_SIZE * GTAP_NUM_WARPS][MAX_PROFILE_DATA];
__device__ int tasks_processed_count[GTAP_GRID_SIZE * GTAP_NUM_WARPS][MAX_PROFILE_DATA];
#endif

#ifdef GTAP_THREAD_HAS_GENERATED_TASK_IDS
constexpr int GTAP_TASK_ID_GEN_QUEUE_STRIDE = (GTAP_MAX_CHILD_TASKS + 1) * GTAP_WARP_SIZE;
constexpr int GTAP_TASK_ID_GEN_WARP_STRIDE = GTAP_NUM_QUEUES * GTAP_TASK_ID_GEN_QUEUE_STRIDE;

__device__ __forceinline__ int get_task_id_generated(int warp_id_global, int queue_idx, int idx) {
    int offset = warp_id_global * GTAP_TASK_ID_GEN_WARP_STRIDE + queue_idx * GTAP_TASK_ID_GEN_QUEUE_STRIDE + idx;
    return d_task_id_generated_by_queue_idx[offset];
}

__device__ __forceinline__ void set_task_id_generated(int warp_id_global, int queue_idx, int idx, int task_id) {
    int offset = warp_id_global * GTAP_TASK_ID_GEN_WARP_STRIDE + queue_idx * GTAP_TASK_ID_GEN_QUEUE_STRIDE + idx;
    d_task_id_generated_by_queue_idx[offset] = task_id;
}
#endif

struct TaskIdFromPool {
    int tid;
    bool first_use;
};

__device__ __forceinline__ TaskIdFromPool get_task_id_from_warp_pool(TaskIdList* tid_list, int* id_list_alloc_pos, int* id_list_free_pos_stale) {
    int old_alloc = atomicAdd(id_list_alloc_pos, 1);
    int warp_id_global = (tid_list - d_task_id_lists);
    int id = 0;
    bool first_use = (old_alloc < GTAP_TOTAL_TASK_IDS_PER_WARP);
    if (first_use) {
        id = warp_id_global * GTAP_TOTAL_TASK_IDS_PER_WARP + old_alloc;
    } else {
        int idx = old_alloc % GTAP_TOTAL_TASK_IDS_PER_WARP;
        if (load_L2_acquire(&tid_list->valid[idx]) == 1) {
            id = load_L2(&tid_list->id_list[idx]);
            store_L2(&tid_list->valid[idx], 0);
        } else {
            gtap_record_runtime_error_and_trap(
                GTAP_ERROR_TASK_ID_POOL_EXHAUSTED, warp_id_global, id, -1,
                old_alloc, GTAP_TOTAL_TASK_IDS_PER_WARP, __LINE__);
        }
    }
    int free_count = *id_list_free_pos_stale - old_alloc;
    if (free_count < GTAP_TASK_ID_POOL_MIN_FREE) {
        int new_free_pos = load_L2(&tid_list->id_list_free_pos);
        *id_list_free_pos_stale = new_free_pos;
        free_count = new_free_pos - old_alloc;
        if (free_count < GTAP_TASK_ID_POOL_MIN_FREE) {
            gtap_record_runtime_error_and_trap(
                GTAP_ERROR_TASK_ID_POOL_EXHAUSTED, warp_id_global, id, -1,
                free_count, GTAP_TASK_ID_POOL_MIN_FREE, __LINE__);
        }
    }
    return TaskIdFromPool{id, first_use};
}

__device__ __forceinline__ void release_task_id_to_warp_pool(int id) {
    int warp_id_global = get_warp_id_global();
    TaskIdList* tid_list = &d_task_id_lists[warp_id_global];
    int old_free = atomicAdd(&tid_list->id_list_free_pos, 1);
    store_L2(&tid_list->id_list[old_free % GTAP_TOTAL_TASK_IDS_PER_WARP], id);
    store_L2(&tid_list->valid[old_free % GTAP_TOTAL_TASK_IDS_PER_WARP], 1);
}

__global__ void init_warp_id_pools_metadata() {
    int warp_id_in_block = get_warp_id_in_block();
    int lane = get_lane_id();
    if (warp_id_in_block < GTAP_NUM_WARPS && lane == 0) {
        int qid = blockIdx.x * GTAP_NUM_WARPS + warp_id_in_block;
        TaskIdList* tid_list = &d_task_id_lists[qid];
        tid_list->id_list_free_pos = GTAP_TOTAL_TASK_IDS_PER_WARP;
    }
    __threadfence();
}

extern "C" __device__ __forceinline__ void __gtap_release_task_id(int tid) {
    release_task_id_to_warp_pool(tid);
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
            GTAP_ERROR_QUEUE_OVERFLOW, get_warp_id_global(), child_tid, -1,
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

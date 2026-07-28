#pragma once

#include "../common/gtap_runtime_common.cuh"

#ifndef __GTAP_WORKER_IS_THREAD
#define __GTAP_WORKER_IS_THREAD
#endif

inline constexpr size_t __gtap_max_task_size = gtap_compile_time_task_data_size_limit();

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
    // Info of parent task
    int        parent_tid;
    uint16_t   parent_generation;
    // Info of child tasks
    int        waiting_child_count;
#endif
};

struct TaskContext {
    int queue_idx;
    int task_id_generated_count_by_queue_idx[GTAP_NUM_QUEUES];
    int* tail_by_queue_idx;
    int* staged_task_ids;
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

__constant__ TaskHeader* d_task_headers;
__constant__ char* d_task_data_bytes;
__constant__ TaskIdList* d_task_id_lists;
#ifdef GTAP_THREAD_HAS_GENERATED_TASK_IDS
__constant__ int* d_task_id_generated_by_queue_idx;
#endif
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
    if (idx >= GTAP_TASK_ID_GEN_QUEUE_STRIDE) {
        GTAP_RECORD_GENERATED_TASK_ID_BUFFER_OVERFLOW(
            task_id, queue_idx, idx, GTAP_TASK_ID_GEN_QUEUE_STRIDE);
    }
    int offset = warp_id_global * GTAP_TASK_ID_GEN_WARP_STRIDE + queue_idx * GTAP_TASK_ID_GEN_QUEUE_STRIDE + idx;
    d_task_id_generated_by_queue_idx[offset] = task_id;
}
#endif

__device__ __forceinline__ int get_task_id_from_warp_pool(TaskIdList* tid_list, int* id_list_alloc_pos, int* id_list_free_pos_stale) {
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
            GTAP_RECORD_TASK_ID_POOL_SLOT_BUSY(id, old_alloc, GTAP_TOTAL_TASK_IDS_PER_WARP);
        }
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

__device__ __forceinline__ void* __gtap_get_task_data(int tid) {
    return d_task_data_bytes + (size_t)tid * gtap_device_task_data_stride();
}

template <typename TaskType>
__device__ __forceinline__ TaskType* __gtap_get_task_data(int tid) {
    return reinterpret_cast<TaskType*>(__gtap_get_task_data(tid));
}

#pragma once

#include "../common/gtap_runtime_common.cuh"

#ifndef __GTAP_WORKER_IS_THREAD
#define __GTAP_WORKER_IS_THREAD
#endif

inline constexpr size_t __gtap_max_task_size = gtap_compile_time_task_data_size_limit();

// #define GTAP_INTERNAL_DEBUG
// #define GTAP_INTERNAL_PROFILE_INIT

struct TaskContext;

struct TaskHeader {
    void (*func)(void* task, int tid, TaskContext* __ctx);
#ifdef GTAP_ASSUME_NO_TASKWAIT
    uint16_t   queue_idx;
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
    int* task_id_generated_count_by_queue_idx;
    int* tail_by_queue_idx;
    int* staged_task_ids;
    int id_list_alloc_pos;
    int id_list_free_pos_stale;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    int task_parent_tids[GTAP_WARP_SIZE];
    uint32_t task_generations[GTAP_WARP_SIZE];
#endif
};

struct TaskIdList {
    int id_list_free_pos;
};

__constant__ TaskHeader* d_task_headers;
__constant__ char* d_task_data_bytes;
__constant__ TaskIdList* d_task_id_lists;
__constant__ int* d_task_id_storage;
__constant__ int* d_task_id_valid;
__device__ int d_first_task_finished;
__device__ int d_all_tasks_finished_flag;
__device__ int d_active_worker_count;

#ifdef GTAP_PROFILE
#ifdef GTAP_EXPERIMENTAL_PROFILE_LEGACY
__constant__ long long* having_task_time;
#endif
__constant__ long long* working_time;
__constant__ int* tasks_processed_count;
__constant__ unsigned long long* profile_dropped_events;
#endif

__device__ __forceinline__ int get_task_id_from_warp_pool(TaskIdList* tid_list, int* id_list_alloc_pos, int* id_list_free_pos_stale) {
    int old_alloc = atomicAdd(id_list_alloc_pos, 1);
    int warp_id_global = (tid_list - d_task_id_lists);
    int id = 0;
    const int task_ids_per_warp = d_gtap_launch_config.tasks_per_worker;
    bool first_use = (old_alloc < task_ids_per_warp);
    if (first_use) {
        id = warp_id_global * task_ids_per_warp + old_alloc;
    } else {
        int idx = old_alloc % task_ids_per_warp;
        const int storage_idx = warp_id_global * task_ids_per_warp + idx;
        if (load_L2_acquire(&d_task_id_valid[storage_idx]) == 1) {
            id = load_L2(&d_task_id_storage[storage_idx]);
            store_L2(&d_task_id_valid[storage_idx], 0);
        } else {
            GTAP_RECORD_TASK_ID_POOL_SLOT_BUSY(id, old_alloc, task_ids_per_warp);
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
    const int task_ids_per_warp = d_gtap_launch_config.tasks_per_worker;
    const int storage_idx =
        warp_id_global * task_ids_per_warp + old_free % task_ids_per_warp;
    store_L2(&d_task_id_storage[storage_idx], id);
    store_L2(&d_task_id_valid[storage_idx], 1);
}

__global__ void init_warp_id_pools_metadata() {
    int warp_id_in_block = get_warp_id_in_block();
    int lane = get_lane_id();
    if (warp_id_in_block < d_gtap_launch_config.warps_per_block && lane == 0) {
        int qid =
            blockIdx.x * d_gtap_launch_config.warps_per_block + warp_id_in_block;
        TaskIdList* tid_list = &d_task_id_lists[qid];
        tid_list->id_list_free_pos = d_gtap_launch_config.tasks_per_worker;
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

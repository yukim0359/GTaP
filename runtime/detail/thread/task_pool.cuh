#pragma once

#include "../common/runtime_error.cuh"
#include "../common/warp_index.cuh"

#include "task_types.cuh"

extern const size_t __gtap_auto_task_data_size;

namespace gtap::detail::thread {

using namespace gtap::detail;

__constant__ size_t d_task_data_stride;

inline size_t host_task_data_stride() {
    return align_up(__gtap_auto_task_data_size, 16);
}

inline cudaError_t init_device_task_data_stride() {
    size_t stride = host_task_data_stride();
    return cudaMemcpyToSymbol(d_task_data_stride, &stride, sizeof(size_t));
}

__constant__ TaskHeader* d_task_headers;         // TaskHeader[num_warps * tasks_per_warp]
__constant__ char* d_task_data_bytes;            // char[num_warps * tasks_per_warp * task_data_stride]
__constant__ int* d_task_id_list_free_positions; // int[num_warps]
__constant__ int* d_task_id_storage;             // int[num_warps * tasks_per_warp]
__constant__ int* d_task_id_valid;               // int[num_warps * tasks_per_warp]

__global__ void init_warp_id_pools_metadata() {
    int warp_id_in_block = get_warp_id_in_block();
    int lane = get_lane_id();
    if (warp_id_in_block < d_launch_config.warps_per_block && lane == 0) {
        int qid =
            blockIdx.x * d_launch_config.warps_per_block + warp_id_in_block;
        d_task_id_list_free_positions[qid] = d_launch_config.tasks_per_scheduling_unit;
    }
    __threadfence();
}

__device__ __forceinline__ int get_task_id_from_warp_pool(
    int* id_list_free_pos,
    int* id_list_alloc_pos,
    int* id_list_free_pos_stale
) {
    int old_alloc = atomicAdd(id_list_alloc_pos, 1);
    int warp_id_global = id_list_free_pos - d_task_id_list_free_positions;
    int id = 0;
    const int task_ids_per_warp = d_launch_config.tasks_per_scheduling_unit;
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
            GTAP_DETAIL_RECORD_TASK_ID_POOL_SLOT_BUSY(id, old_alloc, task_ids_per_warp);
        }
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

__device__ __forceinline__ void release_task_id_to_warp_pool(int id) {
    int warp_id_global = get_warp_id_global();
    int* id_list_free_pos = &d_task_id_list_free_positions[warp_id_global];
    int old_free = atomicAdd(id_list_free_pos, 1);
    const int task_ids_per_warp = d_launch_config.tasks_per_scheduling_unit;
    const int storage_idx =
        warp_id_global * task_ids_per_warp + old_free % task_ids_per_warp;
    store_L2(&d_task_id_storage[storage_idx], id);
    store_L2(&d_task_id_valid[storage_idx], 1);
}

__device__ __forceinline__ void* get_task_data(int tid) {
    return d_task_data_bytes + (size_t)tid * d_task_data_stride;
}

struct task_pool_buffers {
    TaskHeader* headers = nullptr;
    char* task_data = nullptr;
    int* id_list_free_positions = nullptr;
    int* id_storage = nullptr;
    int* id_valid = nullptr;
};

inline size_t task_pool_allocation_bytes(size_t scheduling_units, size_t tasks) {
    return sizeof(TaskHeader) * tasks
        + host_task_data_stride() * tasks
        + sizeof(int) * scheduling_units
        + 2 * sizeof(int) * tasks;
}

// Allocates the pool and starts the async clears. Symbols are published later.
inline cudaError_t stage_task_pool(
    size_t scheduling_units, size_t tasks,
    cudaStream_t stream,
    task_pool_buffers* buffers
) {
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&buffers->headers),
        sizeof(TaskHeader) * tasks));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->headers, 0, sizeof(TaskHeader) * tasks, stream));

    const size_t task_data_size = host_task_data_stride() * tasks;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&buffers->task_data), task_data_size));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->task_data, 0, task_data_size, stream));

    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&buffers->id_list_free_positions),
        sizeof(int) * scheduling_units));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->id_list_free_positions, 0, sizeof(int) * scheduling_units,
        stream));
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&buffers->id_storage), sizeof(int) * tasks));
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&buffers->id_valid), sizeof(int) * tasks));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->id_valid, 0, sizeof(int) * tasks, stream));
    return cudaSuccess;
}

inline cudaError_t publish_task_pool(const task_pool_buffers& buffers) {
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_headers, &buffers.headers, sizeof(TaskHeader*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_data_bytes, &buffers.task_data, sizeof(char*)));
    GTAP_DETAIL_CUDA_TRY(init_device_task_data_stride());
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_list_free_positions, &buffers.id_list_free_positions,
        sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_storage, &buffers.id_storage, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_valid, &buffers.id_valid, sizeof(int*)));
    return cudaSuccess;
}

inline cudaError_t clear_task_pool(
    size_t scheduling_units, size_t tasks,
    cudaStream_t stream
) {
    TaskHeader* headers = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &headers, d_task_headers, sizeof(TaskHeader*)));
    char* task_data = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &task_data, d_task_data_bytes, sizeof(char*)));
    int* id_list_free_positions = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &id_list_free_positions, d_task_id_list_free_positions, sizeof(int*)));
    int* id_valid = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &id_valid, d_task_id_valid, sizeof(int*)));

    if (headers != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            headers, 0, sizeof(TaskHeader) * tasks, stream));
    }
    if (task_data != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            task_data, 0, host_task_data_stride() * tasks, stream));
    }
    if (id_list_free_positions != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            id_list_free_positions, 0, sizeof(int) * scheduling_units, stream));
    }
    if (id_valid != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            id_valid, 0, sizeof(int) * tasks, stream));
    }
    return cudaSuccess;
}

inline cudaError_t free_task_pool() {
    TaskHeader* headers = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &headers, d_task_headers, sizeof(TaskHeader*)));
    char* task_data = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &task_data, d_task_data_bytes, sizeof(char*)));
    int* id_list_free_positions = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &id_list_free_positions, d_task_id_list_free_positions, sizeof(int*)));
    int* id_storage = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &id_storage, d_task_id_storage, sizeof(int*)));
    int* id_valid = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &id_valid, d_task_id_valid, sizeof(int*)));

    if (headers != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(headers));
    if (task_data != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(task_data));
    if (id_list_free_positions != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(id_list_free_positions));
    }
    if (id_storage != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(id_storage));
    if (id_valid != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(id_valid));
    return cudaSuccess;
}

}  // namespace gtap::detail::thread

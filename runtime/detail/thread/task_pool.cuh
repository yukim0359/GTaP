#pragma once

#include "../common/cuda_primitives.cuh"
#include "../common/device_memory.cuh"
#include "../common/runtime_config.cuh"
#include "../common/runtime_error.cuh"
#include "../common/task_pool.cuh"
#include "../common/warp_index.cuh"

#include "task_types.cuh"

// The compiler sets this stride after rounding the task records up to their alignment.
extern const size_t __gtap_task_data_stride;

namespace gtap::detail::thread {

using namespace gtap::detail;

__constant__ size_t d_task_data_stride;

inline cudaError_t publish_task_data_stride() {
    return cudaMemcpyToSymbol(
        d_task_data_stride, &__gtap_task_data_stride, sizeof(size_t));
}

__constant__ TaskHeader* d_task_headers;         // TaskHeader[num_warps * tasks_per_warp]
__constant__ char* d_task_data_bytes;            // char[num_warps * tasks_per_warp * task_data_stride]
__constant__ int* d_task_id_list_free_positions; // int[num_warps]
// High bit is the ring lap. The low 31 bits are the task id.
__constant__ int* d_task_id_storage;             // int[num_warps * tasks_per_warp]

// Launched as <<<number_of_warps, 1>>>.
__global__ void init_warp_id_pools_metadata() {
    if (threadIdx.x == 0) {
        d_task_id_list_free_positions[blockIdx.x] =
            d_launch_config.tasks_per_scheduling_unit;
    }
    __threadfence();
}

__device__ __forceinline__ int get_task_id_from_warp_pool(
    int* id_list_free_pos,
    int* id_list_alloc_pos,
    int* id_list_free_pos_stale
) {
    const int warp_id_global = get_warp_id_global();
    const int old_alloc = atomicAdd(id_list_alloc_pos, 1);
    const int tasks_per_warp = d_launch_config.tasks_per_scheduling_unit;
    const bool first_use = (old_alloc < tasks_per_warp);
    int task_id = 0;
    if (first_use) {
        task_id = warp_id_global * tasks_per_warp + old_alloc;
    } else {
        const int storage_idx = warp_id_global * tasks_per_warp + old_alloc % tasks_per_warp;
        const int packed = load_relaxed(&d_task_id_storage[storage_idx]);
        if (ring_lap_matches(packed, old_alloc, tasks_per_warp)) {
            task_id = unpack_task_id_slot(packed);
        } else {
            GTAP_DETAIL_RECORD_TASK_ID_POOL_SLOT_BUSY(task_id, old_alloc, tasks_per_warp);
        }
    }
    int free_count = *id_list_free_pos_stale - old_alloc;
    if (free_count < GTAP_DETAIL_TASK_ID_POOL_MIN_FREE) {
        const int new_free_pos = load_L2(id_list_free_pos);
        *id_list_free_pos_stale = new_free_pos;
        free_count = new_free_pos - old_alloc;
        if (free_count < GTAP_DETAIL_TASK_ID_POOL_MIN_FREE) {
            GTAP_DETAIL_RECORD_TASK_ID_POOL_LOW_HEADROOM(
                task_id, free_count, GTAP_DETAIL_TASK_ID_POOL_MIN_FREE);
        }
    }
    return task_id;
}

// Writes one recycled id into the slot selected by ticket.
__device__ __forceinline__ void release_task_id_to_warp_pool(int tid, int ticket) {
    const int tasks_per_warp = d_launch_config.tasks_per_scheduling_unit;
    const int return_warp_id = get_warp_id_global();
    const int storage_idx =
        return_warp_id * tasks_per_warp + ticket % tasks_per_warp;
    store_relaxed(
        &d_task_id_storage[storage_idx],
        pack_task_id_slot(tid, ticket, tasks_per_warp));
}

__device__ __forceinline__ void* get_task_data(int tid) {
    return d_task_data_bytes + (size_t)tid * d_task_data_stride;
}

struct task_pool_buffers {
    TaskHeader* headers = nullptr;
    char* task_data = nullptr;
    int* id_list_free_positions = nullptr;
    int* id_storage = nullptr;
};

inline size_t task_pool_allocation_bytes(const launch_config& config) {
    const size_t scheduling_units = static_cast<size_t>(config.total_scheduling_units);
    const size_t tasks =
        scheduling_units * static_cast<size_t>(config.tasks_per_scheduling_unit);
    return sizeof(TaskHeader) * tasks
        + __gtap_task_data_stride * tasks
        + sizeof(int) * scheduling_units
        + sizeof(int) * tasks;
}

// Allocates the pool and starts the async clears. Symbols are published later.
inline cudaError_t stage_task_pool(
    const launch_config& config,
    cudaStream_t stream,
    task_pool_buffers* buffers
) {
    const size_t scheduling_units = static_cast<size_t>(config.total_scheduling_units);
    const size_t tasks =
        scheduling_units * static_cast<size_t>(config.tasks_per_scheduling_unit);
    GTAP_DETAIL_CUDA_TRY(alloc_device(&buffers->headers, sizeof(TaskHeader) * tasks));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->headers, 0, sizeof(TaskHeader) * tasks, stream));

    const size_t task_data_size = __gtap_task_data_stride * tasks;
    GTAP_DETAIL_CUDA_TRY(alloc_device(&buffers->task_data, task_data_size));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->task_data, 0, task_data_size, stream));

    GTAP_DETAIL_CUDA_TRY(alloc_device(
        &buffers->id_list_free_positions, sizeof(int) * scheduling_units));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->id_list_free_positions, 0, sizeof(int) * scheduling_units,
        stream));
    GTAP_DETAIL_CUDA_TRY(alloc_device(&buffers->id_storage, sizeof(int) * tasks));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->id_storage, 0, sizeof(int) * tasks, stream));
    return cudaSuccess;
}

inline cudaError_t publish_task_pool(const task_pool_buffers& buffers) {
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_headers, &buffers.headers, sizeof(TaskHeader*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_data_bytes, &buffers.task_data, sizeof(char*)));
    GTAP_DETAIL_CUDA_TRY(publish_task_data_stride());
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_list_free_positions, &buffers.id_list_free_positions,
        sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_storage, &buffers.id_storage, sizeof(int*)));
    return cudaSuccess;
}

inline cudaError_t clear_task_pool(
    const launch_config& config,
    cudaStream_t stream
) {
    const size_t scheduling_units = static_cast<size_t>(config.total_scheduling_units);
    const size_t tasks =
        scheduling_units * static_cast<size_t>(config.tasks_per_scheduling_unit);
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

    if (headers != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            headers, 0, sizeof(TaskHeader) * tasks, stream));
    }
    if (task_data != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            task_data, 0, __gtap_task_data_stride * tasks, stream));
    }
    if (id_list_free_positions != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            id_list_free_positions, 0, sizeof(int) * scheduling_units, stream));
    }
    if (id_storage != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            id_storage, 0, sizeof(int) * tasks, stream));
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

    if (headers != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(headers));
    if (task_data != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(task_data));
    if (id_list_free_positions != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(id_list_free_positions));
    }
    if (id_storage != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(id_storage));
    return cudaSuccess;
}

inline void release_staged_task_pool(task_pool_buffers* buffers) {
    free_device(buffers->headers);
    free_device(buffers->task_data);
    free_device(buffers->id_list_free_positions);
    free_device(buffers->id_storage);
    TaskHeader* headers = nullptr;
    char* task_data = nullptr;
    int* id_list_free_positions = nullptr;
    int* id_storage = nullptr;
    size_t stride = 0;
    cudaMemcpyToSymbol(d_task_headers, &headers, sizeof(TaskHeader*));
    cudaMemcpyToSymbol(d_task_data_bytes, &task_data, sizeof(task_data));
    cudaMemcpyToSymbol(d_task_data_stride, &stride, sizeof(stride));
    cudaMemcpyToSymbol(
        d_task_id_list_free_positions, &id_list_free_positions,
        sizeof(id_list_free_positions));
    cudaMemcpyToSymbol(d_task_id_storage, &id_storage, sizeof(id_storage));
}

}  // namespace gtap::detail::thread

#pragma once

#include "../common/checked_size.cuh"
#include "../common/cuda_primitives.cuh"
#include "../common/device_memory.cuh"
#include "../common/host_api_error.cuh"
#include "../common/runtime_config.cuh"
#include "../common/runtime_error.cuh"
#include "../common/task_pool.cuh"
#include "../common/warp_index.cuh"

#include "task_types.cuh"
#ifdef GTAP_ENABLE_PROFILING
#include "profile_buffer.cuh"
#endif

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
// High bit is the ring lap. The low 31 bits are the task id.
__constant__ int* d_task_id_storage;             // int[num_warps * tasks_per_warp]

__device__ __forceinline__ int get_task_id_from_warp_pool(
    int* id_list_free_pos,
    int* id_list_alloc_pos
) {
    const int warp_id_global = get_warp_id_global();
    const int old_alloc = atomicAdd(id_list_alloc_pos, 1);
    const int tasks_per_warp = d_launch_config.tasks_per_scheduling_unit;
#ifdef GTAP_ENABLE_PROFILING
    const int unreturned_id_count =
        (old_alloc + 1) + tasks_per_warp - *id_list_free_pos;
    record_unreturned_id_count(warp_id_global, unreturned_id_count);
#endif
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
    const int free_count = *id_list_free_pos - old_alloc;
    if (free_count < GTAP_DETAIL_TASK_ID_POOL_MIN_FREE) {
        GTAP_DETAIL_RECORD_TASK_ID_POOL_LOW_HEADROOM(
            task_id, free_count, GTAP_DETAIL_TASK_ID_POOL_MIN_FREE);
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
    int* id_storage = nullptr;
};

// bytes_per_task is the task record, its header, and one task-id pool slot.
inline bool task_pool_fixed_and_per_task_bytes(
    const launch_config& config,
    size_t* fixed_bytes,
    size_t* bytes_per_task
) {
    (void)config;
    size_t header_and_record = 0;
    size_t per_task = 0;
    if (!checked_add(sizeof(TaskHeader), __gtap_task_data_stride, &header_and_record) ||
        !checked_add(header_and_record, sizeof(int), &per_task)) {
        return false;
    }
    *fixed_bytes = 0;
    *bytes_per_task = per_task;
    return true;
}

// Total bytes for one task-pool component across all task slots.
inline size_t task_pool_component_bytes(
    const launch_config& config,
    size_t bytes_per_task
) {
    size_t bytes = mul_memory_bytes(
        bytes_per_task,
        static_cast<size_t>(config.total_scheduling_units));
    return mul_memory_bytes(
        bytes, static_cast<size_t>(config.tasks_per_scheduling_unit));
}

inline size_t task_pool_allocation_bytes(const launch_config& config) {
    size_t bytes = task_pool_component_bytes(config, sizeof(TaskHeader));
    bytes = add_memory_bytes(
        bytes, task_pool_component_bytes(config, __gtap_task_data_stride));
    return add_memory_bytes(
        bytes, task_pool_component_bytes(config, sizeof(int)));
}

// Allocates the pool and starts the async clears. Symbols are published later.
inline cudaError_t stage_task_pool(
    const launch_config& config,
    cudaStream_t stream,
    task_pool_buffers* buffers,
    host_api_error& error
) {
    const size_t header_bytes = task_pool_component_bytes(config, sizeof(TaskHeader));
    const size_t task_data_bytes = task_pool_component_bytes(config, __gtap_task_data_stride);
    const size_t id_storage_bytes = task_pool_component_bytes(config, sizeof(int));
    if (header_bytes == SIZE_MAX || task_data_bytes == SIZE_MAX ||
        id_storage_bytes == SIZE_MAX) {
        return error.fail(cudaErrorInvalidValue, "task storage size overflowed");
    }

    cudaError_t status = record_error_if_failed(
        error,
        alloc_device(&buffers->headers, header_bytes),
        "failed to allocate task storage (allocation size: %zu bytes)",
        header_bytes);
    if (status != cudaSuccess) return status;
    status = record_error_if_failed(
        error,
        cudaMemsetAsync(buffers->headers, 0, header_bytes, stream),
        "failed to initialize task storage");
    if (status != cudaSuccess) return status;

    status = record_error_if_failed(
        error,
        alloc_device(&buffers->task_data, task_data_bytes),
        "failed to allocate task storage (allocation size: %zu bytes)",
        task_data_bytes);
    if (status != cudaSuccess) return status;
    status = record_error_if_failed(
        error,
        cudaMemsetAsync(buffers->task_data, 0, task_data_bytes, stream),
        "failed to initialize task storage");
    if (status != cudaSuccess) return status;

    status = record_error_if_failed(
        error,
        alloc_device(&buffers->id_storage, id_storage_bytes),
        "failed to allocate task storage (allocation size: %zu bytes)",
        id_storage_bytes);
    if (status != cudaSuccess) return status;
    return record_error_if_failed(
        error,
        cudaMemsetAsync(buffers->id_storage, 0, id_storage_bytes, stream),
        "failed to initialize task storage");
}

inline cudaError_t publish_task_pool(const task_pool_buffers& buffers) {
    cudaError_t status = cudaMemcpyToSymbol(
        d_task_headers, &buffers.headers, sizeof(TaskHeader*));
    if (status != cudaSuccess) return status;
    status = cudaMemcpyToSymbol(
        d_task_data_bytes, &buffers.task_data, sizeof(char*));
    if (status != cudaSuccess) return status;
    status = publish_task_data_stride();
    if (status != cudaSuccess) return status;
    return cudaMemcpyToSymbol(
        d_task_id_storage, &buffers.id_storage, sizeof(int*));
}

inline cudaError_t clear_task_pool(
    const launch_config& config,
    cudaStream_t stream
) {
    const size_t header_bytes = task_pool_component_bytes(config, sizeof(TaskHeader));
    const size_t task_data_bytes = task_pool_component_bytes(config, __gtap_task_data_stride);
    const size_t id_storage_bytes = task_pool_component_bytes(config, sizeof(int));
    if (header_bytes == SIZE_MAX || task_data_bytes == SIZE_MAX ||
        id_storage_bytes == SIZE_MAX) {
        return cudaErrorInvalidValue;
    }

    TaskHeader* headers = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &headers, d_task_headers, sizeof(TaskHeader*)));
    char* task_data = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &task_data, d_task_data_bytes, sizeof(char*)));
    int* id_storage = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &id_storage, d_task_id_storage, sizeof(int*)));

    if (headers != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            headers, 0, header_bytes, stream));
    }
    if (task_data != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            task_data, 0, task_data_bytes, stream));
    }
    if (id_storage != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            id_storage, 0, id_storage_bytes, stream));
    }
    return cudaSuccess;
}

inline cudaError_t free_task_pool() {
    cudaError_t result = cudaSuccess;

    TaskHeader* headers = nullptr;
    cudaError_t headers_read_status = cudaMemcpyFromSymbol(
        &headers, d_task_headers, sizeof(TaskHeader*));
    keep_first_error(headers_read_status, &result);
    char* task_data = nullptr;
    cudaError_t task_data_read_status = cudaMemcpyFromSymbol(
        &task_data, d_task_data_bytes, sizeof(char*));
    keep_first_error(task_data_read_status, &result);
    int* id_storage = nullptr;
    cudaError_t id_storage_read_status = cudaMemcpyFromSymbol(
        &id_storage, d_task_id_storage, sizeof(int*));
    keep_first_error(id_storage_read_status, &result);

    if (headers_read_status == cudaSuccess && headers != nullptr) {
        keep_first_error(cudaFree(headers), &result);
    }
    if (task_data_read_status == cudaSuccess && task_data != nullptr) {
        keep_first_error(cudaFree(task_data), &result);
    }
    if (id_storage_read_status == cudaSuccess && id_storage != nullptr) {
        keep_first_error(cudaFree(id_storage), &result);
    }
    return result;
}

inline void release_staged_task_pool(task_pool_buffers* buffers) {
    free_device(buffers->headers);
    free_device(buffers->task_data);
    free_device(buffers->id_storage);
    TaskHeader* headers = nullptr;
    char* task_data = nullptr;
    int* id_storage = nullptr;
    size_t stride = 0;
    cudaMemcpyToSymbol(d_task_headers, &headers, sizeof(TaskHeader*));
    cudaMemcpyToSymbol(d_task_data_bytes, &task_data, sizeof(task_data));
    cudaMemcpyToSymbol(d_task_data_stride, &stride, sizeof(stride));
    cudaMemcpyToSymbol(d_task_id_storage, &id_storage, sizeof(id_storage));
}

}  // namespace gtap::detail::thread

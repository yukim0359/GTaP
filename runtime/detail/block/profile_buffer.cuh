#pragma once

#include "../common/cuda_primitives.cuh"
#include "../common/device_memory.cuh"
#include "../common/init_error.cuh"
#include "../common/profile_buffer.cuh"
#include "../common/runtime_config.cuh"
#include "../common/runtime_error.cuh"

namespace gtap::detail::block {

using namespace gtap::detail;

#ifdef GTAP_ENABLE_PROFILING
__constant__ long long* working_time;                    // long long[num_blocks * profile_timestamp_capacity]
__constant__ unsigned long long* profile_dropped_events; // unsigned long long[num_blocks]
__constant__ int* task_pool_peak_usage;                  // int[num_blocks]
__constant__ int* task_id_allocation_count;              // int[num_blocks]

// Stores the peak count of pool entries not yet returned to this block.
__device__ __forceinline__ void record_unreturned_id_count(
    int block_id,
    int unreturned_id_count
) {
    if (unreturned_id_count > load_relaxed(&task_pool_peak_usage[block_id])) {
        atomicMax(&task_pool_peak_usage[block_id], unreturned_id_count);
    }
}

__device__ __forceinline__ void record_task_id_allocation_count(
    int block_id,
    int allocation_count
) {
    task_id_allocation_count[block_id] = allocation_count;
}

cudaError_t get_working_time_data(long long* host_working_time) {
    long long* ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, working_time, sizeof(ptr)));
    return cudaMemcpy(
        host_working_time, ptr,
        sizeof(long long) * h_launch_config.grid_size *
            profile_timestamp_capacity(),
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
        sizeof(unsigned long long) * h_launch_config.grid_size,
        cudaMemcpyDeviceToHost);
}

cudaError_t get_block_task_pool_peak_usage(int* host_peak_usage) {
    int* ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &ptr, task_pool_peak_usage, sizeof(ptr)));
    return cudaMemcpy(
        host_peak_usage, ptr,
        sizeof(int) * h_launch_config.grid_size,
        cudaMemcpyDeviceToHost);
}

cudaError_t get_block_task_id_allocation_count(int* host_task_id_allocation_count) {
    int* ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &ptr, task_id_allocation_count, sizeof(ptr)));
    return cudaMemcpy(
        host_task_id_allocation_count, ptr,
        sizeof(int) * h_launch_config.grid_size,
        cudaMemcpyDeviceToHost);
}

cudaError_t get_block_working_time_data(
    int block_id, long long* host_working_time, int max_samples
) {
    long long* ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, working_time, sizeof(ptr)));
    const int count = max_samples < profile_timestamp_capacity() ? max_samples : profile_timestamp_capacity();
    return cudaMemcpy(
        host_working_time,
        ptr + static_cast<size_t>(block_id) * profile_timestamp_capacity(),
        sizeof(long long) * count, cudaMemcpyDeviceToHost);
}

__global__ void get_block_working_time_counts(int* counts) {
    if (threadIdx.x == 0) {
        int count = 0;
        for (int i = 0; i < profile_timestamp_capacity(); i++) {
            if (working_time[blockIdx.x * profile_timestamp_capacity() + i] > 0) {
                count++;
            }
        }
        counts[blockIdx.x] = count;
    }
}
#endif

inline size_t profile_buffer_allocation_bytes(const launch_config& config) {
#ifdef GTAP_ENABLE_PROFILING
    const size_t scheduling_units = static_cast<size_t>(config.total_scheduling_units);
    const size_t timestamps = static_cast<size_t>(config.profile_interval_capacity) * 2;
    return sizeof(long long) * scheduling_units * timestamps
        + sizeof(unsigned long long) * scheduling_units
        + 2 * sizeof(int) * scheduling_units;
#else
    (void)config;
    return 0;
#endif
}

struct profile_buffers {
    long long* working_time = nullptr;
    unsigned long long* dropped_events = nullptr;
    int* peak_usage = nullptr;
    int* task_id_allocation_count = nullptr;
};

inline cudaError_t stage_profile_buffers(
    const launch_config& config,
    cudaStream_t stream,
    profile_buffers* buffers,
    init_error& error
) {
#ifdef GTAP_ENABLE_PROFILING
    const size_t scheduling_units = static_cast<size_t>(config.total_scheduling_units);
    const size_t timestamps = static_cast<size_t>(config.profile_interval_capacity) * 2;
    const size_t profile_bytes = sizeof(long long) * scheduling_units * timestamps;
    const size_t dropped_bytes = sizeof(unsigned long long) * scheduling_units;
    const size_t unit_int_bytes = sizeof(int) * scheduling_units;
    cudaError_t status = check_init(
        error,
        alloc_device(&buffers->working_time, profile_bytes),
        "failed to allocate profile buffers (allocation size: %zu bytes)",
        profile_bytes);
    if (status != cudaSuccess) return status;
    status = check_init(
        error,
        alloc_device(&buffers->dropped_events, dropped_bytes),
        "failed to allocate profile buffers (allocation size: %zu bytes)",
        dropped_bytes);
    if (status != cudaSuccess) return status;
    status = check_init(
        error,
        alloc_device(&buffers->peak_usage, unit_int_bytes),
        "failed to allocate profile buffers (allocation size: %zu bytes)",
        unit_int_bytes);
    if (status != cudaSuccess) return status;
    status = check_init(
        error,
        alloc_device(&buffers->task_id_allocation_count, unit_int_bytes),
        "failed to allocate profile buffers (allocation size: %zu bytes)",
        unit_int_bytes);
    if (status != cudaSuccess) return status;
    status = check_init(
        error,
        cudaMemsetAsync(buffers->working_time, 0, profile_bytes, stream),
        "failed to initialize profile buffers");
    if (status != cudaSuccess) return status;
    status = check_init(
        error,
        cudaMemsetAsync(buffers->dropped_events, 0, dropped_bytes, stream),
        "failed to initialize profile buffers");
    if (status != cudaSuccess) return status;
    status = check_init(
        error,
        cudaMemsetAsync(buffers->peak_usage, 0, unit_int_bytes, stream),
        "failed to initialize profile buffers");
    if (status != cudaSuccess) return status;
    return check_init(
        error,
        cudaMemsetAsync(buffers->task_id_allocation_count, 0, unit_int_bytes, stream),
        "failed to initialize profile buffers");
#else
    (void)config;
    (void)stream;
    (void)buffers;
    (void)error;
    return cudaSuccess;
#endif
}

inline cudaError_t publish_profile_buffers(const profile_buffers& buffers) {
#ifdef GTAP_ENABLE_PROFILING
    cudaError_t status = cudaMemcpyToSymbol(
        working_time, &buffers.working_time, sizeof(buffers.working_time));
    if (status != cudaSuccess) return status;
    status = cudaMemcpyToSymbol(
        profile_dropped_events, &buffers.dropped_events,
        sizeof(buffers.dropped_events));
    if (status != cudaSuccess) return status;
    status = cudaMemcpyToSymbol(
        task_pool_peak_usage, &buffers.peak_usage, sizeof(buffers.peak_usage));
    if (status != cudaSuccess) return status;
    return cudaMemcpyToSymbol(
        task_id_allocation_count, &buffers.task_id_allocation_count,
        sizeof(buffers.task_id_allocation_count));
#else
    (void)buffers;
    return cudaSuccess;
#endif
}

inline cudaError_t clear_profile_buffers(
    const launch_config& config,
    cudaStream_t stream
) {
#ifdef GTAP_ENABLE_PROFILING
    const size_t scheduling_units = static_cast<size_t>(config.total_scheduling_units);
    const size_t timestamps = static_cast<size_t>(config.profile_interval_capacity) * 2;
    long long* working_time_ptr = nullptr;
    unsigned long long* profile_dropped_events_ptr = nullptr;
    int* task_pool_peak_usage_ptr = nullptr;
    int* task_id_allocation_count_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &working_time_ptr, working_time, sizeof(working_time_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &profile_dropped_events_ptr, profile_dropped_events,
        sizeof(profile_dropped_events_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &task_pool_peak_usage_ptr, task_pool_peak_usage,
        sizeof(task_pool_peak_usage_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &task_id_allocation_count_ptr, task_id_allocation_count,
        sizeof(task_id_allocation_count_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        working_time_ptr, 0,
        sizeof(long long) * scheduling_units * timestamps,
        stream));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        profile_dropped_events_ptr, 0,
        sizeof(unsigned long long) * scheduling_units, stream));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        task_pool_peak_usage_ptr, 0, sizeof(int) * scheduling_units, stream));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        task_id_allocation_count_ptr, 0, sizeof(int) * scheduling_units, stream));
#else
    (void)config;
    (void)stream;
#endif
    return cudaSuccess;
}

inline cudaError_t free_profile_buffers() {
#ifdef GTAP_ENABLE_PROFILING
    long long* working_time_ptr = nullptr;
    unsigned long long* profile_dropped_events_ptr = nullptr;
    int* task_pool_peak_usage_ptr = nullptr;
    int* task_id_allocation_count_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &working_time_ptr, working_time, sizeof(working_time_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &profile_dropped_events_ptr, profile_dropped_events,
        sizeof(profile_dropped_events_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &task_pool_peak_usage_ptr, task_pool_peak_usage,
        sizeof(task_pool_peak_usage_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &task_id_allocation_count_ptr, task_id_allocation_count,
        sizeof(task_id_allocation_count_ptr)));
    if (working_time_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(working_time_ptr));
    }
    if (profile_dropped_events_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(profile_dropped_events_ptr));
    }
    if (task_pool_peak_usage_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(task_pool_peak_usage_ptr));
    }
    if (task_id_allocation_count_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(task_id_allocation_count_ptr));
    }
#endif
    return cudaSuccess;
}

inline void release_staged_profile_buffers(profile_buffers* buffers) {
    free_device(buffers->working_time);
    free_device(buffers->dropped_events);
    free_device(buffers->peak_usage);
    free_device(buffers->task_id_allocation_count);
#ifdef GTAP_ENABLE_PROFILING
    long long* working_time_ptr = nullptr;
    unsigned long long* dropped_events = nullptr;
    int* peak_usage = nullptr;
    int* allocation_count = nullptr;
    cudaMemcpyToSymbol(working_time, &working_time_ptr, sizeof(working_time_ptr));
    cudaMemcpyToSymbol(
        profile_dropped_events, &dropped_events, sizeof(dropped_events));
    cudaMemcpyToSymbol(task_pool_peak_usage, &peak_usage, sizeof(peak_usage));
    cudaMemcpyToSymbol(task_id_allocation_count, &allocation_count, sizeof(allocation_count));
#endif
}

}  // namespace gtap::detail::block

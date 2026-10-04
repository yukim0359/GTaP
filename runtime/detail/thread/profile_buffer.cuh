#pragma once

#include "../common/profile_buffer.cuh"
#include "../common/runtime_config.cuh"
#include "../common/runtime_error.cuh"

namespace gtap::detail::thread {

using namespace gtap::detail;

#ifdef GTAP_ENABLE_PROFILING
__constant__ long long* working_time;                    // long long[num_warps * profile_timestamp_capacity]
__constant__ int* tasks_processed_count;                 // int[num_warps * profile_timestamp_capacity]
__constant__ unsigned long long* profile_dropped_events; // unsigned long long[num_warps]

cudaError_t get_warp_working_time_data(long long* host_working_time) {
    long long* ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, working_time, sizeof(ptr)));
    return cudaMemcpy(
        host_working_time, ptr,
        sizeof(long long) * h_launch_config.total_scheduling_units * profile_timestamp_capacity(),
        cudaMemcpyDeviceToHost);
}

cudaError_t get_warp_tasks_processed_count_data(int* host_counts) {
    int* ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, tasks_processed_count, sizeof(ptr)));
    return cudaMemcpy(
        host_counts, ptr,
        sizeof(int) * h_launch_config.total_scheduling_units * profile_timestamp_capacity(),
        cudaMemcpyDeviceToHost);
}

cudaError_t get_warp_profile_dropped_events_data(
    unsigned long long* host_counts
) {
    unsigned long long* ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &ptr, profile_dropped_events, sizeof(ptr)));
    return cudaMemcpy(
        host_counts, ptr,
        sizeof(unsigned long long) * h_launch_config.total_scheduling_units,
        cudaMemcpyDeviceToHost);
}

__global__ void get_warp_working_time_counts(int* counts) {
    if (threadIdx.x == 0) {
        int wid = blockIdx.x;
        int count = 0;
        for (int i = 0; i < profile_timestamp_capacity(); i++) {
            if (working_time[wid * profile_timestamp_capacity() + i] > 0) count++;
        }
        counts[wid] = count;
    }
}
#endif

inline size_t profile_buffer_allocation_bytes(size_t scheduling_units) {
#ifdef GTAP_ENABLE_PROFILING
    return scheduling_units * static_cast<size_t>(profile_timestamp_capacity()) *
            (sizeof(long long) + sizeof(int))
        + scheduling_units * sizeof(unsigned long long);
#else
    (void)scheduling_units;
    return 0;
#endif
}

struct profile_buffers {
    long long* working_time = nullptr;
    int* tasks_processed_count = nullptr;
    unsigned long long* dropped_events = nullptr;
};

inline cudaError_t stage_profile_buffers(
    size_t scheduling_units,
    cudaStream_t working_time_stream,
    cudaStream_t task_count_stream,
    cudaStream_t dropped_stream,
    profile_buffers* buffers
) {
#ifdef GTAP_ENABLE_PROFILING
    const size_t profile_long_bytes =
        sizeof(long long) * scheduling_units * profile_timestamp_capacity();
    const size_t profile_int_bytes =
        sizeof(int) * scheduling_units * profile_timestamp_capacity();
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&buffers->working_time), profile_long_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&buffers->tasks_processed_count),
        profile_int_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&buffers->dropped_events),
        sizeof(unsigned long long) * scheduling_units));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->working_time, 0, profile_long_bytes, working_time_stream));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->tasks_processed_count, 0, profile_int_bytes,
        task_count_stream));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->dropped_events, 0,
        sizeof(unsigned long long) * scheduling_units, dropped_stream));
#else
    (void)scheduling_units;
    (void)working_time_stream;
    (void)task_count_stream;
    (void)dropped_stream;
    (void)buffers;
#endif
    return cudaSuccess;
}

inline cudaError_t publish_profile_buffers(const profile_buffers& buffers) {
#ifdef GTAP_ENABLE_PROFILING
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        working_time, &buffers.working_time, sizeof(buffers.working_time)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        tasks_processed_count, &buffers.tasks_processed_count,
        sizeof(buffers.tasks_processed_count)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        profile_dropped_events, &buffers.dropped_events,
        sizeof(buffers.dropped_events)));
#else
    (void)buffers;
#endif
    return cudaSuccess;
}

inline cudaError_t clear_profile_buffers(
    size_t scheduling_units,
    cudaStream_t working_time_stream,
    cudaStream_t task_count_stream,
    cudaStream_t dropped_stream
) {
#ifdef GTAP_ENABLE_PROFILING
    long long* working_time_ptr = nullptr;
    int* tasks_processed_count_ptr = nullptr;
    unsigned long long* profile_dropped_events_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &working_time_ptr, working_time, sizeof(working_time_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &tasks_processed_count_ptr, tasks_processed_count,
        sizeof(tasks_processed_count_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &profile_dropped_events_ptr, profile_dropped_events,
        sizeof(profile_dropped_events_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        working_time_ptr, 0,
        sizeof(long long) * scheduling_units * profile_timestamp_capacity(),
        working_time_stream));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        tasks_processed_count_ptr, 0,
        sizeof(int) * scheduling_units * profile_timestamp_capacity(),
        task_count_stream));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        profile_dropped_events_ptr, 0,
        sizeof(unsigned long long) * scheduling_units, dropped_stream));
#else
    (void)scheduling_units;
    (void)working_time_stream;
    (void)task_count_stream;
    (void)dropped_stream;
#endif
    return cudaSuccess;
}

inline cudaError_t free_profile_buffers() {
#ifdef GTAP_ENABLE_PROFILING
    long long* working_time_ptr = nullptr;
    int* tasks_processed_count_ptr = nullptr;
    unsigned long long* profile_dropped_events_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &working_time_ptr, working_time, sizeof(working_time_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &tasks_processed_count_ptr, tasks_processed_count,
        sizeof(tasks_processed_count_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &profile_dropped_events_ptr, profile_dropped_events,
        sizeof(profile_dropped_events_ptr)));
    if (working_time_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(working_time_ptr));
    }
    if (tasks_processed_count_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(tasks_processed_count_ptr));
    }
    if (profile_dropped_events_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(profile_dropped_events_ptr));
    }
#endif
    return cudaSuccess;
}

}  // namespace gtap::detail::thread

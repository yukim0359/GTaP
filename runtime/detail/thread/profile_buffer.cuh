#pragma once

#include "../common/runtime.cuh"

namespace gtap::detail::thread {

using namespace gtap::detail;

#ifdef GTAP_ENABLE_PROFILING
__constant__ long long* working_time;
__constant__ int* tasks_processed_count;
__constant__ unsigned long long* profile_dropped_events;
#endif

inline size_t profile_buffer_allocation_bytes(size_t workers) {
#ifdef GTAP_ENABLE_PROFILING
    return workers * static_cast<size_t>(profile_capacity()) *
            (sizeof(long long) + sizeof(int))
        + workers * sizeof(unsigned long long);
#else
    (void)workers;
    return 0;
#endif
}

inline cudaError_t allocate_profile_buffers(
    size_t workers,
    cudaStream_t working_time_stream,
    cudaStream_t task_count_stream,
    cudaStream_t dropped_stream
) {
#ifdef GTAP_ENABLE_PROFILING
    long long* working_time_ptr = nullptr;
    int* tasks_processed_count_ptr = nullptr;
    unsigned long long* profile_dropped_events_ptr = nullptr;
    const size_t profile_long_bytes =
        sizeof(long long) * workers * profile_capacity();
    const size_t profile_int_bytes =
        sizeof(int) * workers * profile_capacity();
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&working_time_ptr), profile_long_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&tasks_processed_count_ptr), profile_int_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&profile_dropped_events_ptr),
        sizeof(unsigned long long) * workers));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        working_time, &working_time_ptr, sizeof(working_time_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        tasks_processed_count, &tasks_processed_count_ptr,
        sizeof(tasks_processed_count_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        profile_dropped_events, &profile_dropped_events_ptr,
        sizeof(profile_dropped_events_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        working_time_ptr, 0, profile_long_bytes, working_time_stream));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        tasks_processed_count_ptr, 0, profile_int_bytes, task_count_stream));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        profile_dropped_events_ptr, 0,
        sizeof(unsigned long long) * workers, dropped_stream));
    GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(dropped_stream));
    GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(working_time_stream));
    GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(task_count_stream));
#else
    (void)workers;
    (void)working_time_stream;
    (void)task_count_stream;
    (void)dropped_stream;
#endif
    return cudaSuccess;
}

inline cudaError_t clear_profile_buffers(
    size_t workers,
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
        sizeof(long long) * workers * profile_capacity(),
        working_time_stream));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        tasks_processed_count_ptr, 0,
        sizeof(int) * workers * profile_capacity(),
        task_count_stream));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        profile_dropped_events_ptr, 0,
        sizeof(unsigned long long) * workers, dropped_stream));
#else
    (void)workers;
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

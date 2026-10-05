#pragma once

#include "../common/device_memory.cuh"
#include "../common/profile_buffer.cuh"
#include "../common/runtime_config.cuh"
#include "../common/runtime_error.cuh"

namespace gtap::detail::block {

using namespace gtap::detail;

#ifdef GTAP_ENABLE_PROFILING
__constant__ long long* working_time;                    // long long[num_blocks * profile_timestamp_capacity]
__constant__ unsigned long long* profile_dropped_events; // unsigned long long[num_blocks]

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

inline size_t profile_buffer_allocation_bytes(size_t scheduling_units) {
#ifdef GTAP_ENABLE_PROFILING
    return sizeof(long long) * scheduling_units * profile_timestamp_capacity()
        + sizeof(unsigned long long) * scheduling_units;
#else
    (void)scheduling_units;
    return 0;
#endif
}

struct profile_buffers {
    long long* working_time = nullptr;
    unsigned long long* dropped_events = nullptr;
};

inline cudaError_t stage_profile_buffers(
    size_t scheduling_units,
    cudaStream_t stream,
    profile_buffers* buffers
) {
#ifdef GTAP_ENABLE_PROFILING
    const size_t profile_bytes =
        sizeof(long long) * scheduling_units * profile_timestamp_capacity();
    GTAP_DETAIL_CUDA_TRY(alloc_device(&buffers->working_time, profile_bytes));
    GTAP_DETAIL_CUDA_TRY(alloc_device(
        &buffers->dropped_events, sizeof(unsigned long long) * scheduling_units));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->working_time, 0, profile_bytes, stream));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->dropped_events, 0,
        sizeof(unsigned long long) * scheduling_units, stream));
#else
    (void)scheduling_units;
    (void)stream;
    (void)buffers;
#endif
    return cudaSuccess;
}

inline cudaError_t publish_profile_buffers(const profile_buffers& buffers) {
#ifdef GTAP_ENABLE_PROFILING
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        working_time, &buffers.working_time, sizeof(buffers.working_time)));
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
    cudaStream_t stream
) {
#ifdef GTAP_ENABLE_PROFILING
    long long* working_time_ptr = nullptr;
    unsigned long long* profile_dropped_events_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &working_time_ptr, working_time, sizeof(working_time_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &profile_dropped_events_ptr, profile_dropped_events,
        sizeof(profile_dropped_events_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        working_time_ptr, 0,
        sizeof(long long) * scheduling_units * profile_timestamp_capacity(),
        stream));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        profile_dropped_events_ptr, 0,
        sizeof(unsigned long long) * scheduling_units, stream));
#else
    (void)scheduling_units;
    (void)stream;
#endif
    return cudaSuccess;
}

inline cudaError_t free_profile_buffers() {
#ifdef GTAP_ENABLE_PROFILING
    long long* working_time_ptr = nullptr;
    unsigned long long* profile_dropped_events_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &working_time_ptr, working_time, sizeof(working_time_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &profile_dropped_events_ptr, profile_dropped_events,
        sizeof(profile_dropped_events_ptr)));
    if (working_time_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(working_time_ptr));
    }
    if (profile_dropped_events_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(profile_dropped_events_ptr));
    }
#endif
    return cudaSuccess;
}

inline void release_staged_profile_buffers(profile_buffers* buffers) {
    free_device(buffers->working_time);
    free_device(buffers->dropped_events);
#ifdef GTAP_ENABLE_PROFILING
    long long* working_time_ptr = nullptr;
    unsigned long long* dropped_events = nullptr;
    cudaMemcpyToSymbol(working_time, &working_time_ptr, sizeof(working_time_ptr));
    cudaMemcpyToSymbol(
        profile_dropped_events, &dropped_events, sizeof(dropped_events));
#endif
}

}  // namespace gtap::detail::block

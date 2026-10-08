#pragma once

#include <climits>

#include "../common/cuda_primitives.cuh"
#include "../common/host_api.cuh"

#include "lifecycle.cuh"

struct gtap_thread_config {
    int grid_size = 4096;
    int block_size = 32;
    // 0 means unset. One set value selects the slot count. Both set takes
    // the minimum. Both unset uses 10000 slots per warp.
    int max_tasks_per_warp = 0;
    size_t max_task_memory_bytes = 0;
    int num_queues = 1;
    int profile_capacity_per_warp = 15000;
    cudaStream_t stream = nullptr;
};

inline cudaError_t gtap_validate_config(const gtap_thread_config& config) {
    if (config.grid_size <= 0) {
        return cudaErrorInvalidConfiguration;
    }
    if (config.block_size <= 0 ||
        config.block_size > GTAP_MAX_THREADS_PER_BLOCK ||
        config.block_size % gtap::detail::warp_size != 0) {
        return cudaErrorInvalidConfiguration;
    }
    if (config.num_queues <= 0 || config.max_tasks_per_warp < 0 ||
        (config.max_tasks_per_warp > 0 &&
         config.max_tasks_per_warp % config.num_queues != 0)) {
        return cudaErrorInvalidValue;
    }
#ifdef GTAP_ENABLE_PROFILING
    if (config.profile_capacity_per_warp <= 0 ||
        config.profile_capacity_per_warp > INT_MAX / 2) {
        return cudaErrorInvalidValue;
    }
#endif
    const int queue_capacity = config.max_tasks_per_warp > 0
        ? config.max_tasks_per_warp / config.num_queues
        : 0;
    gtap::detail::launch_config probe{
        config.grid_size,
        config.block_size,
        config.block_size / gtap::detail::warp_size,
        config.grid_size * (config.block_size / gtap::detail::warp_size),
        config.max_tasks_per_warp,
        config.num_queues,
        queue_capacity,
        config.profile_capacity_per_warp,
        0
    };
    if (gtap::detail::thread::tasks_within_budget(
            probe, config.max_task_memory_bytes) <= 0) {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

inline cudaError_t gtap_initialize(
    const gtap_thread_config& config,
    size_t* device_bytes_allocated = nullptr
);

inline cudaError_t gtap_initialize(size_t* device_bytes_allocated = nullptr) {
    gtap_thread_config config;
    return gtap_initialize(config, device_bytes_allocated);
}

inline cudaError_t gtap_initialize(
    const gtap_thread_config& config,
    size_t* device_bytes_allocated
) {
    cudaError_t validation = gtap_validate_config(config);
    if (validation != cudaSuccess) return validation;
    if (gtap::detail::h_runtime_initialized) return cudaErrorInitializationError;

    gtap::detail::launch_config launch_config{
        config.grid_size,
        config.block_size,
        config.block_size / gtap::detail::warp_size,
        config.grid_size * (config.block_size / gtap::detail::warp_size),
        config.max_tasks_per_warp,
        config.num_queues,
        config.max_tasks_per_warp > 0
            ? config.max_tasks_per_warp / config.num_queues
            : 0,
        config.profile_capacity_per_warp,
        gtap::detail::thread::shared_layout_for(
            config.block_size / gtap::detail::warp_size,
            config.num_queues,
            gtap::detail::thread::include_queue_tails).bytes
    };
    const int tasks = gtap::detail::thread::tasks_within_budget(
        launch_config, config.max_task_memory_bytes);
    if (tasks <= 0) return cudaErrorInvalidValue;
    launch_config.tasks_per_scheduling_unit = tasks;
    launch_config.queue_capacity = tasks / config.num_queues;
    GTAP_DETAIL_CUDA_TRY(gtap::detail::publish_launch_config(launch_config));
    gtap::detail::h_stream = config.stream;
    cudaError_t err = gtap::detail::thread::initialize_runtime();
    if (err != cudaSuccess) {
        gtap::detail::h_stream = nullptr;
        return err;
    }
    gtap::detail::h_runtime_initialized = true;
    gtap::detail::h_runtime_error_reported = false;
    if (device_bytes_allocated != nullptr) {
        *device_bytes_allocated =
            gtap::detail::thread::runtime_device_allocation_bytes();
    }
    return err;
}

inline cudaError_t gtap_finalize() {
    cudaError_t err = gtap::detail::thread::finalize_runtime();
    if (err == cudaSuccess) {
        gtap::detail::h_runtime_initialized = false;
        gtap::detail::h_stream = nullptr;
    }
    return err;
}

inline cudaError_t gtap_reset() {
    return gtap::detail::thread::reset_runtime();
}

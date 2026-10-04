#pragma once

// Public thread host API. Include after the backend scheduler and lifecycle.

#include <climits>

struct gtap_thread_config {
    int grid_size = 4096;
    int block_size = 32;
    int max_tasks_per_warp = 10000;
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
    if (config.max_tasks_per_warp <= 0 ||
        config.num_queues <= 0 ||
        config.max_tasks_per_warp % config.num_queues != 0) {
        return cudaErrorInvalidValue;
    }
#ifdef GTAP_ENABLE_PROFILING
    if (config.profile_capacity_per_warp <= 0 ||
        config.profile_capacity_per_warp > INT_MAX / 2) {
        return cudaErrorInvalidValue;
    }
#endif
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
    if (gtap::detail::initialized_flag()) return cudaErrorInitializationError;

    gtap::detail::launch_config launch_config{
        config.grid_size,
        config.block_size,
        config.block_size / gtap::detail::warp_size,
        config.grid_size * (config.block_size / gtap::detail::warp_size),
        config.max_tasks_per_warp,
        config.num_queues,
        config.max_tasks_per_warp / config.num_queues,
        config.profile_capacity_per_warp,
        gtap::detail::thread::shared_layout_for(
            config.block_size / gtap::detail::warp_size,
            config.num_queues,
            gtap::detail::thread::include_queue_tails).bytes
    };
    GTAP_DETAIL_CUDA_TRY(gtap::detail::publish_launch_config(launch_config));
    gtap::detail::stored_stream() = config.stream;
    cudaError_t err = gtap::detail::thread::initialize_runtime();
    if (err == cudaSuccess) {
        gtap::detail::initialized_flag() = true;
        if (device_bytes_allocated != nullptr) {
            *device_bytes_allocated =
                gtap::detail::thread::runtime_device_allocation_bytes();
        }
    }
    return err;
}

inline cudaError_t gtap_finalize() {
    cudaError_t err = gtap::detail::thread::finalize_runtime();
    if (err == cudaSuccess) {
        gtap::detail::initialized_flag() = false;
        gtap::detail::stored_stream() = nullptr;
    }
    return err;
}

inline cudaError_t gtap_reset() {
    return gtap::detail::thread::reset_runtime();
}

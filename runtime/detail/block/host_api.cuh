#pragma once

#include <climits>

#include "../common/cuda_primitives.cuh"
#include "../common/host_api.cuh"

#include "lifecycle.cuh"

struct gtap_block_config {
    int grid_size = 1024;
    int block_size = 256;
    int max_tasks_per_block = 10000;
    int profile_capacity_per_block = 15000;
    size_t dynamic_shared_bytes = 0;
    cudaStream_t stream = nullptr;
};

inline cudaError_t gtap_validate_config(const gtap_block_config& config) {
    if (config.grid_size <= 0 ||
        config.block_size <= 0 ||
        config.block_size > GTAP_MAX_THREADS_PER_BLOCK ||
        config.block_size % gtap::detail::warp_size != 0) {
        return cudaErrorInvalidConfiguration;
    }
    if (config.max_tasks_per_block <= 0) {
        return cudaErrorInvalidValue;
    }
#ifdef GTAP_ENABLE_PROFILING
    if (config.profile_capacity_per_block <= 0 ||
        config.profile_capacity_per_block > INT_MAX / 2) {
        return cudaErrorInvalidValue;
    }
#endif
    return cudaSuccess;
}

inline cudaError_t gtap_initialize(
    const gtap_block_config& config,
    size_t* device_bytes_allocated = nullptr
);

inline cudaError_t gtap_initialize(size_t* device_bytes_allocated = nullptr) {
    gtap_block_config config;
    return gtap_initialize(config, device_bytes_allocated);
}

inline cudaError_t gtap_initialize(
    const gtap_block_config& config,
    size_t* device_bytes_allocated
) {
    cudaError_t validation = gtap_validate_config(config);
    if (validation != cudaSuccess) return validation;
    if (gtap::detail::initialized_flag()) return cudaErrorInitializationError;

    gtap::detail::launch_config launch_config{
        config.grid_size,
        config.block_size,
        config.block_size / gtap::detail::warp_size,
        config.grid_size,
        config.max_tasks_per_block,
        1,
        config.max_tasks_per_block,
        config.profile_capacity_per_block,
        config.dynamic_shared_bytes
    };
    GTAP_DETAIL_CUDA_TRY(gtap::detail::publish_launch_config(launch_config));
    gtap::detail::h_stream = config.stream;
    cudaError_t err = gtap::detail::block::initialize_runtime();
    if (err == cudaSuccess) {
        gtap::detail::initialized_flag() = true;
        if (device_bytes_allocated != nullptr) {
            *device_bytes_allocated =
                gtap::detail::block::runtime_device_allocation_bytes();
        }
    }
    return err;
}

inline cudaError_t gtap_finalize() {
    cudaError_t err = gtap::detail::block::finalize_runtime();
    if (err == cudaSuccess) {
        gtap::detail::initialized_flag() = false;
        gtap::detail::h_stream = nullptr;
    }
    return err;
}

inline cudaError_t gtap_reset() {
    return gtap::detail::block::reset_runtime();
}

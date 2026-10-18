#pragma once

#include <climits>

#include "../common/cuda_primitives.cuh"
#include "../common/host_api.cuh"
#include "../common/memory_budget.cuh"

#include "lifecycle.cuh"

struct gtap_block_config {
    int grid_size = 1024;
    int block_size = 256;
    size_t memory_budget_bytes = 0;  // 0: use GTAP_MEMORY_BUDGET, or 5 GiB if unset
    int max_tasks_per_block = 0;     // 0: derive from memory_budget_bytes
    int profile_capacity_per_block = 15000;
    size_t dynamic_shared_bytes = 0;
    cudaStream_t stream = nullptr;
};

namespace gtap::detail::block {

inline cudaError_t validate_config(const gtap_block_config& config) {
    if (config.grid_size <= 0 ||
        config.block_size <= 0 ||
        config.block_size > GTAP_MAX_THREADS_PER_BLOCK ||
        config.block_size % gtap::detail::warp_size != 0) {
        return cudaErrorInvalidConfiguration;
    }
    if (config.max_tasks_per_block < 0) {
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

inline cudaError_t resolve_launch_config(
    const gtap_block_config& config,
    launch_config* out
) {
    cudaError_t status = validate_config(config);
    if (status != cudaSuccess) return status;

    const int warps_per_block = config.block_size / warp_size;
    launch_config resolved{
        config.grid_size,
        config.block_size,
        warps_per_block,
        config.grid_size,
        0,
        1,
        0,
        config.profile_capacity_per_block,
        0
    };

    size_t fixed_bytes = 0;
    size_t bytes_per_task = 0;
    if (!runtime_fixed_and_per_task_bytes(resolved, &fixed_bytes, &bytes_per_task)) {
        return cudaErrorInvalidValue;
    }
    status = resolve_tasks_per_unit(
        config.max_tasks_per_block,
        config.memory_budget_bytes,
        resolved.total_scheduling_units,
        resolved.num_queues,
        fixed_bytes,
        bytes_per_task,
        &resolved.tasks_per_scheduling_unit);
    if (status != cudaSuccess) return status;

    resolved.queue_capacity = resolved.tasks_per_scheduling_unit;
    resolved.dynamic_shared_bytes = config.dynamic_shared_bytes;

    *out = resolved;
    return cudaSuccess;
}

}  // namespace gtap::detail::block

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
    if (gtap::detail::h_runtime_initialized) return cudaErrorInitializationError;

    gtap::detail::launch_config resolved{};
    cudaError_t status = gtap::detail::block::resolve_launch_config(config, &resolved);
    if (status != cudaSuccess) return status;
    GTAP_DETAIL_CUDA_TRY(gtap::detail::publish_launch_config(resolved));

    gtap::detail::h_stream = config.stream;
    cudaError_t err = gtap::detail::block::initialize_runtime();
    if (err != cudaSuccess) {
        gtap::detail::h_stream = nullptr;
        return err;
    }

    gtap::detail::h_runtime_initialized = true;
    gtap::detail::h_runtime_error_reported = false;
    if (device_bytes_allocated != nullptr) {
        *device_bytes_allocated =
            gtap::detail::block::runtime_device_allocation_bytes();
    }
    return cudaSuccess;
}

inline cudaError_t gtap_finalize() {
    cudaError_t err = gtap::detail::block::finalize_runtime();
    if (err == cudaSuccess) {
        gtap::detail::h_runtime_initialized = false;
        gtap::detail::h_stream = nullptr;
    }
    return err;
}

inline cudaError_t gtap_reset() {
    return gtap::detail::block::reset_runtime();
}

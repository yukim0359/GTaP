#pragma once

#include <climits>
#include <cstdio>

#include "../common/cuda_primitives.cuh"
#include "../common/host_api.cuh"
#include "../common/memory_budget.cuh"

#include "lifecycle.cuh"

struct gtap_thread_config {
    int grid_size = 4096;
    int block_size = 32;
    size_t memory_budget_bytes = 0;  // 0: use GTAP_MEMORY_BUDGET, or 5 GiB if unset
    int max_tasks_per_warp = 0;      // 0: derive from memory_budget_bytes
    int num_queues = 1;
    int profile_capacity_per_warp = 15000;
    cudaStream_t stream = nullptr;
};

namespace gtap::detail::thread {

inline cudaError_t validate_config(
    const gtap_thread_config& config,
    init_error& error
) {
    if (config.grid_size <= 0) {
        return error.fail(
            cudaErrorInvalidConfiguration,
            "grid_size=%d must be greater than zero",
            config.grid_size);
    }
    if (config.block_size <= 0 ||
        config.block_size > GTAP_MAX_THREADS_PER_BLOCK) {
        return error.fail(
            cudaErrorInvalidConfiguration,
            "block_size=%d must be in (0, %d]",
            config.block_size,
            GTAP_MAX_THREADS_PER_BLOCK);
    }
    if (config.block_size % gtap::detail::warp_size != 0) {
        return error.fail(
            cudaErrorInvalidConfiguration,
            "block_size=%d must be a multiple of %d",
            config.block_size,
            gtap::detail::warp_size);
    }
    const int warps_per_block = config.block_size / gtap::detail::warp_size;
    if (static_cast<long long>(config.grid_size) * warps_per_block > INT_MAX) {
        return error.fail(
            cudaErrorInvalidConfiguration,
            "grid_size=%d and block_size=%d produce more than %d warps",
            config.grid_size,
            config.block_size,
            INT_MAX);
    }
    if (config.num_queues <= 0) {
        return error.fail(
            cudaErrorInvalidValue,
            "num_queues=%d must be greater than zero",
            config.num_queues);
    }
    if (config.max_tasks_per_warp < 0) {
        return error.fail(
            cudaErrorInvalidValue,
            "max_tasks_per_warp=%d must be greater than or equal to zero",
            config.max_tasks_per_warp);
    }
#ifdef GTAP_ENABLE_PROFILING
    if (config.profile_capacity_per_warp <= 0 ||
        config.profile_capacity_per_warp > INT_MAX / 2) {
        return error.fail(
            cudaErrorInvalidValue,
            "profile_capacity_per_warp=%d must be in (0, %d]",
            config.profile_capacity_per_warp,
            INT_MAX / 2);
    }
#endif
    return cudaSuccess;
}

inline cudaError_t resolve_launch_config(
    const gtap_thread_config& config,
    launch_config* out,
    init_error& error
) {
    cudaError_t status = validate_config(config, error);
    if (status != cudaSuccess) return status;

    const int warps_per_block = config.block_size / warp_size;
    launch_config resolved{
        config.grid_size,
        config.block_size,
        warps_per_block,
        config.grid_size * warps_per_block,
        0,
        config.num_queues,
        0,
        config.profile_capacity_per_warp,
        0
    };

    size_t fixed_bytes = 0;
    size_t bytes_per_task = 0;
    if (!runtime_fixed_and_per_task_bytes(resolved, &fixed_bytes, &bytes_per_task)) {
        return error.fail(
            cudaErrorInvalidValue,
            "queue and task-pool size overflowed");
    }
    status = resolve_tasks_per_unit(
        config.max_tasks_per_warp,
        config.memory_budget_bytes,
        resolved.total_scheduling_units,
        resolved.num_queues,
        fixed_bytes,
        bytes_per_task,
        "max_tasks_per_warp",
        &resolved.tasks_per_scheduling_unit,
        error);
    if (status != cudaSuccess) return status;

    resolved.queue_capacity =
        resolved.tasks_per_scheduling_unit / resolved.num_queues;
    resolved.dynamic_shared_bytes = make_shared_layout(
        warps_per_block,
        resolved.num_queues,
        include_queue_tails).bytes;

    *out = resolved;
    return cudaSuccess;
}

inline cudaError_t initialize_impl(
    const gtap_thread_config& config,
    size_t* device_bytes_allocated,
    init_error& error
) {
    if (h_runtime_initialized) {
        return error.fail(
            cudaErrorInitializationError,
            "GTaP is already initialized");
    }

    launch_config resolved{};
    cudaError_t status = resolve_launch_config(config, &resolved, error);
    if (status != cudaSuccess) return status;

    status = publish_launch_config(resolved, error);
    if (status != cudaSuccess) return status;

    h_stream = config.stream;
    status = initialize_runtime(error);
    if (status != cudaSuccess) {
        h_stream = nullptr;
        return status;
    }

    h_runtime_initialized = true;
    h_runtime_error_reported = false;
    if (device_bytes_allocated != nullptr) {
        *device_bytes_allocated = runtime_device_allocation_bytes();
    }
    return cudaSuccess;
}

}  // namespace gtap::detail::thread

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
    gtap::detail::init_error error{};
    cudaError_t status = gtap::detail::thread::initialize_impl(
        config, device_bytes_allocated, error);
    if (status != cudaSuccess) {
        if (error.message[0] != '\0') {
            std::fprintf(
                stderr,
                "GTaP initialization failed: %s (%s)\n",
                error.message,
                cudaGetErrorString(status));
        } else {
            std::fprintf(
                stderr,
                "GTaP initialization failed: %s\n",
                cudaGetErrorString(status));
        }
    }
    return status;
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

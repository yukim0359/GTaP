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

inline cudaError_t validate_config(
    const gtap_block_config& config,
    host_api_error& error
) {
    if (config.grid_size <= 0) {
        return error.fail(
            cudaErrorInvalidConfiguration,
            "grid_size=%d must be greater than zero",
            config.grid_size);
    }
    if (config.block_size <= 0 ||
        config.block_size > GTAP_DETAIL_MAX_THREADS_PER_BLOCK) {
        return error.fail(
            cudaErrorInvalidConfiguration,
            "block_size=%d must be in (0, %d]",
            config.block_size,
            GTAP_DETAIL_MAX_THREADS_PER_BLOCK);
    }
    if (config.block_size % gtap::detail::warp_size != 0) {
        return error.fail(
            cudaErrorInvalidConfiguration,
            "block_size=%d must be a multiple of %d",
            config.block_size,
            gtap::detail::warp_size);
    }
    if (config.max_tasks_per_block < 0) {
        return error.fail(
            cudaErrorInvalidValue,
            "max_tasks_per_block=%d must be greater than or equal to zero",
            config.max_tasks_per_block);
    }
#ifdef GTAP_ENABLE_PROFILING
    if (config.profile_capacity_per_block <= 0 ||
        config.profile_capacity_per_block > INT_MAX / 2) {
        return error.fail(
            cudaErrorInvalidValue,
            "profile_capacity_per_block=%d must be in (0, %d]",
            config.profile_capacity_per_block,
            INT_MAX / 2);
    }
#endif
    return cudaSuccess;
}

inline cudaError_t resolve_launch_config(
    const gtap_block_config& config,
    launch_config* out,
    host_api_error& error
) {
    cudaError_t status = validate_config(config, error);
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
        return error.fail(
            cudaErrorInvalidValue,
            "queue and task storage size overflowed");
    }
    status = resolve_tasks_per_unit(
        config.max_tasks_per_block,
        config.memory_budget_bytes,
        resolved.total_scheduling_units,
        resolved.num_queues,
        fixed_bytes,
        bytes_per_task,
        "max_tasks_per_block",
        &resolved.tasks_per_scheduling_unit,
        error);
    if (status != cudaSuccess) return status;

    resolved.queue_capacity = resolved.tasks_per_scheduling_unit;
    resolved.dynamic_shared_bytes = config.dynamic_shared_bytes;

    *out = resolved;
    return cudaSuccess;
}

inline cudaError_t initialize_impl(
    const gtap_block_config& config,
    size_t* device_bytes_allocated,
    host_api_error& error
) {
    switch (h_runtime_state) {
        case runtime_state::uninitialized:
            break;
        case runtime_state::ready:
        case runtime_state::launched:
            return error.fail(
                cudaErrorInitializationError,
                "GTaP is already initialized");
        case runtime_state::synchronize_failed:
            return error.fail(
                cudaErrorInitializationError,
                "GTaP cannot be initialized after a failed synchronization");
        case runtime_state::reset_failed:
            return error.fail(
                cudaErrorInitializationError,
                "Call gtap_finalize before reinitializing after a failed reset");
        case runtime_state::finalize_failed:
            return error.fail(
                cudaErrorInitializationError,
                "GTaP cannot be initialized again after a failed finalization");
    }

    launch_config resolved{};
    cudaError_t status = resolve_launch_config(config, &resolved, error);
    if (status != cudaSuccess) return status;

    status = record_error_if_failed(
        error,
        publish_launch_config(resolved),
        "failed to publish the launch configuration");
    if (status != cudaSuccess) return status;

    h_stream = config.stream;
    status = initialize_runtime(error);
    if (status != cudaSuccess) {
        h_stream = nullptr;
        return status;
    }

    h_runtime_state = runtime_state::ready;
    if (device_bytes_allocated != nullptr) {
        *device_bytes_allocated = runtime_device_allocation_bytes();
    }
    return cudaSuccess;
}

inline cudaError_t finalize_impl(host_api_error& error) {
    switch (h_runtime_state) {
        case runtime_state::ready:
        case runtime_state::launched:
        case runtime_state::synchronize_failed:
        case runtime_state::reset_failed:
            break;
        case runtime_state::uninitialized:
            return error.fail(
                cudaErrorInitializationError,
                "GTaP is not initialized");
        case runtime_state::finalize_failed:
            return error.fail(
                cudaErrorInitializationError,
                "GTaP cannot be finalized again after a failed finalization");
    }

    cudaError_t status = finalize_runtime(error);
    h_runtime_state = status == cudaSuccess
        ? runtime_state::uninitialized
        : runtime_state::finalize_failed;
    h_stream = nullptr;
    return status;
}

inline cudaError_t reset_impl(host_api_error& error) {
    switch (h_runtime_state) {
        case runtime_state::ready:
        case runtime_state::launched:
            break;
        case runtime_state::uninitialized:
            return error.fail(
                cudaErrorInitializationError,
                "GTaP is not initialized");
        case runtime_state::synchronize_failed:
            return error.fail(
                cudaErrorInitializationError,
                "GTaP cannot be reset after a failed synchronization");
        case runtime_state::reset_failed:
            return error.fail(
                cudaErrorInitializationError,
                "GTaP cannot be reset again after a failed reset");
        case runtime_state::finalize_failed:
            return error.fail(
                cudaErrorInitializationError,
                "GTaP cannot be reset after a failed finalization");
    }

    cudaError_t status = reset_runtime(error);
    h_runtime_state = status == cudaSuccess
        ? runtime_state::ready
        : runtime_state::reset_failed;
    return status;
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
    gtap::detail::host_api_error error{};
    cudaError_t status = gtap::detail::block::initialize_impl(
        config, device_bytes_allocated, error);
    gtap::detail::print_host_api_error("initialization", status, error);
    return status;
}

inline cudaError_t gtap_finalize() {
    gtap::detail::host_api_error error{};
    cudaError_t status = gtap::detail::block::finalize_impl(error);
    gtap::detail::print_host_api_error("finalization", status, error);
    return status;
}

inline cudaError_t gtap_reset() {
    gtap::detail::host_api_error error{};
    cudaError_t status = gtap::detail::block::reset_impl(error);
    gtap::detail::print_host_api_error("reset", status, error);
    return status;
}

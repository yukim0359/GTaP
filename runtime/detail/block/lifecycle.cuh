#pragma once

// Call order for every block backend.
// Queue storage, the task pool, and profile buffers are allocated by their owners.
// All clears use h_stream.
// A failed initialize releases staged buffers. The caller owns h_stream.

#include "../common/checked_size.cuh"
#include "../common/error/device_error.cuh"
#include "../common/error/host_api_error.cuh"
#include "../common/runtime_config.cuh"

#include "profiling/profile_buffer.cuh"
#include "scheduler.cuh"
#include "task_pool.cuh"
#include "termination.cuh"

namespace gtap::detail::block {

using namespace gtap::detail;

static size_t runtime_device_allocation_bytes() {
    const launch_config& config = h_launch_config;
    size_t bytes = queue_storage_allocation_bytes(config);
    bytes = add_memory_bytes(bytes, task_pool_allocation_bytes(config));
    return add_memory_bytes(bytes, profile_buffer_allocation_bytes(config));
}

// Profile buffers are not included. fixed_bytes does not depend on task capacity.
inline bool runtime_fixed_and_per_task_bytes(
    const launch_config& config,
    size_t* fixed_bytes,
    size_t* bytes_per_task
) {
    size_t queue_fixed = 0;
    size_t queue_per_task = 0;
    size_t pool_fixed = 0;
    size_t pool_per_task = 0;
    size_t fixed = 0;
    size_t per_task = 0;
    if (!queue_fixed_and_per_task_bytes(config, &queue_fixed, &queue_per_task) ||
        !task_pool_fixed_and_per_task_bytes(config, &pool_fixed, &pool_per_task) ||
        !checked_add(queue_fixed, pool_fixed, &fixed) ||
        !checked_add(queue_per_task, pool_per_task, &per_task) ||
        per_task == 0) {
        return false;
    }
    *fixed_bytes = fixed;
    *bytes_per_task = per_task;
    return true;
}

inline void abandon_initialize(
    queue_storage_buffers* queues,
    task_pool_buffers* task_pool,
    profile_buffers* profile
) {
    cudaStreamSynchronize(h_stream);
    release_staged_queue_storage(queues);
    release_staged_task_pool(task_pool);
    release_staged_profile_buffers(profile);
    finalize_device_error_record();
    cudaGetLastError();
}

cudaError_t initialize_runtime(host_api_error& error) {
    queue_storage_buffers queues{};
    task_pool_buffers task_pool{};
    profile_buffers profile{};

    auto fail = [&](cudaError_t status) {
        abandon_initialize(&queues, &task_pool, &profile);
        return status;
    };

    cudaError_t status = record_error_if_failed(
        error,
        initialize_device_error_record(),
        "failed to initialize device error reporting");
    if (status != cudaSuccess) return fail(status);
    const launch_config& runtime_config = h_launch_config;

    cudaStream_t stream = h_stream;

    status = stage_queue_storage(runtime_config, stream, &queues, error);
    if (status != cudaSuccess) return fail(status);
    status = stage_task_pool(runtime_config, stream, &task_pool, error);
    if (status != cudaSuccess) return fail(status);
    status = stage_profile_buffers(runtime_config, stream, &profile, error);
    if (status != cudaSuccess) return fail(status);
    status = record_error_if_failed(
        error,
        cudaStreamSynchronize(stream),
        "failed to initialize runtime storage");
    if (status != cudaSuccess) return fail(status);

    status = record_error_if_failed(
        error,
        publish_queue_storage(queues),
        "failed to configure task queues");
    if (status != cudaSuccess) return fail(status);
    status = record_error_if_failed(
        error,
        publish_task_pool(task_pool),
        "failed to configure task storage");
    if (status != cudaSuccess) return fail(status);
    status = record_error_if_failed(
        error,
        publish_profile_buffers(profile),
        "failed to configure profiling");
    if (status != cudaSuccess) return fail(status);

    int zero = 0;
    status = record_error_if_failed(
        error,
        cudaMemcpyToSymbol(d_root_task_finished, &zero, sizeof(int)),
        "failed to configure the runtime");
    if (status != cudaSuccess) return fail(status);
    status = record_error_if_failed(
        error,
        cudaMemcpyToSymbol(d_all_tasks_finished, &zero, sizeof(int)),
        "failed to configure the runtime");
    if (status != cudaSuccess) return fail(status);
    status = record_error_if_failed(
        error,
        cudaMemcpyToSymbol(d_device_error_code, &zero, sizeof(int)),
        "failed to configure the runtime");
    if (status != cudaSuccess) return fail(status);
    status = record_error_if_failed(
        error,
        reset_queue_counters(),
        "failed to initialize queue storage");
    if (status != cudaSuccess) return fail(status);
    int one = 1;
    status = record_error_if_failed(
        error,
        cudaMemcpyToSymbol(d_active_block_count, &one, sizeof(int)),
        "failed to configure the runtime");
    if (status != cudaSuccess) return fail(status);

    init_block_id_pools_metadata<<<runtime_config.grid_size, 1, 0, stream>>>();
    status = record_error_if_failed(
        error,
        cudaGetLastError(),
        "failed to initialize task storage");
    if (status != cudaSuccess) return fail(status);
    // TODO: cudaDeviceSynchronize waits for every stream.
    status = record_error_if_failed(
        error,
        cudaDeviceSynchronize(),
        "failed to initialize task storage");
    if (status != cudaSuccess) return fail(status);
    status = record_error_if_failed(
        error,
        cudaGetLastError(),
        "initialization left a CUDA error");
    if (status != cudaSuccess) return fail(status);
    return cudaSuccess;
}

cudaError_t finalize_runtime(host_api_error& error) {
    cudaError_t result = cudaSuccess;

    auto record_first_error = [&](cudaError_t status, const char* message) {
        if (status != cudaSuccess && result == cudaSuccess) {
            result = error.fail(status, "%s", message);
        }
    };

    record_first_error(free_queue_storage(),
            "failed to release task queues");
    record_first_error(free_task_pool(),
            "failed to release task storage");
    record_first_error(free_profile_buffers(),
            "failed to release profile buffers");
    record_first_error(finalize_device_error_record(),
            "failed to release device error reporting resources");

    return result;
}

cudaError_t reset_runtime(host_api_error& error) {
    reset_device_error_record();
    const launch_config& runtime_config = h_launch_config;
    cudaStream_t stream = h_stream;

    cudaError_t status = record_error_if_failed(
        error,
        clear_queue_storage(runtime_config, stream),
        "failed to reset task queues");
    if (status != cudaSuccess) return status;
    status = record_error_if_failed(
        error,
        clear_task_pool(runtime_config, stream),
        "failed to reset task storage");
    if (status != cudaSuccess) return status;
    status = record_error_if_failed(
        error,
        clear_profile_buffers(runtime_config, stream),
        "failed to reset profile buffers");
    if (status != cudaSuccess) return status;
    status = record_error_if_failed(
        error,
        cudaStreamSynchronize(stream),
        "failed to synchronize the runtime");
    if (status != cudaSuccess) return status;

    int zero = 0;
    status = record_error_if_failed(
        error,
        cudaMemcpyToSymbol(d_root_task_finished, &zero, sizeof(int)),
        "failed to reset the runtime");
    if (status != cudaSuccess) return status;
    status = record_error_if_failed(
        error,
        cudaMemcpyToSymbol(d_all_tasks_finished, &zero, sizeof(int)),
        "failed to reset the runtime");
    if (status != cudaSuccess) return status;
    status = record_error_if_failed(
        error,
        cudaMemcpyToSymbol(d_device_error_code, &zero, sizeof(int)),
        "failed to reset the runtime");
    if (status != cudaSuccess) return status;
    status = record_error_if_failed(
        error,
        reset_queue_counters(),
        "failed to reset task queues");
    if (status != cudaSuccess) return status;
    int one = 1;
    status = record_error_if_failed(
        error,
        cudaMemcpyToSymbol(d_active_block_count, &one, sizeof(int)),
        "failed to reset the runtime");
    if (status != cudaSuccess) return status;

    init_block_id_pools_metadata<<<runtime_config.grid_size, 1, 0, stream>>>();
    status = record_error_if_failed(
        error,
        cudaGetLastError(),
        "failed to reset task storage");
    if (status != cudaSuccess) return status;
    // TODO: cudaDeviceSynchronize waits for every stream.
    status = record_error_if_failed(
        error,
        cudaDeviceSynchronize(),
        "failed to reset task storage");
    if (status != cudaSuccess) return status;
    return record_error_if_failed(
        error, cudaGetLastError(), "failed to reset the runtime");
}

}  // namespace gtap::detail::block

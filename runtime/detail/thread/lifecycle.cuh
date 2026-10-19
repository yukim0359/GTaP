#pragma once

// Call order for every thread backend.
// Queue storage, the task pool, and profile buffers are allocated by their owners.
// All clears use h_stream.
// A failed initialize releases staged buffers. The caller owns h_stream.

#include "../common/checked_size.cuh"
#include "../common/host_api_error.cuh"
#include "../common/runtime_config.cuh"
#include "../common/runtime_error.cuh"

#include "profile_buffer.cuh"
#include "scheduler.cuh"
#include "task_pool.cuh"
#include "termination.cuh"

namespace gtap::detail::thread {

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
    finalize_runtime_error_record();
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
    cudaError_t status = initialize_runtime_error_record();
    if (status != cudaSuccess) {
        return fail(error.fail(status, "failed to initialize runtime error reporting"));
    }
    const launch_config& runtime_config = h_launch_config;
    cudaStream_t stream = h_stream;

#ifdef GTAP_INTERNAL_PROFILE_INIT
    printf("\n=== initialize_runtime detailed profiling ===\n");
    // TODO: Events leak when initialize returns early.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;
#endif

    status = stage_queue_storage(runtime_config, stream, &queues, error);
    if (status != cudaSuccess) return fail(status);
    status = stage_task_pool(runtime_config, stream, &task_pool, error);
    if (status != cudaSuccess) return fail(status);
    status = stage_profile_buffers(runtime_config, stream, &profile, error);
    if (status != cudaSuccess) return fail(status);
    status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
        return fail(error.fail(status, "failed to initialize runtime storage"));
    }
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

#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(start);
#endif
    int zero = 0;
    status = cudaMemcpyToSymbol(d_root_task_finished, &zero, sizeof(int));
    if (status != cudaSuccess) {
        return fail(error.fail(status, "failed to configure the runtime"));
    }
    status = cudaMemcpyToSymbol(d_all_tasks_finished, &zero, sizeof(int));
    if (status != cudaSuccess) {
        return fail(error.fail(status, "failed to configure the runtime"));
    }
    status = cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int));
    if (status != cudaSuccess) {
        return fail(error.fail(status, "failed to configure the runtime"));
    }
#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_root_task_finished): %.3f ms\n", elapsed);
#endif
    // Initialize d_active_warp_count to 1 to prevent early termination
    // before the initial task is pushed by the master thread
    int one = 1;
    status = cudaMemcpyToSymbol(d_active_warp_count, &one, sizeof(int));
    if (status != cudaSuccess) {
        return fail(error.fail(status, "failed to configure the runtime"));
    }
#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_active_warp_count): %.3f ms\n", elapsed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("=== initialize_runtime profiling complete ===\n\n");
#endif

    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return fail(error.fail(status, "initialization left a CUDA error"));
    }
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
    record_first_error(finalize_runtime_error_record(),
            "failed to release runtime error reporting resources");

    return result;
}

cudaError_t reset_runtime() {
    reset_runtime_error_record();
    const launch_config& runtime_config = h_launch_config;
    cudaStream_t stream = h_stream;

    GTAP_DETAIL_CUDA_TRY(clear_queue_storage(runtime_config, stream));
    GTAP_DETAIL_CUDA_TRY(clear_task_pool(runtime_config, stream));

    int zero = 0;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_root_task_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    int one = 1;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_active_warp_count, &one, sizeof(int)));

    GTAP_DETAIL_CUDA_TRY(clear_profile_buffers(runtime_config, stream));
    GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(stream));
    return cudaGetLastError();
}

}  // namespace gtap::detail::thread

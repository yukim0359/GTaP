#pragma once

// Call order for every thread backend.
// Queue storage, the task pool, and profile buffers are allocated by their owners.
// All clears use h_stream.
// A failed initialize releases staged buffers. The caller owns h_stream.

#include <climits>

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
    return queue_storage_allocation_bytes(config) +
           task_pool_allocation_bytes(config) +
           profile_buffer_allocation_bytes(config);
}

inline size_t task_management_bytes(const launch_config& config) {
    return task_pool_allocation_bytes(config) + queue_storage_allocation_bytes(config);
}

// A task count of 0 means unset. A budget of 0 means unset.
// One set value is used as given. Both set takes the smaller slot count.
// Both unset uses default_tasks_per_scheduling_unit.
// The task region is fixed bytes plus a constant increment per queue-group of slots.
inline int tasks_within_budget(launch_config config, size_t budget) {
    if (config.num_queues <= 0) return 0;
    if (budget == 0) {
        const int tasks = config.tasks_per_scheduling_unit == 0
            ? default_tasks_per_scheduling_unit
            : config.tasks_per_scheduling_unit;
        if (tasks <= 0 || tasks % config.num_queues != 0) return 0;
        return tasks;
    }

    int ceiling = config.tasks_per_scheduling_unit;
    if (ceiling == 0) {
        const int units = config.total_scheduling_units;
        if (units <= 0) return 0;
        // id = slot * units + unit must fit in a signed int.
        ceiling = INT_MAX / units;
    }
    if (ceiling < config.num_queues) return 0;
    ceiling -= ceiling % config.num_queues;

    config.tasks_per_scheduling_unit = 0;
    config.queue_capacity = 0;
    const size_t bytes_with_no_slots = task_management_bytes(config);
    if (budget <= bytes_with_no_slots) return 0;
    config.tasks_per_scheduling_unit = config.num_queues;
    config.queue_capacity = 1;
    const size_t bytes_with_one_slot_per_queue = task_management_bytes(config);
    if (bytes_with_one_slot_per_queue <= bytes_with_no_slots) return 0;
    const size_t bytes_per_slot_group =
        bytes_with_one_slot_per_queue - bytes_with_no_slots;
    size_t groups = (budget - bytes_with_no_slots) / bytes_per_slot_group;
    const size_t max_groups = static_cast<size_t>(ceiling / config.num_queues);
    if (groups > max_groups) groups = max_groups;
    return static_cast<int>(groups) * config.num_queues;
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

cudaError_t initialize_runtime() {
    queue_storage_buffers queues{};
    task_pool_buffers task_pool{};
    profile_buffers profile{};
    GTAP_DETAIL_CUDA_TRY_OR(
        initialize_runtime_error_record(),
        abandon_initialize(&queues, &task_pool, &profile));
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

    GTAP_DETAIL_CUDA_TRY_OR(
        stage_queue_storage(runtime_config, stream, &queues),
        abandon_initialize(&queues, &task_pool, &profile));
    GTAP_DETAIL_CUDA_TRY_OR(
        stage_task_pool(runtime_config, stream, &task_pool),
        abandon_initialize(&queues, &task_pool, &profile));
    GTAP_DETAIL_CUDA_TRY_OR(
        stage_profile_buffers(runtime_config, stream, &profile),
        abandon_initialize(&queues, &task_pool, &profile));
    GTAP_DETAIL_CUDA_TRY_OR(
        cudaStreamSynchronize(stream),
        abandon_initialize(&queues, &task_pool, &profile));
    GTAP_DETAIL_CUDA_TRY_OR(
        publish_queue_storage(queues),
        abandon_initialize(&queues, &task_pool, &profile));
    GTAP_DETAIL_CUDA_TRY_OR(
        publish_task_pool(task_pool),
        abandon_initialize(&queues, &task_pool, &profile));
    GTAP_DETAIL_CUDA_TRY_OR(
        publish_profile_buffers(profile),
        abandon_initialize(&queues, &task_pool, &profile));

#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(start);
#endif
    int zero = 0;
    GTAP_DETAIL_CUDA_TRY_OR(
        cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)),
        abandon_initialize(&queues, &task_pool, &profile));
    GTAP_DETAIL_CUDA_TRY_OR(
        cudaMemcpyToSymbol(d_all_tasks_finished, &zero, sizeof(int)),
        abandon_initialize(&queues, &task_pool, &profile));
    GTAP_DETAIL_CUDA_TRY_OR(
        cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)),
        abandon_initialize(&queues, &task_pool, &profile));
#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_first_task_finished): %.3f ms\n", elapsed);
#endif
    // Initialize d_active_warp_count to 1 to prevent early termination
    // before the initial task is pushed by the master thread
    int one = 1;
    GTAP_DETAIL_CUDA_TRY_OR(
        cudaMemcpyToSymbol(d_active_warp_count, &one, sizeof(int)),
        abandon_initialize(&queues, &task_pool, &profile));
#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_active_warp_count): %.3f ms\n", elapsed);
#endif

#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(start, stream);
#endif
    init_warp_id_pools_metadata<<<
        runtime_config.grid_size, runtime_config.block_size, 0, stream>>>();
    GTAP_DETAIL_CUDA_TRY_OR(
        cudaGetLastError(),
        abandon_initialize(&queues, &task_pool, &profile));
#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop, stream);
#endif
    // TODO: cudaDeviceSynchronize waits for every stream.
    GTAP_DETAIL_CUDA_TRY_OR(
        cudaDeviceSynchronize(),
        abandon_initialize(&queues, &task_pool, &profile));
#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  init_warp_id_pools_metadata kernel: %.3f ms\n", elapsed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("=== initialize_runtime profiling complete ===\n\n");
#endif

    GTAP_DETAIL_CUDA_TRY_OR(
        cudaGetLastError(),
        abandon_initialize(&queues, &task_pool, &profile));
    return cudaSuccess;
}

cudaError_t finalize_runtime() {
    GTAP_DETAIL_CUDA_TRY(free_queue_storage());
    GTAP_DETAIL_CUDA_TRY(free_task_pool());
    GTAP_DETAIL_CUDA_TRY(free_profile_buffers());
    GTAP_DETAIL_CUDA_TRY(finalize_runtime_error_record());
    return cudaGetLastError();
}

cudaError_t reset_runtime() {
    reset_runtime_error_record();
    const launch_config& runtime_config = h_launch_config;
    cudaStream_t stream = h_stream;

    GTAP_DETAIL_CUDA_TRY(clear_queue_storage(runtime_config, stream));
    GTAP_DETAIL_CUDA_TRY(clear_task_pool(runtime_config, stream));

    int zero = 0;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    int one = 1;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_active_warp_count, &one, sizeof(int)));

    GTAP_DETAIL_CUDA_TRY(clear_profile_buffers(runtime_config, stream));
    GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(stream));

    init_warp_id_pools_metadata<<<
        runtime_config.grid_size, runtime_config.block_size, 0, stream>>>();
    // TODO: cudaDeviceSynchronize waits for every stream.
    GTAP_DETAIL_CUDA_TRY(cudaDeviceSynchronize());
    return cudaGetLastError();
}

}  // namespace gtap::detail::thread

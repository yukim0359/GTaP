#pragma once

// Call order for every block backend. Include after that backend's scheduler.
// Queue storage, the task pool, and profile buffers are allocated by their owners.
// Each backend chooses how many private streams those clears use.

namespace gtap::detail::block {

using namespace gtap::detail;

static size_t runtime_device_allocation_bytes() {
    const launch_config& c = h_launch_config;
    const size_t workers = static_cast<size_t>(c.total_workers);
    const size_t tasks = workers * c.tasks_per_worker;
    return queue_storage_allocation_bytes(workers, tasks, c.num_queues) +
           task_pool_allocation_bytes(workers, tasks, c.block_size) +
           profile_buffer_allocation_bytes(workers);
}

cudaError_t initialize_runtime() {
    GTAP_DETAIL_CUDA_TRY(initialize_runtime_error_record());
    const launch_config& runtime_config = h_launch_config;
    const size_t total_workers = runtime_config.total_workers;
    const size_t total_tasks = total_workers * runtime_config.tasks_per_worker;

    cudaStream_t streams[runtime_init_stream_count];
    for (int i = 0; i < runtime_init_stream_count; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamCreate(&streams[i]));
    }

    queue_storage_buffers queues{};
    GTAP_DETAIL_CUDA_TRY(stage_queue_storage(
        total_workers, total_tasks, runtime_config.num_queues, streams,
        &queues));
    task_pool_buffers task_pool{};
    GTAP_DETAIL_CUDA_TRY(stage_task_pool(
        total_workers, total_tasks, runtime_config.block_size, streams,
        task_id_free_position_fill, &task_pool));

    for (int i = 0; i < runtime_init_stream_count; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(streams[i]));
    }

    GTAP_DETAIL_CUDA_TRY(publish_queue_storage(queues));
    GTAP_DETAIL_CUDA_TRY(publish_task_pool(task_pool));
    profile_buffers profile{};
    GTAP_DETAIL_CUDA_TRY(stage_profile_buffers(
        total_workers, streams[1], streams[0], &profile));
    GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(streams[0]));
    GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(streams[1]));
    GTAP_DETAIL_CUDA_TRY(publish_profile_buffers(profile));

    for (int i = 0; i < runtime_init_stream_count; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamDestroy(streams[i]));
    }

    int zero = 0;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_first_task_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_all_tasks_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_runtime_error_code, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(reset_queue_counters());
    int one = 1;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_active_block_count, &one, sizeof(int)));

    init_block_id_pools_metadata<<<runtime_config.grid_size, 1>>>();
    return cudaDeviceSynchronize();
}

cudaError_t finalize_runtime() {
    GTAP_DETAIL_CUDA_TRY(free_queue_storage());
    GTAP_DETAIL_CUDA_TRY(free_task_pool());
    GTAP_DETAIL_CUDA_TRY(free_profile_buffers());
    GTAP_DETAIL_CUDA_TRY(finalize_runtime_error_record());
    return cudaGetLastError();
}

cudaError_t reset_runtime() {
    reset_runtime_error_record_host();
    const launch_config& runtime_config = h_launch_config;
    const size_t total_workers = runtime_config.total_workers;
    const size_t total_tasks = total_workers * runtime_config.tasks_per_worker;

    cudaStream_t streams[runtime_init_stream_count];
    for (int i = 0; i < runtime_init_stream_count; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamCreate(&streams[i]));
    }

    GTAP_DETAIL_CUDA_TRY(clear_queue_storage(
        total_workers, total_tasks, runtime_config.num_queues, streams));
    GTAP_DETAIL_CUDA_TRY(clear_task_pool(
        total_workers, total_tasks, runtime_config.block_size, streams,
        task_id_free_position_fill));
    GTAP_DETAIL_CUDA_TRY(clear_profile_buffers(
        total_workers, streams[1], streams[0]));

    for (int i = 0; i < runtime_init_stream_count; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(streams[i]));
    }

    int zero = 0;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_first_task_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_all_tasks_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_runtime_error_code, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(reset_queue_counters());
    int one = 1;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_active_block_count, &one, sizeof(int)));

    init_block_id_pools_metadata<<<runtime_config.grid_size, 1>>>();
    GTAP_DETAIL_CUDA_TRY(cudaDeviceSynchronize());

    for (int i = 0; i < runtime_init_stream_count; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamDestroy(streams[i]));
    }
    return cudaGetLastError();
}

}  // namespace gtap::detail::block

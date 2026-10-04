#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "../common/profile_export.cuh"
#include "../common/runtime_config.cuh"

#include "profile_buffer.cuh"

#ifdef GTAP_ENABLE_PROFILING

static inline gtap_profile_export_result gtap_export_profile(
    const gtap_profile_export_options& options = {}
) {
    gtap_profile_export_result result;
    if (gtap::detail::prepare_profile_output(options, result) !=
        gtap_profile_export_status::success) {
        return result;
    }

    const int blocks = gtap::detail::h_launch_config.grid_size;
    int* device_indices = nullptr;
    int* indices = static_cast<int*>(malloc(sizeof(int) * blocks));
    unsigned long long* dropped = static_cast<unsigned long long*>(
        malloc(sizeof(unsigned long long) * blocks));
    long long* times = static_cast<long long*>(malloc(
        sizeof(long long) * blocks * gtap::detail::profile_timestamp_capacity()));
    if (!indices || !dropped || !times) {
        free(indices); free(dropped); free(times);
        result.status = gtap_profile_export_status::out_of_memory;
        printf("GTaP profile not written: host memory allocation failed\n");
        return result;
    }
    cudaError_t error = cudaMalloc(&device_indices, sizeof(int) * blocks);
    if (error == cudaSuccess) {
        gtap::detail::block::get_block_working_time_counts<<<blocks, 1>>>(device_indices);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error == cudaSuccess) error = cudaMemcpy(
        indices, device_indices, sizeof(int) * blocks, cudaMemcpyDeviceToHost);
    if (error == cudaSuccess) error = gtap::detail::block::get_working_time_data(times);
    if (error == cudaSuccess) {
        error = gtap::detail::block::get_block_profile_dropped_events_data(dropped);
    }
    cudaFree(device_indices);
    if (error != cudaSuccess) {
        free(indices); free(dropped); free(times);
        result.status = gtap_profile_export_status::cuda_error;
        printf("GTaP profile not written: CUDA data transfer failed\n");
        return result;
    }

    long long origin = 0;
    long long profile_end = 0;
    int blocks_with_executed_tasks = 0;
    for (int block = 0; block < blocks; ++block) {
        if (indices[block] > 0) blocks_with_executed_tasks++;
        result.recorded_intervals += static_cast<size_t>(indices[block] / 2);
        result.dropped_intervals += static_cast<size_t>(dropped[block]);
        for (int i = 0; i < indices[block]; ++i) {
            const long long value = times[block * gtap::detail::profile_timestamp_capacity() + i];
            if (value > 0 && (!origin || value < origin)) origin = value;
            if (value > profile_end) profile_end = value;
        }
    }
    if (!result.recorded_intervals) {
        free(indices); free(dropped); free(times);
        result.status = gtap_profile_export_status::no_data;
        printf("GTaP profile not written: no task execution data\n");
        return result;
    }

    double* durations = static_cast<double*>(malloc(
        sizeof(double) * result.recorded_intervals));
    double* all_block_ratios = static_cast<double*>(malloc(
        sizeof(double) * blocks));
    double* active_block_ratios = static_cast<double*>(malloc(
        sizeof(double) * blocks_with_executed_tasks));
    if (!durations || !all_block_ratios || !active_block_ratios) {
        free(indices); free(dropped); free(times);
        free(durations); free(all_block_ratios); free(active_block_ratios);
        result.status = gtap_profile_export_status::out_of_memory;
        printf("GTaP profile not written: host memory allocation failed\n");
        return result;
    }
    const double profile_span = profile_end > origin
        ? static_cast<double>(profile_end - origin) : 0.0;
    size_t task_index = 0;
    size_t active_block_index = 0;
    for (int block = 0; block < blocks; ++block) {
        long long execution_ns = 0;
        for (int i = 0; i < indices[block]; i += 2) {
            const size_t offset =
                static_cast<size_t>(block) * gtap::detail::profile_timestamp_capacity() + i;
            const long long duration = times[offset + 1] - times[offset];
            durations[task_index++] = static_cast<double>(duration);
            execution_ns += duration;
        }
        const double ratio = profile_span > 0.0
            ? static_cast<double>(execution_ns) / profile_span : 0.0;
        all_block_ratios[block] = ratio;
        if (indices[block] > 0)
            active_block_ratios[active_block_index++] = ratio;
    }
    const gtap::detail::profile_distribution duration_stats =
        gtap::detail::compute_distribution(durations, task_index);
    const gtap::detail::profile_distribution all_block_ratio_stats =
        gtap::detail::compute_distribution(all_block_ratios, blocks);
    const gtap::detail::profile_distribution active_block_ratio_stats =
        gtap::detail::compute_distribution(
            active_block_ratios, active_block_index);

    FILE* timeline = fopen(result.intervals_path, "w");
    if (!timeline) {
        free(indices); free(dropped); free(times);
        free(durations); free(all_block_ratios); free(active_block_ratios);
        result.status = gtap_profile_export_status::io_error;
        printf("GTaP profile not written: failed to write %s\n",
               result.result_directory);
        return result;
    }
    fprintf(timeline,
            "block_id,start_ns,end_ns\n");
    for (int block = 0; block < blocks; ++block) {
        for (int i = 0; i < indices[block]; i += 2) {
            const size_t offset =
                static_cast<size_t>(block) * gtap::detail::profile_timestamp_capacity() + i;
            fprintf(timeline, "%d,%lld,%lld\n", block,
                    times[offset] - origin, times[offset + 1] - origin);
        }
    }
    bool io_ok = gtap::detail::close_profile_file(timeline);
    FILE* stats = io_ok ? fopen(result.aggregates_path, "w") : nullptr;
    if (stats) {
        fprintf(stats,
                "block_id,intervals_recorded,intervals_dropped,tasks_executed,"
                "recorded_task_execution_ns,first_execution_ns,"
                "last_execution_ns\n");
        for (int block = 0; block < blocks; ++block) {
            long long recorded_task_execution_ns = 0;
            long long first_execution_ns = 0;
            long long last_execution_ns = 0;
            for (int i = 0; i < indices[block]; i += 2) {
                const size_t offset =
                    static_cast<size_t>(block) * gtap::detail::profile_timestamp_capacity() + i;
                const long long start_ns = times[offset] - origin;
                const long long end_ns = times[offset + 1] - origin;
                recorded_task_execution_ns += end_ns - start_ns;
                if (i == 0) first_execution_ns = start_ns;
                last_execution_ns = end_ns;
            }
            fprintf(stats, "%d,%d,%llu,%d,%lld,%lld,%lld\n", block,
                    indices[block] / 2, dropped[block], indices[block] / 2,
                    recorded_task_execution_ns, first_execution_ns,
                    last_execution_ns);
        }
        io_ok = gtap::detail::close_profile_file(stats);
    } else io_ok = false;

    FILE* metadata = io_ok ? fopen(result.profile_path, "w") : nullptr;
    if (metadata) {
        bool metadata_ok = fputs(
            "{\n  \"schema_version\": 1,\n", metadata) != EOF;
        if (metadata_ok && options.label) {
            metadata_ok = fprintf(
                metadata, "  \"label\": \"%s\",\n", options.label) >= 0;
        }
        metadata_ok = metadata_ok && fprintf(metadata,
            "  \"mode\": \"block\",\n"
            "  \"grid_size\": %d,\n"
            "  \"block_size\": %d,\n"
            "  \"task_execution\": {\n"
            "    \"block_counts\": {\n"
            "      \"total\": %d,\n"
            "      \"with_executed_tasks\": %d\n"
            "    },\n"
            "    \"intervals\": {\n"
            "      \"recording_limit_per_block\": %d,\n"
            "      \"recorded_count\": %zu,\n"
            "      \"dropped_count\": %zu,\n"
            "      \"duration_ns\": {\n"
            "        \"mean\": %.2f, \"stddev\": %.2f,\n"
            "        \"min\": %.0f, \"p50\": %.0f, \"p95\": %.0f,\n"
            "        \"p99\": %.0f, \"max\": %.0f\n"
            "      }\n"
            "    },\n"
            "    \"execution_ratio_per_block\": {\n"
            "      \"all_blocks\": {\n"
            "        \"count\": %zu, \"mean\": %.6f, \"stddev\": %.6f,\n"
            "        \"min\": %.6f, \"p50\": %.6f, \"p95\": %.6f,\n"
            "        \"p99\": %.6f, \"max\": %.6f\n"
            "      },\n"
            "      \"blocks_with_executed_tasks\": {\n"
            "        \"count\": %zu, \"mean\": %.6f, \"stddev\": %.6f,\n"
            "        \"min\": %.6f, \"p50\": %.6f, \"p95\": %.6f,\n"
            "        \"p99\": %.6f, \"max\": %.6f\n"
            "      }\n"
            "    }\n"
            "  }\n"
            "}\n",
            gtap::detail::h_launch_config.grid_size,
            gtap::detail::h_launch_config.block_size,
            blocks, blocks_with_executed_tasks,
            gtap::detail::h_launch_config.profile_interval_capacity,
            result.recorded_intervals, result.dropped_intervals,
            duration_stats.mean, duration_stats.stddev,
            duration_stats.min, duration_stats.p50, duration_stats.p95,
            duration_stats.p99, duration_stats.max,
            all_block_ratio_stats.count, all_block_ratio_stats.mean,
            all_block_ratio_stats.stddev, all_block_ratio_stats.min,
            all_block_ratio_stats.p50, all_block_ratio_stats.p95,
            all_block_ratio_stats.p99, all_block_ratio_stats.max,
            active_block_ratio_stats.count, active_block_ratio_stats.mean,
            active_block_ratio_stats.stddev, active_block_ratio_stats.min,
            active_block_ratio_stats.p50, active_block_ratio_stats.p95,
            active_block_ratio_stats.p99, active_block_ratio_stats.max) >= 0;
        io_ok = metadata_ok && gtap::detail::close_profile_file(metadata);
    } else io_ok = false;

    free(indices); free(dropped); free(times);
    free(durations); free(all_block_ratios); free(active_block_ratios);
    result.status = io_ok ? gtap_profile_export_status::success
                          : gtap_profile_export_status::io_error;
    if (result.status == gtap_profile_export_status::success) {
        printf("GTaP profile written to %s\n", result.result_directory);
    } else {
        printf("GTaP profile not written: failed to write %s\n",
               result.result_directory);
    }
    return result;
}

#else

static inline gtap_profile_export_result gtap_export_profile(
    const gtap_profile_export_options& = {}
) {
    gtap_profile_export_result result;
    result.status = gtap_profile_export_status::profiling_disabled;
    return result;
}

#endif

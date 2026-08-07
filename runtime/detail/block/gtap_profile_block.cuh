#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/gtap_profile_export.cuh"
#include "../common/gtap_runtime_common.cuh"
#include "gtap_profile_host_api.cuh"

#ifdef GTAP_PROFILE

static inline gtap_profile_export_result gtap_export_profile(
    const gtap_profile_export_options& options = {}
) {
    gtap_profile_export_result result;
    if (!gtap_profile_valid_label(options.label)) {
        result.status = gtap_profile_export_status::invalid_label;
        printf("GTaP profile not written: invalid label\n");
        return result;
    }
    if (!gtap_profile_resolve_output(
            options.output_directory, result.result_directory,
            sizeof(result.result_directory))) {
        result.status = gtap_profile_export_status::invalid_output_directory;
        printf("GTaP profile not written: invalid output directory\n");
        return result;
    }
    const int n0 = snprintf(result.profile_path, sizeof(result.profile_path),
        "%s/profile.json", result.result_directory);
    const int n1 = snprintf(result.timeline_path, sizeof(result.timeline_path),
        "%s/task_execution_intervals.csv", result.result_directory);
    const int n2 = snprintf(result.statistics_path, sizeof(result.statistics_path),
        "%s/task_execution_aggregates.csv", result.result_directory);
    if (n0 < 0 || n1 < 0 || n2 < 0 ||
        static_cast<size_t>(n0) >= sizeof(result.profile_path) ||
        static_cast<size_t>(n1) >= sizeof(result.timeline_path) ||
        static_cast<size_t>(n2) >= sizeof(result.statistics_path)) {
        result.status = gtap_profile_export_status::path_too_long;
        printf("GTaP profile not written: output path is too long\n");
        return result;
    }
    if (!options.overwrite &&
        (gtap_profile_path_exists(result.profile_path) ||
         gtap_profile_path_exists(result.timeline_path) ||
         gtap_profile_path_exists(result.statistics_path))) {
        result.status = gtap_profile_export_status::already_exists;
        printf("GTaP profile not written: files already exist in %s\n",
               result.result_directory);
        return result;
    }

    const int blocks = gtap_stored_launch_config().grid_size;
    int* device_indices = nullptr;
    int* indices = static_cast<int*>(malloc(sizeof(int) * blocks));
    unsigned long long* dropped = static_cast<unsigned long long*>(
        malloc(sizeof(unsigned long long) * blocks));
    long long* times = static_cast<long long*>(malloc(
        sizeof(long long) * blocks * gtap_profile_capacity()));
    if (!indices || !dropped || !times) {
        free(indices); free(dropped); free(times);
        result.status = gtap_profile_export_status::out_of_memory;
        printf("GTaP profile not written: host memory allocation failed\n");
        return result;
    }
    cudaError_t error = cudaMalloc(&device_indices, sizeof(int) * blocks);
    if (error == cudaSuccess) {
        get_final_working_time_indices<<<blocks, 1>>>(device_indices);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error == cudaSuccess) error = cudaMemcpy(
        indices, device_indices, sizeof(int) * blocks, cudaMemcpyDeviceToHost);
    if (error == cudaSuccess) error = get_working_time_data(times);
    if (error == cudaSuccess) {
#ifdef GTAP_PROFILE_HAS_DROPPED_COUNTER
        error = get_block_profile_dropped_events_data(dropped);
#else
        memset(dropped, 0, sizeof(unsigned long long) * blocks);
#endif
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
            const long long value = times[block * gtap_profile_capacity() + i];
            if (value > 0 && (!origin || value < origin)) origin = value;
            if (value > profile_end) profile_end = value;
        }
    }
    result.complete = result.dropped_intervals == 0;
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
                static_cast<size_t>(block) * gtap_profile_capacity() + i;
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
    const gtap_profile_distribution duration_stats =
        gtap_profile_compute_distribution(durations, task_index);
    const gtap_profile_distribution all_block_ratio_stats =
        gtap_profile_compute_distribution(all_block_ratios, blocks);
    const gtap_profile_distribution active_block_ratio_stats =
        gtap_profile_compute_distribution(
            active_block_ratios, active_block_index);

    FILE* timeline = fopen(result.timeline_path, "w");
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
                static_cast<size_t>(block) * gtap_profile_capacity() + i;
            fprintf(timeline, "%d,%lld,%lld\n", block,
                    times[offset] - origin, times[offset + 1] - origin);
        }
    }
    bool io_ok = !ferror(timeline) && fclose(timeline) == 0;
    if (io_ok) result.generated_files++;
    FILE* stats = io_ok ? fopen(result.statistics_path, "w") : nullptr;
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
                    static_cast<size_t>(block) * gtap_profile_capacity() + i;
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
        io_ok = !ferror(stats) && fclose(stats) == 0;
        if (io_ok) result.generated_files++;
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
            gtap_stored_launch_config().grid_size,
            gtap_stored_launch_config().block_size,
            blocks, blocks_with_executed_tasks,
            gtap_stored_launch_config().profile_interval_capacity,
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
        const bool metadata_error = ferror(metadata) != 0;
        const bool metadata_closed = fclose(metadata) == 0;
        io_ok = metadata_ok && !metadata_error && metadata_closed;
        if (io_ok) result.generated_files++;
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

#endif

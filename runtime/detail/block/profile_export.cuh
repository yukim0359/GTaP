#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "../common/profile_export.cuh"
#include "../common/runtime_config.cuh"

#include "profile_buffer.cuh"

#ifdef GTAP_ENABLE_PROFILING

namespace gtap::detail::block {

static inline gtap_profile_export_result export_profile_impl(
    const gtap_profile_export_options& options,
    profile_export_error& error
) {
    gtap_profile_export_result result;
    const char* rejection = nullptr;
    switch (h_runtime_state) {
        case runtime_state::launched:
            break;
        case runtime_state::ready:
            rejection = "no kernel has been launched since initialization or reset";
            break;
        case runtime_state::uninitialized:
            rejection = "GTaP is not initialized";
            break;
        case runtime_state::reset_failed:
            rejection = "profile data is unavailable after a failed reset";
            break;
        case runtime_state::finalize_failed:
            rejection = "profile data is unavailable after a failed finalization";
            break;
    }
    if (rejection != nullptr) {
        error.fail("%s", rejection);
        return result;
    }
    if (!prepare_profile_output(options, result, error)) {
        return result;
    }

    const int blocks = h_launch_config.grid_size;
    int* device_indices = nullptr;
    int* indices = static_cast<int*>(malloc(sizeof(int) * blocks));
    unsigned long long* dropped = static_cast<unsigned long long*>(
        malloc(sizeof(unsigned long long) * blocks));
    long long* times = static_cast<long long*>(malloc(
        sizeof(long long) * blocks * profile_timestamp_capacity()));
    int* task_pool_peak_usage = static_cast<int*>(malloc(sizeof(int) * blocks));
    int* task_id_allocation_count = static_cast<int*>(malloc(sizeof(int) * blocks));
    if (!indices || !dropped || !times || !task_pool_peak_usage ||
        !task_id_allocation_count) {
        free(indices); free(dropped); free(times); free(task_pool_peak_usage); free(task_id_allocation_count);
        error.fail("host memory allocation failed");
        return result;
    }
    cudaError_t status = cudaMalloc(&device_indices, sizeof(int) * blocks);
    const char* cuda_failure = nullptr;
    if (status != cudaSuccess) {
        cuda_failure = "failed to allocate device memory for profile export: %s";
    }
    if (status == cudaSuccess) {
        get_block_working_time_counts<<<blocks, 1>>>(device_indices);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    if (status == cudaSuccess) status = cudaMemcpy(
        indices, device_indices, sizeof(int) * blocks, cudaMemcpyDeviceToHost);
    if (status == cudaSuccess) status = get_working_time_data(times);
    if (status == cudaSuccess) {
        status = get_block_profile_dropped_events_data(dropped);
    }
    if (status == cudaSuccess) {
        status = get_block_task_pool_peak_usage(task_pool_peak_usage);
    }
    if (status == cudaSuccess) {
        status = get_block_task_id_allocation_count(task_id_allocation_count);
    }
    if (status != cudaSuccess && cuda_failure == nullptr) {
        cuda_failure = "failed to collect profile data: %s";
    }
    if (device_indices != nullptr) {
        const cudaError_t free_status = cudaFree(device_indices);
        if (status == cudaSuccess) {
            status = free_status;
            if (status != cudaSuccess) {
                cuda_failure = "failed to release temporary device memory: %s";
            }
        }
    }
    if (status != cudaSuccess) {
        free(indices); free(dropped); free(times); free(task_pool_peak_usage); free(task_id_allocation_count);
        error.fail(cuda_failure, cudaGetErrorString(status));
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
            const long long value = times[static_cast<size_t>(block) * profile_timestamp_capacity() + i];
            if (value > 0 && (!origin || value < origin)) origin = value;
            if (value > profile_end) profile_end = value;
        }
    }
    if (!result.recorded_intervals) {
        free(indices); free(dropped); free(times); free(task_pool_peak_usage); free(task_id_allocation_count);
        error.fail("no profile intervals were recorded");
        return result;
    }

    double* durations = static_cast<double*>(malloc(
        sizeof(double) * result.recorded_intervals));
    double* all_block_ratios = static_cast<double*>(malloc(
        sizeof(double) * blocks));
    double* active_block_ratios = static_cast<double*>(malloc(
        sizeof(double) * blocks_with_executed_tasks));
    double* peak_usages = static_cast<double*>(malloc(
        sizeof(double) * blocks));
    double* task_id_allocation_counts = static_cast<double*>(malloc(
        sizeof(double) * blocks));
    if (!durations || !all_block_ratios || !active_block_ratios ||
        !peak_usages || !task_id_allocation_counts) {
        free(indices); free(dropped); free(times); free(task_pool_peak_usage); free(task_id_allocation_count);
        free(durations); free(all_block_ratios); free(active_block_ratios);
        free(peak_usages); free(task_id_allocation_counts);
        error.fail("host memory allocation failed");
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
                static_cast<size_t>(block) * profile_timestamp_capacity() + i;
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
    const profile_distribution duration_stats =
        compute_distribution(durations, task_index);
    const profile_distribution all_block_ratio_stats =
        compute_distribution(all_block_ratios, blocks);
    const profile_distribution active_block_ratio_stats =
        compute_distribution(
            active_block_ratios, active_block_index);
    for (int block = 0; block < blocks; ++block) {
        peak_usages[block] =
            static_cast<double>(task_pool_peak_usage[block]);
        task_id_allocation_counts[block] =
            static_cast<double>(task_id_allocation_count[block]);
    }
    const profile_distribution peak_usage_stats =
        compute_distribution(peak_usages, blocks);
    const profile_distribution task_id_allocation_count_stats =
        compute_distribution(task_id_allocation_counts, blocks);

    FILE* timeline = fopen(result.intervals_path, "w");
    bool io_ok = timeline != nullptr;
    if (!io_ok) {
        record_profile_open_failure(error, result.intervals_path);
    }
    if (io_ok) {
        bool wrote = fprintf(timeline,
                "block_id,start_ns,end_ns\n") >= 0;
        for (int block = 0; wrote && block < blocks; ++block) {
            for (int i = 0; wrote && i < indices[block]; i += 2) {
                const size_t offset =
                    static_cast<size_t>(block) * profile_timestamp_capacity() + i;
                wrote = fprintf(timeline, "%d,%lld,%lld\n", block,
                        times[offset] - origin, times[offset + 1] - origin) >= 0;
            }
        }
        if (!wrote) {
            record_profile_write_failure(error, result.intervals_path);
            io_ok = false;
        }
        if (fclose(timeline) != 0 && io_ok) {
            record_profile_write_failure(error, result.intervals_path);
            io_ok = false;
        }
    }
    FILE* stats = nullptr;
    if (io_ok) {
        stats = fopen(result.aggregates_path, "w");
        if (!stats) {
            record_profile_open_failure(error, result.aggregates_path);
            io_ok = false;
        }
    }
    if (stats) {
        bool wrote = fprintf(stats,
                "block_id,intervals_recorded,intervals_dropped,tasks_executed,"
                "recorded_task_execution_ns,first_execution_ns,"
                "last_execution_ns,task_pool_peak_usage,task_id_allocation_count\n") >= 0;
        for (int block = 0; wrote && block < blocks; ++block) {
            long long recorded_task_execution_ns = 0;
            long long first_execution_ns = 0;
            long long last_execution_ns = 0;
            for (int i = 0; i < indices[block]; i += 2) {
                const size_t offset =
                    static_cast<size_t>(block) * profile_timestamp_capacity() + i;
                const long long start_ns = times[offset] - origin;
                const long long end_ns = times[offset + 1] - origin;
                recorded_task_execution_ns += end_ns - start_ns;
                if (i == 0) first_execution_ns = start_ns;
                last_execution_ns = end_ns;
            }
            if (fprintf(stats, "%d,%d,%llu,%d,%lld,%lld,%lld,%d,%d\n", block,
                    indices[block] / 2, dropped[block], indices[block] / 2,
                    recorded_task_execution_ns, first_execution_ns,
                    last_execution_ns, task_pool_peak_usage[block],
                    task_id_allocation_count[block]) < 0) {
                wrote = false;
                break;
            }
        }
        if (!wrote) {
            record_profile_write_failure(error, result.aggregates_path);
            io_ok = false;
        }
        if (fclose(stats) != 0 && io_ok) {
            record_profile_write_failure(error, result.aggregates_path);
            io_ok = false;
        }
    }

    FILE* metadata = nullptr;
    if (io_ok) {
        metadata = fopen(result.profile_path, "w");
        if (!metadata) {
            record_profile_open_failure(error, result.profile_path);
            io_ok = false;
        }
    }
    if (metadata) {
        bool wrote = fputs(
            "{\n  \"schema_version\": 2,\n", metadata) != EOF;
        if (wrote && options.label) {
            wrote = fprintf(
                metadata, "  \"label\": \"%s\",\n", options.label) >= 0;
        }
        if (wrote) {
            wrote = fprintf(metadata,
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
            "  },\n"
            "  \"task_pool\": {\n"
            "    \"scheduling_unit\": \"block\",\n"
            "    \"capacity\": %d,\n"
            "    \"peak_usage\": {\n"
            "      \"count\": %zu, \"mean\": %.2f, \"stddev\": %.2f,\n"
            "      \"min\": %.0f, \"p50\": %.0f, \"p95\": %.0f,\n"
            "      \"p99\": %.0f, \"max\": %.0f\n"
            "    },\n"
            "    \"task_id_allocation_count\": {\n"
            "      \"count\": %zu, \"mean\": %.2f, \"stddev\": %.2f,\n"
            "      \"min\": %.0f, \"p50\": %.0f, \"p95\": %.0f,\n"
            "      \"p99\": %.0f, \"max\": %.0f\n"
            "    }\n"
            "  }\n"
            "}\n",
            h_launch_config.grid_size,
            h_launch_config.block_size,
            blocks, blocks_with_executed_tasks,
            h_launch_config.profile_interval_capacity,
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
            active_block_ratio_stats.p99, active_block_ratio_stats.max,
            h_launch_config.tasks_per_scheduling_unit,
            peak_usage_stats.count, peak_usage_stats.mean,
            peak_usage_stats.stddev, peak_usage_stats.min,
            peak_usage_stats.p50, peak_usage_stats.p95,
            peak_usage_stats.p99, peak_usage_stats.max,
            task_id_allocation_count_stats.count, task_id_allocation_count_stats.mean,
            task_id_allocation_count_stats.stddev, task_id_allocation_count_stats.min,
            task_id_allocation_count_stats.p50, task_id_allocation_count_stats.p95,
            task_id_allocation_count_stats.p99, task_id_allocation_count_stats.max) >= 0;
        }
        if (!wrote) {
            record_profile_write_failure(error, result.profile_path);
            io_ok = false;
        }
        if (fclose(metadata) != 0 && io_ok) {
            record_profile_write_failure(error, result.profile_path);
            io_ok = false;
        }
    }

    free(indices); free(dropped); free(times); free(task_pool_peak_usage); free(task_id_allocation_count);
    free(durations); free(all_block_ratios); free(active_block_ratios);
    free(peak_usages); free(task_id_allocation_counts);
    result.success = io_ok;
    return result;
}

}  // namespace gtap::detail::block

static inline gtap_profile_export_result gtap_export_profile(
    const gtap_profile_export_options& options = {}
) {
    gtap::detail::profile_export_error error{};
    gtap_profile_export_result result =
        gtap::detail::block::export_profile_impl(options, error);
    gtap::detail::print_profile_export_result(result, error);
    return result;
}

#else

static inline gtap_profile_export_result gtap_export_profile(
    const gtap_profile_export_options& = {}
) {
    gtap::detail::profile_export_error error{};
    error.fail("profiling is disabled; rebuild with GTAP_ENABLE_PROFILING");
    gtap_profile_export_result result;
    gtap::detail::print_profile_export_result(result, error);
    return result;
}

#endif

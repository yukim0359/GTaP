#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <utility>

#include "../common/profile_export.cuh"
#include "../common/runtime_config.cuh"

#include "profile_buffer.cuh"

#ifdef GTAP_ENABLE_PROFILING

namespace gtap::detail::thread {

static inline bool export_profile_impl(
    const gtap_profile_export_options& options,
    profile_export_error& error,
    std::string& resolved_directory
) {
    std::string result_directory;
    std::string profile_path;
    std::string intervals_path;
    std::string aggregates_path;
    size_t recorded_intervals = 0;
    size_t dropped_intervals = 0;

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
        return false;
    }
    if (!prepare_profile_output(
            options, result_directory, profile_path, intervals_path,
            aggregates_path, error)) {
        return false;
    }

    const int warps = h_launch_config.total_scheduling_units;
    int* device_indices = nullptr;
    int* indices = static_cast<int*>(malloc(sizeof(int) * warps));
    unsigned long long* dropped = static_cast<unsigned long long*>(
        malloc(sizeof(unsigned long long) * warps));
    long long* times = static_cast<long long*>(malloc(
        sizeof(long long) * warps * profile_timestamp_capacity()));
    int* task_counts = static_cast<int*>(malloc(
        sizeof(int) * warps * profile_timestamp_capacity()));
    int* task_pool_peak_usage = static_cast<int*>(malloc(sizeof(int) * warps));
    int* task_id_allocation_count = static_cast<int*>(malloc(sizeof(int) * warps));
    if (!indices || !dropped || !times || !task_counts || !task_pool_peak_usage ||
        !task_id_allocation_count) {
        free(indices); free(dropped); free(times); free(task_counts);
        free(task_pool_peak_usage); free(task_id_allocation_count);
        error.fail("host memory allocation failed");
        return false;
    }

    cudaError_t status = cudaMalloc(&device_indices, sizeof(int) * warps);
    const char* cuda_failure = nullptr;
    if (status != cudaSuccess) {
        cuda_failure = "failed to allocate device memory for profile export: %s";
    }
    if (status == cudaSuccess) {
        get_warp_working_time_counts<<<warps, 1>>>(device_indices);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    if (status == cudaSuccess) {
        status = cudaMemcpy(indices, device_indices, sizeof(int) * warps,
                            cudaMemcpyDeviceToHost);
    }
    if (status == cudaSuccess) status = get_warp_working_time_data(times);
    if (status == cudaSuccess) {
        status = get_warp_tasks_processed_count_data(task_counts);
    }
    if (status == cudaSuccess) {
        status = get_warp_profile_dropped_events_data(dropped);
    }
    if (status == cudaSuccess) {
        status = get_warp_task_pool_peak_usage(task_pool_peak_usage);
    }
    if (status == cudaSuccess) {
        status = get_warp_task_id_allocation_count(task_id_allocation_count);
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
        free(indices); free(dropped); free(times); free(task_counts);
        free(task_pool_peak_usage); free(task_id_allocation_count);
        error.fail(cuda_failure, cudaGetErrorString(status));
        return false;
    }

    long long origin = 0;
    long long profile_end = 0;
    int warps_with_executed_tasks = 0;
    for (int warp = 0; warp < warps; ++warp) {
        if (indices[warp] > 0) warps_with_executed_tasks++;
        recorded_intervals += static_cast<size_t>(indices[warp] / 2);
        dropped_intervals += static_cast<size_t>(dropped[warp]);
        for (int i = 0; i < indices[warp]; ++i) {
            const long long value =
                times[static_cast<size_t>(warp) * profile_timestamp_capacity() + i];
            if (value > 0 && (origin == 0 || value < origin)) origin = value;
            if (value > profile_end) profile_end = value;
        }
    }
    if (recorded_intervals == 0) {
        free(indices); free(dropped); free(times); free(task_counts);
        free(task_pool_peak_usage); free(task_id_allocation_count);
        error.fail("no profile intervals were recorded");
        return false;
    }

    double* durations = static_cast<double*>(malloc(
        sizeof(double) * recorded_intervals));
    double* batch_sizes = static_cast<double*>(malloc(
        sizeof(double) * recorded_intervals));
    double* all_warp_ratios = static_cast<double*>(malloc(
        sizeof(double) * warps));
    double* active_warp_ratios = static_cast<double*>(malloc(
        sizeof(double) * warps_with_executed_tasks));
    double* peak_usages = static_cast<double*>(malloc(
        sizeof(double) * warps));
    double* task_id_allocation_counts = static_cast<double*>(malloc(
        sizeof(double) * warps));
    if (!durations || !batch_sizes || !all_warp_ratios ||
        !active_warp_ratios || !peak_usages || !task_id_allocation_counts) {
        free(indices); free(dropped); free(times); free(task_counts);
        free(task_pool_peak_usage); free(task_id_allocation_count);
        free(durations); free(batch_sizes); free(all_warp_ratios);
        free(active_warp_ratios);
        free(peak_usages); free(task_id_allocation_counts);
        error.fail("host memory allocation failed");
        return false;
    }

    const double profile_span = profile_end > origin
        ? static_cast<double>(profile_end - origin) : 0.0;
    size_t batch_index = 0;
    size_t active_warp_index = 0;
    for (int warp = 0; warp < warps; ++warp) {
        long long execution_ns = 0;
        for (int i = 0; i < indices[warp]; i += 2) {
            const size_t offset =
                static_cast<size_t>(warp) * profile_timestamp_capacity() + i;
            const long long duration = times[offset + 1] - times[offset];
            durations[batch_index] = static_cast<double>(duration);
            batch_sizes[batch_index] = static_cast<double>(task_counts[offset]);
            execution_ns += duration;
            ++batch_index;
        }
        const double ratio = profile_span > 0.0
            ? static_cast<double>(execution_ns) / profile_span : 0.0;
        all_warp_ratios[warp] = ratio;
        if (indices[warp] > 0)
            active_warp_ratios[active_warp_index++] = ratio;
    }
    const profile_distribution duration_stats =
        compute_distribution(durations, batch_index);
    const profile_distribution batch_size_stats =
        compute_distribution(batch_sizes, batch_index);
    const profile_distribution all_warp_ratio_stats =
        compute_distribution(all_warp_ratios, warps);
    const profile_distribution active_warp_ratio_stats =
        compute_distribution(
            active_warp_ratios, active_warp_index);
    for (int warp = 0; warp < warps; ++warp) {
        peak_usages[warp] =
            static_cast<double>(task_pool_peak_usage[warp]);
        task_id_allocation_counts[warp] =
            static_cast<double>(task_id_allocation_count[warp]);
    }
    const profile_distribution peak_usage_stats =
        compute_distribution(peak_usages, warps);
    const profile_distribution task_id_allocation_count_stats =
        compute_distribution(task_id_allocation_counts, warps);

    FILE* timeline = fopen(intervals_path.c_str(), "w");
    bool io_ok = timeline != nullptr;
    if (!io_ok) {
        record_profile_open_failure(error, intervals_path.c_str());
    }
    if (io_ok) {
        bool wrote = fprintf(timeline,
                "warp_id,start_ns,end_ns,batch_task_count\n") >= 0;
        for (int warp = 0; wrote && warp < warps; ++warp) {
            for (int i = 0; wrote && i < indices[warp]; i += 2) {
                const size_t offset =
                    static_cast<size_t>(warp) * profile_timestamp_capacity() + i;
                wrote = fprintf(timeline, "%d,%lld,%lld,%d\n", warp,
                        times[offset] - origin, times[offset + 1] - origin,
                        task_counts[offset]) >= 0;
            }
        }
        if (!wrote) {
            record_profile_write_failure(error, intervals_path.c_str());
            io_ok = false;
        }
        if (fclose(timeline) != 0 && io_ok) {
            record_profile_write_failure(error, intervals_path.c_str());
            io_ok = false;
        }
    }
    FILE* stats = nullptr;
    if (io_ok) {
        stats = fopen(aggregates_path.c_str(), "w");
        if (!stats) {
            record_profile_open_failure(error, aggregates_path.c_str());
            io_ok = false;
        }
    }
    if (stats) {
        bool wrote = fprintf(stats,
                "warp_id,intervals_recorded,intervals_dropped,tasks_executed,"
                "recorded_task_execution_ns,first_execution_ns,"
                "last_execution_ns,task_pool_peak_usage,task_id_allocation_count\n") >= 0;
        for (int warp = 0; wrote && warp < warps; ++warp) {
            long long tasks_executed = 0;
            long long recorded_task_execution_ns = 0;
            long long first_execution_ns = 0;
            long long last_execution_ns = 0;
            for (int i = 0; i < indices[warp]; i += 2) {
                const size_t offset =
                    static_cast<size_t>(warp) * profile_timestamp_capacity() + i;
                const long long start_ns = times[offset] - origin;
                const long long end_ns = times[offset + 1] - origin;
                recorded_task_execution_ns += end_ns - start_ns;
                if (i == 0) first_execution_ns = start_ns;
                last_execution_ns = end_ns;
                tasks_executed +=
                    task_counts[offset + 1];
            }
            if (fprintf(stats, "%d,%d,%llu,%lld,%lld,%lld,%lld,%d,%d\n", warp,
                    indices[warp] / 2, dropped[warp], tasks_executed,
                    recorded_task_execution_ns, first_execution_ns,
                    last_execution_ns, task_pool_peak_usage[warp],
                    task_id_allocation_count[warp]) < 0) {
                wrote = false;
                break;
            }
        }
        if (!wrote) {
            record_profile_write_failure(error, aggregates_path.c_str());
            io_ok = false;
        }
        if (fclose(stats) != 0 && io_ok) {
            record_profile_write_failure(error, aggregates_path.c_str());
            io_ok = false;
        }
    }

    FILE* metadata = nullptr;
    if (io_ok) {
        metadata = fopen(profile_path.c_str(), "w");
        if (!metadata) {
            record_profile_open_failure(error, profile_path.c_str());
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
            "  \"mode\": \"thread\",\n"
            "  \"grid_size\": %d,\n"
            "  \"block_size\": %d,\n"
            "  \"task_execution\": {\n"
            "    \"warp_counts\": {\n"
            "      \"total\": %d,\n"
            "      \"with_executed_tasks\": %d\n"
            "    },\n"
            "    \"intervals\": {\n"
            "      \"recording_limit_per_warp\": %d,\n"
            "      \"recorded_count\": %zu,\n"
            "      \"dropped_count\": %zu,\n"
            "      \"duration_ns\": {\n"
            "        \"mean\": %.2f, \"stddev\": %.2f,\n"
            "        \"min\": %.0f, \"p50\": %.0f, \"p95\": %.0f,\n"
            "        \"p99\": %.0f, \"max\": %.0f\n"
            "      },\n"
            "      \"tasks_executed_per_interval\": {\n"
            "        \"mean\": %.2f, \"stddev\": %.2f,\n"
            "        \"min\": %.0f, \"p50\": %.0f, \"p95\": %.0f,\n"
            "        \"p99\": %.0f, \"max\": %.0f\n"
            "      }\n"
            "    },\n"
            "    \"execution_ratio_per_warp\": {\n"
            "      \"all_warps\": {\n"
            "        \"count\": %zu, \"mean\": %.6f, \"stddev\": %.6f,\n"
            "        \"min\": %.6f, \"p50\": %.6f, \"p95\": %.6f,\n"
            "        \"p99\": %.6f, \"max\": %.6f\n"
            "      },\n"
            "      \"warps_with_executed_tasks\": {\n"
            "        \"count\": %zu, \"mean\": %.6f, \"stddev\": %.6f,\n"
            "        \"min\": %.6f, \"p50\": %.6f, \"p95\": %.6f,\n"
            "        \"p99\": %.6f, \"max\": %.6f\n"
            "      }\n"
            "    }\n"
            "  },\n"
            "  \"task_pool\": {\n"
            "    \"scheduling_unit\": \"warp\",\n"
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
            warps, warps_with_executed_tasks,
            h_launch_config.profile_interval_capacity,
            recorded_intervals, dropped_intervals,
            duration_stats.mean, duration_stats.stddev,
            duration_stats.min, duration_stats.p50, duration_stats.p95,
            duration_stats.p99, duration_stats.max,
            batch_size_stats.mean, batch_size_stats.stddev,
            batch_size_stats.min, batch_size_stats.p50, batch_size_stats.p95,
            batch_size_stats.p99, batch_size_stats.max,
            all_warp_ratio_stats.count, all_warp_ratio_stats.mean,
            all_warp_ratio_stats.stddev, all_warp_ratio_stats.min,
            all_warp_ratio_stats.p50, all_warp_ratio_stats.p95,
            all_warp_ratio_stats.p99, all_warp_ratio_stats.max,
            active_warp_ratio_stats.count, active_warp_ratio_stats.mean,
            active_warp_ratio_stats.stddev, active_warp_ratio_stats.min,
            active_warp_ratio_stats.p50, active_warp_ratio_stats.p95,
            active_warp_ratio_stats.p99, active_warp_ratio_stats.max,
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
            record_profile_write_failure(error, profile_path.c_str());
            io_ok = false;
        }
        if (fclose(metadata) != 0 && io_ok) {
            record_profile_write_failure(error, profile_path.c_str());
            io_ok = false;
        }
    }

    free(indices); free(dropped); free(times); free(task_counts);
    free(task_pool_peak_usage); free(task_id_allocation_count);
    free(durations); free(batch_sizes); free(all_warp_ratios);
    free(active_warp_ratios);
    free(peak_usages); free(task_id_allocation_counts);
    if (io_ok) {
        resolved_directory = std::move(result_directory);
    }
    return io_ok;
}

}  // namespace gtap::detail::thread

static inline bool gtap_export_profile(
    const gtap_profile_export_options& options = {},
    std::string* result_directory = nullptr
) {
    gtap::detail::profile_export_error error{};
    std::string resolved_directory;
    const bool success = gtap::detail::thread::export_profile_impl(
        options, error, resolved_directory);
    gtap::detail::print_profile_export_result(
        success, resolved_directory.c_str(), error);
    if (success && result_directory != nullptr) {
        *result_directory = std::move(resolved_directory);
    }
    return success;
}

#else

static inline bool gtap_export_profile(
    const gtap_profile_export_options& = {},
    std::string* = nullptr
) {
    gtap::detail::profile_export_error error{};
    error.fail("profiling is disabled; rebuild with GTAP_ENABLE_PROFILING");
    gtap::detail::print_profile_export_result(false, nullptr, error);
    return false;
}

#endif

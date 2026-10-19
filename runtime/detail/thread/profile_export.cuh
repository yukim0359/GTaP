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

struct profile_data {
    int* indices = nullptr;
    unsigned long long* dropped = nullptr;
    long long* times = nullptr;
    int* task_counts = nullptr;
    int* task_pool_peak_usage = nullptr;
    int* task_id_allocation_count = nullptr;
};

struct profile_statistics {
    long long origin = 0;
    int warps_with_executed_tasks = 0;
    size_t recorded_intervals = 0;
    size_t dropped_intervals = 0;
    profile_distribution duration_stats;
    profile_distribution batch_size_stats;
    profile_distribution all_warp_ratio_stats;
    profile_distribution active_warp_ratio_stats;
    profile_distribution peak_usage_stats;
    profile_distribution task_id_allocation_count_stats;
};

static inline bool write_profile_intervals(
    profile_export_error& error,
    const std::string& path,
    int warps,
    const profile_data& data,
    long long origin
) {
    FILE* file = fopen(path.c_str(), "w");
    if (file == nullptr) {
        record_profile_open_failure(error, path.c_str());
        return false;
    }
    bool wrote = fprintf(file,
            "warp_id,start_ns,end_ns,batch_task_count\n") >= 0;
    for (int warp = 0; wrote && warp < warps; ++warp) {
        for (int i = 0; wrote && i < data.indices[warp]; i += 2) {
            const size_t offset =
                static_cast<size_t>(warp) * profile_timestamp_capacity() + i;
            wrote = fprintf(file, "%d,%lld,%lld,%d\n", warp,
                    data.times[offset] - origin, data.times[offset + 1] - origin,
                    data.task_counts[offset]) >= 0;
        }
    }
    if (!wrote) {
        record_profile_write_failure(error, path.c_str());
    }
    if (fclose(file) != 0 && wrote) {
        record_profile_write_failure(error, path.c_str());
        wrote = false;
    }
    return wrote;
}

static inline bool write_profile_aggregates(
    profile_export_error& error,
    const std::string& path,
    int warps,
    const profile_data& data,
    long long origin
) {
    FILE* file = fopen(path.c_str(), "w");
    if (file == nullptr) {
        record_profile_open_failure(error, path.c_str());
        return false;
    }
    bool wrote = fprintf(file,
            "warp_id,intervals_recorded,intervals_dropped,tasks_executed,"
            "recorded_task_execution_ns,first_execution_ns,"
            "last_execution_ns,task_pool_peak_usage,task_id_allocation_count\n") >= 0;
    for (int warp = 0; wrote && warp < warps; ++warp) {
        long long tasks_executed = 0;
        long long recorded_task_execution_ns = 0;
        long long first_execution_ns = 0;
        long long last_execution_ns = 0;
        for (int i = 0; i < data.indices[warp]; i += 2) {
            const size_t offset =
                static_cast<size_t>(warp) * profile_timestamp_capacity() + i;
            const long long start_ns = data.times[offset] - origin;
            const long long end_ns = data.times[offset + 1] - origin;
            recorded_task_execution_ns += end_ns - start_ns;
            if (i == 0) first_execution_ns = start_ns;
            last_execution_ns = end_ns;
            tasks_executed += data.task_counts[offset + 1];
        }
        if (fprintf(file, "%d,%d,%llu,%lld,%lld,%lld,%lld,%d,%d\n", warp,
                data.indices[warp] / 2, data.dropped[warp], tasks_executed,
                recorded_task_execution_ns, first_execution_ns,
                last_execution_ns, data.task_pool_peak_usage[warp],
                data.task_id_allocation_count[warp]) < 0) {
            wrote = false;
            break;
        }
    }
    if (!wrote) {
        record_profile_write_failure(error, path.c_str());
    }
    if (fclose(file) != 0 && wrote) {
        record_profile_write_failure(error, path.c_str());
        wrote = false;
    }
    return wrote;
}

static inline bool write_profile_metadata(
    profile_export_error& error,
    const std::string& path,
    const gtap_profile_export_options& options,
    int warps,
    const profile_statistics& statistics
) {
    FILE* file = fopen(path.c_str(), "w");
    if (file == nullptr) {
        record_profile_open_failure(error, path.c_str());
        return false;
    }
    bool wrote = fputs("{\n  \"schema_version\": 2,\n", file) != EOF;
    if (wrote && options.label) {
        wrote = fprintf(file, "  \"label\": \"%s\",\n", options.label) >= 0;
    }
    if (wrote) {
        wrote = fprintf(file,
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
            warps, statistics.warps_with_executed_tasks,
            h_launch_config.profile_interval_capacity,
            statistics.recorded_intervals, statistics.dropped_intervals,
            statistics.duration_stats.mean, statistics.duration_stats.stddev,
            statistics.duration_stats.min, statistics.duration_stats.p50,
            statistics.duration_stats.p95, statistics.duration_stats.p99,
            statistics.duration_stats.max,
            statistics.batch_size_stats.mean, statistics.batch_size_stats.stddev,
            statistics.batch_size_stats.min, statistics.batch_size_stats.p50,
            statistics.batch_size_stats.p95, statistics.batch_size_stats.p99,
            statistics.batch_size_stats.max,
            statistics.all_warp_ratio_stats.count,
            statistics.all_warp_ratio_stats.mean,
            statistics.all_warp_ratio_stats.stddev,
            statistics.all_warp_ratio_stats.min,
            statistics.all_warp_ratio_stats.p50,
            statistics.all_warp_ratio_stats.p95,
            statistics.all_warp_ratio_stats.p99,
            statistics.all_warp_ratio_stats.max,
            statistics.active_warp_ratio_stats.count,
            statistics.active_warp_ratio_stats.mean,
            statistics.active_warp_ratio_stats.stddev,
            statistics.active_warp_ratio_stats.min,
            statistics.active_warp_ratio_stats.p50,
            statistics.active_warp_ratio_stats.p95,
            statistics.active_warp_ratio_stats.p99,
            statistics.active_warp_ratio_stats.max,
            h_launch_config.tasks_per_scheduling_unit,
            statistics.peak_usage_stats.count, statistics.peak_usage_stats.mean,
            statistics.peak_usage_stats.stddev, statistics.peak_usage_stats.min,
            statistics.peak_usage_stats.p50, statistics.peak_usage_stats.p95,
            statistics.peak_usage_stats.p99, statistics.peak_usage_stats.max,
            statistics.task_id_allocation_count_stats.count,
            statistics.task_id_allocation_count_stats.mean,
            statistics.task_id_allocation_count_stats.stddev,
            statistics.task_id_allocation_count_stats.min,
            statistics.task_id_allocation_count_stats.p50,
            statistics.task_id_allocation_count_stats.p95,
            statistics.task_id_allocation_count_stats.p99,
            statistics.task_id_allocation_count_stats.max) >= 0;
    }
    if (!wrote) {
        record_profile_write_failure(error, path.c_str());
    }
    if (fclose(file) != 0 && wrote) {
        record_profile_write_failure(error, path.c_str());
        wrote = false;
    }
    return wrote;
}

static inline void free_profile_data(profile_data& data) {
    free(data.indices);
    free(data.dropped);
    free(data.times);
    free(data.task_counts);
    free(data.task_pool_peak_usage);
    free(data.task_id_allocation_count);
    data = {};
}

static inline bool collect_profile_data(
    profile_export_error& error,
    int warps,
    profile_data& data
) {
    int* device_indices = nullptr;
    data.indices = static_cast<int*>(malloc(sizeof(int) * warps));
    data.dropped = static_cast<unsigned long long*>(
        malloc(sizeof(unsigned long long) * warps));
    data.times = static_cast<long long*>(malloc(
        sizeof(long long) * warps * profile_timestamp_capacity()));
    data.task_counts = static_cast<int*>(malloc(
        sizeof(int) * warps * profile_timestamp_capacity()));
    data.task_pool_peak_usage = static_cast<int*>(malloc(sizeof(int) * warps));
    data.task_id_allocation_count = static_cast<int*>(malloc(sizeof(int) * warps));
    if (!data.indices || !data.dropped || !data.times || !data.task_counts ||
        !data.task_pool_peak_usage || !data.task_id_allocation_count) {
        free_profile_data(data);
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
        status = cudaMemcpy(data.indices, device_indices, sizeof(int) * warps,
                            cudaMemcpyDeviceToHost);
    }
    if (status == cudaSuccess) status = get_warp_working_time_data(data.times);
    if (status == cudaSuccess) {
        status = get_warp_tasks_processed_count_data(data.task_counts);
    }
    if (status == cudaSuccess) {
        status = get_warp_profile_dropped_events_data(data.dropped);
    }
    if (status == cudaSuccess) {
        status = get_warp_task_pool_peak_usage(data.task_pool_peak_usage);
    }
    if (status == cudaSuccess) {
        status = get_warp_task_id_allocation_count(data.task_id_allocation_count);
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
        free_profile_data(data);
        error.fail(cuda_failure, cudaGetErrorString(status));
        return false;
    }
    return true;
}

static inline bool compute_profile_statistics(
    profile_export_error& error,
    int warps,
    const profile_data& data,
    profile_statistics& statistics
) {
    statistics = {};
    long long profile_end = 0;
    for (int warp = 0; warp < warps; ++warp) {
        if (data.indices[warp] > 0) statistics.warps_with_executed_tasks++;
        statistics.recorded_intervals += static_cast<size_t>(data.indices[warp] / 2);
        statistics.dropped_intervals += static_cast<size_t>(data.dropped[warp]);
        for (int i = 0; i < data.indices[warp]; ++i) {
            const long long value =
                data.times[static_cast<size_t>(warp) * profile_timestamp_capacity() + i];
            if (value > 0 && (statistics.origin == 0 || value < statistics.origin)) {
                statistics.origin = value;
            }
            if (value > profile_end) profile_end = value;
        }
    }
    if (statistics.recorded_intervals == 0) {
        error.fail("no profile intervals were recorded");
        return false;
    }

    double* durations = static_cast<double*>(malloc(
        sizeof(double) * statistics.recorded_intervals));
    double* batch_sizes = static_cast<double*>(malloc(
        sizeof(double) * statistics.recorded_intervals));
    double* all_warp_ratios = static_cast<double*>(malloc(
        sizeof(double) * warps));
    double* active_warp_ratios = static_cast<double*>(malloc(
        sizeof(double) * statistics.warps_with_executed_tasks));
    double* peak_usages = static_cast<double*>(malloc(
        sizeof(double) * warps));
    double* task_id_allocation_counts = static_cast<double*>(malloc(
        sizeof(double) * warps));
    if (!durations || !batch_sizes || !all_warp_ratios ||
        !active_warp_ratios || !peak_usages || !task_id_allocation_counts) {
        free(durations);
        free(batch_sizes);
        free(all_warp_ratios);
        free(active_warp_ratios);
        free(peak_usages);
        free(task_id_allocation_counts);
        error.fail("host memory allocation failed");
        return false;
    }

    const double profile_span = profile_end > statistics.origin
        ? static_cast<double>(profile_end - statistics.origin) : 0.0;
    size_t batch_index = 0;
    size_t active_warp_index = 0;
    for (int warp = 0; warp < warps; ++warp) {
        long long execution_ns = 0;
        for (int i = 0; i < data.indices[warp]; i += 2) {
            const size_t offset =
                static_cast<size_t>(warp) * profile_timestamp_capacity() + i;
            const long long duration = data.times[offset + 1] - data.times[offset];
            durations[batch_index] = static_cast<double>(duration);
            batch_sizes[batch_index] = static_cast<double>(data.task_counts[offset]);
            execution_ns += duration;
            ++batch_index;
        }
        const double ratio = profile_span > 0.0
            ? static_cast<double>(execution_ns) / profile_span : 0.0;
        all_warp_ratios[warp] = ratio;
        if (data.indices[warp] > 0)
            active_warp_ratios[active_warp_index++] = ratio;
    }
    statistics.duration_stats =
        compute_distribution(durations, batch_index);
    statistics.batch_size_stats =
        compute_distribution(batch_sizes, batch_index);
    statistics.all_warp_ratio_stats =
        compute_distribution(all_warp_ratios, warps);
    statistics.active_warp_ratio_stats =
        compute_distribution(active_warp_ratios, active_warp_index);
    for (int warp = 0; warp < warps; ++warp) {
        peak_usages[warp] =
            static_cast<double>(data.task_pool_peak_usage[warp]);
        task_id_allocation_counts[warp] =
            static_cast<double>(data.task_id_allocation_count[warp]);
    }
    statistics.peak_usage_stats =
        compute_distribution(peak_usages, warps);
    statistics.task_id_allocation_count_stats =
        compute_distribution(task_id_allocation_counts, warps);
    free(durations);
    free(batch_sizes);
    free(all_warp_ratios);
    free(active_warp_ratios);
    free(peak_usages);
    free(task_id_allocation_counts);
    return true;
}

static inline bool export_profile_impl(
    const gtap_profile_export_options& options,
    profile_export_error& error,
    std::string& resolved_directory
) {
    std::string result_directory;
    std::string profile_path;
    std::string intervals_path;
    std::string aggregates_path;

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
        case runtime_state::synchronize_failed:
            rejection = "profile data is unavailable after a failed synchronization";
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
    profile_data data;
    if (!collect_profile_data(error, warps, data)) {
        return false;
    }

    profile_statistics statistics;
    if (!compute_profile_statistics(error, warps, data, statistics)) {
        free_profile_data(data);
        return false;
    }

    bool io_ok = write_profile_intervals(
        error, intervals_path, warps, data, statistics.origin);
    if (io_ok) {
        io_ok = write_profile_aggregates(
            error, aggregates_path, warps, data, statistics.origin);
    }
    if (io_ok) {
        io_ok = write_profile_metadata(
            error, profile_path, options, warps, statistics);
    }

    free_profile_data(data);
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

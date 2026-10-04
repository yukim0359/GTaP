#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/profile_export.cuh"
#include "../common/runtime.cuh"
#include "profile_buffer.cuh"

#ifdef GTAP_ENABLE_PROFILING

namespace gtap::detail::thread {

cudaError_t get_warp_working_time_data(long long* host_working_time) {
    long long* ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, working_time, sizeof(ptr)));
    return cudaMemcpy(
        host_working_time, ptr,
        sizeof(long long) * stored_launch_config().total_workers * profile_capacity(),
        cudaMemcpyDeviceToHost);
}

cudaError_t get_warp_tasks_processed_count_data(int* host_counts) {
    int* ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, tasks_processed_count, sizeof(ptr)));
    return cudaMemcpy(
        host_counts, ptr,
        sizeof(int) * stored_launch_config().total_workers * profile_capacity(),
        cudaMemcpyDeviceToHost);
}

cudaError_t get_warp_profile_dropped_events_data(
    unsigned long long* host_counts
) {
    unsigned long long* ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &ptr, profile_dropped_events, sizeof(ptr)));
    return cudaMemcpy(
        host_counts, ptr,
        sizeof(unsigned long long) *
            stored_launch_config().total_workers,
        cudaMemcpyDeviceToHost);
}

__global__ void get_warp_working_time_counts(int* counts) {
    if (threadIdx.x == 0) {
        int wid = blockIdx.x;
        int count = 0;
        for (int i = 0; i < profile_capacity(); i++) {
            if (working_time[wid * profile_capacity() + i] > 0) count++;
        }
        counts[wid] = count;
    }
}

}  // namespace gtap::detail::thread

static inline gtap_profile_export_result gtap_export_profile(
    const gtap_profile_export_options& options = {}
) {
    gtap_profile_export_result result;
    if (gtap::detail::prepare_profile_output(options, result) !=
        gtap_profile_export_status::success) {
        return result;
    }

    const int warps = gtap::detail::stored_launch_config().total_workers;
    int* device_indices = nullptr;
    int* indices = static_cast<int*>(malloc(sizeof(int) * warps));
    unsigned long long* dropped = static_cast<unsigned long long*>(
        malloc(sizeof(unsigned long long) * warps));
    long long* times = static_cast<long long*>(malloc(
        sizeof(long long) * warps * gtap::detail::profile_capacity()));
    int* task_counts = static_cast<int*>(malloc(
        sizeof(int) * warps * gtap::detail::profile_capacity()));
    if (!indices || !dropped || !times || !task_counts) {
        free(indices); free(dropped); free(times); free(task_counts);
        result.status = gtap_profile_export_status::out_of_memory;
        printf("GTaP profile not written: host memory allocation failed\n");
        return result;
    }

    cudaError_t error = cudaMalloc(&device_indices, sizeof(int) * warps);
    if (error == cudaSuccess) {
        gtap::detail::thread::get_warp_working_time_counts<<<warps, 1>>>(device_indices);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error == cudaSuccess) {
        error = cudaMemcpy(indices, device_indices, sizeof(int) * warps,
                           cudaMemcpyDeviceToHost);
    }
    if (error == cudaSuccess) error = gtap::detail::thread::get_warp_working_time_data(times);
    if (error == cudaSuccess) {
        error = gtap::detail::thread::get_warp_tasks_processed_count_data(task_counts);
    }
    if (error == cudaSuccess) {
#ifdef GTAP_PROFILE_HAS_DROPPED_COUNTER
        error = gtap::detail::thread::get_warp_profile_dropped_events_data(dropped);
#else
        memset(dropped, 0, sizeof(unsigned long long) * warps);
#endif
    }
    cudaFree(device_indices);
    if (error != cudaSuccess) {
        free(indices); free(dropped); free(times); free(task_counts);
        result.status = gtap_profile_export_status::cuda_error;
        printf("GTaP profile not written: CUDA data transfer failed\n");
        return result;
    }

    long long origin = 0;
    long long profile_end = 0;
    int warps_with_executed_tasks = 0;
    for (int warp = 0; warp < warps; ++warp) {
        if (indices[warp] > 0) warps_with_executed_tasks++;
        result.recorded_intervals += static_cast<size_t>(indices[warp] / 2);
        result.dropped_intervals += static_cast<size_t>(dropped[warp]);
        for (int i = 0; i < indices[warp]; ++i) {
            const long long value =
                times[warp * gtap::detail::profile_capacity() + i];
            if (value > 0 && (origin == 0 || value < origin)) origin = value;
            if (value > profile_end) profile_end = value;
        }
    }
    if (result.recorded_intervals == 0) {
        free(indices); free(dropped); free(times); free(task_counts);
        result.status = gtap_profile_export_status::no_data;
        printf("GTaP profile not written: no task execution data\n");
        return result;
    }

    double* durations = static_cast<double*>(malloc(
        sizeof(double) * result.recorded_intervals));
    double* batch_sizes = static_cast<double*>(malloc(
        sizeof(double) * result.recorded_intervals));
    double* all_warp_ratios = static_cast<double*>(malloc(
        sizeof(double) * warps));
    double* active_warp_ratios = static_cast<double*>(malloc(
        sizeof(double) * warps_with_executed_tasks));
    if (!durations || !batch_sizes || !all_warp_ratios ||
        !active_warp_ratios) {
        free(indices); free(dropped); free(times); free(task_counts);
        free(durations); free(batch_sizes); free(all_warp_ratios);
        free(active_warp_ratios);
        result.status = gtap_profile_export_status::out_of_memory;
        printf("GTaP profile not written: host memory allocation failed\n");
        return result;
    }

    const double profile_span = profile_end > origin
        ? static_cast<double>(profile_end - origin) : 0.0;
    size_t batch_index = 0;
    size_t active_warp_index = 0;
    for (int warp = 0; warp < warps; ++warp) {
        long long execution_ns = 0;
        for (int i = 0; i < indices[warp]; i += 2) {
            const size_t offset =
                static_cast<size_t>(warp) * gtap::detail::profile_capacity() + i;
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
    const gtap::detail::profile_distribution duration_stats =
        gtap::detail::compute_distribution(durations, batch_index);
    const gtap::detail::profile_distribution batch_size_stats =
        gtap::detail::compute_distribution(batch_sizes, batch_index);
    const gtap::detail::profile_distribution all_warp_ratio_stats =
        gtap::detail::compute_distribution(all_warp_ratios, warps);
    const gtap::detail::profile_distribution active_warp_ratio_stats =
        gtap::detail::compute_distribution(
            active_warp_ratios, active_warp_index);

    FILE* timeline = fopen(result.intervals_path, "w");
    if (!timeline) {
        free(indices); free(dropped); free(times); free(task_counts);
        free(durations); free(batch_sizes); free(all_warp_ratios);
        free(active_warp_ratios);
        result.status = gtap_profile_export_status::io_error;
        printf("GTaP profile not written: failed to write %s\n",
               result.result_directory);
        return result;
    }
    fprintf(timeline,
            "warp_id,start_ns,end_ns,batch_task_count\n");
    for (int warp = 0; warp < warps; ++warp) {
        for (int i = 0; i < indices[warp]; i += 2) {
            const size_t offset =
                static_cast<size_t>(warp) * gtap::detail::profile_capacity() + i;
            fprintf(timeline, "%d,%lld,%lld,%d\n", warp,
                    times[offset] - origin, times[offset + 1] - origin,
                    task_counts[offset]);
        }
    }
    bool io_ok = gtap::detail::close_profile_file(timeline);
    FILE* stats = io_ok ? fopen(result.aggregates_path, "w") : nullptr;
    if (stats) {
        fprintf(stats,
                "warp_id,intervals_recorded,intervals_dropped,tasks_executed,"
                "recorded_task_execution_ns,first_execution_ns,"
                "last_execution_ns\n");
        for (int warp = 0; warp < warps; ++warp) {
            long long tasks_executed = 0;
            long long recorded_task_execution_ns = 0;
            long long first_execution_ns = 0;
            long long last_execution_ns = 0;
            for (int i = 0; i < indices[warp]; i += 2) {
                const size_t offset =
                    static_cast<size_t>(warp) * gtap::detail::profile_capacity() + i;
                const long long start_ns = times[offset] - origin;
                const long long end_ns = times[offset + 1] - origin;
                recorded_task_execution_ns += end_ns - start_ns;
                if (i == 0) first_execution_ns = start_ns;
                last_execution_ns = end_ns;
                tasks_executed +=
                    task_counts[offset + 1];
            }
            fprintf(stats, "%d,%d,%llu,%lld,%lld,%lld,%lld\n", warp,
                    indices[warp] / 2, dropped[warp], tasks_executed,
                    recorded_task_execution_ns, first_execution_ns,
                    last_execution_ns);
        }
        io_ok = gtap::detail::close_profile_file(stats);
    } else {
        io_ok = false;
    }

    FILE* metadata = io_ok ? fopen(result.profile_path, "w") : nullptr;
    if (metadata) {
        bool metadata_ok = fputs(
            "{\n  \"schema_version\": 1,\n", metadata) != EOF;
        if (metadata_ok && options.label) {
            metadata_ok = fprintf(
                metadata, "  \"label\": \"%s\",\n", options.label) >= 0;
        }
        metadata_ok = metadata_ok && fprintf(metadata,
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
            "  }\n"
            "}\n",
            gtap::detail::stored_launch_config().grid_size,
            gtap::detail::stored_launch_config().block_size,
            warps, warps_with_executed_tasks,
            gtap::detail::stored_launch_config().profile_interval_capacity,
            result.recorded_intervals, result.dropped_intervals,
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
            active_warp_ratio_stats.p99, active_warp_ratio_stats.max) >= 0;
        io_ok = metadata_ok && gtap::detail::close_profile_file(metadata);
    } else {
        io_ok = false;
    }

    free(indices); free(dropped); free(times); free(task_counts);
    free(durations); free(batch_sizes); free(all_warp_ratios);
    free(active_warp_ratios);
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

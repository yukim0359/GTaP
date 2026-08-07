#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

enum class gtap_profile_export_status {
    success = 0,
    invalid_label,
    invalid_output_directory,
    path_too_long,
    already_exists,
    cuda_error,
    out_of_memory,
    io_error,
    no_data
};

struct gtap_profile_export_options {
    const char* output_directory = "./profile";
    const char* label = nullptr;
    bool overwrite = false;
};

static inline bool gtap_profile_valid_label(const char* label) {
    if (!label) return true;
    const size_t length = strlen(label);
    if (length == 0 || length > 128) return false;
    for (const unsigned char* p =
             reinterpret_cast<const unsigned char*>(label); *p; ++p) {
        if (!((*p >= 'A' && *p <= 'Z') ||
               (*p >= 'a' && *p <= 'z') ||
               (*p >= '0' && *p <= '9') ||
               *p == '-' || *p == '_' || *p == '.')) {
            return false;
        }
    }
    return true;
}

struct gtap_profile_export_result {
    gtap_profile_export_status status =
        gtap_profile_export_status::io_error;
    cudaError_t cuda_error = cudaSuccess;
    int system_error = 0;
    char error_path[512] = {};
    size_t recorded_intervals = 0;
    size_t dropped_intervals = 0;
    bool complete = false;
    int generated_files = 0;
    char result_directory[512] = {};
    char profile_path[512] = {};
    char timeline_path[512] = {};
    char statistics_path[512] = {};
};

struct gtap_profile_distribution {
    size_t count = 0;
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double max = 0.0;
};

static inline int gtap_profile_compare_double(const void* lhs, const void* rhs) {
    const double a = *static_cast<const double*>(lhs);
    const double b = *static_cast<const double*>(rhs);
    return (a > b) - (a < b);
}

static inline double gtap_profile_nearest_rank(
    const double* sorted, size_t count, double quantile
) {
    if (!count) return 0.0;
    const double exact_rank = quantile * static_cast<double>(count);
    size_t rank = static_cast<size_t>(exact_rank);
    if (static_cast<double>(rank) < exact_rank) ++rank;
    if (rank == 0) rank = 1;
    if (rank > count) rank = count;
    return sorted[rank - 1];
}

// Avoid imposing a libm link dependency on programs that enable profiling.
static inline double gtap_profile_sqrt(double value) {
    if (value <= 0.0) return 0.0;
    double estimate = value >= 1.0 ? value : 1.0;
    for (int iteration = 0; iteration < 64; ++iteration) {
        const double next = 0.5 * (estimate + value / estimate);
        if (next == estimate) break;
        estimate = next;
    }
    return estimate;
}

// Sorts values in place.
static inline gtap_profile_distribution gtap_profile_compute_distribution(
    double* values, size_t count
) {
    gtap_profile_distribution stats;
    stats.count = count;
    if (!count) return stats;

    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) sum += values[i];
    stats.mean = sum / static_cast<double>(count);

    double squared_deviation_sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double deviation = values[i] - stats.mean;
        squared_deviation_sum += deviation * deviation;
    }
    stats.stddev = gtap_profile_sqrt(
        squared_deviation_sum / static_cast<double>(count));

    qsort(values, count, sizeof(double), gtap_profile_compare_double);
    stats.min = values[0];
    stats.p50 = gtap_profile_nearest_rank(values, count, 0.50);
    stats.p95 = gtap_profile_nearest_rank(values, count, 0.95);
    stats.p99 = gtap_profile_nearest_rank(values, count, 0.99);
    stats.max = values[count - 1];
    return stats;
}

static inline void gtap_profile_set_error_path(
    gtap_profile_export_result* result, const char* path
) {
    if (!path) return;
    snprintf(result->error_path, sizeof(result->error_path), "%s", path);
}

static inline bool gtap_profile_is_directory(const char* path) {
    struct stat st = {};
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static inline bool gtap_profile_path_exists(const char* path) {
    struct stat st = {};
    return stat(path, &st) == 0;
}

static inline bool gtap_profile_create_parents(const char* path) {
    if (!path || !path[0] || strlen(path) >= 512) return false;
    char copy[512] = {};
    memcpy(copy, path, strlen(path) + 1);
    for (char* p = copy + 1; *p; ++p) {
        if (*p != '/') continue;
        *p = '\0';
        if (copy[0] && mkdir(copy, 0755) != 0 &&
            !(errno == EEXIST && gtap_profile_is_directory(copy))) {
            return false;
        }
        *p = '/';
    }
    return true;
}

static inline bool gtap_profile_resolve_output(
    const char* pattern, char* output, size_t output_size
) {
    if (!pattern || !pattern[0] || strlen(pattern) >= output_size) return false;
    const char* marker = strstr(pattern, "%i");
    if (marker && strstr(marker + 2, "%i")) return false;
    if (marker) {
        const size_t prefix = static_cast<size_t>(marker - pattern);
        for (int index = 1; index < INT_MAX; ++index) {
            const int written = snprintf(
                output, output_size, "%.*s%d%s", static_cast<int>(prefix),
                pattern, index, marker + 2);
            if (written < 0 || static_cast<size_t>(written) >= output_size) {
                return false;
            }
            struct stat st = {};
            if (stat(output, &st) != 0 && errno == ENOENT) break;
        }
    } else {
        memcpy(output, pattern, strlen(pattern) + 1);
        struct stat st = {};
        if (stat(output, &st) == 0) {
            return S_ISDIR(st.st_mode);
        }
        if (errno != ENOENT) return false;
    }
    if (!gtap_profile_create_parents(output)) return false;
    return mkdir(output, 0755) == 0;
}

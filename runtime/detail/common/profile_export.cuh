#pragma once

#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

struct gtap_profile_export_options {
    const char* output_directory = "./profile";
    const char* label = nullptr;
    bool overwrite = false;
};

struct gtap_profile_export_result {
    bool success = false;
    size_t recorded_intervals = 0;
    size_t dropped_intervals = 0;
    char result_directory[512] = {};
    char profile_path[512] = {};
    char intervals_path[512] = {};
    char aggregates_path[512] = {};
};

namespace gtap::detail {

struct profile_distribution {
    size_t count = 0;
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double max = 0.0;
};

static inline bool valid_label(const char* label) {
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

static inline int compare_double(const void* lhs, const void* rhs) {
    const double a = *static_cast<const double*>(lhs);
    const double b = *static_cast<const double*>(rhs);
    return (a > b) - (a < b);
}

static inline double nearest_rank(
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
static inline double square_root(double value) {
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
static inline profile_distribution compute_distribution(
    double* values, size_t count
) {
    profile_distribution stats;
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
    stats.stddev = square_root(
        squared_deviation_sum / static_cast<double>(count));

    qsort(values, count, sizeof(double), compare_double);
    stats.min = values[0];
    stats.p50 = nearest_rank(values, count, 0.50);
    stats.p95 = nearest_rank(values, count, 0.95);
    stats.p99 = nearest_rank(values, count, 0.99);
    stats.max = values[count - 1];
    return stats;
}

static inline bool is_directory(const char* path) {
    struct stat st = {};
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static inline bool path_exists(const char* path) {
    struct stat st = {};
    return stat(path, &st) == 0;
}

static inline bool create_parents(const char* path) {
    if (!path || !path[0] || strlen(path) >= 512) return false;
    char copy[512] = {};
    memcpy(copy, path, strlen(path) + 1);
    for (char* p = copy + 1; *p; ++p) {
        if (*p != '/') continue;
        *p = '\0';
        if (copy[0] && mkdir(copy, 0755) != 0 &&
            !(errno == EEXIST && is_directory(copy))) {
            return false;
        }
        *p = '/';
    }
    return true;
}

static inline bool resolve_output(
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
    if (!create_parents(output)) return false;
    return mkdir(output, 0755) == 0;
}

// Host-side reason for one profile export failure. The first recorded message
// is kept. Callers still return gtap_profile_export_result.
struct profile_export_error {
    char message[512] = {};

    void fail(const char* format, ...) {
        if (message[0] != '\0') return;
        va_list args;
        va_start(args, format);
        vsnprintf(message, sizeof(message), format, args);
        va_end(args);
    }
};

static inline void print_profile_export_result(
    const gtap_profile_export_result& result,
    const profile_export_error& error
) {
    if (result.success) {
        printf("GTaP profile written to %s\n", result.result_directory);
        return;
    }
    if (error.message[0] != '\0') {
        fprintf(stderr, "GTaP profile not written: %s\n", error.message);
    }
}

static inline bool prepare_profile_output(
    const gtap_profile_export_options& options,
    gtap_profile_export_result& result,
    profile_export_error& error
) {
    if (!valid_label(options.label)) {
        error.fail("invalid label");
        return false;
    }
    if (!resolve_output(
            options.output_directory, result.result_directory,
            sizeof(result.result_directory))) {
        error.fail("invalid output directory");
        return false;
    }
    const int metadata_len = snprintf(
        result.profile_path, sizeof(result.profile_path), "%s/profile.json",
        result.result_directory);
    const int timeline_len = snprintf(
        result.intervals_path, sizeof(result.intervals_path),
        "%s/task_execution_intervals.csv", result.result_directory);
    const int statistics_len = snprintf(
        result.aggregates_path, sizeof(result.aggregates_path),
        "%s/task_execution_aggregates.csv", result.result_directory);
    if (metadata_len < 0 || timeline_len < 0 || statistics_len < 0 ||
        static_cast<size_t>(metadata_len) >= sizeof(result.profile_path) ||
        static_cast<size_t>(timeline_len) >= sizeof(result.intervals_path) ||
        static_cast<size_t>(statistics_len) >= sizeof(result.aggregates_path)) {
        error.fail("output path is too long");
        return false;
    }
    if (!options.overwrite &&
        (path_exists(result.profile_path) ||
         path_exists(result.intervals_path) ||
         path_exists(result.aggregates_path))) {
        error.fail(
            "files already exist in %s", result.result_directory);
        return false;
    }
    return true;
}

// Reads errno immediately. Call before any later libc call can replace it.
static inline void record_profile_open_failure(
    profile_export_error& error,
    const char* path
) {
    const int open_errno = errno;
    error.fail(
        "failed to open '%s' for writing: %s",
        path,
        strerror(open_errno));
}

static inline void record_profile_write_failure(
    profile_export_error& error,
    const char* path
) {
    const int write_errno = errno;
    error.fail(
        "failed to write '%s': %s",
        path,
        strerror(write_errno));
}

}  // namespace gtap::detail

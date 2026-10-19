#pragma once

#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

#include "profile_export_error.cuh"

struct gtap_profile_export_options {
    const char* output_directory = "./profile";
    const char* label = nullptr;
    bool overwrite = false;
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
    if (!path || !path[0]) return false;
    std::string copy(path);
    for (size_t i = 1; i < copy.size(); ++i) {
        if (copy[i] != '/') continue;
        copy[i] = '\0';
        if (copy[0] != '\0' && mkdir(copy.c_str(), 0755) != 0 &&
            !(errno == EEXIST && is_directory(copy.c_str()))) {
            return false;
        }
        copy[i] = '/';
    }
    return true;
}

static inline bool resolve_output(const char* pattern, std::string& output) {
    if (!pattern || !pattern[0]) return false;
    const char* marker = strstr(pattern, "%i");
    if (marker && strstr(marker + 2, "%i")) return false;
    if (marker) {
        const size_t prefix = static_cast<size_t>(marker - pattern);
        for (int index = 1; index < INT_MAX; ++index) {
            output = std::string(pattern, prefix) + std::to_string(index) + (marker + 2);
            struct stat st = {};
            if (stat(output.c_str(), &st) == 0) continue;
            if (errno == ENOENT) break;
            return false;
        }
    } else {
        output = pattern;
        struct stat st = {};
        if (stat(output.c_str(), &st) == 0) {
            return S_ISDIR(st.st_mode);
        }
        if (errno != ENOENT) return false;
    }
    if (!create_parents(output.c_str())) return false;
    return mkdir(output.c_str(), 0755) == 0;
}

static inline void print_profile_export_result(
    bool success,
    const char* result_directory,
    const profile_export_error& error
) {
    if (success) {
        printf("GTaP profile written to %s\n", result_directory);
        return;
    }
    if (error.message[0] != '\0') {
        fprintf(stderr, "GTaP profile not written: %s\n", error.message);
    }
}

static inline bool prepare_profile_output(
    const gtap_profile_export_options& options,
    std::string& result_directory,
    std::string& profile_path,
    std::string& intervals_path,
    std::string& aggregates_path,
    profile_export_error& error
) {
    if (!valid_label(options.label)) {
        error.fail("invalid label");
        return false;
    }
    if (!resolve_output(options.output_directory, result_directory)) {
        error.fail("invalid output directory");
        return false;
    }
    profile_path = result_directory + "/profile.json";
    intervals_path = result_directory + "/task_execution_intervals.csv";
    aggregates_path = result_directory + "/task_execution_aggregates.csv";
    if (!options.overwrite &&
        (path_exists(profile_path.c_str()) ||
         path_exists(intervals_path.c_str()) ||
         path_exists(aggregates_path.c_str()))) {
        error.fail("files already exist in %s", result_directory.c_str());
        return false;
    }
    return true;
}

}  // namespace gtap::detail

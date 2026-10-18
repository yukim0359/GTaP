#pragma once

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>

#include "checked_size.cuh"
#include "runtime_config.cuh"

namespace gtap::detail {

// Management buffers only. Profile buffers stay on the profile-capacity fields.
inline constexpr size_t default_memory_budget_bytes = 5ull << 30;

// Bytes, or K, M, G, or T. An iB suffix and a bare letter are powers of 1024.
// A B suffix is a power of 1000. 5G and 5GiB are 5 * 1024^3. 5GB is 5 * 1000^3.
inline bool parse_memory_budget(const char* text, size_t* bytes) {
    if (text == nullptr) return false;
    while (*text == ' ' || *text == '\t') ++text;
    if (*text < '0' || *text > '9') return false;

    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || errno == ERANGE) return false;
    while (*end == ' ' || *end == '\t') ++end;

    char token[4] = {};
    int length = 0;
    while (*end != '\0' && *end != ' ' && *end != '\t') {
        if (length == 3) return false;
        char c = *end;
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        if (c < 'a' || c > 'z') return false;
        token[length++] = c;
        ++end;
    }
    while (*end == ' ' || *end == '\t') ++end;
    if (*end != '\0') return false;

    unsigned long long scale = 1;
    if (length != 0) {
        int exponent = 0;
        if (token[0] == 'b' && length == 1) {
            exponent = 0;
        } else if (token[0] == 'k') {
            exponent = 1;
        } else if (token[0] == 'm') {
            exponent = 2;
        } else if (token[0] == 'g') {
            exponent = 3;
        } else if (token[0] == 't') {
            exponent = 4;
        } else {
            return false;
        }
        const bool bare = length == 1;
        const bool with_b = length == 2 && token[1] == 'b';
        const bool with_ib = length == 3 && token[1] == 'i' && token[2] == 'b';
        if (token[0] == 'b') {
            if (!bare) return false;
        } else if (!bare && !with_b && !with_ib) {
            return false;
        }
        if (with_b) {
            for (int i = 0; i < exponent; ++i) {
                if (scale > SIZE_MAX / 1000) return false;
                scale *= 1000;
            }
        } else {
            scale = 1ull << (10 * exponent);
        }
    }

    if (value > SIZE_MAX / scale) return false;
    *bytes = static_cast<size_t>(value * scale);
    return true;
}

inline bool read_memory_budget_env(size_t* bytes, bool* is_set) {
    const char* text = std::getenv("GTAP_MEMORY_BUDGET");
    if (text == nullptr) {
        *is_set = false;
        return true;
    }
    *is_set = true;
    return parse_memory_budget(text, bytes);
}

// TODO: Each cudaErrorInvalidValue return below is a different check, but
// gtap_initialize only hands the caller that code. cudaGetErrorString then
// says "invalid argument". Keep a host-side reason and report it from
// gtap_initialize. validate_config has the same gap.
inline cudaError_t resolve_tasks_per_unit(
    int requested_tasks,
    size_t config_budget_bytes,
    int scheduling_units,
    int num_queues,
    size_t fixed_bytes,
    size_t bytes_per_task,
    int* tasks_per_unit
) {
    if (requested_tasks < 0 || scheduling_units <= 0 || num_queues <= 0 ||
        bytes_per_task == 0) {
        return cudaErrorInvalidValue;
    }

    const size_t units = static_cast<size_t>(scheduling_units);
    const size_t queues = static_cast<size_t>(num_queues);
    // The scheduling-unit count times tasks per unit is computed in signed int.
    // The first recycle lap's 2 * pool_size - 1 fits in int when pool_size <= 1 << 30.
    const size_t max_tasks_by_total_count = static_cast<size_t>(INT_MAX) / units;
    const size_t max_tasks_by_ticket = static_cast<size_t>(1) << 30;
    const size_t max_tasks_per_unit =
        max_tasks_by_ticket < max_tasks_by_total_count
            ? max_tasks_by_ticket
            : max_tasks_by_total_count;
    size_t resolved_tasks_per_unit = 0;

    // requested_tasks > 0 ignores the budget.
    if (requested_tasks > 0) {
        resolved_tasks_per_unit = static_cast<size_t>(requested_tasks);
    } else {
        size_t budget_bytes = config_budget_bytes;
        if (budget_bytes == 0) {
            bool env_is_set = false;
            size_t env_budget_bytes = 0;
            if (!read_memory_budget_env(&env_budget_bytes, &env_is_set)) {
                return cudaErrorInvalidValue;
            }
            budget_bytes = env_is_set ? env_budget_bytes : default_memory_budget_bytes;
        }
        if (budget_bytes < fixed_bytes) return cudaErrorInvalidValue;

        size_t bytes_per_capacity_increment = 0;
        if (!checked_mul(bytes_per_task, units, &bytes_per_capacity_increment)) {
            return cudaErrorInvalidValue;
        }
        resolved_tasks_per_unit =
            (budget_bytes - fixed_bytes) / bytes_per_capacity_increment;

        if (resolved_tasks_per_unit > max_tasks_per_unit) {
            resolved_tasks_per_unit = max_tasks_per_unit;
        }
        resolved_tasks_per_unit = (resolved_tasks_per_unit / queues) * queues;
    }

    // Explicit capacities are not clamped above, so reject one past this limit.
    if (resolved_tasks_per_unit > max_tasks_per_unit ||
        resolved_tasks_per_unit % queues != 0) {
        return cudaErrorInvalidValue;
    }

    const size_t queue_capacity = resolved_tasks_per_unit / queues;
    if (queue_capacity <= static_cast<size_t>(GTAP_DETAIL_QUEUE_MARGIN) ||
        resolved_tasks_per_unit <
            static_cast<size_t>(GTAP_DETAIL_TASK_ID_POOL_MIN_FREE)) {
        return cudaErrorInvalidValue;
    }

    *tasks_per_unit = static_cast<int>(resolved_tasks_per_unit);
    return cudaSuccess;
}

}  // namespace gtap::detail

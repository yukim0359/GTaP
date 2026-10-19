#pragma once

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>

#include "checked_size.cuh"
#include "init_error.cuh"
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

// config_budget_bytes, then GTAP_MEMORY_BUDGET, then the 5 GiB default.
inline cudaError_t resolve_memory_budget(
    size_t config_budget_bytes,
    size_t* budget_bytes,
    init_error& error
) {
    if (config_budget_bytes != 0) {
        *budget_bytes = config_budget_bytes;
        return cudaSuccess;
    }
    const char* text = std::getenv("GTAP_MEMORY_BUDGET");
    if (text == nullptr) {
        *budget_bytes = default_memory_budget_bytes;
        return cudaSuccess;
    }
    if (!parse_memory_budget(text, budget_bytes)) {
        return error.fail(
            cudaErrorInvalidValue,
            "invalid or out-of-range GTAP_MEMORY_BUDGET=\"%s\"; example: 5GiB",
            text);
    }
    return cudaSuccess;
}

// Bytes required for the minimum task capacity. Overflow returns SIZE_MAX.
inline size_t minimum_required_bytes(
    size_t fixed_bytes,
    size_t bytes_per_task,
    size_t units,
    size_t queues
) {
    const size_t min_queue_slots =
        static_cast<size_t>(GTAP_DETAIL_QUEUE_MARGIN) + 1;
    size_t min_tasks = 0;
    if (!checked_mul(queues, min_queue_slots, &min_tasks)) return SIZE_MAX;
    if (min_tasks < static_cast<size_t>(GTAP_DETAIL_TASK_ID_POOL_MIN_FREE)) {
        min_tasks = static_cast<size_t>(GTAP_DETAIL_TASK_ID_POOL_MIN_FREE);
    }
    size_t bytes = mul_memory_bytes(bytes_per_task, min_tasks);
    bytes = mul_memory_bytes(bytes, units);
    return add_memory_bytes(fixed_bytes, bytes);
}

inline cudaError_t fail_insufficient_budget(
    init_error& error,
    size_t budget_bytes,
    size_t required_bytes
) {
    if (required_bytes == SIZE_MAX) {
        return error.fail(
            cudaErrorInvalidValue,
            "memory budget of %zu bytes cannot cover the minimum task capacity for this configuration",
            budget_bytes);
    }
    return error.fail(
        cudaErrorInvalidValue,
        "memory budget of %zu bytes is below the %zu bytes required for this configuration",
        budget_bytes,
        required_bytes);
}

inline cudaError_t resolve_tasks_per_unit(
    int requested_tasks,
    size_t config_budget_bytes,
    int scheduling_units,
    int num_queues,
    size_t fixed_bytes,
    size_t bytes_per_task,
    const char* task_capacity_name,
    int* tasks_per_unit,
    init_error& error
) {
    // validate_config rejects a negative task capacity, a non-positive
    // scheduling-unit count, and a non-positive queue count.
    if (bytes_per_task == 0) {
        return error.fail(
            cudaErrorInvalidValue,
            "internal error: invalid task-storage size");
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
        if (resolved_tasks_per_unit > max_tasks_per_unit) {
            return error.fail(
                cudaErrorInvalidValue,
                "%s=%d exceeds the limit of %zu tasks per scheduling unit",
                task_capacity_name,
                requested_tasks,
                max_tasks_per_unit);
        }
        if (resolved_tasks_per_unit % queues != 0) {
            return error.fail(
                cudaErrorInvalidValue,
                "%s=%d must be a multiple of num_queues=%d",
                task_capacity_name,
                requested_tasks,
                num_queues);
        }
        const size_t queue_capacity = resolved_tasks_per_unit / queues;
        if (queue_capacity <= static_cast<size_t>(GTAP_DETAIL_QUEUE_MARGIN)) {
            return error.fail(
                cudaErrorInvalidValue,
                "%s=%d provides %zu slots per queue; more than %d are required when num_queues=%d",
                task_capacity_name,
                requested_tasks,
                queue_capacity,
                GTAP_DETAIL_QUEUE_MARGIN,
                num_queues);
        }
        if (resolved_tasks_per_unit <
            static_cast<size_t>(GTAP_DETAIL_TASK_ID_POOL_MIN_FREE)) {
            return error.fail(
                cudaErrorInvalidValue,
                "%s=%d is below the minimum of %d tasks per scheduling unit",
                task_capacity_name,
                requested_tasks,
                GTAP_DETAIL_TASK_ID_POOL_MIN_FREE);
        }
    } else {
        size_t budget_bytes = 0;
        cudaError_t status = resolve_memory_budget(
            config_budget_bytes, &budget_bytes, error);
        if (status != cudaSuccess) return status;
        if (budget_bytes < fixed_bytes) {
            return fail_insufficient_budget(
                error,
                budget_bytes,
                minimum_required_bytes(
                    fixed_bytes, bytes_per_task, units, queues));
        }

        size_t bytes_per_capacity_increment = 0;
        if (!checked_mul(bytes_per_task, units, &bytes_per_capacity_increment)) {
            return error.fail(
                cudaErrorInvalidValue,
                "queue and task storage size overflowed");
        }
        resolved_tasks_per_unit =
            (budget_bytes - fixed_bytes) / bytes_per_capacity_increment;

        if (resolved_tasks_per_unit > max_tasks_per_unit) {
            resolved_tasks_per_unit = max_tasks_per_unit;
        }
        resolved_tasks_per_unit = (resolved_tasks_per_unit / queues) * queues;
        const size_t queue_capacity = resolved_tasks_per_unit / queues;

        // TODO: Distinguish insufficient memory budgets from configurations whose
        // minimum task capacity exceeds the task-count limit. Increasing the budget
        // cannot fix the latter.
        if (queue_capacity <= static_cast<size_t>(GTAP_DETAIL_QUEUE_MARGIN) ||
            resolved_tasks_per_unit <
                static_cast<size_t>(GTAP_DETAIL_TASK_ID_POOL_MIN_FREE)) {
            return fail_insufficient_budget(
                error,
                budget_bytes,
                minimum_required_bytes(
                    fixed_bytes, bytes_per_task, units, queues));
        }
    }

    *tasks_per_unit = static_cast<int>(resolved_tasks_per_unit);
    return cudaSuccess;
}

}  // namespace gtap::detail

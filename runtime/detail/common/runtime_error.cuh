#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

namespace gtap::detail {

// Runtime error codes. Keep this order aligned with error_short_message
// and error_detail_formatter.
enum class runtime_error_code {
    none = 0,
    invalid_queue_idx,
    invalid_queue_idx_after_join,
    queue_overflow,
    task_id_pool_slot_busy,
    task_id_pool_low_headroom,
    // Experimental global_queue only: child task-ID spill buffer overflow.
    generated_task_id_buffer_overflow,
};

constexpr int runtime_error_count =
    static_cast<int>(runtime_error_code::generated_task_id_buffer_overflow) + 1;

struct runtime_error_record {
    int valid;
    int code;
    int block_idx;
    int thread_idx;
    int tid;
    int queue_idx;
    int value;
    int limit;
};

static void format_detail_invalid_queue_idx(
    char*, size_t, const runtime_error_record*);
static void format_detail_invalid_queue_idx_after_join(
    char*, size_t, const runtime_error_record*);
static void format_detail_queue_overflow(
    char*, size_t, const runtime_error_record*);
static void format_detail_task_id_pool_slot_busy(
    char*, size_t, const runtime_error_record*);
static void format_detail_task_id_pool_low_headroom(
    char*, size_t, const runtime_error_record*);
static void format_detail_generated_task_id_buffer_overflow(
    char*, size_t, const runtime_error_record*);

static const char* const error_short_message[runtime_error_count] = {
    "no error",
    "invalid queue index",
    "invalid queue index after join",
    "task queue limit exceeded",
    "task ID pool exhausted",
    "task ID pool exhausted",
    "generated task-ID buffer overflow",
};

static void (*const error_detail_formatter[runtime_error_count])(
    char*, size_t, const runtime_error_record*
) = {
    nullptr,
    format_detail_invalid_queue_idx,
    format_detail_invalid_queue_idx_after_join,
    format_detail_queue_overflow,
    format_detail_task_id_pool_slot_busy,
    format_detail_task_id_pool_low_headroom,
    format_detail_generated_task_id_buffer_overflow,
};

// 0: no error, >0: error code
__device__ int d_runtime_error_code;
__constant__ runtime_error_record* d_runtime_error_record;

static runtime_error_record* h_runtime_error_record = nullptr;

inline cudaError_t initialize_runtime_error_record() {
    if (h_runtime_error_record == nullptr) {
        cudaError_t st = cudaHostAlloc(reinterpret_cast<void**>(&h_runtime_error_record),
                                       sizeof(runtime_error_record),
                                       cudaHostAllocMapped);
        if (st != cudaSuccess) return st;
    }

    memset(h_runtime_error_record, 0, sizeof(runtime_error_record));
    runtime_error_record* d_record = nullptr;
    cudaError_t st = cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_record),
                                              h_runtime_error_record, 0);
    if (st != cudaSuccess) return st;
    return cudaMemcpyToSymbol(d_runtime_error_record, &d_record,
                              sizeof(runtime_error_record*));
}

inline void reset_runtime_error_record() {
    if (h_runtime_error_record != nullptr) {
        memset(h_runtime_error_record, 0, sizeof(runtime_error_record));
    }
}

inline cudaError_t finalize_runtime_error_record() {
    cudaError_t st = cudaSuccess;
    if (h_runtime_error_record != nullptr) {
        st = cudaFreeHost(h_runtime_error_record);
        h_runtime_error_record = nullptr;
    }
    runtime_error_record* record = nullptr;
    cudaError_t clear = cudaMemcpyToSymbol(
        d_runtime_error_record, &record, sizeof(runtime_error_record*));
    if (st == cudaSuccess) st = clear;
    return st;
}

inline bool runtime_error_code_is_valid(int error_code) {
    return error_code >= static_cast<int>(runtime_error_code::none) &&
           error_code < runtime_error_count;
}

static void format_detail_invalid_queue_idx(
    char* out, size_t n, const runtime_error_record* r
) {
    snprintf(out, n, "  block=%d, thread=%d, task_id=%d, queue=%d, num_queues=%d",
        r->block_idx, r->thread_idx, r->tid, r->queue_idx, r->limit);
}

static void format_detail_invalid_queue_idx_after_join(
    char* out, size_t n, const runtime_error_record* r
) {
    snprintf(out, n, "  block=%d, thread=%d, task_id=%d, queue=%d, num_queues=%d",
        r->block_idx, r->thread_idx, r->tid, r->queue_idx, r->limit);
}

static void format_detail_queue_overflow(
    char* out, size_t n, const runtime_error_record* r
) {
    if (r->queue_idx >= 0) {
        snprintf(out, n,
            "  block=%d, thread=%d, task_id=%d, queue=%d, usage=%d, limit=%d",
            r->block_idx, r->thread_idx, r->tid, r->queue_idx, r->value, r->limit);
    } else if (r->tid >= 0) {
        snprintf(out, n, "  block=%d, thread=%d, task_id=%d, usage=%d, limit=%d",
            r->block_idx, r->thread_idx, r->tid, r->value, r->limit);
    } else {
        snprintf(out, n, "  block=%d, thread=%d, kind=%d, usage=%d, limit=%d",
            r->block_idx, r->thread_idx, r->queue_idx, r->value, r->limit);
    }
}

static void format_detail_task_id_pool_slot_busy(
    char* out, size_t n, const runtime_error_record* r
) {
    const int unreleased_slot =
        (r->limit > 0) ? (r->value % r->limit) : r->value;
    snprintf(out, n, "  block=%d, thread=%d, slot=%d, pool_size=%d",
        r->block_idx, r->thread_idx, unreleased_slot, r->limit);
}

static void format_detail_task_id_pool_low_headroom(
    char* out, size_t n, const runtime_error_record* r
) {
    snprintf(out, n, "  block=%d, thread=%d, task_id=%d, headroom=%d, limit=%d",
        r->block_idx, r->thread_idx, r->tid, r->value, r->limit);
}

static void format_detail_generated_task_id_buffer_overflow(
    char* out, size_t n, const runtime_error_record* r
) {
    if (r->queue_idx >= 0) {
        snprintf(out, n,
            "  block=%d, thread=%d, task_id=%d, queue=%d, index=%d, limit=%d",
            r->block_idx, r->thread_idx, r->tid, r->queue_idx, r->value, r->limit);
    } else {
        snprintf(out, n, "  block=%d, thread=%d, task_id=%d, index=%d, limit=%d",
            r->block_idx, r->thread_idx, r->tid, r->value, r->limit);
    }
}

__device__ __forceinline__ void record_runtime_error_and_trap(
    runtime_error_code code,
    int tid,
    int queue_idx,
    int value,
    int limit
) {
    const int code_int = static_cast<int>(code);
    const int none = static_cast<int>(runtime_error_code::none);
    if (atomicCAS(&d_runtime_error_code, none, code_int) == none) {
        runtime_error_record* record = d_runtime_error_record;
        if (record != nullptr) {
            record->code = code_int;
            record->block_idx = blockIdx.x;
            record->thread_idx = threadIdx.x;
            record->tid = tid;
            record->queue_idx = queue_idx;
            record->value = value;
            record->limit = limit;
            __threadfence_system();
            record->valid = 1;
            __threadfence_system();
        }
    }
    __trap();
}

inline const char* get_runtime_error_string(int error_code) {
    if (runtime_error_code_is_valid(error_code)) {
        return error_short_message[error_code];
    }
    return "unknown error";
}

inline bool read_runtime_error_record(runtime_error_record* record) {
    if (h_runtime_error_record != nullptr &&
        h_runtime_error_record->valid != 0) {
        *record = *h_runtime_error_record;
        return true;
    }
    *record = {};
    return false;
}

static void format_runtime_error_fields(
    char* out,
    size_t n,
    const runtime_error_record& record
) {
    if (runtime_error_code_is_valid(record.code) &&
        error_detail_formatter[record.code] != nullptr) {
        error_detail_formatter[record.code](out, n, &record);
        return;
    }
    snprintf(out, n,
        "  block=%d, thread=%d, task_id=%d, queue=%d, value=%d, limit=%d",
        record.block_idx, record.thread_idx, record.tid,
        record.queue_idx, record.value, record.limit);
}

inline void print_runtime_error_fields(const runtime_error_record& record) {
    char fields[256];
    format_runtime_error_fields(fields, sizeof(fields), record);
    fprintf(stderr, "%s\n", fields);
}

}  // namespace gtap::detail

#define GTAP_DETAIL_RECORD_INVALID_QUEUE_IDX(tid, queue_idx, num_queues) \
    gtap::detail::record_runtime_error_and_trap( \
        gtap::detail::runtime_error_code::invalid_queue_idx, \
        (tid), (queue_idx), (queue_idx), (num_queues))

#define GTAP_DETAIL_RECORD_INVALID_QUEUE_IDX_AFTER_JOIN(tid, queue_idx, num_queues) \
    gtap::detail::record_runtime_error_and_trap( \
        gtap::detail::runtime_error_code::invalid_queue_idx_after_join, \
        (tid), (queue_idx), (queue_idx), (num_queues))

#define GTAP_DETAIL_RECORD_QUEUE_OVERFLOW(tid, queue_idx, usage, capacity) \
    gtap::detail::record_runtime_error_and_trap( \
        gtap::detail::runtime_error_code::queue_overflow, \
        (tid), (queue_idx), (usage), (capacity))

#define GTAP_DETAIL_RECORD_TASK_ID_POOL_SLOT_BUSY(tid, alloc_count, pool_size) \
    gtap::detail::record_runtime_error_and_trap( \
        gtap::detail::runtime_error_code::task_id_pool_slot_busy, \
        (tid), -1, (alloc_count), (pool_size))

#define GTAP_DETAIL_RECORD_TASK_ID_POOL_LOW_HEADROOM(tid, headroom, min_headroom) \
    gtap::detail::record_runtime_error_and_trap( \
        gtap::detail::runtime_error_code::task_id_pool_low_headroom, \
        (tid), -1, (headroom), (min_headroom))

#define GTAP_DETAIL_RECORD_GENERATED_TASK_ID_BUFFER_OVERFLOW(tid, queue_idx, index, capacity) \
    gtap::detail::record_runtime_error_and_trap( \
        gtap::detail::runtime_error_code::generated_task_id_buffer_overflow, \
        (tid), (queue_idx), (index), (capacity))

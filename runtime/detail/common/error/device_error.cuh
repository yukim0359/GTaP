#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

namespace gtap::detail {

// Device-side error codes. Keep this order aligned with error_short_message
// and error_detail_formatter.
enum class device_error_code {
    none = 0,
    invalid_queue_idx_task,
    invalid_queue_idx_taskwait,
    task_queue_overflow,
    task_id_pool_slot_not_ready,
    task_id_pool_exhausted,
    // Experimental global_queue only: child task-ID spill buffer overflow.
    generated_task_id_buffer_overflow,
};

constexpr int device_error_count =
    static_cast<int>(device_error_code::generated_task_id_buffer_overflow) + 1;

struct device_error_record {
    int valid;
    int code;
    int block_idx;
    int thread_idx;
    int tid;
    int queue_idx;
    int value;
    int limit;
};

static void format_detail_invalid_queue_idx_task(
    char*, size_t, const device_error_record*);
static void format_detail_invalid_queue_idx_taskwait(
    char*, size_t, const device_error_record*);
static void format_detail_task_queue_overflow(
    char*, size_t, const device_error_record*);
static void format_detail_task_id_pool_slot_not_ready(
    char*, size_t, const device_error_record*);
static void format_detail_task_id_pool_exhausted(
    char*, size_t, const device_error_record*);
static void format_detail_generated_task_id_buffer_overflow(
    char*, size_t, const device_error_record*);

static const char* const error_short_message[device_error_count] = {
    "no error",
    "invalid queue index specified by task",
    "invalid queue index specified by taskwait",
    "task queue capacity exceeded",
    "task ID pool exhausted",
    "task ID pool exhausted",
    "generated task-ID buffer overflow",
};

static void (*const error_detail_formatter[device_error_count])(
    char*, size_t, const device_error_record*
) = {
    nullptr,
    format_detail_invalid_queue_idx_task,
    format_detail_invalid_queue_idx_taskwait,
    format_detail_task_queue_overflow,
    format_detail_task_id_pool_slot_not_ready,
    format_detail_task_id_pool_exhausted,
    format_detail_generated_task_id_buffer_overflow,
};

// 0: no error, >0: error code
__device__ int d_device_error_code;
__constant__ device_error_record* d_device_error_record;
inline device_error_record* h_device_error_record = nullptr;

inline cudaError_t initialize_device_error_record() {
    if (h_device_error_record == nullptr) {
        cudaError_t st = cudaHostAlloc(reinterpret_cast<void**>(&h_device_error_record),
                                       sizeof(device_error_record),
                                       cudaHostAllocMapped);
        if (st != cudaSuccess) return st;
    }

    memset(h_device_error_record, 0, sizeof(device_error_record));
    device_error_record* d_record = nullptr;
    cudaError_t st = cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_record),
                                              h_device_error_record, 0);
    if (st != cudaSuccess) return st;
    return cudaMemcpyToSymbol(d_device_error_record, &d_record,
                              sizeof(device_error_record*));
}

inline void reset_device_error_record() {
    if (h_device_error_record != nullptr) {
        memset(h_device_error_record, 0, sizeof(device_error_record));
    }
}

inline cudaError_t finalize_device_error_record() {
    cudaError_t st = cudaSuccess;
    if (h_device_error_record != nullptr) {
        st = cudaFreeHost(h_device_error_record);
        h_device_error_record = nullptr;
    }
    device_error_record* record = nullptr;
    cudaError_t clear = cudaMemcpyToSymbol(
        d_device_error_record, &record, sizeof(device_error_record*));
    if (st == cudaSuccess) st = clear;
    return st;
}

inline bool device_error_code_is_valid(int error_code) {
    return error_code >= static_cast<int>(device_error_code::none) &&
           error_code < device_error_count;
}

static void format_detail_invalid_queue_idx_task(
    char* out, size_t n, const device_error_record* r
) {
    snprintf(out, n, "  block=%d, thread=%d, task_id=%d, queue=%d, num_queues=%d",
        r->block_idx, r->thread_idx, r->tid, r->queue_idx, r->limit);
}

static void format_detail_invalid_queue_idx_taskwait(
    char* out, size_t n, const device_error_record* r
) {
    snprintf(out, n, "  block=%d, thread=%d, task_id=%d, queue=%d, num_queues=%d",
        r->block_idx, r->thread_idx, r->tid, r->queue_idx, r->limit);
}

static void format_detail_task_queue_overflow(
    char* out, size_t n, const device_error_record* r
) {
    if (r->queue_idx >= 0) {
        snprintf(out, n,
            "  block=%d, thread=%d, queue=%d, queue_capacity=%d",
            r->block_idx, r->thread_idx, r->queue_idx, r->limit);
    } else {
        snprintf(out, n,
            "  block=%d, thread=%d, queue_capacity=%d",
            r->block_idx, r->thread_idx, r->limit);
    }
}

static void format_detail_task_id_pool_slot_not_ready(
    char* out, size_t n, const device_error_record* r
) {
    snprintf(out, n, "  block=%d, thread=%d, pool_capacity=%d",
        r->block_idx, r->thread_idx, r->limit);
}

static void format_detail_task_id_pool_exhausted(
    char* out, size_t n, const device_error_record* r
) {
    snprintf(out, n, "  block=%d, thread=%d, task_id=%d, pool_capacity=%d",
        r->block_idx, r->thread_idx, r->tid, r->limit);
}

static void format_detail_generated_task_id_buffer_overflow(
    char* out, size_t n, const device_error_record* r
) {
    if (r->queue_idx >= 0) {
        snprintf(out, n,
            "  block=%d, thread=%d, task_id=%d, queue=%d, buffer_capacity=%d",
            r->block_idx, r->thread_idx, r->tid, r->queue_idx, r->limit);
    } else {
        snprintf(out, n,
            "  block=%d, thread=%d, task_id=%d, buffer_capacity=%d",
            r->block_idx, r->thread_idx, r->tid, r->limit);
    }
}

__device__ __forceinline__ void record_device_error_and_trap(
    device_error_code code,
    int tid,
    int queue_idx,
    int value,
    int limit
) {
    const int code_int = static_cast<int>(code);
    const int none = static_cast<int>(device_error_code::none);
    if (atomicCAS(&d_device_error_code, none, code_int) == none) {
        device_error_record* record = d_device_error_record;
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

inline const char* get_device_error_string(int error_code) {
    if (device_error_code_is_valid(error_code)) {
        return error_short_message[error_code];
    }
    return "unknown error";
}

inline bool read_device_error_record(device_error_record* record) {
    if (h_device_error_record != nullptr &&
        h_device_error_record->valid != 0) {
        *record = *h_device_error_record;
        return true;
    }
    *record = {};
    return false;
}

static void format_device_error_fields(
    char* out,
    size_t n,
    const device_error_record& record
) {
    if (device_error_code_is_valid(record.code) &&
        error_detail_formatter[record.code] != nullptr) {
        error_detail_formatter[record.code](out, n, &record);
        return;
    }
    snprintf(out, n, "  block=%d, thread=%d, error_code=%d",
        record.block_idx, record.thread_idx, record.code);
}

inline void print_device_error_fields(const device_error_record& record) {
    char fields[256];
    format_device_error_fields(fields, sizeof(fields), record);
    fprintf(stderr, "%s\n", fields);
}

}  // namespace gtap::detail

#define GTAP_DETAIL_RECORD_INVALID_QUEUE_IDX_TASK(tid, queue_idx, num_queues) \
    gtap::detail::record_device_error_and_trap( \
        gtap::detail::device_error_code::invalid_queue_idx_task, \
        (tid), (queue_idx), (queue_idx), (num_queues))

#define GTAP_DETAIL_RECORD_INVALID_QUEUE_IDX_TASKWAIT(tid, queue_idx, num_queues) \
    gtap::detail::record_device_error_and_trap( \
        gtap::detail::device_error_code::invalid_queue_idx_taskwait, \
        (tid), (queue_idx), (queue_idx), (num_queues))

#define GTAP_DETAIL_RECORD_TASK_QUEUE_OVERFLOW(tid, queue_idx, usage, capacity) \
    gtap::detail::record_device_error_and_trap( \
        gtap::detail::device_error_code::task_queue_overflow, \
        (tid), (queue_idx), (usage), (capacity))

#define GTAP_DETAIL_RECORD_TASK_ID_POOL_SLOT_NOT_READY(tid, pool_capacity) \
    gtap::detail::record_device_error_and_trap( \
        gtap::detail::device_error_code::task_id_pool_slot_not_ready, \
        (tid), -1, -1, (pool_capacity))

#define GTAP_DETAIL_RECORD_TASK_ID_POOL_EXHAUSTED(tid, pool_capacity) \
    gtap::detail::record_device_error_and_trap( \
        gtap::detail::device_error_code::task_id_pool_exhausted, \
        (tid), -1, -1, (pool_capacity))

#define GTAP_DETAIL_RECORD_GENERATED_TASK_ID_BUFFER_OVERFLOW(tid, queue_idx, index, capacity) \
    gtap::detail::record_device_error_and_trap( \
        gtap::detail::device_error_code::generated_task_id_buffer_overflow, \
        (tid), (queue_idx), (index), (capacity))

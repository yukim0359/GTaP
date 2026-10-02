#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

namespace gtap::detail {

// Runtime error codes (keep declaration order aligned with tables below).
enum class runtime_error_code {
    none = 0,
    invalid_queue_idx,
    invalid_queue_idx_after_join,
    queue_overflow,
    task_id_pool_slot_busy,
    task_id_pool_low_headroom,
    generated_task_id_buffer_overflow,
};

constexpr int runtime_error_count =
    static_cast<int>(runtime_error_code::generated_task_id_buffer_overflow) + 1;

struct runtime_error_record {
    int valid;
    int code;
    int src_line;
    int block_idx;
    int thread_idx;
    int tid;
    int queue_idx;
    int value;
    int limit;
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

inline void reset_runtime_error_record_host() {
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
    return st;
}

inline bool runtime_error_code_is_valid(int error_code) {
    return error_code >= static_cast<int>(runtime_error_code::none) &&
           error_code < runtime_error_count;
}

static void print_detail_invalid_queue_idx(const runtime_error_record* r) {
    fprintf(stderr, "Invalid queue index %d for task tid=%d (num_queues=%d)",
        r->queue_idx, r->tid, r->limit);
}

static void print_detail_invalid_queue_idx_after_join(const runtime_error_record* r) {
    fprintf(stderr, "Invalid queue index %d after join for task tid=%d (num_queues=%d)",
        r->queue_idx, r->tid, r->limit);
}

static void print_detail_queue_overflow(const runtime_error_record* r) {
    if (r->queue_idx >= 0) {
        fprintf(stderr, "Task queue %d overflow for task tid=%d "
            "(usage=%d, capacity=%d)",
            r->queue_idx, r->tid, r->value, r->limit);
    } else if (r->tid >= 0) {
        fprintf(stderr, "Task queue overflow for task tid=%d "
            "(usage=%d, capacity=%d)",
            r->tid, r->value, r->limit);
    } else {
        fprintf(stderr, "Task queue overflow (kind=%d, usage=%d, capacity=%d)",
            r->queue_idx, r->value, r->limit);
    }
}

static void print_detail_task_id_pool_slot_busy(const runtime_error_record* r) {
    const int unreleased_slot =
        (r->limit > 0) ? (r->value % r->limit) : r->value;
    fprintf(stderr, "Task ID pool exhausted: reuse slot %d still in use "
        "(alloc_count=%d, pool_size=%d, task_tid=%d)",
        unreleased_slot, r->value, r->limit, r->tid);
}

static void print_detail_task_id_pool_low_headroom(const runtime_error_record* r) {
    fprintf(stderr, "Task ID pool exhausted: headroom=%d below minimum %d "
        "(task_tid=%d)",
        r->value, r->limit, r->tid);
}

static void print_detail_generated_task_id_buffer_overflow(
    const runtime_error_record* r
) {
    if (r->queue_idx >= 0) {
        fprintf(stderr, "Generated task-ID buffer overflow for task tid=%d "
            "(queue=%d, index=%d, capacity=%d)",
            r->tid, r->queue_idx, r->value, r->limit);
    } else {
        fprintf(stderr, "Generated task-ID buffer overflow for task tid=%d "
            "(index=%d, capacity=%d)",
            r->tid, r->value, r->limit);
    }
}

static const char* const error_short_message[runtime_error_count] = {
    "No error",
    "Invalid queue index",
    "Invalid queue index after join",
    "Queue overflow",
    "Task ID pool exhausted",
    "Task ID pool exhausted",
    "Generated task ID buffer overflow",
};

static void (*const error_detail_printer[runtime_error_count])(const runtime_error_record*) = {
    nullptr,
    print_detail_invalid_queue_idx,
    print_detail_invalid_queue_idx_after_join,
    print_detail_queue_overflow,
    print_detail_task_id_pool_slot_busy,
    print_detail_task_id_pool_low_headroom,
    print_detail_generated_task_id_buffer_overflow,
};

__device__ __forceinline__ void record_runtime_error_and_trap(
    runtime_error_code code,
    int tid,
    int queue_idx,
    int value,
    int limit,
    int src_line
) {
    const int code_int = static_cast<int>(code);
    const int none = static_cast<int>(runtime_error_code::none);
    if (atomicCAS(&d_runtime_error_code, none, code_int) == none) {
        runtime_error_record* record = d_runtime_error_record;
        if (record != nullptr) {
            record->code = code_int;
            record->src_line = src_line;
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
    return "Unknown error";
}

static void print_error_details(const runtime_error_record* r) {
    if (runtime_error_code_is_valid(r->code) &&
        error_detail_printer[r->code] != nullptr) {
        error_detail_printer[r->code](r);
        return;
    }
    fprintf(stderr,
        "%s (code=%d, tid=%d, queue=%d, value=%d, limit=%d)",
        get_runtime_error_string(r->code), r->code,
        r->tid, r->queue_idx, r->value, r->limit);
}

inline bool read_error_report(runtime_error_record* record) {
    if (h_runtime_error_record != nullptr &&
        h_runtime_error_record->valid != 0) {
        *record = *h_runtime_error_record;
        return true;
    }
    *record = {};
    return false;
}

inline void print_error_report(const runtime_error_record* r) {
    fprintf(stderr,
        "GTaP Runtime Error at block %d, thread %d: ",
        r->block_idx, r->thread_idx);
    print_error_details(r);
    fprintf(stderr, " (source_line: %d)\n", r->src_line);
}

}  // namespace gtap::detail

inline cudaError_t gtap_synchronize() {
    cudaError_t st = cudaDeviceSynchronize();
    gtap::detail::runtime_error_record record{};
    if (gtap::detail::read_error_report(&record)) {
        gtap::detail::print_error_report(&record);
        return st;
    }
    if (st != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR: %s\n", cudaGetErrorString(st));
        return st;
    }
    return cudaSuccess;
}

#define GTAP_DETAIL_RECORD_INVALID_QUEUE_IDX(tid, queue_idx, num_queues) \
    gtap::detail::record_runtime_error_and_trap( \
        gtap::detail::runtime_error_code::invalid_queue_idx, \
        (tid), (queue_idx), (queue_idx), (num_queues), __LINE__)

#define GTAP_DETAIL_RECORD_INVALID_QUEUE_IDX_AFTER_JOIN(tid, queue_idx, num_queues) \
    gtap::detail::record_runtime_error_and_trap( \
        gtap::detail::runtime_error_code::invalid_queue_idx_after_join, \
        (tid), (queue_idx), (queue_idx), (num_queues), __LINE__)

#define GTAP_DETAIL_RECORD_QUEUE_OVERFLOW(tid, queue_idx, usage, capacity) \
    gtap::detail::record_runtime_error_and_trap( \
        gtap::detail::runtime_error_code::queue_overflow, (tid), (queue_idx), (usage), (capacity), __LINE__)

#define GTAP_DETAIL_RECORD_TASK_ID_POOL_SLOT_BUSY(tid, alloc_count, pool_size) \
    gtap::detail::record_runtime_error_and_trap( \
        gtap::detail::runtime_error_code::task_id_pool_slot_busy, \
        (tid), -1, (alloc_count), (pool_size), __LINE__)

#define GTAP_DETAIL_RECORD_TASK_ID_POOL_LOW_HEADROOM(tid, headroom, min_headroom) \
    gtap::detail::record_runtime_error_and_trap( \
        gtap::detail::runtime_error_code::task_id_pool_low_headroom, \
        (tid), -1, (headroom), (min_headroom), __LINE__)

#define GTAP_DETAIL_RECORD_GENERATED_TASK_ID_BUFFER_OVERFLOW(tid, queue_idx, index, capacity) \
    gtap::detail::record_runtime_error_and_trap( \
        gtap::detail::runtime_error_code::generated_task_id_buffer_overflow, \
        (tid), (queue_idx), (index), (capacity), __LINE__)

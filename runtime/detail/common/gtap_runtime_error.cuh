#pragma once

#include <cuda_runtime.h>
#include <string.h>

// Runtime error codes (keep declaration order aligned with tables below).
enum GTapRuntimeError {
    GTAP_ERROR_NONE = 0,
    GTAP_ERROR_INVALID_QUEUE_IDX = 1,
    GTAP_ERROR_INVALID_QUEUE_IDX_AFTER_JOIN = 2,
    GTAP_ERROR_QUEUE_OVERFLOW = 3,
    GTAP_ERROR_RESULT_HANDLE_OVERFLOW = 4,
    GTAP_ERROR_TASK_ID_POOL_SLOT_BUSY = 5,
    GTAP_ERROR_TASK_ID_POOL_LOW_HEADROOM = 6,
    GTAP_ERROR_GENERATED_TASK_ID_BUFFER_OVERFLOW = 7,
    GTAP_ERROR_INVALID_TASKWAIT = 8,
    GTAP_ERROR_COUNT
};

struct GTapRuntimeErrorReport {
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

// Global error code
__device__ int d_runtime_error_code;  // 0: no error, >0: error code
__constant__ GTapRuntimeErrorReport* d_runtime_error_report;

static GTapRuntimeErrorReport* h_runtime_error_report = nullptr;

inline cudaError_t gtap_get_runtime_error_code(int* error_code) {
    return cudaMemcpyFromSymbol(error_code, d_runtime_error_code, sizeof(int));
}

inline cudaError_t gtap_init_runtime_error_report() {
    if (h_runtime_error_report == nullptr) {
        cudaError_t st = cudaHostAlloc(reinterpret_cast<void**>(&h_runtime_error_report),
                                       sizeof(GTapRuntimeErrorReport),
                                       cudaHostAllocMapped);
        if (st != cudaSuccess) return st;
    }

    memset(h_runtime_error_report, 0, sizeof(GTapRuntimeErrorReport));
    GTapRuntimeErrorReport* d_report = nullptr;
    cudaError_t st = cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_report),
                                              h_runtime_error_report, 0);
    if (st != cudaSuccess) return st;
    return cudaMemcpyToSymbol(d_runtime_error_report, &d_report,
                              sizeof(GTapRuntimeErrorReport*));
}

inline void gtap_reset_runtime_error_report_host() {
    if (h_runtime_error_report != nullptr) {
        memset(h_runtime_error_report, 0, sizeof(GTapRuntimeErrorReport));
    }
}

inline cudaError_t gtap_finalize_runtime_error_report() {
    cudaError_t st = cudaSuccess;
    if (h_runtime_error_report != nullptr) {
        st = cudaFreeHost(h_runtime_error_report);
        h_runtime_error_report = nullptr;
    }
    return st;
}

inline bool gtap_runtime_error_code_is_valid(int error_code) {
    return error_code >= GTAP_ERROR_NONE && error_code < GTAP_ERROR_COUNT;
}

static void gtap_print_detail_invalid_queue_idx(const GTapRuntimeErrorReport* r) {
    printf(
        "Invalid queue index %d for task tid=%d (num_queues=%d)",
        r->queue_idx, r->tid, r->limit);
}

static void gtap_print_detail_invalid_queue_idx_after_join(const GTapRuntimeErrorReport* r) {
    printf(
        "Invalid queue index %d after join for task tid=%d (num_queues=%d)",
        r->queue_idx, r->tid, r->limit);
}

static void gtap_print_detail_queue_overflow(const GTapRuntimeErrorReport* r) {
    if (r->queue_idx >= 0) {
        printf(
            "Task queue %d overflow for task tid=%d "
            "(usage=%d, capacity=%d)",
            r->queue_idx, r->tid, r->value, r->limit);
    } else if (r->tid >= 0) {
        printf(
            "Task queue overflow for task tid=%d "
            "(usage=%d, capacity=%d)",
            r->tid, r->value, r->limit);
    } else {
        printf(
            "Task queue overflow (kind=%d, usage=%d, capacity=%d)",
            r->queue_idx, r->value, r->limit);
    }
}

static void gtap_print_detail_result_handle_overflow(const GTapRuntimeErrorReport* r) {
    if (r->queue_idx >= 0) {
        printf(
            "Result-handle table overflow for parent tid=%d "
            "(child_tid=%d, slot=%d, capacity=%d)",
            r->tid, r->queue_idx, r->value, r->limit);
    } else {
        printf(
            "Result-handle table overflow for parent tid=%d "
            "(slot=%d, capacity=%d)",
            r->tid, r->value, r->limit);
    }
}

static void gtap_print_detail_task_id_pool_slot_busy(const GTapRuntimeErrorReport* r) {
    const int unreleased_slot =
        (r->limit > 0) ? (r->value % r->limit) : r->value;
    printf(
        "Task ID pool exhausted: reuse slot %d still in use "
        "(alloc_count=%d, pool_size=%d, task_tid=%d)",
        unreleased_slot, r->value, r->limit, r->tid);
}

static void gtap_print_detail_task_id_pool_low_headroom(const GTapRuntimeErrorReport* r) {
    printf(
        "Task ID pool exhausted: headroom=%d below minimum %d "
        "(task_tid=%d)",
        r->value, r->limit, r->tid);
}

static void gtap_print_detail_generated_task_id_buffer_overflow(
    const GTapRuntimeErrorReport* r
) {
    if (r->queue_idx >= 0) {
        printf(
            "Generated task-ID buffer overflow for task tid=%d "
            "(queue=%d, index=%d, capacity=%d)",
            r->tid, r->queue_idx, r->value, r->limit);
    } else {
        printf(
            "Generated task-ID buffer overflow for task tid=%d "
            "(index=%d, capacity=%d)",
            r->tid, r->value, r->limit);
    }
}

static void gtap_print_detail_invalid_taskwait(const GTapRuntimeErrorReport* r) {
    printf(
        "Invalid taskwait for parent tid=%d "
        "(child_index=%d, max_children=%d)",
        r->tid, r->value, r->limit);
}

typedef void (*GTapRuntimeErrorDetailFn)(const GTapRuntimeErrorReport*);

static const char* const kGtapRuntimeErrorShortMessage[GTAP_ERROR_COUNT] = {
    "No error",
    "Invalid queue index",
    "Invalid queue index after join",
    "Queue overflow",
    "Result handle table overflow",
    "Task ID pool exhausted",
    "Task ID pool exhausted",
    "Generated task ID buffer overflow",
    "Invalid taskwait lowering",
};

static const GTapRuntimeErrorDetailFn kGtapRuntimeErrorDetailPrinter[GTAP_ERROR_COUNT] = {
    nullptr,
    gtap_print_detail_invalid_queue_idx,
    gtap_print_detail_invalid_queue_idx_after_join,
    gtap_print_detail_queue_overflow,
    gtap_print_detail_result_handle_overflow,
    gtap_print_detail_task_id_pool_slot_busy,
    gtap_print_detail_task_id_pool_low_headroom,
    gtap_print_detail_generated_task_id_buffer_overflow,
    gtap_print_detail_invalid_taskwait,
};

inline const char* gtap_get_runtime_error_string(int error_code) {
    if (gtap_runtime_error_code_is_valid(error_code)) {
        return kGtapRuntimeErrorShortMessage[error_code];
    }
    return "Unknown error";
}

inline void gtap_print_runtime_error_details(const GTapRuntimeErrorReport* r) {
    if (gtap_runtime_error_code_is_valid(r->code) &&
        kGtapRuntimeErrorDetailPrinter[r->code] != nullptr) {
        kGtapRuntimeErrorDetailPrinter[r->code](r);
        return;
    }
    printf(
        "%s (code=%d, tid=%d, queue=%d, value=%d, limit=%d)",
        gtap_get_runtime_error_string(r->code), r->code,
        r->tid, r->queue_idx, r->value, r->limit);
}

inline bool gtap_print_runtime_error_report() {
    if (h_runtime_error_report == nullptr || h_runtime_error_report->valid == 0) {
        return false;
    }
    const GTapRuntimeErrorReport* r = h_runtime_error_report;
    printf(
        "GTaP Runtime Error at block %d, thread %d: ",
        r->block_idx, r->thread_idx);
    gtap_print_runtime_error_details(r);
    printf(" (source_line: %d)\n", r->src_line);
    return true;
}

inline cudaError_t gtap_check_runtime_error() {
    if (gtap_print_runtime_error_report()) {
        return cudaSuccess;
    }

    int error_code = 0;
    cudaError_t cuda_err = gtap_get_runtime_error_code(&error_code);
    if (cuda_err != cudaSuccess) {
        printf("GTaP Runtime Error: Unable to read error code (CUDA error: %s)\n", cudaGetErrorString(cuda_err));
        return cuda_err;
    }
    if (error_code != GTAP_ERROR_NONE) {
        printf("GTaP Runtime Error: %s (code: %d)\n", gtap_get_runtime_error_string(error_code), error_code);
    }
    return cudaSuccess;
}

inline cudaError_t gtap_report_cuda_error(cudaError_t st) {
    if (st != cudaSuccess) {
        if (!gtap_print_runtime_error_report()) {
            printf("CUDA ERROR: %s\n", cudaGetErrorString(st));
        }
        return st;
    }
    return gtap_check_runtime_error();
}

inline cudaError_t gtap_synchronize() {
    cudaError_t st = cudaDeviceSynchronize();
    return gtap_report_cuda_error(st);
}

__device__ __forceinline__ void gtap_record_runtime_error_and_trap(
    GTapRuntimeError code,
    int tid,
    int queue_idx,
    int value,
    int limit,
    int src_line
) {
    const int code_int = static_cast<int>(code);
    if (atomicCAS(&d_runtime_error_code, GTAP_ERROR_NONE, code_int) == GTAP_ERROR_NONE) {
        GTapRuntimeErrorReport* report = d_runtime_error_report;
        if (report != nullptr) {
            report->code = code_int;
            report->src_line = src_line;
            report->block_idx = blockIdx.x;
            report->thread_idx = threadIdx.x;
            report->tid = tid;
            report->queue_idx = queue_idx;
            report->value = value;
            report->limit = limit;
            __threadfence_system();
            report->valid = 1;
            __threadfence_system();
        }
    }
    __trap();
}

#define GTAP_RECORD_INVALID_QUEUE_IDX(tid, queue_idx, num_queues) \
    gtap_record_runtime_error_and_trap( \
        GTAP_ERROR_INVALID_QUEUE_IDX, \
        (tid), (queue_idx), (queue_idx), (num_queues), __LINE__)

#define GTAP_RECORD_INVALID_QUEUE_IDX_AFTER_JOIN(tid, queue_idx, num_queues) \
    gtap_record_runtime_error_and_trap( \
        GTAP_ERROR_INVALID_QUEUE_IDX_AFTER_JOIN, \
        (tid), (queue_idx), (queue_idx), (num_queues), __LINE__)

#define GTAP_RECORD_QUEUE_OVERFLOW(tid, queue_idx, usage, capacity) \
    gtap_record_runtime_error_and_trap( \
        GTAP_ERROR_QUEUE_OVERFLOW, (tid), (queue_idx), (usage), (capacity), __LINE__)

#define GTAP_RECORD_RESULT_HANDLE_OVERFLOW(parent_tid, child_tid, slot, capacity) \
    gtap_record_runtime_error_and_trap( \
        GTAP_ERROR_RESULT_HANDLE_OVERFLOW, \
        (parent_tid), (child_tid), (slot), (capacity), __LINE__)

#define GTAP_RECORD_TASK_ID_POOL_SLOT_BUSY(tid, alloc_count, pool_size) \
    gtap_record_runtime_error_and_trap( \
        GTAP_ERROR_TASK_ID_POOL_SLOT_BUSY, \
        (tid), -1, (alloc_count), (pool_size), __LINE__)

#define GTAP_RECORD_TASK_ID_POOL_LOW_HEADROOM(tid, headroom, min_headroom) \
    gtap_record_runtime_error_and_trap( \
        GTAP_ERROR_TASK_ID_POOL_LOW_HEADROOM, \
        (tid), -1, (headroom), (min_headroom), __LINE__)

#define GTAP_RECORD_GENERATED_TASK_ID_BUFFER_OVERFLOW(tid, queue_idx, index, capacity) \
    gtap_record_runtime_error_and_trap( \
        GTAP_ERROR_GENERATED_TASK_ID_BUFFER_OVERFLOW, \
        (tid), (queue_idx), (index), (capacity), __LINE__)

#define GTAP_RECORD_INVALID_TASKWAIT(parent_tid, child_index, max_children) \
    gtap_record_runtime_error_and_trap( \
        GTAP_ERROR_INVALID_TASKWAIT, \
        (parent_tid), -1, (child_index), (max_children), __LINE__)

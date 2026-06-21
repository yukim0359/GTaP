#pragma once

#include <cuda_runtime.h>
#include <string.h>

// Runtime error codes
enum GTapRuntimeError {
    GTAP_ERROR_NONE = 0,
    GTAP_ERROR_INVALID_QUEUE_IDX = 1,
    GTAP_ERROR_QUEUE_OVERFLOW = 2,
    GTAP_ERROR_TASK_ID_POOL_EXHAUSTED = 3,
    GTAP_ERROR_INVALID_QUEUE_IDX_AFTER_JOIN = 4,
    GTAP_ERROR_INVALID_TASKWAIT = 5,
    GTAP_ERROR_GENERATED_TASK_ID_BUFFER_OVERFLOW = 6
};

struct GTapRuntimeErrorReport {
    int valid;
    int code;
    int line;
    int block_idx;
    int thread_idx;
    int worker_idx;
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

// Get error code and return error message string
inline const char* gtap_get_runtime_error_string(int error_code) {
    switch (error_code) {
        case GTAP_ERROR_NONE:
            return "No error";
        case GTAP_ERROR_INVALID_QUEUE_IDX:
            return "Invalid queue index";
        case GTAP_ERROR_QUEUE_OVERFLOW:
            return "Queue overflow";
        case GTAP_ERROR_TASK_ID_POOL_EXHAUSTED:
            return "Task ID pool exhausted";
        case GTAP_ERROR_INVALID_QUEUE_IDX_AFTER_JOIN:
            return "Invalid queue index after join";
        case GTAP_ERROR_INVALID_TASKWAIT:
            return "Invalid taskwait lowering";
        case GTAP_ERROR_GENERATED_TASK_ID_BUFFER_OVERFLOW:
            return "Generated task ID buffer overflow";
        default:
            return "Unknown error";
    }
}

inline bool gtap_print_runtime_error_report() {
    if (h_runtime_error_report == nullptr || h_runtime_error_report->valid == 0) {
        return false;
    }
    const GTapRuntimeErrorReport* r = h_runtime_error_report;
    printf("GTaP Runtime Error: %s (code: %d, worker: %d, block: %d, thread: %d, tid: %d, queue: %d, value: %d, limit: %d, line: %d)\n",
           gtap_get_runtime_error_string(r->code), r->code, r->worker_idx,
           r->block_idx, r->thread_idx, r->tid, r->queue_idx,
           r->value, r->limit, r->line);
    return true;
}

// Convenience function to check and print error if any
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
    int code,
    int worker_idx,
    int tid,
    int queue_idx,
    int value,
    int limit,
    int line
) {
    if (atomicCAS(&d_runtime_error_code, GTAP_ERROR_NONE, code) == GTAP_ERROR_NONE) {
        GTapRuntimeErrorReport* report = d_runtime_error_report;
        if (report != nullptr) {
            report->code = code;
            report->line = line;
            report->block_idx = blockIdx.x;
            report->thread_idx = threadIdx.x;
            report->worker_idx = worker_idx;
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

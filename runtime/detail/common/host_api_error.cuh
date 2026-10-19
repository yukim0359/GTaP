#pragma once

#include <cstdarg>
#include <cstdio>
#include <cuda_runtime.h>

namespace gtap::detail {

// Host-side reason for one public API failure. The first recorded message is
// kept. Callers still return cudaError_t.
struct host_api_error {
    char message[512] = {};

    cudaError_t fail(cudaError_t status, const char* format, ...) {
        if (status != cudaSuccess && message[0] == '\0') {
            va_list args;
            va_start(args, format);
            std::vsnprintf(message, sizeof(message), format, args);
            va_end(args);
        }
        return status;
    }
};

inline void print_host_api_error(
    const char* operation,
    cudaError_t status,
    const host_api_error& error
) {
    if (status == cudaSuccess) return;
    if (error.message[0] != '\0') {
        std::fprintf(
            stderr,
            "GTaP %s failed: %s (%s)\n",
            operation,
            error.message,
            cudaGetErrorString(status));
    } else {
        std::fprintf(
            stderr,
            "GTaP %s failed: %s\n",
            operation,
            cudaGetErrorString(status));
    }
}

inline void keep_first_error(cudaError_t status, cudaError_t* result) {
    if (*result == cudaSuccess) {
        *result = status;
    }
}

template <typename... Args>
inline cudaError_t record_error_if_failed(
    host_api_error& error,
    cudaError_t status,
    const char* format,
    const Args&... args
) {
    if (status == cudaSuccess) return cudaSuccess;
    return error.fail(status, format, args...);
}

}  // namespace gtap::detail

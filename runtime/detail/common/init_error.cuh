#pragma once

#include <cstdarg>
#include <cstdio>
#include <cuda_runtime.h>

namespace gtap::detail {

// Host-side reason for one gtap_initialize failure. The first recorded
// message is kept. Callers still return cudaError_t.
struct init_error {
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

template <typename... Args>
inline cudaError_t check_init(
    init_error& error,
    cudaError_t status,
    const char* format,
    const Args&... args
) {
    if (status == cudaSuccess) return cudaSuccess;
    return error.fail(status, format, args...);
}

}  // namespace gtap::detail

#pragma once

#include <memory>

#include "runtime_config.cuh"
#include "runtime_error.cuh"

template<class Kernel, class... Args>
inline cudaError_t gtap_launch(Kernel kernel, Args&&... args) {
    if (!gtap::detail::h_runtime_initialized) {
        return cudaErrorInitializationError;
    }
    const gtap::detail::launch_config& config = gtap::detail::h_launch_config;
    if constexpr (sizeof...(Args) == 0) {
        return cudaLaunchKernel(
            reinterpret_cast<const void*>(kernel),
            dim3(static_cast<unsigned int>(config.grid_size)),
            dim3(static_cast<unsigned int>(config.block_size)),
            nullptr,
            config.dynamic_shared_bytes,
            gtap::detail::h_stream
        );
    } else {
        void* packed_arguments[] = {
            const_cast<void*>(
                static_cast<const void*>(std::addressof(args))
            )...
        };
        return cudaLaunchKernel(
            reinterpret_cast<const void*>(kernel),
            dim3(static_cast<unsigned int>(config.grid_size)),
            dim3(static_cast<unsigned int>(config.block_size)),
            packed_arguments,
            config.dynamic_shared_bytes,
            gtap::detail::h_stream
        );
    }
}

inline cudaError_t gtap_synchronize() {
    cudaError_t st = cudaDeviceSynchronize();
    gtap::detail::runtime_error_record record{};
    if (gtap::detail::read_error_report(&record)) {
        gtap::detail::print_error_report(&record);
        gtap::detail::reset_runtime_error_record_host();
        return st;
    }
    if (st != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR: %s\n", cudaGetErrorString(st));
        return st;
    }
    return cudaSuccess;
}

#pragma once

#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include "host_api_error.cuh"
#include "runtime_config.cuh"
#include "runtime_error.cuh"

namespace gtap::detail {

// cudaLaunchKernel copies the bytes at each pointer. The copy is the kernel
// parameter type, so the object here has to be that type already.
template<class Param, class Arg>
inline Param convert_launch_argument(Arg&& arg) {
    static_assert(std::is_convertible<Arg, Param>::value,
        "gtap_launch argument is not convertible to the kernel parameter type");
    return std::forward<Arg>(arg);
}

template<class... Params, std::size_t... Index>
inline cudaError_t launch_prepared_kernel(
    void (*kernel)(Params...),
    std::tuple<Params...>& prepared,
    std::index_sequence<Index...>
) {
    void* packed_arguments[] = {
        static_cast<void*>(std::addressof(std::get<Index>(prepared)))...
    };
    const launch_config& config = h_launch_config;
    return cudaLaunchKernel(
        reinterpret_cast<const void*>(kernel),
        dim3(static_cast<unsigned int>(config.grid_size)),
        dim3(static_cast<unsigned int>(config.block_size)),
        packed_arguments,
        config.dynamic_shared_bytes,
        h_stream);
}

}  // namespace gtap::detail

// Converts each argument to the kernel parameter type, then passes those
// objects to cudaLaunchKernel. A different argument count, or a type that
// does not convert, is rejected while compiling.
template<class... Params, class... Args>
inline cudaError_t gtap_launch(void (*kernel)(Params...), Args&&... args) {
    if constexpr (sizeof...(Params) != sizeof...(Args)) {
        static_assert(sizeof...(Params) == sizeof...(Args),
            "gtap_launch argument count does not match the kernel");
        return cudaErrorInvalidValue;
    } else {
        gtap::detail::host_api_error error{};
        const char* rejection = nullptr;
        switch (gtap::detail::h_runtime_state) {
            case gtap::detail::runtime_state::ready:
                break;
            case gtap::detail::runtime_state::uninitialized:
                rejection = "GTaP is not initialized; call gtap_initialize before launching";
                break;
            case gtap::detail::runtime_state::launched:
                rejection = "call gtap_reset before launching again";
                break;
            case gtap::detail::runtime_state::reset_failed:
                rejection =
                    "call gtap_finalize before reinitializing after a failed reset";
                break;
            case gtap::detail::runtime_state::finalize_failed:
                rejection = "GTaP cannot be launched after a failed finalization";
                break;
        }
        if (rejection != nullptr) {
            cudaError_t status = error.fail(
                cudaErrorInitializationError, "%s", rejection);
            gtap::detail::print_host_api_error("launch", status, error);
            return status;
        }
        cudaError_t status = cudaSuccess;
        if constexpr (sizeof...(Params) == 0) {
            const gtap::detail::launch_config& config = gtap::detail::h_launch_config;
            status = cudaLaunchKernel(
                reinterpret_cast<const void*>(kernel),
                dim3(static_cast<unsigned int>(config.grid_size)),
                dim3(static_cast<unsigned int>(config.block_size)),
                nullptr,
                config.dynamic_shared_bytes,
                gtap::detail::h_stream);
        } else {
            std::tuple<Params...> prepared{
                gtap::detail::convert_launch_argument<Params>(
                    std::forward<Args>(args))...
            };
            status = gtap::detail::launch_prepared_kernel(
                kernel, prepared, std::index_sequence_for<Params...>{});
        }
        status = gtap::detail::record_error_if_failed(
            error, status, "failed to launch the kernel");
        if (status == cudaSuccess) {
            gtap::detail::h_runtime_state = gtap::detail::runtime_state::launched;
        }
        gtap::detail::print_host_api_error("launch", status, error);
        return status;
    }
}

inline cudaError_t gtap_synchronize() {
    cudaError_t st = cudaDeviceSynchronize();
    gtap::detail::print_failed_cuda_call(st);
    return st;
}

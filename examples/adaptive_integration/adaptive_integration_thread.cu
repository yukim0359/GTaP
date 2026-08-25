#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "gtap_thread.cuh"

// A higher frequency creates many adaptive subintervals without setting the
// tolerance below the useful precision of double-precision arithmetic.
constexpr double kDefaultIntegrationTolerance = 1.0e-12;
constexpr double kValidationTolerance = 1.0e-7;
constexpr double kDefaultFrequency = 300.0;

__device__ double d_result;

__device__ __forceinline__ double integrand(double x, double frequency) {
    return cos(frequency * x);
}

__device__ __forceinline__ double trap(double a, double b,
                                       double frequency) {
    return 0.5 * (b - a) *
           (integrand(a, frequency) + integrand(b, frequency));
}

#pragma gtap function
__device__ double integrate(double a, double b, double tolerance,
                            double frequency) {
    const double m = 0.5 * (a + b);
    const double coarse = trap(a, b, frequency);
    const double fine =
        trap(a, m, frequency) + trap(m, b, frequency);

    if (fabs(fine - coarse) < tolerance) {
        // Richardson extrapolation of the two trapezoidal estimates.
        return (4.0 * fine - coarse) / 3.0;
    }

    double left;
    double right;
    #pragma gtap task
    left = integrate(a, m, tolerance, frequency);
    #pragma gtap task
    right = integrate(m, b, tolerance, frequency);
    #pragma gtap taskwait
    return left + right;
}

__global__ void exec_kernel(double lo, double hi, double tolerance,
                            double frequency) {
    #pragma gtap entry
    d_result = integrate(lo, hi, tolerance, frequency);
}

int main(int argc, char** argv) {
    double lo = 0.0;
    double hi = 1.0;
    double tolerance = kDefaultIntegrationTolerance;
    double frequency = kDefaultFrequency;
    if (argc >= 2) lo = std::strtod(argv[1], nullptr);
    if (argc >= 3) hi = std::strtod(argv[2], nullptr);
    if (argc >= 4) tolerance = std::strtod(argv[3], nullptr);
    if (argc >= 5) frequency = std::strtod(argv[4], nullptr);
    if (!std::isfinite(lo) || !std::isfinite(hi) ||
        !std::isfinite(tolerance) || tolerance <= 0.0 ||
        !std::isfinite(frequency)) {
        std::fprintf(stderr,
                     "usage: %s [lo [hi [tolerance [frequency]]]]\n",
                     argv[0]);
        return 1;
    }

    gtap_thread_config config{
        .grid_size = 2000,
        .block_size = 32,
        .max_tasks_per_warp = 100000,
        .num_queues = 1,
    };

    cudaError_t err = gtap_initialize(config);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "gtap_initialize failed: %s\n",
                     cudaGetErrorString(err));
        return 1;
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    err = gtap_launch(exec_kernel, lo, hi, tolerance, frequency);
    if (err == cudaSuccess) {
        cudaEventRecord(stop);
        err = gtap_synchronize();
    }
    if (err != cudaSuccess) {
        std::fprintf(stderr, "adaptive integration failed: %s\n",
                     cudaGetErrorString(err));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        gtap_finalize();
        return 1;
    }

    double result = 0.0;
    err = cudaMemcpyFromSymbol(&result, d_result, sizeof(result));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "copying the result failed: %s\n",
                     cudaGetErrorString(err));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        gtap_finalize();
        return 1;
    }

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    const double reference = frequency == 0.0
        ? hi - lo
        : (std::sin(frequency * hi) - std::sin(frequency * lo)) /
              frequency;
    const double abs_error = std::fabs(result - reference);
    const double error_limit =
        kValidationTolerance * std::fmax(1.0, std::fabs(reference));
    const bool passed = abs_error <= error_limit;

    std::printf("Adaptive integration (thread mode) on [%.6g, %.6g], "
                "tolerance %.3e, frequency %.6g\n",
                lo, hi, tolerance, frequency);
    std::printf("Result: %.15g (reference %.15g)\n", result, reference);
    std::printf("Absolute error: %.3e [%s]\n", abs_error,
                passed ? "PASSED" : "FAILED");
    std::printf("Execution time: %.3f ms\n", elapsed_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    const cudaError_t finalize_err = gtap_finalize();
    return passed && finalize_err == cudaSuccess ? 0 : 1;
}

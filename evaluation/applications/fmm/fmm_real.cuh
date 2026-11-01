#ifndef FMM_REAL_CUH
#define FMM_REAL_CUH

#include <cmath>

#ifndef FMM3D_SINGLE
#define FMM3D_SINGLE 0
#endif

// Match exafmm-beta/include/types.h for Real (EXAFMM_SINGLE => float, default double).
// EXA_EPS / H_EPS2: gtap_fmm uses a single legacy scaling (1e-6) in both precisions.
// ExaFMM sets EPS=1e-16 for double, but this port's M2M/M2L/L2L tables were tuned at 1e-6.
#if FMM3D_SINGLE
using Real = float;
static constexpr Real EXA_EPS = 1e-6f;
static constexpr Real H_EPS2 = 1e-6f;
#else
using Real = double;
static constexpr Real EXA_EPS = 1e-6;
static constexpr Real H_EPS2 = 1e-6;
#endif

__host__ __device__ inline Real real_abs(Real x) {
#if defined(__CUDA_ARCH__)
#if FMM3D_SINGLE
  return fabsf(x);
#else
  return fabs(x);
#endif
#else
  return std::abs(x);
#endif
}

__host__ __device__ inline Real real_sqrt(Real x) {
#if defined(__CUDA_ARCH__)
#if FMM3D_SINGLE
  return sqrtf(x);
#else
  return sqrt(x);
#endif
#else
  return std::sqrt(x);
#endif
}

__host__ __device__ inline Real real_rsqrt(Real x) {
#if defined(__CUDA_ARCH__)
#if FMM3D_SINGLE
  return rsqrtf(x);
#else
  return rsqrt(x);
#endif
#else
  return Real(1) / std::sqrt(x);
#endif
}

__host__ __device__ inline Real real_cos(Real x) {
#if defined(__CUDA_ARCH__)
#if FMM3D_SINGLE
  return cosf(x);
#else
  return cos(x);
#endif
#else
  return std::cos(x);
#endif
}

__host__ __device__ inline Real real_sin(Real x) {
#if defined(__CUDA_ARCH__)
#if FMM3D_SINGLE
  return sinf(x);
#else
  return sin(x);
#endif
#else
  return std::sin(x);
#endif
}

__host__ __device__ inline Real real_acos(Real x) {
#if defined(__CUDA_ARCH__)
#if FMM3D_SINGLE
  return acosf(x);
#else
  return acos(x);
#endif
#else
  return std::acos(x);
#endif
}

__host__ __device__ inline Real real_atan2(Real y, Real x) {
#if defined(__CUDA_ARCH__)
#if FMM3D_SINGLE
  return atan2f(y, x);
#else
  return atan2(y, x);
#endif
#else
  return std::atan2(y, x);
#endif
}

__host__ __device__ inline Real real_fmin(Real a, Real b) {
#if defined(__CUDA_ARCH__)
#if FMM3D_SINGLE
  return fminf(a, b);
#else
  return fmin(a, b);
#endif
#else
  return std::fmin(a, b);
#endif
}

__host__ __device__ inline Real real_fmax(Real a, Real b) {
#if defined(__CUDA_ARCH__)
#if FMM3D_SINGLE
  return fmaxf(a, b);
#else
  return fmax(a, b);
#endif
#else
  return std::fmax(a, b);
#endif
}

__host__ __device__ inline Real real_copysign(Real x, Real y) {
#if defined(__CUDA_ARCH__)
#if FMM3D_SINGLE
  return copysignf(x, y);
#else
  return copysign(x, y);
#endif
#else
  return std::copysign(x, y);
#endif
}

#endif  // FMM_REAL_CUH

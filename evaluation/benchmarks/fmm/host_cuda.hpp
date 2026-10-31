#pragma once

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(stmt)                                                        \
  do {                                                                          \
    cudaError_t _e = (stmt);                                                    \
    if (_e != cudaSuccess) {                                                    \
      std::printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                  cudaGetErrorString(_e));                                      \
      return 1;                                                                 \
    }                                                                           \
  } while (0)

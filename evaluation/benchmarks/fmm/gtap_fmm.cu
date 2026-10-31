// 3D Laplace FMM-style benchmark for GTaP.
//
// The numerical kernel is still compact, but the data path now uses a small
// fixed-order spherical-harmonic Laplace expansion following exafmm-beta's
// kernels/laplace.h indexing and translation formulas, while preserving the
// DTT comparison axis.

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

// #define GTAP_ENABLE_PROFILING
#include "gtap_thread.cuh"
#include "config_defaults.cuh"
#include "fmm_types.cuh"
#include "host_sph.hpp"
#include "tree/host_octree.hpp"
#include "tree/gpu_morton.cuh"
#include "dtt/host.cuh"
#include "host_timing.hpp"
#include "device_symbols.cuh"


#include "numerics/device_sph.cuh"
#include "numerics/host_upward.hpp"
#include "numerics/host_l2l.hpp"
#include "numerics/kernels.cuh"
#include "numerics/validate.cuh"
#include "dtt/device.cuh"

#include "device_bind.cuh"
#include "host_types.hpp"
#include "host_options.hpp"
#include "host_setup.cuh"
#include "host_run.cuh"
#include "host_validate.hpp"
#include "host_report.hpp"

int main(int argc, char **argv) {
  FmmRun run;
  if (!parse_options(argc, argv, run.options)) return 1;
  if (init_cuda()) return 1;

  run.timings.run_start = std::chrono::high_resolution_clock::now();

  const auto t_tree_build_start = std::chrono::high_resolution_clock::now();
  if (build_tree(run)) return 1;
  run.timings.ms_tree_build = h_elapsed_ms(
      t_tree_build_start, std::chrono::high_resolution_clock::now());
  if (run_cpu_upward(run)) return 1;

  const auto t_metadata_start = std::chrono::high_resolution_clock::now();
  if (build_metadata(run)) return 1;
  run.timings.ms_metadata = h_elapsed_ms(
      t_metadata_start, std::chrono::high_resolution_clock::now());

  const auto t_h2d_bind_start = std::chrono::high_resolution_clock::now();
  if (alloc_and_bind(run)) return 1;
  run.timings.ms_h2d_bind = h_elapsed_ms(
      t_h2d_bind_start, std::chrono::high_resolution_clock::now());

  if (run_gpu_upward(run)) return 1;
  if (init_dtt(run)) return 1;
  if (run_dtt(run)) return 1;
  if (run_m2l_l2l_eval(run)) return 1;
  if (copy_results(run)) return 1;
  if (validate_results(run)) return 1;
  print_run_report(run);
  release_run(run);
  return 0;
}

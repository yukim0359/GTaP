// 3D Laplace FMM-style benchmark for GTaP.
//
// The numerical kernel is still compact, but the data path now uses a small
// fixed-order spherical-harmonic Laplace expansion following exafmm-beta's
// kernels/laplace.h indexing and translation formulas, while preserving the
// DTT comparison axis.

#include <cuda_runtime.h>
#include <chrono>

// #define GTAP_ENABLE_PROFILING
#include "config_defaults.cuh"

#include "../../common/gtap_launch.hpp"
#include "host_options.hpp"
#include "host_setup.cuh"
#include "host_run.cuh"
#include "host_validate.cuh"
#include "host_report.hpp"

int main(int argc, char **argv) {
  FmmRun run;
  if (!parse_options(argc, argv, run.options)) return 1;
  int sample_n = run.options.direct_sample_n;
  if (!parse_int_env("FMM3D_DIRECT_SAMPLE_N", 0, sample_n)) return 1;
  run.options.direct_sample_n = std::min(run.options.n, sample_n);
  if (!init_cuda()) return 1;

  run.timings.run_start = std::chrono::high_resolution_clock::now();

  if (!build_tree(run)) return 1;
  if (!build_metadata(run)) return 1;
  if (!alloc_and_bind(run)) return 1;
  if (!run_upward(run)) return 1;
  if (!init_dtt(run)) return 1;
  if (!run_dtt(run)) return 1;
  if (!run_m2l_l2l_eval(run)) return 1;
  if (!copy_results(run)) return 1;
  if (!validate_results(run)) return 1;
  print_run_report(run);
  release_run(run);
  return 0;
}

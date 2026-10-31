#pragma once

#include "config_defaults.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>

static constexpr int DIRECT_SAMPLE_N_DEFAULT = FMM3D_DIRECT_SAMPLE_N;

struct FmmOptions {
  int n = 0;
  Real theta = Real(0.5);
  int direct_sample_n = 0;
};

static int h_get_env_int(const char *name, int fallback) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0') return fallback;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value) return fallback;
  if (parsed < 0) return 0;
  if (parsed > std::numeric_limits<int>::max()) return std::numeric_limits<int>::max();
  return static_cast<int>(parsed);
}

static bool parse_options(int argc, char **argv, FmmOptions &options) {
  if (argc < 2) {
    std::printf("Usage: %s <n> [theta]\n", argv[0]);
    return false;
  }
  options.n = std::atoi(argv[1]);
  options.theta = (argc >= 3) ? static_cast<Real>(std::atof(argv[2])) : Real(0.5);
  if (options.n <= 0) return false;
  options.direct_sample_n =
      std::min(options.n, h_get_env_int("FMM3D_DIRECT_SAMPLE_N", DIRECT_SAMPLE_N_DEFAULT));
  return true;
}

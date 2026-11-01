#pragma once

#include "config_defaults.cuh"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <limits>

static constexpr int DIRECT_SAMPLE_N_DEFAULT = FMM3D_DIRECT_SAMPLE_N;

struct FmmOptions {
  int n = 0;
  Real theta = Real(0.5);
  int direct_sample_n = 0;
};

static bool is_integer_string(const char *s) {
  if (s == nullptr || *s == '\0') return false;
  int i = (s[0] == '-' || s[0] == '+') ? 1 : 0;
  if (s[i] == '\0') return false;
  for (; s[i] != '\0'; ++i) {
    if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
  }
  return true;
}

static bool parse_int_env(const char *name, int minimum, int &out) {
  const char *s = std::getenv(name);
  if (s == nullptr || *s == '\0') return true;
  if (!is_integer_string(s)) {
    std::fprintf(stderr, "%s must be an integer >= %d, got %s\n", name, minimum, s);
    return false;
  }
  const long value = std::strtol(s, nullptr, 10);
  if (value < minimum || value > std::numeric_limits<int>::max()) {
    std::fprintf(stderr, "%s must be an integer in [%d, %d], got %s\n", name, minimum,
                 std::numeric_limits<int>::max(), s);
    return false;
  }
  out = static_cast<int>(value);
  return true;
}

static bool apply_launch_env(int &grid_size, int &block_size, int &max_tasks_per_warp) {
  return parse_int_env("GTAP_GRID_SIZE", 1, grid_size) &&
         parse_int_env("GTAP_BLOCK_SIZE", 1, block_size) &&
         parse_int_env("GTAP_MAX_TASKS_PER_WARP", 0, max_tasks_per_warp);
}

static bool parse_options(int argc, char **argv, FmmOptions &options) {
  if (argc < 2) {
    std::printf("Usage: %s <n> [theta]\n", argv[0]);
    return false;
  }
  options.n = std::atoi(argv[1]);
  options.theta = (argc >= 3) ? static_cast<Real>(std::atof(argv[2])) : Real(0.5);
  if (options.n <= 0) return false;
  int sample_n = DIRECT_SAMPLE_N_DEFAULT;
  if (!parse_int_env("FMM3D_DIRECT_SAMPLE_N", 0, sample_n)) return false;
  options.direct_sample_n = std::min(options.n, sample_n);
  return true;
}

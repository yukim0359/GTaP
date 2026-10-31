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

static bool parse_positive_int_env(const char *name, int &out) {
  const char *s = std::getenv(name);
  if (s == nullptr || *s == '\0') return true;
  if (!is_integer_string(s)) {
    std::fprintf(stderr, "%s must be a positive integer, got %s\n", name, s);
    return false;
  }
  int value = std::atoi(s);
  if (value < 1) {
    std::fprintf(stderr, "%s must be positive, got %d\n", name, value);
    return false;
  }
  out = value;
  return true;
}

static bool apply_launch_env(int &grid_size, int &block_size, int &max_tasks_per_warp) {
  return parse_positive_int_env("GTAP_GRID_SIZE", grid_size) &&
         parse_positive_int_env("GTAP_BLOCK_SIZE", block_size) &&
         parse_positive_int_env("GTAP_MAX_TASKS_PER_WARP", max_tasks_per_warp);
}

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

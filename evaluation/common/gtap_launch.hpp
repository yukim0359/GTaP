#pragma once

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <limits>

#ifndef GTAP_GRID_SIZE
#define GTAP_GRID_SIZE 1
#endif
#ifndef GTAP_BLOCK_SIZE
#define GTAP_BLOCK_SIZE 32
#endif
#ifndef GTAP_MAX_TASKS_PER_WARP
#define GTAP_MAX_TASKS_PER_WARP 0
#endif
#ifndef GTAP_MAX_TASKS_PER_BLOCK
#define GTAP_MAX_TASKS_PER_BLOCK 0
#endif
#ifndef GTAP_NUM_QUEUES
#define GTAP_NUM_QUEUES 1
#endif

inline bool parse_int_env(const char *name, int minimum, int &out) {
  const char *s = std::getenv(name);
  if (s == nullptr || *s == '\0') {
    return true;
  }
  int i = (s[0] == '-' || s[0] == '+') ? 1 : 0;
  if (s[i] == '\0') {
    std::fprintf(stderr, "%s must be an integer >= %d, got %s\n", name, minimum, s);
    return false;
  }
  for (; s[i] != '\0'; ++i) {
    if (!std::isdigit(static_cast<unsigned char>(s[i]))) {
      std::fprintf(stderr, "%s must be an integer >= %d, got %s\n", name, minimum, s);
      return false;
    }
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

inline bool apply_thread_launch_env(int &grid_size, int &block_size, int &max_tasks_per_warp) {
  return parse_int_env("GTAP_GRID_SIZE", 1, grid_size) &&
         parse_int_env("GTAP_BLOCK_SIZE", 1, block_size) &&
         parse_int_env("GTAP_MAX_TASKS_PER_WARP", 0, max_tasks_per_warp);
}

inline bool apply_block_launch_env(int &grid_size, int &block_size, int &max_tasks_per_block) {
  return parse_int_env("GTAP_GRID_SIZE", 1, grid_size) &&
         parse_int_env("GTAP_BLOCK_SIZE", 1, block_size) &&
         parse_int_env("GTAP_MAX_TASKS_PER_BLOCK", 0, max_tasks_per_block);
}

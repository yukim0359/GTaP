#pragma once

#include <chrono>

static double h_elapsed_ms(std::chrono::high_resolution_clock::time_point begin,
                           std::chrono::high_resolution_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

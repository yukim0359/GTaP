#pragma once

#include <time.h>

static double elapsed_ms(const timespec& start, const timespec& stop) {
    return (double)(stop.tv_sec - start.tv_sec) * 1000.0 +
           (double)(stop.tv_nsec - start.tv_nsec) / 1000000.0;
}

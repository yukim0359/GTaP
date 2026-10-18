#pragma once

#include <cstddef>
#include <cstdint>

namespace gtap::detail {

inline bool checked_mul(size_t a, size_t b, size_t* out) {
    if (b != 0 && a > SIZE_MAX / b) return false;
    *out = a * b;
    return true;
}

inline bool checked_add(size_t a, size_t b, size_t* out) {
    if (a > SIZE_MAX - b) return false;
    *out = a + b;
    return true;
}

inline size_t mul_memory_bytes(size_t a, size_t b) {
    size_t bytes = 0;
    if (a == SIZE_MAX || b == SIZE_MAX || !checked_mul(a, b, &bytes)) {
        return SIZE_MAX;
    }
    return bytes;
}

inline size_t add_memory_bytes(size_t a, size_t b) {
    size_t bytes = 0;
    return checked_add(a, b, &bytes) ? bytes : SIZE_MAX;
}

}  // namespace gtap::detail

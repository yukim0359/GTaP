#pragma once

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

namespace gtap::detail {

// Host-side reason for one profile export failure. The first recorded message
// is kept. Callers still return bool.
struct profile_export_error {
    char message[512] = {};

    void fail(const char* format, ...) {
        if (message[0] != '\0') return;
        va_list args;
        va_start(args, format);
        vsnprintf(message, sizeof(message), format, args);
        va_end(args);
    }
};

// Reads errno immediately. Call before any later libc call can replace it.
static inline void record_profile_open_failure(
    profile_export_error& error,
    const char* path
) {
    const int open_errno = errno;
    error.fail(
        "failed to open '%s' for writing: %s",
        path,
        strerror(open_errno));
}

static inline void record_profile_write_failure(
    profile_export_error& error,
    const char* path
) {
    const int write_errno = errno;
    error.fail(
        "failed to write '%s': %s",
        path,
        strerror(write_errno));
}

}  // namespace gtap::detail

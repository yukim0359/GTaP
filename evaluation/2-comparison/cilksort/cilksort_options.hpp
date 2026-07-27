#pragma once

#include <cstdlib>
#include <cstring>

inline bool cilksort_is_validate_flag(const char* arg) {
    return std::strcmp(arg, "--no-validate") == 0 ||
           std::strcmp(arg, "--skip-validate") == 0;
}

inline bool cilksort_validate_enabled(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (cilksort_is_validate_flag(argv[i])) {
            return false;
        }
    }
    const char* env = std::getenv("CILKSORT_VALIDATE");
    if (env && (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 ||
              std::strcmp(env, "FALSE") == 0)) {
        return false;
    }
    return true;
}

inline const char* cilksort_data_file_arg(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] != '-') {
            return argv[i];
        }
    }
    return nullptr;
}

inline const char* cilksort_reference_file_arg(int argc, char** argv) {
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            continue;
        }
        if (++positional == 2) {
            return argv[i];
        }
    }
    return nullptr;
}

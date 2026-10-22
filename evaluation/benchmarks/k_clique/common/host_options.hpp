#pragma once

#include "gpu_orient.cuh"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int parse_int_arg(char** argv, int argc, int index, int default_value) {
    return argc > index ? atoi(argv[index]) : default_value;
}

static bool is_integer_string(const char* s) {
    if (s == nullptr || *s == '\0') return false;
    int i = (s[0] == '-' || s[0] == '+') ? 1 : 0;
    if (s[i] == '\0') return false;
    for (; s[i] != '\0'; ++i) {
        if (!std::isdigit((unsigned char)s[i])) return false;
    }
    return true;
}

static GtapOrientMode parse_gtap_orient_from_env(GtapOrientMode default_mode = GTAP_ORIENT_DEGREE) {
    const char* s = std::getenv("GTAP_ORIENT");
    if (s == nullptr || *s == '\0') return default_mode;
    if (std::strcmp(s, "degen") == 0 || std::strcmp(s, "degeneracy") == 0) {
        return GTAP_ORIENT_DEGEN;
    }
    if (std::strcmp(s, "degree") == 0) {
        return GTAP_ORIENT_DEGREE;
    }
    fprintf(stderr,
            "Unknown GTAP_ORIENT=%s; using %s\n",
            s,
            gtap_orient_mode_name(default_mode));
    return default_mode;
}

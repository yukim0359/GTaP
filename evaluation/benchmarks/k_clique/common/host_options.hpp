#pragma once

#include "config_defaults.cuh"
#include "gpu_orient.cuh"
#include "host_graph.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct HostOptions {
    const char* graph_path = nullptr;
    int k = 7;
};

static void graph_file_stem(const char* path, char* out, size_t out_size) {
    if (out == nullptr || out_size == 0) return;
    const char* base = path == nullptr ? "" : path;
    for (const char* p = base; *p != '\0'; ++p) {
        if (*p == '/' || *p == '\\') base = p + 1;
    }
    size_t len = std::strlen(base);
    const char* dot = std::strrchr(base, '.');
    if (dot != nullptr && dot != base) len = (size_t)(dot - base);
    if (len >= out_size) len = out_size - 1;
    std::memcpy(out, base, len);
    out[len] = '\0';
    if (out[0] == '\0') {
        std::snprintf(out, out_size, "graph");
    }
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

static bool parse_options(int argc, char** argv, HostOptions& options) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s graph_path [k]\n", argv[0]);
        return false;
    }
    options.graph_path = argv[1];
    options.k = 7;
    if (argc >= 3) {
        if (!is_integer_string(argv[2])) {
            fprintf(stderr, "k must be an integer, got %s\n", argv[2]);
            return false;
        }
        options.k = atoi(argv[2]);
    }
    return true;
}

static bool validate_problem(const HostOptions& options, const HostGraph& graph) {
    if (graph.n < 1) {
        fprintf(stderr, "n must be positive, got %d\n", graph.n);
        return false;
    }
    if (options.k < 3 || options.k > graph.n) {
        fprintf(stderr, "k must be in [3, n], got k=%d n=%d\n", options.k, graph.n);
        return false;
    }
    return true;
}

static bool parse_positive_int_env(const char* name, int& out) {
    const char* s = std::getenv(name);
    if (s == nullptr || *s == '\0') return true;
    if (!is_integer_string(s)) {
        fprintf(stderr, "%s must be a positive integer, got %s\n", name, s);
        return false;
    }
    int value = atoi(s);
    if (value < 1) {
        fprintf(stderr, "%s must be positive, got %d\n", name, value);
        return false;
    }
    out = value;
    return true;
}

static bool apply_launch_env(int& grid_size, int& block_size, int& max_tasks_per_warp) {
    return parse_positive_int_env("GTAP_GRID_SIZE", grid_size) &&
           parse_positive_int_env("GTAP_BLOCK_SIZE", block_size) &&
           parse_positive_int_env("GTAP_MAX_TASKS_PER_WARP", max_tasks_per_warp);
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

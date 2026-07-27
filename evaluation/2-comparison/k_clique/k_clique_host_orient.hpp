#pragma once

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <queue>
#include <utility>
#include <vector>

enum GtapOrientMode {
    GTAP_ORIENT_DEGREE = 0,
    GTAP_ORIENT_DEGEN = 1,
};

static const char* gtap_orient_mode_name(GtapOrientMode mode) {
    switch (mode) {
        case GTAP_ORIENT_DEGEN: return "degen";
        case GTAP_ORIENT_DEGREE:
        default: return "degree";
    }
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

static HostGraph build_oriented_csr_from_edges(
    int n, size_t undirected_edges, std::vector<std::pair<int, int>> edges) {
    HostGraph oriented;
    oriented.n = n;
    oriented.undirected_edges = undirected_edges;
    oriented.oriented = true;
    oriented.row_ptr.assign((size_t)n + 1, 0);
    for (auto e : edges) ++oriented.row_ptr[(size_t)e.first + 1];
    for (int i = 0; i < n; ++i) {
        oriented.row_ptr[(size_t)i + 1] += oriented.row_ptr[(size_t)i];
    }
    oriented.col_idx.assign(edges.size(), 0);
    std::vector<int> cursor = oriented.row_ptr;
    for (auto e : edges) {
        oriented.col_idx[(size_t)cursor[(size_t)e.first]++] = e.second;
    }
    for (int u = 0; u < n; ++u) {
        std::sort(oriented.col_idx.begin() + oriented.row_ptr[(size_t)u],
                  oriented.col_idx.begin() + oriented.row_ptr[(size_t)u + 1]);
    }
    return oriented;
}

static std::vector<int> compute_kcore_peel_priority(const HostGraph& graph) {
    const int n = graph.n;
    std::vector<int> deg((size_t)n);
    std::priority_queue<
        std::pair<int, int>,
        std::vector<std::pair<int, int>>,
        std::greater<std::pair<int, int>>> heap;
    for (int u = 0; u < n; ++u) {
        deg[(size_t)u] = graph.row_ptr[(size_t)u + 1] - graph.row_ptr[(size_t)u];
        heap.push(std::make_pair(deg[(size_t)u], u));
    }

    std::vector<char> removed((size_t)n, 0);
    std::vector<int> priority((size_t)n, 0);
    int peel_priority = 0;

    while (!heap.empty()) {
        int d = heap.top().first;
        int u = heap.top().second;
        heap.pop();
        if (removed[(size_t)u] || d != deg[(size_t)u]) continue;

        removed[(size_t)u] = 1;
        priority[(size_t)u] = peel_priority++;
        for (int ei = graph.row_ptr[(size_t)u]; ei < graph.row_ptr[(size_t)u + 1]; ++ei) {
            int v = graph.col_idx[(size_t)ei];
            if (removed[(size_t)v]) continue;
            --deg[(size_t)v];
            heap.push(std::make_pair(deg[(size_t)v], v));
        }
    }
    return priority;
}

static HostGraph orient_from_peel_priority(
    const HostGraph& graph, const std::vector<int>& priority) {
    std::vector<std::pair<int, int>> edges;
    edges.reserve(graph.undirected_edges);
    for (int u = 0; u < graph.n; ++u) {
        for (int ei = graph.row_ptr[(size_t)u]; ei < graph.row_ptr[(size_t)u + 1]; ++ei) {
            int v = graph.col_idx[(size_t)ei];
            if (u >= v) continue;
            bool u_before_v = priority[(size_t)u] < priority[(size_t)v] ||
                              (priority[(size_t)u] == priority[(size_t)v] && u < v);
            if (u_before_v) edges.push_back(std::make_pair(u, v));
            else edges.push_back(std::make_pair(v, u));
        }
    }
    return build_oriented_csr_from_edges(graph.n, graph.undirected_edges, std::move(edges));
}

static HostGraph orient_by_degeneracy(const HostGraph& graph) {
    std::vector<int> priority = compute_kcore_peel_priority(graph);
    return orient_from_peel_priority(graph, priority);
}

static HostGraph orient_graph(const HostGraph& graph, GtapOrientMode mode) {
    switch (mode) {
        case GTAP_ORIENT_DEGEN:
            return orient_by_degeneracy(graph);
        case GTAP_ORIENT_DEGREE:
        default:
            return orient_by_degree(graph);
    }
}

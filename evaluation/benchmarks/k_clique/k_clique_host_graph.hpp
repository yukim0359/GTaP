#pragma once

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <time.h>
#include <vector>

struct HostGraph {
    int n;
    size_t undirected_edges;
    bool oriented;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
};

static double elapsed_ms(const timespec& start, const timespec& stop) {
    return (double)(stop.tv_sec - start.tv_sec) * 1000.0 +
           (double)(stop.tv_nsec - start.tv_nsec) / 1000000.0;
}

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

static HostGraph build_csr_from_edges(
    int n, const std::vector<std::pair<int, int>>& undirected_edges) {
    HostGraph graph;
    graph.n = n;
    graph.undirected_edges = 0;
    graph.oriented = false;
    std::vector<std::vector<int>> adj((size_t)n);

    for (auto e : undirected_edges) {
        int u = e.first;
        int v = e.second;
        if (u < 0 || v < 0 || u >= n || v >= n || u == v) continue;
        adj[(size_t)u].push_back(v);
        adj[(size_t)v].push_back(u);
    }

    graph.row_ptr.resize((size_t)n + 1);
    for (int u = 0; u < n; ++u) {
        std::vector<int>& row = adj[(size_t)u];
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        graph.row_ptr[(size_t)u + 1] = graph.row_ptr[(size_t)u] + (int)row.size();
        graph.undirected_edges += row.size();
    }
    graph.undirected_edges /= 2;

    graph.col_idx.resize((size_t)graph.row_ptr[(size_t)n]);
    for (int u = 0; u < n; ++u) {
        const std::vector<int>& row = adj[(size_t)u];
        std::copy(row.begin(), row.end(), graph.col_idx.begin() + graph.row_ptr[(size_t)u]);
    }
    return graph;
}

static HostGraph orient_by_degree(const HostGraph& graph) {
    std::vector<std::pair<int, int>> edges;
    edges.reserve(graph.undirected_edges);
    std::vector<int> degree((size_t)graph.n);
    for (int u = 0; u < graph.n; ++u) {
        degree[(size_t)u] = graph.row_ptr[(size_t)u + 1] - graph.row_ptr[(size_t)u];
    }

    for (int u = 0; u < graph.n; ++u) {
        for (int ei = graph.row_ptr[(size_t)u]; ei < graph.row_ptr[(size_t)u + 1]; ++ei) {
            int v = graph.col_idx[(size_t)ei];
            if (u >= v) continue;
            bool u_before_v = degree[(size_t)u] < degree[(size_t)v] ||
                              (degree[(size_t)u] == degree[(size_t)v] && u < v);
            if (u_before_v) edges.push_back(std::make_pair(u, v));
            else edges.push_back(std::make_pair(v, u));
        }
    }

    HostGraph oriented;
    oriented.n = graph.n;
    oriented.undirected_edges = graph.undirected_edges;
    oriented.oriented = true;
    oriented.row_ptr.assign((size_t)graph.n + 1, 0);
    for (auto e : edges) ++oriented.row_ptr[(size_t)e.first + 1];
    for (int i = 0; i < graph.n; ++i) {
        oriented.row_ptr[(size_t)i + 1] += oriented.row_ptr[(size_t)i];
    }
    oriented.col_idx.assign(edges.size(), 0);
    std::vector<int> cursor = oriented.row_ptr;
    for (auto e : edges) {
        oriented.col_idx[(size_t)cursor[(size_t)e.first]++] = e.second;
    }
    for (int u = 0; u < graph.n; ++u) {
        std::sort(oriented.col_idx.begin() + oriented.row_ptr[(size_t)u],
                  oriented.col_idx.begin() + oriented.row_ptr[(size_t)u + 1]);
    }
    return oriented;
}

#include "k_clique_host_orient.hpp"

static HostGraph generate_random_graph(int n, int edge_probability_percent, unsigned seed) {
    std::vector<std::pair<int, int>> edges;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 99);
    for (int u = 0; u < n; ++u) {
        for (int v = u + 1; v < n; ++v) {
            if (dist(rng) < edge_probability_percent) edges.push_back(std::make_pair(u, v));
        }
    }
    return build_csr_from_edges(n, edges);
}

static bool load_graph_file(const char* path, HostGraph& graph) {
    std::ifstream in(path);
    if (!in.is_open()) {
        fprintf(stderr, "Failed to open graph file: %s\n", path);
        return false;
    }

    std::vector<std::pair<int, int>> edges;
    std::string line;
    bool header_seen = false;
    bool saw_zero = false;
    long long declared_n = -1;
    long long max_vertex = -1;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '%' || line[0] == '#') continue;

        if (!header_seen) {
            std::istringstream hdr(line);
            long long rows = 0, cols = 0, nnz = 0;
            std::string extra;
            if ((hdr >> rows >> cols >> nnz) && !(hdr >> extra) && rows == cols) {
                declared_n = rows;
                header_seen = true;
                continue;
            }
            header_seen = true;
        }

        std::istringstream iss(line);
        long long u = 0, v = 0;
        if (!(iss >> u >> v)) continue;
        if (u == v) continue;
        if (u == 0 || v == 0) saw_zero = true;
        if (u > max_vertex) max_vertex = u;
        if (v > max_vertex) max_vertex = v;
        edges.push_back(std::make_pair((int)u, (int)v));
    }

    if (declared_n < 0) declared_n = max_vertex + (saw_zero ? 1 : 0);
    if (declared_n <= 0 || declared_n > INT_MAX) {
        fprintf(stderr, "Graph file is empty, invalid, or too large: %s\n", path);
        return false;
    }

    bool one_based = !saw_zero;
    if (one_based) {
        for (auto& e : edges) {
            --e.first;
            --e.second;
        }
    }
    graph = build_csr_from_edges((int)declared_n, edges);
    return true;
}

static bool cpu_is_edge(const HostGraph& graph, int u, int v) {
    const int* begin = graph.col_idx.data() + graph.row_ptr[(size_t)u];
    const int* end = graph.col_idx.data() + graph.row_ptr[(size_t)u + 1];
    return std::binary_search(begin, end, v);
}

static unsigned long long cpu_count_k_rec(
    const HostGraph& graph, int k, int depth, int begin, int end, int* selected) {
    if (depth == k) return 1ULL;

    unsigned long long count = 0ULL;
    for (int i = begin; i < end; ++i) {
        int u = graph.col_idx[(size_t)i];
        bool ok = true;
        for (int j = 0; j < depth; ++j) {
            if (!cpu_is_edge(graph, selected[j], u)) {
                ok = false;
                break;
            }
        }
        if (!ok) continue;
        selected[depth] = u;
        count += cpu_count_k_rec(
            graph,
            k,
            depth + 1,
            graph.row_ptr[(size_t)u],
            graph.row_ptr[(size_t)u + 1],
            selected);
    }
    return count;
}

static unsigned long long cpu_count_k_oriented(const HostGraph& graph, int k) {
    unsigned long long count = 0ULL;
    std::vector<int> selected((size_t)k);
    for (int u = 0; u < graph.n; ++u) {
        selected[(size_t)0] = u;
        count += cpu_count_k_rec(
            graph,
            k,
            1,
            graph.row_ptr[(size_t)u],
            graph.row_ptr[(size_t)u + 1],
            selected.data());
    }
    return count;
}

static std::vector<int> make_edge_sources(const HostGraph& graph) {
    std::vector<int> edge_src(graph.col_idx.size());
    for (int u = 0; u < graph.n; ++u) {
        for (int ei = graph.row_ptr[(size_t)u]; ei < graph.row_ptr[(size_t)u + 1]; ++ei) {
            edge_src[(size_t)ei] = u;
        }
    }
    return edge_src;
}

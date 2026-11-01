#pragma once

#include <algorithm>
#include <climits>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

struct HostGraph {
    int n;
    size_t undirected_edges;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
};

static HostGraph build_csr_from_edges(
    int n, const std::vector<std::pair<int, int>>& undirected_edges) {
    HostGraph graph;
    graph.n = n;
    graph.undirected_edges = 0;
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

#pragma once

#include "k_clique_config_defaults.cuh"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static inline size_t gtap_k_ncr_table_elems() {
    return (size_t)GTAP_K_NCR_ROWS * (size_t)GTAP_K_NCR_COLS;
}

static inline std::string gtap_k_default_ncr_path(const char* source_file) {
    std::string path(source_file);
    const std::string::size_type pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return "data/nCr.txt";
    return path.substr(0, pos + 1) + "data/nCr.txt";
}

static inline bool gtap_k_load_ncr_table_from_file(
    const char* path,
    std::vector<unsigned long long>& table) {
    FILE* infile = fopen(path, "r");
    if (infile == nullptr) return false;

    table.assign(gtap_k_ncr_table_elems(), 0ULL);
    for (int row = 0; row < GTAP_K_NCR_ROWS; ++row) {
        for (int col = 0; col < GTAP_K_NCR_COLS; ++col) {
            double value = 0.0;
            if (fscanf(infile, "%lf,", &value) != 1) {
                fclose(infile);
                return false;
            }
            table[(size_t)row * (size_t)GTAP_K_NCR_COLS + (size_t)col] =
                (unsigned long long)value;
        }
    }
    fclose(infile);
    return true;
}

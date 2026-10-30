#pragma once

#include "config_defaults.cuh"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <string>
#include <vector>

static inline std::string default_ncr_path() {
    std::string path(__FILE__);
    const std::string::size_type pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return "../data/nCr.txt";
    return path.substr(0, pos + 1) + "../data/nCr.txt";
}

static inline bool load_ncr_table_from_file(
    const char* path,
    std::vector<unsigned long long>& table) {
    FILE* infile = fopen(path, "r");
    if (infile == nullptr) return false;

    table.assign((size_t)GTAP_NCR_ROWS * (size_t)GTAP_NCR_COLS, 0ULL);
    for (int row = 0; row < GTAP_NCR_ROWS; ++row) {
        for (int col = 0; col < GTAP_NCR_COLS; ++col) {
            double value = 0.0;
            if (fscanf(infile, "%lf,", &value) != 1) {
                fclose(infile);
                return false;
            }
            table[(size_t)row * (size_t)GTAP_NCR_COLS + (size_t)col] =
                (unsigned long long)value;
        }
    }
    fclose(infile);
    return true;
}

static inline bool load_ncr_to_device(unsigned long long** d_nCr) {
    std::vector<unsigned long long> ncr_host;
    const std::string ncr_path = default_ncr_path();
    const char* ncr_file = std::getenv("GTAP_NCR_FILE");
    if (ncr_file == nullptr || ncr_file[0] == '\0') ncr_file = ncr_path.c_str();
    if (!load_ncr_table_from_file(ncr_file, ncr_host)) {
        fprintf(stderr, "Failed to load nCr table from %s\n", ncr_file);
        return false;
    }
    const size_t bytes =
        (size_t)GTAP_NCR_ROWS * (size_t)GTAP_NCR_COLS * sizeof(unsigned long long);
    cudaError_t err = cudaMalloc(d_nCr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc nCr failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    err = cudaMemcpy(*d_nCr, ncr_host.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy nCr failed: %s\n", cudaGetErrorString(err));
        cudaFree(*d_nCr);
        *d_nCr = nullptr;
        return false;
    }
    return true;
}

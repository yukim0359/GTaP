#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <time.h>

#include "cilksort_options.hpp"

static inline double diff_sec(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}

std::vector<int> load_array(const char* filename, size_t& n) {
    std::vector<int> data;
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s for reading\n", filename);
        return data;
    }

    if (fread(&n, sizeof(size_t), 1, fp) != 1) {
        fprintf(stderr, "Error: Cannot read size\n");
        fclose(fp);
        return data;
    }

    data.resize(n);
    if (fread(data.data(), sizeof(int), n, fp) != n) {
        fprintf(stderr, "Error: Cannot read array\n");
        data.clear();
        fclose(fp);
        return data;
    }

    fclose(fp);
    printf("Loaded %zu elements from %s\n", n, filename);
    return data;
}

static void merge_seq(int* a, int a_len,
                      int* b, int b_len,
                      int* dst) {
    int i = 0, j = 0, k = 0;
    while (i < a_len && j < b_len) {
        if (a[i] <= b[j]) dst[k++] = a[i++];
        else              dst[k++] = b[j++];
    }
    while (i < a_len) dst[k++] = a[i++];
    while (j < b_len) dst[k++] = b[j++];
}

static void cilksort_rec(int* arr, int* tmp, int n) {
    if (n < 2) return;

    int n12  = n / 2;
    int n1   = n12 / 2;
    int n2   = n12 - n1;
    int n34  = n - n12;
    int n3   = n34 / 2;
    int n4   = n34 - n3;

    cilksort_rec(arr,             tmp,             n1);
    cilksort_rec(arr + n1,        tmp + n1,        n2);
    cilksort_rec(arr + n12,       tmp + n12,       n3);
    cilksort_rec(arr + n12 + n3,  tmp + n12 + n3,  n4);

    merge_seq(arr, n1, arr + n1, n2, tmp);
    merge_seq(arr + n12, n3, arr + n12 + n3, n4, tmp + n12);

    merge_seq(tmp, n12, tmp + n12, n34, arr);
}

void cilksort(std::vector<int>& a) {
    if (a.empty()) return;
    std::vector<int> tmp(a.size());
    cilksort_rec(a.data(), tmp.data(), (int)a.size());
}

int main(int argc, char** argv) {
    const bool validate = cilksort_validate_enabled(argc, argv);
    const char* data_file = cilksort_data_file_arg(argc, argv);

    if (!data_file) {
        printf("Usage: %s [--no-validate] <data_file>\n", argv[0]);
        printf("  --no-validate: Skip correctness check (benchmark mode)\n");
        return 1;
    }

    size_t N;
    std::vector<int> a = load_array(data_file, N);
    if (a.empty()) return 1;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    cilksort(a);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_sec = diff_sec(t0, t1);

    int exit_code = 0;
    if (validate) {
        bool ok = std::is_sorted(a.begin(), a.end());
        printf("Cilksort(%zu) = %s\n", N, ok ? "Correct" : "Incorrect");
        exit_code = ok ? 0 : 1;
    }
    printf("Execution time: %.3f ms\n", elapsed_sec * 1000.0);

    return exit_code;
}

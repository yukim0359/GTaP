#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

static unsigned long long total_solutions = 0ULL;

void serial_search(int row, uint32_t col, uint32_t ld, uint32_t rd, int n) {
    if (row == n) {
        #pragma omp atomic
        total_solutions++;
        return;
    }
    uint32_t mask = (n < 32 ? (1u << n) - 1 : 0xFFFFFFFFu);
    uint32_t avail = mask & ~(col | ld | rd);
    while (avail) {
        uint32_t p = avail & -avail;
        avail -= p;
        serial_search(row + 1, col | p, (ld | p) << 1, (rd | p) >> 1, n);
    }
}

void solve(int row, uint32_t col, uint32_t ld, uint32_t rd, int n, int cutoff) {
    if (row > cutoff) {
        serial_search(row, col, ld, rd, n);
        return;
    }

    uint32_t mask = (n < 32 ? (1u << n) - 1 : 0xFFFFFFFFu);
    uint32_t avail = mask & ~(col | ld | rd);

    while (avail) {
        uint32_t p = avail & -avail;
        avail -= p;
        #pragma omp task firstprivate(row, col, ld, rd, p, n, cutoff)
        solve(row + 1, col | p, (ld | p) << 1, (rd | p) >> 1, n, cutoff);
    }
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s <n> [cutoff]\n", prog);
    fprintf(stderr, "  n: board size (default 16)\n");
    fprintf(stderr, "  cutoff: parallel depth limit (default 7)\n");
}

int main(int argc, char **argv) {
    int n = 16;
    int cutoff = 7;
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) cutoff = atoi(argv[2]);
    if (argc > 3 || n < 0 || cutoff < 0) {
        usage(argv[0]);
        return 1;
    }

    double start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        solve(0, 0, 0, 0, n, cutoff);
    }
    double end = omp_get_wtime();

    printf("N-Queens(%d) = %llu\n", n, total_solutions);
    printf("Execution time: %.3f ms\n", (end - start) * 1000.0);
    return 0;
}

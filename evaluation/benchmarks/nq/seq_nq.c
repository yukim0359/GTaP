#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

static inline double diff_sec(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}

static unsigned long long total_solutions = 0ULL;

void solve(int row, uint32_t col, uint32_t ld, uint32_t rd, int n) {
    if (row == n) {
        total_solutions++;
        return;
    }

    uint32_t mask = (n < 32 ? (1u << n) - 1u : 0xFFFFFFFFu);
    uint32_t avail = mask & ~(col | ld | rd);

    while (avail) {
        uint32_t p = avail & -avail;
        avail -= p;
        solve(row + 1, col | p, (ld | p) << 1, (rd | p) >> 1, n);
    }
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s <n> [cutoff]\n", prog);
    fprintf(stderr, "  n: board size (default 16)\n");
    fprintf(stderr, "  cutoff: ignored (present for benchmark script compatibility)\n");
}

int main(int argc, char **argv) {
    int n = 16;
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 3 || n < 0) {
        usage(argv[0]);
        return 1;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    solve(0, 0, 0, 0, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    printf("N-Queens(%d) = %llu\n", n, total_solutions);
    printf("Execution time: %.3f ms\n", diff_sec(t0, t1) * 1000.0);

    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cilk/cilk.h>

static unsigned long long total_solutions = 0ULL;

void serial_search(int row, uint32_t col, uint32_t ld, uint32_t rd, int n) {
    if (row == n) {
        __atomic_fetch_add(&total_solutions, 1ULL, __ATOMIC_RELAXED);
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
        cilk_spawn solve(row + 1, col | p, (ld | p) << 1, (rd | p) >> 1, n, cutoff);
    }
    cilk_sync;
}

static inline double diff_sec(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
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

    /* Warm up the Cilk runtime; compare_nq.sh averages many timed runs. */
    solve(0, 0, 0, 0, 2, cutoff);
    total_solutions = 0ULL;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    solve(0, 0, 0, 0, n, cutoff);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    printf("N-Queens(%d) = %llu\n", n, total_solutions);
    printf("Execution time: %.3f ms\n", diff_sec(t0, t1) * 1000.0);
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

static inline double diff_sec(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}

static uint32_t g_answer = 0;

void solve(int row, uint32_t column, uint32_t left, uint32_t right, int GRID_SIZE) {
    if (row == GRID_SIZE) {
        g_answer++;
        return;
    }

    uint32_t mask  = (GRID_SIZE < 32 ? (1u << GRID_SIZE) - 1u : 0xFFFFFFFFu);
    uint32_t avail = mask & ~(column | left | right);

    while (avail) {
        uint32_t p = avail & -avail;
        avail -= p;

        solve(
            row + 1,
            column | p,
            (left  | p) << 1,
            (right | p) >> 1,
            GRID_SIZE
        );
    }
}

int main(int argc, char **argv) {
    int GRID_SIZE = (argc > 1 ? atoi(argv[1]) : 16);
    
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    solve(0, 0, 0, 0, GRID_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed_sec = diff_sec(t0, t1);

    printf("N-Queens(%d) = %u\n", GRID_SIZE, g_answer);
    printf("Execution time: %.3f ms\n", elapsed_sec * 1000.0);

    return 0;
}

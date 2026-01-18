#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

static uint32_t total_solutions = 0;

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
    // if (row == n) {
    //     #pragma omp atomic
    //     total_solutions++;
    //     return;
    // }

    uint32_t mask = (n < 32 ? (1u << n) - 1 : 0xFFFFFFFFu);
    uint32_t avail = mask & ~(col | ld | rd);

    while (avail) {
        uint32_t p = avail & -avail;
        avail -= p;
        #pragma omp task firstprivate(row, col, ld, rd, p, n, cutoff)
        solve(row + 1, col | p, (ld | p) << 1, (rd | p) >> 1, n, cutoff);
    }
}

int main(int argc, char **argv) {
    int n = (argc > 1 ? atoi(argv[1]) : 16);
    int cutoff = (argc > 2 ? atoi(argv[2]) : 7);

    #pragma omp parallel
    {
        #pragma omp single
        {
            /* no op */
        }
    }

    double start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        solve(0, 0, 0, 0, n, cutoff);
    }
    double end = omp_get_wtime();

    printf("N-Queens(%d) = %u\n", n, total_solutions);
    printf("Execution time: %.3f ms\n", (end - start) * 1000);
    return 0;
}

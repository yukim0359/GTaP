#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cilk/cilk.h>

int fib(int n) {
    if (n < 2) return n;

    int a, b;
    a = cilk_spawn fib(n - 1);
    b = cilk_spawn fib(n - 2);
    cilk_sync;
    return a + b;
}

static inline double diff_sec(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}

int main(int argc, char** argv) {
    int n = 40;
    if (argc >= 2) n = atoi(argv[1]);

    /* Warm up the Cilk runtime (similar to omp_fib.c). */
    (void)fib(2);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int result = fib(n);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed_ms = diff_sec(t0, t1) * 1000.0;

    printf("Fibonacci of %d is %d\n", n, result);
    printf("Execution time: %.3f ms\n", elapsed_ms);
    return 0;
}

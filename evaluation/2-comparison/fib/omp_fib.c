#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

int fib(int n) {
    if (n < 2) return n;

    int a, b;
    #pragma omp task shared(a)
    a = fib(n - 1);
    #pragma omp task shared(b)
    b = fib(n - 2);
    #pragma omp taskwait
    return a + b;
}

int main(int argc, char** argv) {
    int n = 40; // default value
    if (argc >= 2) n = atoi(argv[1]);

    #pragma omp parallel
    {
        #pragma omp master
        {
            /* no op */
        }
    }

    int result = 0;
    double start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp master
        {
            result = fib(n);
        }
    }
    double end = omp_get_wtime();
    double time_used = end - start;

    printf("Fibonacci of %d is %d\n", n, result);
    printf("Execution time: %.3f ms\n", time_used * 1000);

    return 0;
}

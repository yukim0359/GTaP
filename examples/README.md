# GTaP Examples

Minimal, runnable GTaP programs. Each subdirectory contains one example and a `Makefile` to build it.

## Prerequisites

- **GTaP compiler**: Build the Clang-based GTaP compiler first; see [`../clang-gtap/README.md`](../clang-gtap/README.md).
- **CUDA Toolkit**: Set `CUDA_PATH` (or `CUDA_HOME`) to the CUDA installation root.
- **Runtime headers**: Located at `../runtime` relative to each example.


## Building

```bash
# Optional: set environment variables (can also be passed directly to make)
export CUDA_PATH=/path/to/cuda
export CUDA_ARCH=sm_90   # sm_80, sm_70, etc.

cd examples/fib
make
./bin/fib_thread
```

Or override variables on the command line:

```bash
cd examples/spmv
make GTAP_ROOT=/path/to/gtap CUDA_PATH=/path/to/cuda
./bin/spmv_thread
./bin/spmv_block
```

## Examples


| Example | Description | Mode |
|---------|-------------|------|
| [adaptive_integration](adaptive_integration/) | Adaptive numerical integration from the paper | thread / block |
| [fib](fib/) | Fibonacci with thread/block modes and optional profiling visualization | thread / block |
| [nq](nq/) | N-Queens solver with task spawning and cutoff | thread |
| [mergesort](mergesort/) | Recursive parallel mergesort | thread |
| [cilksort](cilksort/) | Parallel mergesort (Cilk-style) | thread |
| [binary_tree](binary_tree/) | Synthetic tree workload (memory + compute) | thread / block |
| [spmv](spmv/) | Sparse matrix–vector multiplication (divide-and-conquer) | thread / block |
| [asynchronous_bfs](asynchronous_bfs/) | Asynchronous breadth-first search on real-world graphs | thread / block |

Examples with both thread and block variants provide two source files and two build targets (`make thread` / `make block`).


## API Reference

### Example program: Fibonacci

```c
#include "gtap_thread.cuh"

__device__ int d_result;

#pragma gtap function
__device__ int fib(int n) {
    if (n < 2) return n;
    int a, b;
    #pragma gtap task
    a = fib(n - 1);
    #pragma gtap task
    b = fib(n - 2);
    #pragma gtap taskwait
    return a + b;
}

__global__ void exec_kernel(int n) {
    #pragma gtap entry
    d_result = fib(n);
}

int main() {
    gtap_thread_config config{
        .grid_size = 4000,
        .block_size = 32,
        .max_tasks_per_warp = 150000,
    };
    gtap_initialize(config);
    gtap_launch(exec_kernel, 40);
    gtap_synchronize();
    gtap_finalize();
}
```

### Pragmas

| Pragma | Description |
|--------|-------------|
| `#pragma gtap function` | Marks a `__device__` function as a *task function*. The compiler transforms it into a state-machine so it can suspend at `taskwait` and resume later. |
| `#pragma gtap task [queue(expr)]` | Spawns a child task. Must be placed immediately before a call to a task function. The parent continues; the child is enqueued by the runtime. The optional `queue(expr)` hint enables DAQ (thread mode only). |
| `#pragma gtap taskwait [queue(expr)]` | Suspends the current task until all direct child tasks spawned since the last `taskwait` have completed. `queue(expr)` selects the queue for the re-enqueued continuation. |
| `#pragma gtap entry` | Enqueues the initial (root) task and starts execution inside the persistent kernel. Must be immediately followed by a call to a task function. |


### Runtime Functions

| Function | Description |
|----------|-------------|
| `gtap_initialize(config)` | Stores the launch configuration and allocates runtime memory. Returns `cudaError_t`. |
| `gtap_launch(kernel, args...)` | Launches a kernel using the configuration supplied at initialization. |
| `gtap_synchronize()` | Waits for device execution and reports runtime errors. |
| `gtap_finalize()` | Releases memory allocated by `gtap_initialize()`. |
| `gtap_reset()` | Resets runtime state without re-allocating memory (useful for multiple runs). |
| `gtap_export_profile([options])` | Exports profiling results with `-DGTAP_ENABLE_PROFILING` and returns whether the profile was written. Without that macro, returns false and standard error reports that profiling is disabled. |


## Runtime Configuration

Thread mode uses `gtap_thread_config`:

| Field | Description |
|-------|-------------|
| `grid_size` | Number of thread blocks. |
| `block_size` | Number of threads per block. Must be a multiple of the warp size. |
| `memory_budget_bytes` | Device-memory budget for automatic task capacity. `0` reads `GTAP_MEMORY_BUDGET`, then the 5 GiB default. |
| `max_tasks_per_warp` | Task slots per warp. `0` derives this from the memory budget. |
| `num_queues` | Number of DAQ queues. Default `1`. Set this only with `#pragma gtap task queue(expr)`. |
| `profile_capacity_per_warp` | Profiling slots per warp when built with `-DGTAP_ENABLE_PROFILING`. |
| `stream` | CUDA stream for launch and synchronization. `nullptr` uses the default stream. |

Block mode uses `gtap_block_config`:

| Field | Description |
|-------|-------------|
| `grid_size` | Number of thread blocks. |
| `block_size` | Number of threads per block. Must be a multiple of the warp size. |
| `memory_budget_bytes` | Same meaning as in thread mode. |
| `max_tasks_per_block` | Task slots per block. `0` derives this from the memory budget. |
| `profile_capacity_per_block` | Profiling slots per block when built with `-DGTAP_ENABLE_PROFILING`. |
| `dynamic_shared_bytes` | Extra dynamic shared memory reserved for the application. |
| `stream` | CUDA stream for launch and synchronization. `nullptr` uses the default stream. |

The compiler option `-fgtap-no-taskwait` enables the compact runtime mode
for programs that never execute `taskwait`.

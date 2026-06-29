# GTaP Examples

Minimal, runnable GTaP programs. Each subdirectory contains one example and a `Makefile` to build it.

## Prerequisites

- **GTaP compiler**: Build the Clang-based GTaP compiler first; see [../clang-gtap/README.md](../clang-gtap/README.md).
- **CUDA Toolkit**: Set `CUDA_PATH` (or `CUDA_HOME`) to the CUDA installation root.
- **Runtime headers**: Located at `../runtime` relative to each example.


## Building

```bash
# Optional: set environment variables (can also be passed directly to make)
export CUDA_PATH=/path/to/cuda
export CUDA_ARCH=sm_90   # sm_80, sm_70, etc.

cd examples/fib
make
./bin/fib
```

Or override variables on the command line:

```bash
cd examples/spmv
make GTAP_ROOT=/path/to/gtap CUDA_PATH=/path/to/cuda
./bin/spmv_thread
./bin/spmv_block
```

## Examples

| Example | Description | Worker |
|---------|-------------|--------|
| [fib](fib/) | Fibonacci: basic `task` / `taskwait` usage | thread |
| [fib_profile](fib_profile/) | Fibonacci with profiling enabled; includes a Python visualization script | thread |
| [nq](nq/) | N-Queens solver with task spawning and cutoff | thread |
| [mergesort](mergesort/) | Recursive parallel mergesort | thread |
| [cilksort](cilksort/) | Parallel mergesort (Cilk-style) | thread |
| [binary_tree](binary_tree/) | Synthetic tree workload (memory + compute) | thread / block |
| [spmv](spmv/) | Sparse matrix–vector multiplication (divide-and-conquer) | thread / block |
| [asynchronous_bfs](asynchronous_bfs/) | Asynchronous breadth-first search on real-world graphs | thread / block |

Examples with both thread and block variants (e.g. `binary_tree`, `spmv`, `asynchronous_bfs`) provide two source files and two build targets (`make thread` / `make block`).


## API Reference

### Example program: Fibonacci

```c
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
```

### Pragmas

| Pragma | Description |
|--------|-------------|
| `#pragma gtap function` | Marks a `__device__` function as a *task function*. The compiler transforms it into a state-machine so it can suspend at `taskwait` and resume later. |
| `#pragma gtap task [queue(expr)]` | Spawns a child task. Must be placed immediately before a call to a task function. The parent continues; the child is enqueued by the runtime. The optional `queue(expr)` hint enables EPAQ (thread workers only). |
| `#pragma gtap taskwait [queue(expr)]` | Suspends the current task until all direct child tasks spawned since the last `taskwait` have completed. `queue(expr)` selects the queue for the re-enqueued continuation. |
| `#pragma gtap entry` | Enqueues the initial (root) task and starts execution inside the persistent kernel. Must be immediately followed by a call to a task function. |


### Runtime Functions

| Function | Description |
|----------|-------------|
| `gtap_initialize()` | Allocates and initializes runtime memory (call once before the kernel launch). Returns `cudaError_t`. |
| `gtap_finalize()` | Releases memory allocated by `gtap_initialize()`. |
| `gtap_reset()` | Resets runtime state without re-allocating memory (useful for multiple runs). |
| `gtap_visualize_profile(name)` | Dumps profiling data to CSV files in `./profile/`. Available only when compiled with `-DPROFILE`. |


## Compile-Time Parameters

GTaP requires several compile-time configuration macros to control memory allocation and performance; default values are provided but may not be appropriate for all programs.
These macros must be defined before including `gtap_thread.cuh` / `gtap_block.cuh` (or passed as `-D` flags to the compiler). 

| Macro | Applies to | Description |
|-------|-----------|-------------|
| `GTAP_GRID_SIZE` | both | Number of thread blocks used to launch the kernel (grid size, 1-D). |
| `GTAP_BLOCK_SIZE` | both | Number of threads per block (block size, 1-D). |
| `GTAP_MAX_TASKS_PER_WARP` | thread | Maximum number of pending tasks that can be held per warp. Determines the size of the per-warp task pool. |
| `GTAP_MAX_TASKS_PER_BLOCK` | block | Maximum number of pending tasks that can be held per block. |
| `GTAP_MAX_CHILD_TASKS` | both | Maximum number of child tasks a single task may spawn within one task function invocation. |
| `GTAP_NUM_QUEUES` | thread | Number of EPAQ queues. Default is `1`. Increase when `queue(expr)` hints are used. |
| `GTAP_ASSUME_NO_TASKWAIT` | both | When defined, omits join metadata (child task IDs). Safe only for programs that never execute `taskwait`. Reduces per-task memory overhead. |

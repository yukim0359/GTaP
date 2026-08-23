# Quickstart

This tutorial walks through a minimal GTaP program. It introduces the two
execution modes, builds and runs thread-mode Fibonacci, and then explains
task functions, child tasks, joins, and host-side launch.

All commands are run from the root of the cloned GTaP repository and assume
that the CUDA Toolkit is available.

## 1. GTaP's two execution modes

GTaP provides two execution modes:

| Mode | One task runs on | Use when |
| --- | --- | --- |
| [Thread mode](./execution-modes#thread-mode) | One CUDA thread | The task body is fine-grained and mostly sequential |
| [Block mode](./execution-modes#block-mode) | One CUDA thread block | The task body uses shared memory, block synchronization, or cooperative parallel work |

This tutorial uses thread mode because each Fibonacci task is fine-grained and
sequential. See [Execution Modes](./execution-modes) for further details and
the block-mode programming model.

## 2. Build and run Fibonacci

Set `CUDA_PATH` to the CUDA Toolkit root and choose the GPU architecture.
Then build and run the example:

```bash
export CUDA_PATH=/path/to/cuda
export CUDA_ARCH=sm_90

"${CUDA_PATH}/bin/nvcc" --version

cd examples/fib
make
./bin/fib_thread
```

You can override paths and architecture settings on the command line:

```bash
make GTAP_ROOT=/path/to/GTaP \
     CUDA_PATH=/path/to/cuda \
     CUDA_ARCH=sm_90
```

## 3. Understand the programming model

The example includes the thread-mode runtime:

```cpp
#include "gtap_thread.cuh"
```

The work performed by a GTaP task must be defined in a separate `__device__`
function, called a **task function**. Place `#pragma gtap function` immediately
before its definition:

```cpp
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
```

This function uses three GTaP concepts:

1. `#pragma gtap function` declares a task function.
2. `#pragma gtap task` spawns the immediately following task-function call as
   a child task.
3. `#pragma gtap taskwait` waits for the direct child tasks and then resumes the
   parent.

A `task` spawn does not block the parent. A `taskwait` waits only for direct
children spawned since the previous `taskwait`; it is not a device-wide
barrier.

Place `#pragma gtap entry` before the root task-function call in a `__global__`
kernel. This kernel is the entry point for the GTaP computation:

```cpp
__global__ void exec_kernel(int n) {
    #pragma gtap entry
    d_result = fib(n);
}
```

These four pragmas are sufficient for the first program. See
[Rules and Pitfalls](./rules-and-pitfalls) for execution constraints and the
[Pragma Reference](../reference/pragmas) for the exact syntax and restrictions
of each pragma.

## 4. Launch the computation

The host initializes GTaP, then uses `gtap_launch` to launch the CUDA kernel
containing the `entry` directive. This starts the root task.
`gtap_synchronize` waits for the computation to finish, and `gtap_finalize`
releases the runtime:

```cpp
gtap_thread_config config{
    .grid_size = 4000,
    .block_size = 32,
    .max_tasks_per_warp = 100000,
    .num_queues = 1,
};

cudaError_t status = gtap_initialize(config);
if (status != cudaSuccess) return 1;

status = gtap_launch(exec_kernel, 40);
if (status == cudaSuccess)
    status = gtap_synchronize();

gtap_finalize();
```

For every function argument, return value, error, and repeated-run behavior,
see the [Runtime API Reference](../reference/runtime-functions). See the
[Configuration Reference](../reference/configuration) for all configuration
fields and validation rules.

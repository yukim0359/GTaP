# GTaP

GTaP is a pragma-based task-parallel programming system for GPUs, implemented in CUDA C++.

GTaP consists of:

- A header-only runtime library
- A Clang extension that lowers GTaP pragmas into CUDA device code

GTaP enables structured fork-join parallelism directly on GPUs using a pragma-based programming model.

> 🔬 GTaP is a research prototype under active development.
> Its interfaces and internal mechanisms may change.

## Features

- **Fork-join task parallelism on GPUs**:  
  Programmers express fork-join using `#pragma gtap task` and `#pragma gtap taskwait`.
  GTaP realizes fork-join parallelism by representing each task function as a switch-statement-based state machine.  
  The Clang extension automatically generates these state machines and manages task data across join points.  

- **Two modes**:  
  GTaP supports two execution modes: **thread mode** and **block mode**.
  In thread mode, a task function runs on a single CUDA thread and is written like ordinary sequential code.
  In block mode, each task is executed cooperatively by all threads in one thread block, allowing CUDA-style data-parallel code using `threadIdx`, shared memory, and block-wide synchronization.

  In block mode, each thread that reaches `#pragma gtap task` independently spawns a child task. A spawn intended to create exactly one child should therefore be guarded by a designated thread.
  `#pragma gtap taskwait` is collective and must be reached by all threads in the block.

  The runtime provides `gtap_thread.cuh` and `gtap_block.cuh` for these modes.

- **Divergence-aware queueing (DAQ)**:  
  In thread mode, programmers can optionally specify a queue index using `#pragma gtap task queue(expr)` at spawn or `#pragma gtap taskwait queue(expr)` for the post-join continuation.
  Tasks expected to follow similar execution paths can be placed in the same queue, reducing inter-task warp divergence.

- **Task schedulers**:  
  GTaP uses GPU-resident randomized work-stealing.
  In thread mode, a warp acquires up to 32 runnable tasks via a warp-cooperative batched pop/steal.


## Repository layout

| Directory | Description |
|-----------|-------------|
| **`clang-gtap/`** | Clang fork that compiles GTaP programs. See [`clang-gtap/README.md`](https://github.com/yukim0359/clang-gtap/blob/main/README.md) for build and usage. |
| **`runtime/`** | Header-only GTaP runtime library. |
| **`evaluation/`** | Benchmarks and scripts used for performance evaluation. |
| **`examples/`** | Example GTaP programs (fib, n-queens, mergesort, cilksort, tree workloads, etc.). |


## Getting Started

GTaP has been tested on one NVIDIA GH200 node of the [Miyabi-G supercomputer](https://www.cc.u-tokyo.ac.jp/en/supercomputer/miyabi/service/).

Tested environment:

- NVIDIA GH200
- Compute capability 9.0 (`sm_90`)
- Clang 21.1.8
- CUDA Toolkit 12.9
- Linux kernel `5.14.0-427.13.1.el9_4.aarch64`

### 1. Clone the repository:

```bash
git clone https://github.com/yukim0359/GTaP.git --recursive
cd GTaP
```

### 2. Build the compiler:

Follow [`clang-gtap/README.md`](https://github.com/yukim0359/clang-gtap/blob/main/README.md) to build the GTaP-enabled Clang.

### 3. Compile programs:

Example: Fibonacci

```bash
cd examples/fib
make
./bin/fib
```

Compilation flags, runtime configuration parameters, and required preprocessor definitions are described in [`examples/README.md`](examples/README.md).

## Reproducing Evaluation Results

Detailed instructions for reproducing experimental results are provided in [`evaluation/README.md`](evaluation/README.md).


## Profiling

GTaP includes a profiler for inspecting GPU-side task scheduling.

Define `GTAP_PROFILE` before including the runtime header:

```cpp
#define GTAP_PROFILE
#include "gtap_thread.cuh"  // or "gtap_block.cuh"
```

After executing and synchronizing the GTaP kernel, call:

```cpp
gtap_export_profile("app_name");
```

The profiler writes CSV files under `./profile/`, including task timelines and summary statistics for warps or blocks.

`GTAP_PROFILE_CAPACITY_PER_WARP` (thread mode) or `GTAP_PROFILE_CAPACITY_PER_BLOCK` (block mode) can be adjusted when more samples are required.

See [`examples/fib_profile`](examples/fib_profile) for an example.


## License

- **clang-gtap**: Based on the LLVM Project and distributed under the Apache License v2.0 with LLVM Exceptions. See [`clang-gtap/LICENSE.TXT`](https://github.com/yukim0359/clang-gtap/blob/main/LICENSE.TXT).
- **Other components**: See [`LICENSE`](LICENSE) at the repository root.

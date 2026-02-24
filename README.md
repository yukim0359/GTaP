# GTaP

GTaP is a directive-based fork-join task-parallel runtime system for GPUs, implemented in CUDA C++.
It consists of:
- A header-only runtime library
- A Clang-based compiler extension that translates GTaP directives into CUDA device code

GTaP enables structured fork-join parallelism directly on GPUs using a pragma-based programming model.

> 🔬 GTaP is a research prototype under active development.
> Interfaces and internal mechanisms may evolve over time.

## Features

- **Fork-join task parallelism on GPUs**:  
  Programmers express fork-join using `#pragma gtap task` and `#pragma gtap taskwait`.
  GTaP realizes fork-join parallelism by representing each task function as a switch-statement-based state machine.  
  The Clang extension automatically generates these state machines and manages task data across join points.  

- **Two granularities**:  
  GTaP supports two execution modes for task execution: thread-executed (thread-level workers) and block-cooperative (block-level workers).
  In the thread-executed mode, a task function runs on a single CUDA thread and is written like ordinary sequential code.
  In the block-cooperative mode, a task function runs cooperatively on all threads in one thread block; programmers write it in a GPU-style data-parallel manner using `threadIdx` / `blockIdx`.
  The runtime provides `gtap_thread.cuh` and `gtap_block.cuh` for these modes.

- **Execution-path-aware queueing (EPAQ)**:  
  Programmers can optionally specify a queue index as `#pragma gtap task queue(expr)` (at spawn) or `#pragma gtap taskwait queue(expr)` (at re-entry after a join).
  This allows tasks that are expected to follow different execution paths to be separated before they run.

- **Task schedulers**:  
  GTaP uses randomized work-stealing.
  In thread-executed mode, a warp acquires up to 32 runnable tasks via a warp-cooperative batched pop/steal.


## Repository layout

| Directory | Description |
|-----------|-------------|
| **clang-gtap/** | Clang fork that compiles GTaP programs. See [clang-gtap/README.md](clang-gtap/README.md) for build and usage. |
| **runtime/** | Header-only GTaP runtime library. |
| **evaluation/** | Benchmarks and scripts used for performance evaluation. |
| **examples/** | Example GTaP programs (fib, n-queens, mergesort, cilksort, tree workloads, etc.). |


## Getting Started

We have verified build and basic functionality on a single GH200 node of the [Miyabi-G](https://www.cc.u-tokyo.ac.jp/en/supercomputer/miyabi/service/) supercomputer (1× NVIDIA GH200, compute capability 9.0 / sm_90; Clang 21.1.8, CUDA Toolkit 12.9, Linux kernel 5.14.0-427.13.1.el9_4.aarch64).

1. Clone the repository:

```bash
git clone https://github.com/yukim0359/GTaP.git --recursive
cd GTaP
```

2. Build the compiler:

Follow [clang-gtap/README.md](clang-gtap/README.md) to build the GTaP-enabled Clang.

3. Compile programs:

Example: Fibonacci

```bash
cd examples/fib
make
./bin/fib
```

Compilation flags and required preprocessor macros are described in [examples/README.md](examples/README.md).


## Reproducing Evaluation Results

Detailed instructions for reproducing experimental results are provided in [evaluation/README.md](evaluation/README.md).


## Profiling

GTaP has a built-in profiler for inspecting how tasks are scheduled on the GPU.

- **Data**: `gtap_visualize_profile("app_name")` writes CSV files under `./profile/` with warp/block timelines and summary statistics.  
- **Usage**:
  ```cpp
  #define PROFILE  // enable GTaP profiling
  #include "gtap_thread.cuh"  // or "gtap_block.cuh"

  int main(){
    // ... launch GTaP kernel and synchronize ...

    gtap_visualize_profile("fib");
  }
  ```
- **Programmer responsibilities**: define `PROFILE` in the translation unit you profile, and optionally tune `MAX_PROFILE_DATA` if you need more samples.

For further details, see `examples/fib_profile`.


## License

- **clang-gtap**: Based on the LLVM Project; see [clang-gtap/LICENSE.TXT](clang-gtap/LICENSE.TXT) (Apache License v2.0 with LLVM Exceptions).
- **Other components**: See [LICENSE](LICENSE) at the repository root.

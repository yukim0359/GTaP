---
title: GTaP
description: A pragma-based task-parallel programming system for GPUs.
sidebar: false
footer: false
outline: 2
---

# GTaP Documentation

GTaP is a pragma-based task-parallel programming system for GPUs, implemented in CUDA C++.
In GTaP, tasks execute within a persistent CUDA kernel; each task is not a separate kernel launch.

GTaP combines a header-only CUDA runtime with a Clang extension that lowers GTaP pragmas into CUDA device code.
The source code is available on [GitHub](https://github.com/yukim0359/GTaP).

> GTaP is a research prototype under active development. Its interfaces and
> internal mechanisms may change.

## Documentation

### Tutorial

Start with [Installation](./tutorial/installation), then run the first program
in [Quickstart](./tutorial/quickstart). Continue with
[Execution Modes](./tutorial/execution-modes),
[Rules and Pitfalls](./tutorial/rules-and-pitfalls), and
[Profiling](./tutorial/profiling) as needed.

### API Reference

Look up exact syntax and interfaces in [Pragmas](./reference/pragmas),
[Runtime Functions](./reference/runtime-functions),
[Configuration](./reference/configuration), and the
[Profiling API](./reference/profiling).

## Programming at a Glance

A CUDA kernel uses `entry` to create the root task. A task function uses `task`
to spawn direct children and `taskwait` to join them:

```cpp
__device__ int d_result;

#pragma gtap function
__device__ int fib(int n) {
    if (n < 2) return n;
    int left, right;
    #pragma gtap task
    left = fib(n - 1);
    #pragma gtap task
    right = fib(n - 2);
    #pragma gtap taskwait
    return left + right;
}

__global__ void exec_kernel(int n) {
    #pragma gtap entry
    d_result = fib(n);
}
```

## Features

- **Fork-join task parallelism on GPUs.** Express task creation and join points
  with `#pragma gtap task` and `#pragma gtap taskwait`.
- **Thread and block modes.** Execute one task on one CUDA thread, or
  cooperatively with an entire CUDA thread block.
- **GPU-resident scheduling.** A GPU-resident scheduler distributes runnable
  tasks through randomized work stealing.
- **Divergence-aware queueing.** Group tasks with similar expected execution
  paths to reduce inter-task warp divergence in thread mode.

## Publication

Yuki Maeda and Kenjiro Taura.<br>
[**GTaP: A GPU-Resident Fork-Join Task-Parallel Runtime with a Pragma-Based
Interface.**](https://arxiv.org/abs/2604.05982)<br>
arXiv:2604.05982, 2026.

# Rules and Pitfalls

This chapter collects the rules and pitfalls that matter when writing GTaP
programs.

## Task storage is fixed at initialization

GTaP preallocates task records and task queues during `gtap_initialize`. Their
capacities do not grow during execution. In thread mode, `num_queues` fixes how
many [divergence-aware queues (DAQ)](./execution-modes#divergence-aware-queueing)
are allocated at initialization. If a workload exhausts the task-ID pool or a
queue, execution reports a runtime error.

Size `max_tasks_per_warp` or `max_tasks_per_block` for the maximum number of
simultaneously live tasks. See the
[Configuration Reference](../reference/configuration) for the thread-mode
per-queue capacity formula and validation rules.

## Task functions must be declared explicitly

Only a `__device__` function annotated with `#pragma gtap function` is a GTaP
task function.

```cpp
#pragma gtap function
__device__ int visit(Node* node) {
    // ...
}
```

`#pragma gtap task` and `#pragma gtap taskwait` may appear only inside such a
task function. An ordinary `__device__` function may still be called as a
helper, but it cannot spawn or wait for GTaP tasks.

## Every task-function call needs a GTaP pragma

A task function is not called like an ordinary device function. Every call
must be one of the following:

- a child-task spawn immediately preceded by `#pragma gtap task`
- the root-task call immediately preceded by `#pragma gtap entry`

```cpp
// Child task: valid inside a task function.
#pragma gtap task
result = visit(child);

// Root task: valid inside the persistent kernel.
#pragma gtap entry
d_result = visit(root);
```

A direct call is not supported, even when it appears inside another task
function:

```cpp
// Invalid: visit is a GTaP task function.
result = visit(child);
```

The call following `task` or `entry` must be a direct call, optionally on the
right-hand side of an assignment. A task-function call cannot be hidden inside
another expression or nested as an argument:

```cpp
#pragma gtap task
result = visit(visit(child));  // Invalid nested task-function call.
```

## `taskwait` may resume on a different thread or block

`#pragma gtap taskwait` is a suspension point. The runtime records the live
task state, returns control to the scheduler, and re-enqueues the continuation
after the direct children finish. Work stealing may therefore resume the task
on:

- a different CUDA thread in thread mode
- a different CUDA thread block in block mode

Do not carry state tied to the particular CUDA thread or CUDA thread block
across `taskwait`. In particular:

- do not assume that `blockIdx` identifies the same CUDA thread block before and after
  the wait
- do not retain a pointer or reference to shared memory across the wait
- do not expect shared-memory contents to survive the wait

```cpp
extern __shared__ int scratch[];

scratch[threadIdx.x] = make_value();
int* saved = &scratch[threadIdx.x];

#pragma gtap taskwait

consume(*saved);  // Invalid assumption: this may be a different CUDA thread block.
```

Store persistent data in global memory, or reconstruct block-local shared
state after the join. Ordinary local values that remain live across the wait
are saved in the task record and restored by the compiler; in block mode,
their per-thread values are preserved logically even if another block resumes
the task.

## `taskwait` joins the current task's children

A `taskwait` waits only for direct children spawned since the previous
`taskwait` in the same task function. It is not a device-wide barrier and does
not wait for unrelated tasks.

## `taskwait` is collective in block mode

In block mode, `taskwait` is collective, like `__syncthreads()`. Every thread
in the CUDA thread block must reach the same wait. Placing it in a branch taken
by only some threads is invalid.

```cpp
// Invalid in block mode when the condition differs between threads.
if (threadIdx.x == 0) {
    #pragma gtap taskwait
}
```

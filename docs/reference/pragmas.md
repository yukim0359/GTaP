# Pragmas

GTaP's compiler interface consists of four pragmas.

## `#pragma gtap function`

Marks a CUDA device function as a GTaP task function.

```cpp
#pragma gtap function
__device__ int solve(Node* node) {
    // ...
}
```

The Clang extension transforms the function into a resumable state machine so
that execution can suspend at a `taskwait` and continue after its children
finish.

### Requirements

- Apply it to a `__device__` function.
- Use `task` and `taskwait` only inside a task function.
- Calls annotated with `task` or `entry` must target a task function.
- Do not call a task function directly: use `task` for a child or `entry` for
  the root task.
- Any argument or local variable whose value is needed after a `taskwait` must
  be trivially copyable. This is because a task may suspend at the wait,
  requiring the compiler to save these values in the task record and restore
  them when the task resumes.

## `#pragma gtap task`

Spawns the immediately following task-function call as a child of the current
task.

```cpp
#pragma gtap task
left_result = visit(node->left);
```

The parent continues after enqueueing the child. A later `taskwait` receives
the child result and joins the child lifetime.

The directive must be followed by a direct task-function call, optionally as
the right-hand side of an assignment. Do not use the assigned result before
the corresponding `taskwait` completes.

### Thread mode queue clause

```cpp
#pragma gtap task queue(queue_expression)
visit(child);
```

`queue_expression` is evaluated for the spawn and selects a divergence-aware
queue. The resulting index must satisfy:

```text
0 <= queue_expression < config.num_queues
```

Queue selection is a scheduling hint and must not affect program correctness.

### Block mode rule

Every thread that reaches a `task` pragma independently spawns a child. Guard
the spawn when exactly one child is intended:

```cpp
if (threadIdx.x == 0) {
    #pragma gtap task
    result = child(input);
}
```

Only threads that execute the spawn receive that child's return value.

## `#pragma gtap taskwait`

Suspends the current task until all direct children spawned since the previous
`taskwait` have completed.

```cpp
#pragma gtap taskwait
consume(left_result, right_result);
```

The continuation is re-enqueued and may resume on a different CUDA thread in
thread mode or a different CUDA thread block in block mode. CUDA state tied to
the previous thread or block, including shared memory and pointers into it,
must not cross the wait.

Due to a current implementation limitation, a `taskwait` inside a `switch`
statement is not supported.

### Thread mode queue clause

```cpp
#pragma gtap taskwait queue(queue_expression)
```

The clause selects the queue used when the suspended continuation becomes
runnable. The index must be within `config.num_queues`.

### Block mode rule

`taskwait` is collective in block mode. Every thread in the block must reach
the same wait. Do not place it in a branch taken by only part of the block.

## `#pragma gtap entry`

Creates the initial root task from the CUDA kernel passed to `gtap_launch`.

```cpp
__global__ void exec_kernel(int input) {
    #pragma gtap entry
    d_result = solve(input);
}
```

### Requirements

- Place it inside the CUDA global function passed to `gtap_launch`.
- The immediately following expression must call a GTaP task function.
- Every thread in the launched grid must reach the directive. A normal launch
  creates one logical root task for the computation.

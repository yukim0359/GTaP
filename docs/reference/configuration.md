# Configuration

GTaP uses separate configuration structures for thread and block execution
modes. The structures provide defaults for all runtime fields, while a few
settings differ because the unit of execution is a CUDA thread or thread
block.

## Settings by execution mode

| Purpose | Thread mode | Block mode |
| --- | --- | --- |
| Grid size | `grid_size = 4096` | `grid_size = 1024` |
| CUDA block size | `block_size = 32` | `block_size = 256` |
| Task capacity | `max_tasks_per_warp = 0` | `max_tasks_per_block = 0` |
| Task-memory budget | `max_task_memory_bytes = 0` | `max_task_memory_bytes = 0` |
| Profile capacity | `profile_capacity_per_warp = 15000` | `profile_capacity_per_block = 15000` |
| CUDA stream | `stream = nullptr` | `stream = nullptr` |
| DAQ queues | `num_queues = 1` | — |
| Dynamic shared memory per block | — | `dynamic_shared_bytes = 0` |

`grid_size` is the number of CUDA thread blocks launched by
[`gtap_launch`](./runtime-functions#gtap-launch).
`nullptr` selects the default CUDA stream. Profile capacity is used only when
profiling is enabled.

`0` on either task field means that field is unset. The live slot count is
chosen as follows:

- only the task count is set: use that count
- only `max_task_memory_bytes` is set: use the largest count whose task region fits
- both are set: use the smaller of the two
- neither is set: use 10000 slots per warp or block

Task capacity limits simultaneously live tasks assigned to each CUDA warp in
thread mode or CUDA thread block in block mode, not the total number of tasks
executed. The byte budget covers task slots, headers, id rings, and queues.
Profile buffers and the block entry-result buffer are outside this budget.

## Configuration structures

### Thread mode

`gtap_thread_config` is defined by `gtap_thread.cuh`:

```cpp
struct gtap_thread_config {
    int grid_size = 4096;
    int block_size = 32;
    int max_tasks_per_warp = 0;
    size_t max_task_memory_bytes = 0;
    int num_queues = 1;
    int profile_capacity_per_warp = 15000;
    cudaStream_t stream = nullptr;
};
```

Task capacity is allocated to each CUDA warp and divided equally among the
configured queues:

```text
capacity per queue = max_tasks_per_warp / num_queues
```

### Block mode

`gtap_block_config` is defined by `gtap_block.cuh`:

```cpp
struct gtap_block_config {
    int grid_size = 1024;
    int block_size = 256;
    int max_tasks_per_block = 0;
    size_t max_task_memory_bytes = 0;
    int profile_capacity_per_block = 15000;
    size_t dynamic_shared_bytes = 0;
    cudaStream_t stream = nullptr;
};
```

Task capacity is allocated to each CUDA thread block. `dynamic_shared_bytes`
specifies the dynamic shared memory supplied to each block.

## `gtap_validate_config`

Validates a configuration without allocating runtime resources or changing
the GTaP initialization state.

```cpp
cudaError_t gtap_validate_config(const gtap_thread_config& config);
cudaError_t gtap_validate_config(const gtap_block_config& config);
```

Both modes require:

- `grid_size > 0`
- a CUDA block size in `(0, GTAP_MAX_THREADS_PER_BLOCK]`
- `block_size` to be a multiple of 32
- a task count that is unset (`0`) or positive
- when profiling is enabled, a profile capacity in `(0, INT_MAX / 2]`

Thread mode additionally requires:

- `num_queues > 0`
- a positive `max_tasks_per_warp` to be divisible by `num_queues`
- the resolved slot count to be divisible by `num_queues`, including the
  10000 used when both task fields are unset

The function returns `cudaSuccess` for a valid configuration,
`cudaErrorInvalidConfiguration` for invalid launch geometry, and
`cudaErrorInvalidValue` for invalid task, queue, or profile capacities.

## Compile-time settings

| Setting | Kind | Purpose |
| --- | --- | --- |
| `GTAP_ENABLE_PROFILING` | Preprocessor macro | Enables collection of profiling data |
| `-fgtap-no-taskwait` | GTaP Clang option | Selects the compact runtime for programs that do not use `taskwait` |

For profiling, compile with `-DGTAP_ENABLE_PROFILING`; see the
[Profiling API Reference](./profiling).

`-fgtap-no-taskwait` removes join-state support. Do not use it if the program
contains `#pragma gtap taskwait`.

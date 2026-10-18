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
| Memory budget | `memory_budget_bytes = 0` | `memory_budget_bytes = 0` |
| Task capacity | `max_tasks_per_warp = 0` | `max_tasks_per_block = 0` |
| Profile capacity | `profile_capacity_per_warp = 15000` | `profile_capacity_per_block = 15000` |
| CUDA stream | `stream = nullptr` | `stream = nullptr` |
| DAQ queues | `num_queues = 1` | — |
| Dynamic shared memory per block | — | `dynamic_shared_bytes = 0` |

The runtime memory budget defaults to 5 GiB. Set `GTAP_MEMORY_BUDGET` to change
it, or set a nonzero `memory_budget_bytes` to override the environment
variable.

Task capacity limits the number of simultaneously live tasks assigned to each
CUDA warp in thread mode or CUDA thread block in block mode, not the total
number of tasks executed. Larger capacities consume more device memory.
At `0`, GTaP selects the largest valid capacity within the budget. A positive
value specifies the capacity directly and overrides the budget. Profiling
buffers are configured separately and excluded from the budget.

`grid_size` is the number of CUDA thread blocks launched by
[`gtap_launch`](./runtime-functions#gtap-launch).
`nullptr` selects the default CUDA stream. Profile capacity is used only when
profiling is enabled.

## Configuration structures

### Thread mode

`gtap_thread_config` is defined by `gtap_thread.cuh`:

```cpp
struct gtap_thread_config {
    int grid_size = 4096;
    int block_size = 32;
    size_t memory_budget_bytes = 0;
    int max_tasks_per_warp = 0;
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
    size_t memory_budget_bytes = 0;
    int max_tasks_per_block = 0;
    int profile_capacity_per_block = 15000;
    size_t dynamic_shared_bytes = 0;
    cudaStream_t stream = nullptr;
};
```

Task capacity is allocated to each CUDA thread block. `dynamic_shared_bytes`
specifies the dynamic shared memory supplied to each block.

## Accepted configurations

[`gtap_initialize`](./runtime-functions#gtap-initialize) rejects a configuration
before it allocates runtime resources.

Both modes require:

- `grid_size > 0`
- a CUDA block size in `(0, GTAP_MAX_THREADS_PER_BLOCK]`
- `block_size` to be a multiple of 32
- a non-negative task capacity (`0` selects the memory budget)
- when profiling is enabled, a profile capacity in `(0, INT_MAX / 2]`

Thread mode additionally requires:

- `num_queues > 0`
- `max_tasks_per_warp % num_queues == 0`

Invalid launch geometry returns `cudaErrorInvalidConfiguration`. An invalid
task, queue, or profile capacity returns `cudaErrorInvalidValue`.

## Compile-time settings

| Setting | Kind | Purpose |
| --- | --- | --- |
| `GTAP_ENABLE_PROFILING` | Preprocessor macro | Enables collection of profiling data |
| `-fgtap-no-taskwait` | GTaP Clang option | Selects the compact runtime for programs that do not use `taskwait` |

For profiling, compile with `-DGTAP_ENABLE_PROFILING`; see the
[Profiling API Reference](./profiling).

`-fgtap-no-taskwait` removes join-state support. Do not use it if the program
contains `#pragma gtap taskwait`.

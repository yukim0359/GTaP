# Configuration

GTaP uses separate configuration structures for thread and block execution
modes. The structures provide defaults for all runtime fields, while a few
settings differ because the unit of execution is a CUDA thread or thread
block.

## Settings by execution mode

| Purpose | Thread mode | Block mode |
| --- | --- | --- |
| Grid size | `grid_size = 1024` | `grid_size = 1024` |
| CUDA block size | `block_size = 256` | `GTAP_BLOCK_SIZE` (compile time) |
| Task capacity | `max_tasks_per_warp = 150000` | `max_tasks_per_block = 10000` |
| Profile capacity | `profile_capacity_per_warp = 15000` | `profile_capacity_per_block = 15000` |
| CUDA stream | `stream = nullptr` | `stream = nullptr` |
| DAQ queues | `num_queues = 1` | — |
| Dynamic shared memory per block | — | `dynamic_shared_bytes = 0` |

`grid_size` is the number of CUDA thread blocks launched by
[`gtap_launch`](./runtime-functions#gtap-launch).
`nullptr` selects the default CUDA stream. Profile capacity is used only when
profiling is enabled.

Task capacity limits the number of simultaneously live tasks assigned to each
CUDA warp in thread mode or CUDA thread block in block mode, not the total
number of tasks executed. Larger capacities consume more device memory.

## Configuration structures

### Thread mode

`gtap_thread_config` is defined by `gtap_thread.cuh`:

```cpp
struct gtap_thread_config {
    int grid_size = 1024;
    int block_size = 256;
    int max_tasks_per_warp = 150000;
    int num_queues = 1;
    int profile_capacity_per_warp = 15000;
    cudaStream_t stream = nullptr;
};
```

`block_size` must be a multiple of 32. Task capacity is allocated to each CUDA
warp and divided equally among the configured queues:

```text
capacity per queue = max_tasks_per_warp / num_queues
```

### Block mode

`gtap_block_config` is defined by `gtap_block.cuh`:

```cpp
struct gtap_block_config {
    int grid_size = 1024;
    int max_tasks_per_block = 100000;
    int profile_capacity_per_block = 15000;
    size_t dynamic_shared_bytes = 0;
    cudaStream_t stream = nullptr;
};
```

Task capacity is allocated to each CUDA thread block. `dynamic_shared_bytes`
specifies the dynamic shared memory supplied to each block. Block size is set
with `GTAP_BLOCK_SIZE`; see [Compile-time settings](#compile-time-settings).

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
- a positive task capacity
- when profiling is enabled, a profile capacity in `(0, INT_MAX / 2]`

Thread mode additionally requires:

- `block_size` to be a multiple of `GTAP_WARP_SIZE` (32 on CUDA)
- `num_queues > 0`
- `max_tasks_per_warp % num_queues == 0`

The function returns `cudaSuccess` for a valid configuration,
`cudaErrorInvalidConfiguration` for invalid launch geometry, and
`cudaErrorInvalidValue` for invalid task, queue, or profile capacities.

## Compile-time settings

| Setting | Kind | Purpose |
| --- | --- | --- |
| `GTAP_BLOCK_SIZE` | Preprocessor macro | Required in block mode; sets the number of threads in each CUDA thread block |
| `GTAP_ENABLE_PROFILING` | Preprocessor macro | Enables collection of profiling data |
| `-fgtap-no-taskwait` | GTaP Clang option | Selects the compact runtime for programs that do not use `taskwait` |

Define `GTAP_BLOCK_SIZE` before including `gtap_block.cuh`, either in the
source or on the compiler command line:

```text
-DGTAP_BLOCK_SIZE=256
```

The selected value is exposed as `gtap_compiled_block_size`. For profiling,
compile with `-DGTAP_ENABLE_PROFILING`; see the
[Profiling API Reference](./profiling).

`-fgtap-no-taskwait` removes join-state support. Do not use it if the program
contains `#pragma gtap taskwait`.

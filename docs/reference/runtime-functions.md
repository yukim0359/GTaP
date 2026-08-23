# Runtime Functions

The thread and block runtimes expose the same host functions. Their
configuration types and launch geometry differ, but the required call order is
the same.

## `gtap_initialize`

Initializes the selected runtime and allocates its device-side task storage,
queues, scheduler metadata, error report, and optional profiling buffers.

### Common forms

```cpp
gtap_initialize();                 // Default configuration
gtap_initialize(config);           // Custom configuration
gtap_initialize(config, &bytes);   // Also report allocated device memory
```

In thread mode, `config` is a
[`gtap_thread_config`](./configuration#thread-mode). In block mode, it is a
[`gtap_block_config`](./configuration#block-mode). The optional device-memory
output can also be used with the default configuration by calling
`gtap_initialize(&bytes)`.

### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `config` | `const gtap_thread_config&` or `const gtap_block_config&` | Execution-mode configuration. See the [Configuration Reference](./configuration). |
| `device_bytes_allocated` | `size_t*` | Optional output. On successful initialization, receives the number of bytes allocated by the GTaP runtime on the device. Pass `nullptr` when this value is not needed. |

The reported byte count covers GTaP's runtime allocations. It does not include
application allocations or the CUDA context.

### Return value

| Result | Meaning |
| --- | --- |
| `cudaSuccess` | Runtime initialization completed successfully |
| `cudaErrorInvalidConfiguration` | Launch geometry is invalid |
| `cudaErrorInvalidValue` | A task capacity, queue count, or profiling capacity is invalid |
| `cudaErrorInitializationError` | GTaP is already initialized in this process |
| Other CUDA error | A CUDA allocation, copy, symbol, or initialization operation failed |

### Preconditions

- Include exactly one execution-mode header for the runtime being initialized.
- For block mode, define a valid `GTAP_BLOCK_SIZE` before including the header.
- Do not call [`gtap_initialize`](#gtap-initialize) again until
  [`gtap_finalize`](#gtap-finalize) has completed successfully. Use
  [`gtap_reset`](#gtap-reset) to reuse an existing allocation.

### Example

```cpp
gtap_thread_config config{
    .grid_size = 4000,
    .block_size = 32,
    .max_tasks_per_warp = 150000,
    .num_queues = 1,
};

size_t runtime_bytes = 0;
cudaError_t status = gtap_initialize(config, &runtime_bytes);
```

## `gtap_launch`

Launches a CUDA global function with the grid, block, dynamic shared-memory,
and stream settings stored by [`gtap_initialize`](#gtap-initialize).

### Signature

```cpp
template<class Kernel, class... Args>
cudaError_t gtap_launch(Kernel kernel, Args&&... args);
```

### Parameters

| Parameter | Description |
| --- | --- |
| `kernel` | CUDA `__global__` function containing a `#pragma gtap entry` call |
| `args...` | Kernel arguments, in exactly the order and with types compatible with the kernel signature |

### Return value

- `cudaErrorInitializationError` if GTaP has not been initialized.
- Otherwise, the result of `cudaLaunchKernel`.

The launch is asynchronous. A successful return means the kernel was submitted,
not that execution finished successfully. Call [`gtap_synchronize`](#gtap-synchronize)
before reading results or exporting a profile.

### Example

```cpp
__global__ void exec_kernel(int n) {
    #pragma gtap entry
    d_result = fib(n);
}

cudaError_t status = gtap_launch(exec_kernel, 40);
```

## `gtap_synchronize`

Waits for all preceding device work to complete and reports CUDA or GTaP
runtime failures.

### Signature

```cpp
cudaError_t gtap_synchronize();
```

### Return value

Returns the result of device synchronization. If CUDA reports a failure, GTaP
prints the captured task-runtime details when available and returns that CUDA
error.

### Notes

- Synchronization is device-wide because the implementation uses
  `cudaDeviceSynchronize`, even when `config.stream` is non-null.
- Call it after every [`gtap_launch`](#gtap-launch) before reading application
  results.
- Call it before
  [`gtap_export_profile`](./profiling#gtap-export-profile),
  [`gtap_reset`](#gtap-reset), or [`gtap_finalize`](#gtap-finalize).

```cpp
cudaError_t status = gtap_synchronize();
if (status != cudaSuccess) {
    fprintf(stderr, "GTaP execution failed: %s\n",
            cudaGetErrorString(status));
}
```

## `gtap_reset`

Clears task queues, task IDs, task storage, completion state, error state, and
profiling buffers without releasing and reallocating the runtime memory.

### Signature

```cpp
cudaError_t gtap_reset();
```

### Return value

Returns `cudaSuccess` when all runtime state was reset. Otherwise, returns the
CUDA error produced by a memory operation, helper-kernel launch, synchronization,
or stream operation used during reset.

### Preconditions

- GTaP must already be initialized.
- No GTaP kernel may still be executing. Call
  [`gtap_synchronize()`](#gtap-synchronize) first.
- Application-owned device data is not reset; only GTaP runtime state is
  cleared.

### Repeated execution

```cpp
gtap_initialize(config);

for (int run = 0; run < 10; ++run) {
    if (run != 0) gtap_reset();
    gtap_launch(exec_kernel, run);
    gtap_synchronize();
}

gtap_finalize();
```

## `gtap_finalize`

Releases device and host memory owned by the GTaP runtime. On success, it also
clears the initialized state and stored CUDA stream.

### Signature

```cpp
cudaError_t gtap_finalize();
```

### Return value

Returns `cudaSuccess` after all runtime resources have been released. Otherwise,
returns the CUDA error produced during cleanup. The runtime only clears its
initialized flag when finalization succeeds.

### Preconditions

- The runtime must have been initialized.
- Complete outstanding execution with
  [`gtap_synchronize()`](#gtap-synchronize) before finalizing.
- Export required profiling data before finalization because the GTaP profiler
  buffers are released here.

```cpp
gtap_launch(exec_kernel, input);
gtap_synchronize();
gtap_finalize();
```

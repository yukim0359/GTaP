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
| `cudaErrorInvalidValue` | A task capacity, queue count, memory budget, or profiling capacity is invalid |
| `cudaErrorInitializationError` | GTaP is already initialized, or a previous finalization or reset failed and the runtime cannot be initialized again |
| Other CUDA error | A CUDA allocation, copy, symbol, or initialization operation failed |

On failure, `gtap_initialize` writes an error message to standard error. The
message names the rejected field, budget, or allocation and appends the CUDA
error string.

### Preconditions

- Include exactly one execution-mode header for the runtime being initialized.
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

If GTaP is uninitialized, a kernel has already been launched since initialization
or reset, or a previous reset or finalization failed, returns
`cudaErrorInitializationError`. These failures are reported to standard error
without launching a kernel or changing the runtime state. Otherwise, returns
the result of `cudaLaunchKernel`.

A successful return means the kernel was submitted, not that execution finished.
Call [`gtap_synchronize`](#gtap-synchronize) before reading results, exporting a
profile, resetting, or finalizing. Call [`gtap_reset`](#gtap-reset) before
launching again.

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

Returns the result of `cudaDeviceSynchronize` and reports failures to standard
error, including captured GTaP task-runtime details when available.

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

Resets GTaP's runtime state and profiling buffers without reallocating memory.
Application-owned data is unchanged.

### Signature

```cpp
cudaError_t gtap_reset();
```

### Return value

Returns `cudaSuccess` on success. If an operation during reset fails, returns
its CUDA error and leaves the runtime unusable.
[`gtap_finalize`](#gtap-finalize) remains available for cleanup.

If GTaP is uninitialized or a previous reset or finalization failed, returns
`cudaErrorInitializationError` without changing the runtime state.

All failures are reported to standard error.

### Preconditions

- GTaP must already be initialized.
- No GTaP kernel may still be executing. Call
  [`gtap_synchronize()`](#gtap-synchronize) first.

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

Attempts to release all runtime-owned device and host memory and clears the
stored CUDA stream. On success, the runtime can be initialized again. If cleanup
fails, it cannot be reused or finalized again, and unreleased memory remains
allocated until the process exits.

### Signature

```cpp
cudaError_t gtap_finalize();
```

### Return value

Returns `cudaSuccess` on success, or the first CUDA error encountered during
cleanup.

If GTaP is uninitialized or a previous finalization failed, returns
`cudaErrorInitializationError` without releasing any resources.

All failures are reported to standard error.

### Preconditions

- The runtime is initialized, or a reset has failed.
- Complete outstanding execution with
  [`gtap_synchronize()`](#gtap-synchronize) before finalizing.
- Export required profiling data before finalization because the GTaP profiler
  buffers are released here.

```cpp
gtap_launch(exec_kernel, input);
gtap_synchronize();
gtap_finalize();
```

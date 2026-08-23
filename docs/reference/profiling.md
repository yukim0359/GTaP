# Profiling

Compile with `-DGTAP_ENABLE_PROFILING` to collect task-execution intervals in
either execution mode.

## `gtap_export_profile`

Copies recorded profiling data to the host, computes summary statistics, and
writes a result directory.

### Signature

```cpp
gtap_profile_export_result gtap_export_profile(
    const gtap_profile_export_options& options = {}
);
```

### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `options` | `const gtap_profile_export_options&` | Output location, optional label, and overwrite policy. Uses defaults when omitted. |

### Return value

Returns a [`gtap_profile_export_result`](#gtap-profile-export-result). Inspect
its `status` before using any generated path.

### Requirements

- Compile with `GTAP_ENABLE_PROFILING` to record data.
- Call `gtap_synchronize()` before exporting.
- Export before `gtap_reset()` or `gtap_finalize()`, which clear or release the
  GTaP profiler buffers.

### Example

```cpp
gtap_profile_export_result result = gtap_export_profile({
    .output_directory = "./profile/fib_%i",
    .overwrite = false,
});

if (result.status != gtap_profile_export_status::success) {
    fprintf(stderr, "Profile export failed: %s\n",
            gtap_profile_export_status_string(result.status));
}
```

## `gtap_profile_export_options`

```cpp
struct gtap_profile_export_options {
    const char* output_directory = "./profile";
    const char* label = nullptr;
    bool overwrite = false;
};
```

### Fields

| Field | Default | Description |
| --- | --- | --- |
| `output_directory` | `"./profile"` | Result directory. Parent directories are created when possible. One `%i` marker selects the first non-existing numbered directory. |
| `label` | `nullptr` | Optional label written to `profile.json` |
| `overwrite` | `false` | Allow the three generated output files to replace existing files |

A non-null label must contain 1–128 characters. Only ASCII letters, digits,
hyphen (`-`), underscore (`_`), and period (`.`) are accepted.

`output_directory` must be non-empty and fit within the 512-byte result path
buffers. At most one `%i` marker is allowed.

## `gtap_profile_export_result`

```cpp
struct gtap_profile_export_result {
    gtap_profile_export_status status;
    size_t recorded_intervals;
    size_t dropped_intervals;
    char result_directory[512];
    char profile_path[512];
    char intervals_path[512];
    char aggregates_path[512];
};
```

### Fields

| Field | Description |
| --- | --- |
| `status` | Final export status |
| `recorded_intervals` | Number of complete task-execution intervals written |
| `dropped_intervals` | Number of intervals omitted because a CUDA warp or thread block's profile buffer was full |
| `result_directory` | Resolved output directory |
| `profile_path` | Path to `profile.json` |
| `intervals_path` | Path to `task_execution_intervals.csv` |
| `aggregates_path` | Path to `task_execution_aggregates.csv` |

Paths are meaningful only when the corresponding stage of export succeeded.

## `gtap_profile_export_status`

```cpp
enum class gtap_profile_export_status {
    success = 0,
    invalid_label,
    invalid_output_directory,
    path_too_long,
    already_exists,
    cuda_error,
    out_of_memory,
    io_error,
    no_data,
    profiling_disabled
};
```

| Value | Meaning |
| --- | --- |
| `success` | All output files were written successfully |
| `invalid_label` | `label` is empty, too long, or contains unsupported characters |
| `invalid_output_directory` | The directory pattern is invalid or cannot be created/resolved |
| `path_too_long` | A generated output-file path does not fit in its 512-byte buffer |
| `already_exists` | An output file exists and `overwrite` is false |
| `cuda_error` | Copying or preparing profile data failed in CUDA |
| `out_of_memory` | A required host allocation failed |
| `io_error` | Opening, writing, or closing an output file failed |
| `no_data` | No complete task-execution interval was recorded |
| `profiling_disabled` | The program was compiled without `GTAP_ENABLE_PROFILING` |

Use `gtap_profile_export_status_string(status)` to obtain a printable lowercase
status name.

## Generated files

| File | Contents |
| --- | --- |
| `profile.json` | Mode, launch geometry, interval counts, dropped counts, and statistical summaries |
| `task_execution_intervals.csv` | Per-warp or per-block start and end times; thread mode also records the executed task count for each interval |
| `task_execution_aggregates.csv` | Per-warp or per-block totals and recorded execution span |

# Profiling

Compile with `-DGTAP_ENABLE_PROFILING` to collect task-execution intervals in
either execution mode.

## `gtap_export_profile`

Copies recorded profiling data to the host, computes summary statistics, and
writes a result directory.

### Signature

```cpp
bool gtap_export_profile(
    const gtap_profile_export_options& options = {},
    std::string* result_directory = nullptr
);
```

### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `options` | `const gtap_profile_export_options&` | Output location, optional label, and overwrite policy. Uses defaults when omitted. |
| `result_directory` | `std::string*` | Optional output parameter. Receives the resolved output directory on success; unchanged on failure. |

### Return value

Returns `true` when the profile was written. Otherwise,
returns `false` and reports the reason to standard error once, including when
profiling is disabled. Partial output files may remain on failure.

### Preconditions

- Export after [`gtap_launch`](./runtime-functions#gtap-launch) and a successful
  [`gtap_synchronize`](./runtime-functions#gtap-synchronize), and before
  [`gtap_reset`](./runtime-functions#gtap-reset) or
  [`gtap_finalize`](./runtime-functions#gtap-finalize). Reset and finalization
  clear or release the GTaP profiler buffers.

### Example

```cpp
std::string directory;
bool success = gtap_export_profile(
    {.output_directory = "./profile/fib_%i", .overwrite = false},
    &directory);
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

`output_directory` must be non-empty. At most one `%i` marker is allowed.

## Generated files

| File | Contents |
| --- | --- |
| `profile.json` | Statistical summaries of execution intervals and per-unit task-pool usage |
| `task_execution_intervals.csv` | Per-warp or per-block start and end times; thread mode also records the executed task count for each interval |
| `task_execution_aggregates.csv` | Per-warp or per-block total executed task count, execution time, and task-pool usage |

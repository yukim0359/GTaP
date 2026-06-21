# GTaP runtime headers

The public runtime entry points are:

- `gtap_thread.cuh` for thread-executed GTaP tasks.
- `gtap_block.cuh` for block-cooperative GTaP tasks.

Headers under `detail/` implement the compiler/runtime ABI used by generated
GTaP code. Types such as `TaskHeader`, `TaskContext`, `WarpTaskQueue`,
`TaskIdList`, and `GTaPResultHandle` are intentionally visible to generated CUDA
code, but they are not stable user-facing APIs.

The old implementation include paths under `common/`, `thread/`, and `block/`
remain as compatibility includes. New applications should include only the
public wrappers above.

Experimental scheduler variants are explicit opt-in wrappers:

- `experimental/gtap_thread_gq.cuh`
- `experimental/gtap_thread_chaselev.cuh`
- `experimental/gtap_block_gq.cuh`

The top-level `gtap_thread_gq.cuh`, `gtap_thread_chaselev.cuh`, and
`gtap_block_gq.cuh` headers remain only as compatibility includes for existing
benchmarks.

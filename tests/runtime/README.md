# GTaP GPU runtime regression tests

These tests compile small GTaP programs and execute them on a CUDA GPU.  Each
program computes its expected result independently and exits nonzero on a
runtime error or result mismatch.

Covered behavior:

- recursive thread tasks, child-result delivery, and `taskwait`;
- recursive block tasks, collective `taskwait`, and spawning-thread-only result
  delivery;
- transitive thread/block joins over a complete binary task tree;
- multiple suspension/resumption points and non-default spawn/resume queues.

Run the suite with:

```sh
make -C tests/runtime check CUDA_HOME=/path/to/cuda CUDA_ARCH=sm_90
```

`CLANG_BIN`, `GT_INC`, `CUDA_PATH`, and `CUDA_ARCH` can be overridden.  These
are GPU tests and are intentionally separate from Clang's compile-only `lit`
tests under `clang-gtap/clang/test/GTaP`.
